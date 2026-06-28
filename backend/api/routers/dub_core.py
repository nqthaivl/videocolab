import os
import re
import uuid
import asyncio
import logging
import shutil
import subprocess
import soundfile as sf
import torch
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Body, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from core.db import db_conn
from core.config import PREVIEW_DIR
from core.tasks import task_manager
from core import event_bus
from schemas.requests import DubIngestUrlRequest
from services.model_manager import get_model, _gpu_pool, _cpu_pool, get_diarization_pipeline, offload_tts_for_asr, restore_tts_after_asr
from services.audio_io import _safe_soundfile_write
from services.ffmpeg_utils import find_ffmpeg
from services.segmentation import (
    segment_transcript,
    assign_speakers_from_diarization,
    assign_speakers_from_turns,
    assign_speakers_heuristic,
    clean_up_segments,
)
from services.onset_align import snap_segment_starts, split_segments_at_silence, needs_silence_resplit
from services import dub_pipeline

router = APIRouter()
logger = logging.getLogger("omnivoice.api")

# ── Legacy-name aliases to services/dub_pipeline.py ────────────────────────
# Phase 2.4 moved the business logic into a service. Other routers
# (dub_generate, dub_translate, dub_export) + internal call sites below still
# reference the `_get_job` / `_save_job` / `_active_procs` names; those
# aliases let the transition happen without a repo-wide rename pass.
#
# New code should import from `services.dub_pipeline` directly. Aliases can
# disappear once every caller updates.
#
_dub_jobs           = dub_pipeline._dub_jobs
_active_procs       = dub_pipeline._active_procs
_active_procs_lock  = dub_pipeline._active_procs_lock
_DUB_DIR_REAL       = dub_pipeline._DUB_DIR_REAL

_compute_file_hash = dub_pipeline.compute_file_hash
_find_cached_job   = dub_pipeline.find_cached_job
_safe_job_dir      = dub_pipeline.safe_job_dir
_register_proc     = dub_pipeline.register_proc
_unregister_proc   = dub_pipeline.unregister_proc
_kill_job_procs    = dub_pipeline.kill_job_procs
_get_job           = dub_pipeline.get_job
_save_job          = dub_pipeline.save_job


@router.post("/dub/create-srt-job")
async def create_srt_job(
    job_id: Optional[str] = Body(None),
    filename: str = Body("movie.srt"),
    duration: float = Body(0.0),
):
    """Initialize an empty dummy job for SRT-based movie dubbing without a source video."""
    job_id = job_id or str(uuid.uuid4())[:8]
    job_dir = _safe_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=400, detail="Mã job_id không hợp lệ.")
    
    os.makedirs(job_dir, exist_ok=True)
    
    job = {
        "video_path": "",
        "audio_path": "",
        "vocals_path": "",
        "no_vocals_path": None,
        "thumb_path": None,
        "duration": duration,
        "filename": filename,
        "segments": [],
        "dubbed_tracks": {},
        "scene_cuts": [],
        "youtube_subs": None,
        "input_type": "audio",
    }
    
    dub_pipeline.put_job(job_id, job)
    _save_job(job_id, job, filename=filename, duration=duration)
    return {"job_id": job_id}


@router.post("/dub/import-srt/{job_id}")
async def dub_import_srt(job_id: str, file: UploadFile = File(...)):
    """Replace `job["segments"]` with timestamps + text parsed from an SRT
    file. Used as a fallback when Whisper mis-transcribes — the user can
    point at their own pre-synced subtitles and skip ASR entirely.

    Returns the new segment list plus counts of any cues we had to skip or
    re-time (overlap shifts). The caller surfaces these so the user knows
    if the import wasn't lossless.
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}") from e
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded SRT file is empty.")
    from services.srt_parser import _TIMING_RE, parse_srt
    
    # Try decoding: utf-8 (with/without BOM), utf-16 (with/without BOM), then fallback to latin-1.
    # We check if the decoded text actually contains the timing pattern to pick the correct encoding.
    text = None
    for enc in ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"]:
        try:
            t = raw_bytes.decode(enc)
            if _TIMING_RE.search(t):
                text = t
                break
        except UnicodeDecodeError:
            continue
            
    if text is None:
        # Fall back to latin-1 so legacy Windows-encoded subs don't blow up
        try:
            text = raw_bytes.decode("latin-1", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode file: {e}")

    result = parse_srt(text)
    if not result.segments:
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid cues found in the uploaded file. "
                f"Skipped {result.skipped_cues} malformed cue(s). "
                "Expected SubRip (.srt) format: index, then 'HH:MM:SS,ms --> HH:MM:SS,ms', then text, blank line."
            ),
        )

    # Clamp cues that run past the source media's known duration. Pipeline
    # downstream code assumes segment.end <= duration; without this, dub
    # generation would try to time-stretch into negative slack.
    duration = float(job.get("duration") or 0.0)
    clamped = 0
    if duration > 0:
        kept = []
        for seg in result.segments:
            if seg["start"] >= duration:
                continue
            if seg["end"] > duration:
                seg = {**seg, "end": round(duration, 3)}
                clamped += 1
            kept.append(seg)
        # Re-id after clamp drops.
        segments = [{**s, "id": i} for i, s in enumerate(kept)]
    else:
        segments = result.segments

    # SRT-only jobs are created with duration=0. Downstream mix/export needs a
    # non-zero timeline or the final dubbed WAV is empty (#48 guard).
    cue_end = max((float(s["end"]) for s in segments), default=0.0)
    if cue_end > 0:
        job["duration"] = max(duration, cue_end)

    job["segments"] = segments
    if not job.get("source_lang") or job.get("source_lang") in ("en", "auto"):
        from api.routers.dub_translate import _guess_lang_from_text

        class _Cue:
            def __init__(self, text: str):
                self.text = text

        guessed = _guess_lang_from_text([_Cue(s["text"]) for s in segments[:12]])
        if guessed:
            job["source_lang"] = guessed
    _save_job(job_id, job)
    logger.info(
        "Imported %d cue(s) from .srt for job %s (skipped=%d, overlap_shifted=%d, clamped=%d)",
        len(segments), job_id, result.skipped_cues, result.dropped_overlaps, clamped,
    )
    return {
        "segments": segments,
        "stats": {
            "imported": len(segments),
            "skipped_malformed": result.skipped_cues,
            "dropped_overlap": result.dropped_overlaps,
            "clamped_to_duration": clamped,
        },
    }


@router.post("/dub/detect-text/{job_id}")
async def detect_video_text_regions(
    job_id: str,
    max_frames: int = Query(16, ge=4, le=80, description="Max frames to analyze"),
    sample_fps: float = Query(0.5, ge=0.1, le=2.0, description="Frames per second to sample"),
):
    """OCR-style detection of on-screen text regions for blur overlay."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job id")
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    video_path = job.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail="Video not found for this job")

    from services.video_text_detect import detect_text_regions

    result = await detect_text_regions(video_path, max_frames=max_frames, sample_fps=sample_fps)
    if result.get("error") and not result.get("regions"):
        return result
    return result


@router.post("/dub/extract-ocr-subtitles/{job_id}")
async def dub_extract_ocr_subtitles(
    job_id: str,
    max_frames: int = Query(40, ge=8, le=120, description="Max frames to OCR"),
    sample_fps: float = Query(1.0, ge=0.2, le=3.0, description="Frames per second to sample"),
    language: Optional[str] = Query(None, description="Source language hint (auto/vi/zh/...)"),
):
    """Read burned-in on-screen subtitles via OCR → dub segments (skip ASR)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job id")
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    video_path = job.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail="Video not found for this job")

    from services.video_text_detect import extract_ocr_subtitle_segments
    from services.hunyuan_ocr import is_hunyuan_model_cached, is_hunyuan_model_loaded

    result = await extract_ocr_subtitle_segments(
        video_path,
        max_frames=max_frames,
        sample_fps=sample_fps,
        language=language,
    )
    result["model_cached"] = is_hunyuan_model_cached()
    result["model_loaded"] = is_hunyuan_model_loaded()
    segments = result.get("segments") or []
    if segments:
        job["segments"] = segments
        if segments:
            job["duration"] = max(float(job.get("duration") or 0), float(segments[-1]["end"]))
        _save_job(job_id, job)
    return result


@router.post("/dub/cleanup-segments/{job_id}")
def dub_cleanup_segments(job_id: str):
    """Re-run merge/stitch passes on a job's existing segments to drop fragments."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    segments = job.get("segments") or []
    cleaned = clean_up_segments(segments)
    job["segments"] = cleaned
    _save_job(job_id, job)
    return {"segments": cleaned, "before": len(segments), "after": len(cleaned)}


@router.post("/dub/abort/{job_id}")
def dub_abort(job_id: str):
    """Cancel in-flight upload/transcribe subprocesses for a job."""
    with _active_procs_lock:
        had_procs = bool(_active_procs.get(job_id))
    _kill_job_procs(job_id)
    job = _dub_jobs.get(job_id)
    if job is not None:
        job["aborted"] = True
    try:
        task_manager.cancel_task(job_id)
    except Exception:
        pass
    return {"aborted": True, "had_active_procs": had_procs}


@router.get("/dub/history")
def list_dub_history():
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM dub_history ORDER BY created_at DESC LIMIT 30").fetchall()
    return [dict(r) for r in rows]

@router.delete("/dub/history")
def clear_dub_history():
    """Delete persisted dub rows and their on-disk dirs (scoped to known IDs)."""
    with db_conn() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM dub_history").fetchall()]
        conn.execute("DELETE FROM dub_history")
    for jid in ids:
        safe = _safe_job_dir(jid)
        if safe and os.path.isdir(safe):
            shutil.rmtree(safe, ignore_errors=True)
    event_bus.emit("dub_history")
    return {"cleared": True, "count": len(ids)}

@router.delete("/dub/history/{history_id}")
def delete_single_dub_history(history_id: str):
    with db_conn() as conn:
        conn.execute("DELETE FROM dub_history WHERE id=?", (history_id,))
    safe = _safe_job_dir(history_id)
    if safe and os.path.isdir(safe):
        shutil.rmtree(safe, ignore_errors=True)
    _dub_jobs.pop(history_id, None)
    event_bus.emit("dub_history", {"action": "deleted", "id": history_id})
    return {"deleted": True}

@router.post("/preview/upload")
async def preview_upload(video: UploadFile = File(...)):
    ext = os.path.splitext(video.filename or "video.mp4")[1].lower()
    safe_name = f"{uuid.uuid4().hex[:12]}"
    vid_path = os.path.join(PREVIEW_DIR, f"{safe_name}{ext}")
    wav_path = os.path.join(PREVIEW_DIR, f"{safe_name}.wav")
    
    with open(vid_path, "wb") as f:
        f.write(await video.read())
        
    has_audio = False
    if ext not in [".wav", ".mp3", ".m4a", ".aac"]:
        try:
            ffmpeg_cmd = [
                find_ffmpeg(), "-y", "-i", vid_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                wav_path
            ]
            subprocess.run(
                ffmpeg_cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=300,
            )
            has_audio = True
        except Exception as e:
            logger.warning(f"FFmpeg extraction failed: {e}")
            pass

    return {
        "url": f"/preview/{safe_name}{ext}",
        "audioUrl": f"/preview/{safe_name}.wav" if has_audio else f"/preview/{safe_name}{ext}",
        "filename": video.filename,
    }

@router.get("/preview/{filename}")
async def preview_serve(filename: str):
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid preview filename")
    preview_real = os.path.realpath(PREVIEW_DIR)
    path = os.path.realpath(os.path.join(PREVIEW_DIR, filename))
    if not path.startswith(preview_real + os.sep):
        raise HTTPException(400, "Invalid preview filename")
    if not os.path.isfile(path):
        raise HTTPException(404, "Preview not found")
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        ".mp4": "video/mp4", ".mov": "video/quicktime", 
        ".mkv": "video/x-matroska", ".webm": "video/webm", 
        ".avi": "video/x-msvideo", ".wav": "audio/wav", 
        ".mp3": "audio/mpeg"
    }
    return FileResponse(path, media_type=media_types.get(ext, "application/octet-stream"))

# ── Legacy aliases for the extracted ingest pipeline (Phase 2.4 finish) ────
_run_proc_factory = dub_pipeline.run_proc_factory
_yt_download_sync = dub_pipeline.yt_download_sync
_prep_event       = dub_pipeline.prep_event
_ingest_gen       = dub_pipeline.ingest_pipeline


#: Recognised audio extensions for audio-only dubbing (#119). When the client
#: declares input_type=audio we refuse anything that isn't a known audio
#: container so a mislabelled video can't slip past the video-skipping branch.
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


@router.post("/dub/upload")
async def dub_upload(
    video: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
    input_type: str = Form("video"),
    skip_demucs: bool = Form(False),
    srt_mode: bool = Form(False),
):
    """Accept a media upload, write to disk, queue background prep task.

    `input_type` is "video" (default) or "audio". Audio-only jobs (#119) skip
    scene detection, thumbnailing, and the final video mux — the transcribe →
    translate → TTS core is identical.

    `srt_mode=true` (Clone phim + file SRT): skip Demucs + scene detection —
    only extract audio/video for mux; subtitles come from the uploaded SRT.

    Returns 202 with {job_id, task_id, filename}. Client should open SSE on
    /tasks/stream/{task_id} to monitor extract/demucs stages and wait for the
    'ready' event before starting transcription.
    """
    input_type = (input_type or "video").lower()
    if input_type not in ("video", "audio"):
        raise HTTPException(status_code=400, detail="input_type must be 'video' or 'audio'")

    if srt_mode:
        skip_demucs = True

    job_id = job_id or str(uuid.uuid4())[:8]
    job_dir = _safe_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid job_id. Must be alphanumeric + hyphens/underscores only, ≤64 chars. Generate a fresh job_id or omit it to auto-create one.",
        )
    ext = os.path.splitext(video.filename or "video.mp4")[1]
    if input_type == "audio" and ext.lower() not in _AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio-only dubbing needs an audio file ({', '.join(sorted(_AUDIO_EXTS))}); got '{ext or 'no extension'}'.",
        )

    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, f"original{ext}")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    filename = video.filename or f"video{ext}"
    task_id = f"prep_{job_id}"
    await task_manager.add_task(
        task_id, "prep",
        _ingest_gen, job_id, job_dir,
        {"kind": "file", "path": video_path, "input_type": input_type, "skip_demucs": skip_demucs, "srt_mode": srt_mode}, filename,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "task_id": task_id, "filename": filename},
    )


@router.post("/dub/ingest-url")
async def dub_ingest_url(req: DubIngestUrlRequest):
    """Ingest a remote video URL via yt-dlp. Queues background prep task.

    Returns 202 immediately with {job_id, task_id}. All work (download,
    audio extract, Demucs, scene detect, thumbnail) happens in the background
    task and progress is streamed via /tasks/stream/{task_id}.
    """
    url = (req.url or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://. Paste a full video link (e.g. https://youtube.com/watch?v=…) or drop a local file instead.",
        )

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="URL ingest needs yt-dlp, but it isn't installed. Install it (`pip install yt-dlp`) and restart the server — or drop a local video file instead.",
        )

    job_id = req.job_id or str(uuid.uuid4())[:8]
    job_dir = _safe_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid job_id. Must be alphanumeric + hyphens/underscores only, ≤64 chars. Generate a fresh job_id or omit it to auto-create one.",
        )
    os.makedirs(job_dir, exist_ok=True)

    task_id = f"prep_{job_id}"
    source = {
        "kind": "url",
        "url": url,
        "fetch_subs": bool(req.fetch_subs),
        "sub_langs": req.sub_langs or None,
    }
    await task_manager.add_task(
        task_id, "prep",
        _ingest_gen, job_id, job_dir,
        source, None,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "task_id": task_id, "filename": ""},
    )


TRANSCRIBE_CHUNK_S = float(os.environ.get("OMNIVOICE_TRANSCRIBE_CHUNK_S", "30.0"))
TRANSCRIBE_CHUNK_TIMEOUT_S = float(os.environ.get("OMNIVOICE_TRANSCRIBE_CHUNK_TIMEOUT_S", "120.0"))


_sse_event = dub_pipeline.sse_event
_prep_event_helper = dub_pipeline.prep_event  # alias; we keep the module-local _prep_event below for the inline one-liner shape


@router.get("/dub/transcribe-stream/{job_id}")
async def dub_transcribe_stream(
    job_id: str,
    num_speakers: Optional[int] = None,
    per_segment_refs: bool = True,
    language: Optional[str] = None,
):
    """Stream per-chunk segments via SSE, then emit diarized final pass.

    Pre-flight checks (missing job, missing audio, ASR not loaded) are emitted
    as in-stream `error` events rather than HTTP errors, because EventSource
    on the client can't read non-2xx response bodies — a 503 there surfaces
    as an opaque "network error" instead of the actionable message we want.

    `num_speakers` is an optional hint passed straight to pyannote. Left unset,
    pyannote auto-detects the count — but its auto-detect can collapse a
    multi-speaker clip to a single speaker (issue #274). When the user knows
    the exact count, supplying it forces pyannote to return that many speakers.
    """
    if language == "auto" or language == "":
        language = None

    # Clamp to a sane range; ignore anything non-positive / absurd so a bad
    # query string can never break the diarization call. None → auto-detect.
    if num_speakers is not None:
        try:
            num_speakers = int(num_speakers)
            num_speakers = num_speakers if 1 <= num_speakers <= 20 else None
        except (TypeError, ValueError):
            num_speakers = None

    job = _get_job(job_id)

    preflight_error: Optional[str] = None
    asr_audio_target: Optional[str] = None
    _asr_backend = None
    scene_cuts: list = []

    if not job:
        preflight_error = "Job not found. It may have been cleaned up or was never created."
    else:
        # Guard the model load: if it raises, the SSE stream would otherwise die
        # before emitting any event, and the UI shows a misleading generic
        # "stream dropped" message instead of the real cause (issue #255).
        try:
            _model = await get_model()
        except Exception as e:
            logger.exception("transcribe preflight: model load failed (job=%s)", job_id)
            from core.failure import build_failure
            f = build_failure(e, stage="transcribe-preflight", include_diagnostic=False)
            preflight_error = f["reason"] + (f" — {f['hint']}" if f.get("hint") else "")
            _model = None
        if _model is not None:
            asr_audio_target = job.get("vocals_path")
            if not asr_audio_target or not os.path.exists(asr_audio_target):
                asr_audio_target = job.get("audio_path")
            if not asr_audio_target or not os.path.exists(asr_audio_target):
                preflight_error = "No audio available for transcription."
            else:
                from services.asr_backend import get_active_asr_backend
                try:
                    # The PyTorch-Whisper backend lazily builds its own pipeline
                    # when no preloaded `_asr_pipe` is present (issue #255), so it
                    # no longer needs OMNIVOICE_PRELOAD_TTS_ASR=1 — don't reject it
                    # here; any load failure surfaces per-chunk with a real cause.
                    _asr_backend = get_active_asr_backend(asr_pipe=getattr(_model, "_asr_pipe", None))
                except Exception as e:
                    from core.failure import build_failure
                    f = build_failure(e, stage="transcribe-preflight", include_diagnostic=False)
                    preflight_error = "ASR backend initialization failed: " + f["reason"] + (
                        f" — {f['hint']}" if f.get("hint") else ""
                    )
                scene_cuts = job.get("scene_cuts") or []

    async def _gen_body():
        if preflight_error:
            yield _sse_event("error", {"detail": preflight_error})
            return
        import math
        import tempfile
        loop = asyncio.get_running_loop()

        def _load():
            audio_np, sr = sf.read(asr_audio_target, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            return audio_np, sr

        try:
            audio_np, sr = await loop.run_in_executor(_cpu_pool, _load)
        except Exception as e:
            yield _sse_event("error", {"detail": f"audio load failed: {e}"})
            return

        total = float(len(audio_np)) / float(sr) if sr else 0.0
        # FunASR: built-in VAD + sentence_info. Whole-file mode is most accurate
        # but blocks the SSE stream until the full clip finishes. For long
        # videos, chunk at ASR_FUNASR_CHUNK_S (default 90s when duration > 3min)
        # so segments appear progressively without the old 30s Whisper boundaries.
        funasr_chunk_s = 0.0
        if _asr_backend.id == "funasr":
            raw_chunk = os.environ.get("ASR_FUNASR_CHUNK_S", "").strip()
            if raw_chunk:
                funasr_chunk_s = max(0.0, float(raw_chunk))
            elif total > 180.0:
                funasr_chunk_s = 90.0
        use_funasr_whole_file = _asr_backend.id == "funasr" and funasr_chunk_s <= 0
        if _asr_backend.id == "funasr" and funasr_chunk_s > 0:
            chunks_n = max(1, int(math.ceil(total / funasr_chunk_s))) if total > 0 else 1
            chunk_s = funasr_chunk_s
        elif use_funasr_whole_file:
            chunks_n = 1
            chunk_s = total if total > 0 else TRANSCRIBE_CHUNK_S
        else:
            chunks_n = max(1, int(math.ceil(total / TRANSCRIBE_CHUNK_S))) if total > 0 else 1
            chunk_s = TRANSCRIBE_CHUNK_S
        yield _sse_event("start", {"duration": total, "chunks": chunks_n, "chunk_s": chunk_s})

        # Free VRAM: move TTS model to CPU so WhisperX + VAD can fit.
        # Only offloads when free GPU memory is < 4 GB (e.g. laptop GPUs).
        # Non-fatal: an offload failure must not drop the stream (#255) —
        # transcription can still proceed (it just has less headroom).
        try:
            await loop.run_in_executor(_cpu_pool, offload_tts_for_asr)
        except Exception as e:
            logger.warning("offload_tts_for_asr failed (continuing): %s", e)

        all_segments: list[dict] = []
        detected_lang = None
        lang_for_asr = language  # reuse after first auto-detect to skip re-detect per chunk
        next_seg_id = 0
        chunk_errors: list[str] = []
        # Speaker turns from an ASR backend that diarizes inline (FunASR cam++).
        # When present, _diarize() uses them and skips pyannote (Phase 2, #182).
        asr_speaker_turns: list[dict] = []

        for i in range(chunks_n):
            if job.get("aborted"):
                yield _sse_event("aborted", {})
                return
            if use_funasr_whole_file:
                t0, t1 = 0.0, total
                chunk_arr = None
            else:
                t0 = i * chunk_s
                t1 = min(total, t0 + chunk_s)
                s_from = int(t0 * sr)
                s_to = int(t1 * sr)
                chunk_arr = audio_np[s_from:s_to]
                if len(chunk_arr) == 0:
                    continue

            def _transcribe_chunk(arr=chunk_arr, offset=t0, local_sr=sr, whole_file=use_funasr_whole_file, lang=lang_for_asr):
                # Route through the active backend (WhisperX by default).
                # Backends all take a file path — write chunk wav unless FunASR
                # whole-file mode (VAD needs full context).
                try:
                    if whole_file:
                        audio_path = asr_audio_target
                    else:
                        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        tmp.close()
                        audio_path = tmp.name
                        _safe_soundfile_write(audio_path, arr, local_sr)
                    try:
                        r = _asr_backend.transcribe(audio_path, word_timestamps=True, language=lang)
                    finally:
                        if not whole_file:
                            try:
                                os.remove(audio_path)
                            except OSError:
                                pass
                    shifted = []
                    for c in r.get("chunks", []) or []:
                        ts = c.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
                        a0 = (ts[0] if ts[0] is not None else 0.0) + offset
                        a1 = (ts[1] if ts[1] is not None else 0.0) + offset
                        shifted.append({"text": c.get("text", ""), "timestamp": (a0, a1)})
                    # Shift segments by offset so word-level timestamps (e.g. from WhisperX)
                    # are preserved and offset correctly.
                    shifted_segments = []
                    for seg in r.get("segments", []) or []:
                        s0 = seg.get("start")
                        s1 = seg.get("end")
                        seg_start = (s0 if s0 is not None else 0.0) + offset
                        seg_end = (s1 if s1 is not None else 0.0) + offset
                        shifted_words = []
                        for w in seg.get("words", []) or []:
                            w_word = w.get("word") or w.get("text") or ""
                            w_start = w.get("start")
                            w_end = w.get("end")
                            if w_start is not None:
                                w_start = w_start + offset
                            if w_end is not None:
                                w_end = w_end + offset
                            shifted_words.append({
                                "word": w_word,
                                "start": w_start,
                                "end": w_end,
                                "probability": w.get("probability", 1.0)
                            })
                        shifted_segments.append({
                            "text": seg.get("text", ""),
                            "start": seg_start,
                            "end": seg_end,
                            "words": shifted_words
                        })
                    # Inline-diarization speaker turns (FunASR cam++), offset-shifted
                    # to the full-audio timeline so _diarize() can use them.
                    turns = []
                    for seg in r.get("segments", []) or []:
                        spk = seg.get("speaker")
                        s0, s1 = seg.get("start"), seg.get("end")
                        if spk is None or s0 is None or s1 is None:
                            continue
                        turns.append({"start": s0 + offset, "end": s1 + offset, "speaker": spk})
                    return {
                        "chunks": shifted,
                        "segments": shifted_segments,
                        "language": r.get("language"),
                        "speaker_turns": turns
                    }
                except Exception as e:
                    logger.exception("chunk transcribe failed (backend=%s)", _asr_backend.id)
                    return {"chunks": [], "language": None, "error": str(e)}

            try:
                # wait_for in a loop to yield pings so the EventSource connection doesn't drop
                fut = loop.run_in_executor(_gpu_pool, _transcribe_chunk)
                waited = 0.0
                part = None
                chunk_timeout = (
                    max(TRANSCRIBE_CHUNK_TIMEOUT_S, total * 3.0)
                    if use_funasr_whole_file
                    else TRANSCRIBE_CHUNK_TIMEOUT_S
                )
                while True:
                    done, pending = await asyncio.wait([fut], timeout=5.0)
                    if done:
                        part = done.pop().result()
                        break
                    yield _sse_event("ping", {})
                    waited += 5.0
                    if waited >= chunk_timeout:
                        # Re-raise TimeoutError if we exceed the overall limit
                        raise asyncio.TimeoutError()
            except asyncio.TimeoutError:
                logger.error(
                    "Transcribe chunk %d/%d timed out after %.0fs (job=%s)",
                    i + 1, chunks_n, chunk_timeout, job_id,
                )
                part = {
                    "chunks": [], "language": None,
                    "error": f"Chunk {i+1} timed out after {chunk_timeout:.0f}s — "
                             f"ASR backend may be stuck. Try restarting the server.",
                }
            if part.get("error"):
                chunk_errors.append(part["error"])
                logger.warning("Chunk %d/%d error: %s", i + 1, chunks_n, part["error"])
            if detected_lang is None and part.get("language"):
                detected_lang = part["language"]
                lang_for_asr = (detected_lang.split("_")[0][:2] or detected_lang).lower()
            asr_speaker_turns.extend(part.get("speaker_turns") or [])
            chunk_segs = segment_transcript(part, duration=t1, scene_cuts=scene_cuts)
            # #280: Whisper often stretches a segment's start back over
            # leading music/silence (classic case: speech begins at 0:03,
            # transcript says 0.0 → the dub plays 3 s early). Snap starts
            # forward to the actual speech onset. `audio_np` is the same
            # track ASR ran on — vocals.wav when Demucs succeeded.
            try:
                snap_segment_starts(chunk_segs, audio_np, sr)
            except Exception as e:
                logger.warning("onset alignment skipped for chunk %d: %s", i, e)
            if needs_silence_resplit(chunk_segs):
                try:
                    chunk_segs = split_segments_at_silence(chunk_segs, audio_np, sr)
                except Exception as e:
                    logger.warning("silence split skipped for chunk %d: %s", i, e)
            chunk_segs = assign_speakers_heuristic(chunk_segs)
            for s in chunk_segs:
                s["id"] = f"s{next_seg_id:05x}"
                s["text_original"] = s.get("text", "")
                next_seg_id += 1
            all_segments.extend(chunk_segs)
            yield _sse_event("segments", {
                "chunk": i, "total_chunks": chunks_n,
                "segments": chunk_segs,
                "progress": (i + 1) / chunks_n,
                "error": part.get("error"),
            })

        if job.get("aborted"):
            yield _sse_event("aborted", {})
            return

        # Empty-transcription guard: if every chunk came back with zero
        # segments we can't proceed to diarization/clone extraction. Emit an
        # actionable error so the UI can surface a Retry instead of silently
        # landing in an empty editor. Commonly caused by a first-run model
        # download failure, a PyTorch 2.6 weights_only regression inside
        # whisperx's VAD load, or an unsupported audio format.
        if not all_segments:
            # Deduplicate while preserving order so one root cause doesn't
            # repeat N times in the UI toast. Sanitize each message so home
            # paths / tokens from a backend traceback never leak (#255).
            from core.failure import sanitize, build_failure
            seen = set()
            uniq: list[str] = []
            for msg in chunk_errors:
                s = sanitize(msg)
                if s and s not in seen:
                    seen.add(s)
                    uniq.append(s)
            if uniq:
                detail = "Transcription produced no segments. " + " | ".join(uniq[:3])
                # Add the actionable hint for a recognized failure class
                # (e.g. pkg_resources missing → install setuptools).
                hint = build_failure(" ".join(uniq), stage="transcribe", include_diagnostic=False).get("hint")
                if hint:
                    detail += f" — {hint}"
            else:
                detail = (
                    "Transcription produced no segments. The audio may be silent, "
                    "too short, or in an unsupported format. Try re-uploading or "
                    "check that the source has an audible speech track."
                )
            logger.error("transcribe yielded 0 segments (job=%s): %s", job_id, detail)
            yield _sse_event("error", {"detail": detail, "retryable": True})
            yield _sse_event("done", {})
            return

        def _diarize():
            """Returns (segments, warning_payload_or_None).

            `warning_payload` is a structured dict
            `{detail, error_class, docs_url}` whenever we silently fell back
            to the silence-gap heuristic (no HF_TOKEN, model unavailable,
            license not accepted, or pyannote raised). The heuristic only
            detects speaker turns from >1.2s silences, so a rapid-fire
            man↔woman exchange will read as one speaker. Issue #78 — we
            attach an `error_class` so the front-end's errorDocsMap can
            render a "See docs" deeplink instead of a dead-end toast.
            """
            # The active ASR backend already diarized inline (FunASR cam++):
            # use its speaker turns directly and skip pyannote entirely (#182).
            if asr_speaker_turns:
                logger.info("Using inline ASR diarization (%d turns); skipping pyannote.", len(asr_speaker_turns))
                return assign_speakers_from_turns(all_segments, asr_speaker_turns), None

            from services.model_manager import (
                DIARIZATION_ERR_LICENSE,
                DIARIZATION_ERR_NO_TOKEN,
            )
            from core import error_docs_map

            diar_pipe, err_sentinel = get_diarization_pipeline(return_error=True)
            if not diar_pipe:
                # Phase 1 AUTH-01: ask the resolver (App → Env → HF-CLI),
                # not just the env var. This is the #35 fix — users who
                # ran `huggingface-cli login` previously saw the "no
                # HF_TOKEN" branch even though the library would have
                # read the token. Now the cascade is honoured.
                from services import token_resolver
                resolved = token_resolver.resolve()

                if err_sentinel == DIARIZATION_ERR_NO_TOKEN or not resolved:
                    detail = (
                        "Speaker diarization is disabled because no HuggingFace token "
                        "was found in any source (Settings → API Keys, the HF_TOKEN "
                        "env var, or ~/.cache/huggingface/token from `huggingface-cli "
                        "login`). To detect multiple speakers, set a token in one of "
                        "those places and accept the pyannote/speaker-diarization-3.1 "
                        "license at huggingface.co. Falling back to a silence-gap "
                        "heuristic — turns with no audible pause between them will "
                        "be merged into one speaker."
                    )
                    error_class = "HF_AUTH_FAILED"
                elif err_sentinel == DIARIZATION_ERR_LICENSE:
                    who = resolved.username or "(whoami suppressed)"
                    detail = (
                        f"Speaker diarization model is gated — the "
                        f"pyannote/speaker-diarization-3.1 license has not been "
                        f"accepted on HuggingFace by this account "
                        f"(source={resolved.source}, user={who}). Visit "
                        f"huggingface.co/pyannote/speaker-diarization-3.1 AND "
                        f"huggingface.co/pyannote/segmentation-3.0 while signed "
                        f"in and click 'Agree and access repository' on both, "
                        f"then restart this dub job. Falling back to a "
                        f"silence-gap heuristic; rapid speaker turns may be "
                        f"merged into one speaker."
                    )
                    error_class = "PYANNOTE_LICENSE_REQUIRED"
                else:
                    # err_sentinel == DIARIZATION_ERR_LOAD (or unexpected None
                    # with a resolved token — historical safety net).
                    who = resolved.username or "(whoami suppressed)"
                    detail = (
                        f"Speaker diarization model failed to load even though an HF "
                        f"token was found (source={resolved.source}, user={who}). "
                        f"Most common causes: the pyannote/speaker-diarization-3.1 "
                        f"license has not been accepted on HuggingFace, or there is "
                        f"a pyannote/torch version mismatch. See backend logs for "
                        f"the underlying error. Falling back to a silence-gap "
                        f"heuristic; rapid speaker turns may be merged."
                    )
                    error_class = "PYANNOTE_LICENSE_REQUIRED"
                return (
                    assign_speakers_heuristic(all_segments),
                    {
                        "detail": detail,
                        "error_class": error_class,
                        "docs_url": error_docs_map.lookup(error_class),
                    },
                )
            try:
                # Pass the user's speaker-count hint through to pyannote when
                # provided (#274). pyannote's apply() accepts num_speakers;
                # omit it entirely when None so we don't depend on the kwarg
                # existing in every pyannote build.
                if num_speakers:
                    logger.info("Diarizing with num_speakers=%d (user hint)", num_speakers)
                    diar = diar_pipe(asr_audio_target, num_speakers=num_speakers)
                else:
                    diar = diar_pipe(asr_audio_target)
                return assign_speakers_from_diarization(all_segments, diar), None
            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                # Mid-run failure — classify against the same sentinels so a
                # post-load 401 (rare but possible after a token rotation)
                # still gets the right docs deeplink.
                from services.model_manager import _classify_diarization_error
                err_class_post = _classify_diarization_error(e)
                error_class = (
                    "PYANNOTE_LICENSE_REQUIRED"
                    if err_class_post == DIARIZATION_ERR_LICENSE
                    else "PYANNOTE_LICENSE_REQUIRED"  # LOAD failures land here too
                )
                return (
                    assign_speakers_heuristic(all_segments),
                    {
                        "detail": (
                            f"Speaker diarization crashed mid-run "
                            f"({type(e).__name__}); falling back to a silence-gap "
                            f"heuristic. Rapid speaker turns may be merged."
                        ),
                        "error_class": error_class,
                        "docs_url": error_docs_map.lookup(error_class),
                    },
                )

        fut_diar = loop.run_in_executor(_gpu_pool, _diarize)
        final_segs = None
        diar_warning = None
        while True:
            done, pending = await asyncio.wait([fut_diar], timeout=5.0)
            if done:
                final_segs, diar_warning = done.pop().result()
                break
            yield _sse_event("ping", {})
        if diar_warning:
            logger.warning("diarization fallback: %s", diar_warning.get("detail"))
            yield _sse_event("warning", {
                "detail": diar_warning.get("detail"),
                "source": "diarization",
                "error_class": diar_warning.get("error_class"),
                "docs_url": diar_warning.get("docs_url"),
            })

        job["segments"] = final_segs

        # Auto-speaker-clone: sample each detected speaker's voice from the
        # Demucs-isolated vocals track and assign `auto:speaker_N` as the
        # default profile for their segments. This is what lets a user add a
        # new target language and have the ORIGINAL speaker speak it — the
        # central pro-grade dubbing promise.
        try:
            from services.speaker_clone import extract_speaker_clones, auto_profile_id
            vocals_for_clone = job.get("vocals_path") or asr_audio_target
            fut_clones = loop.run_in_executor(
                _cpu_pool, extract_speaker_clones,
                vocals_for_clone, final_segs, os.path.dirname(vocals_for_clone),
            )
            clones = None
            while True:
                done, pending = await asyncio.wait([fut_clones], timeout=5.0)
                if done:
                    clones = done.pop().result()
                    break
                yield _sse_event("ping", {})
            # Wave 3.2: per-segment clone refs. Cut each long-enough segment's
            # own reference from the vocals so the dub of each line matches the
            # prosody of its source line. Short lines fall back to the
            # per-speaker clone below. Default on; the user can force
            # per-speaker by disabling it (job["per_segment_refs"]).
            seg_clones = {}
            job["per_segment_refs"] = per_segment_refs
            if per_segment_refs:
                try:
                    from services.speaker_clone import extract_segment_refs
                    seg_ids_for_clone = [s.get("id", i) for i, s in enumerate(final_segs)]
                    seg_clones = await loop.run_in_executor(
                        _cpu_pool, lambda: extract_segment_refs(
                            vocals_for_clone, final_segs,
                            os.path.dirname(vocals_for_clone),
                            seg_ids=seg_ids_for_clone,
                        ),
                    )
                    if seg_clones:
                        job["segment_clones"] = seg_clones
                except Exception as e:
                    logger.warning("per-segment clone refs skipped: %s", e)

            if clones or seg_clones:
                if clones:
                    job["speaker_clones"] = clones
                # Default each segment's profile_id to its detected speaker's
                # auto-clone — but only if the user hasn't already assigned
                # something. (#486)
                #
                # We prefer the UI-visible `auto:{speaker}` id over the
                # per-segment `auto-seg:{id}` id even when a per-segment ref
                # exists, because the dub editor's Voice dropdown only renders
                # `auto:` options ("From Video → Speaker N"). An `auto-seg:`
                # value matches no <option>, so the row silently read
                # "Default" while the speaker was actually bound — exactly the
                # reported bug. The per-segment ref is NOT lost: dub_generate's
                # `auto:` branch transparently prefers this segment's own
                # per-segment ref (job["segment_clones"][seg_id]) when present,
                # so a row shown as "Speaker 1" still clones from its own line
                # when that line is long enough.
                for s in final_segs:
                    if s.get("profile_id"):
                        continue
                    spk = s.get("speaker_id") or "Speaker 1"
                    if spk in clones:
                        s["profile_id"] = auto_profile_id(spk)
                        continue
                    # No per-speaker clone for this speaker (too little usable
                    # audio overall) but this single line was long enough for
                    # its own ref — fall back to the per-segment id. The editor
                    # can't render it, but generation still clones correctly.
                    sid = str(s.get("id", ""))
                    if sid and sid in seg_clones:
                        s["profile_id"] = f"auto-seg:{sid}"
        except Exception as e:
            logger.warning("speaker_clone extraction skipped: %s", e)

        job["source_lang"] = ((detected_lang or "en").split("_")[0][:2] or "en").lower()
        job["full_transcript"] = " ".join(s.get("text", "") for s in final_segs)
        _save_job(job_id, job)

        # Restore TTS model to GPU now that ASR is done
        if _asr_backend:
            try:
                _asr_backend.unload()
            except Exception as e:
                logger.warning("Failed to unload ASR backend: %s", e)

        await loop.run_in_executor(_cpu_pool, restore_tts_after_asr)

        if torch.backends.mps.is_available():
            try: torch.mps.empty_cache()
            except Exception: pass

        yield _sse_event("final", {
            "segments": final_segs,
            "source_lang": job["source_lang"],
            "full_transcript": job["full_transcript"],
            "speaker_clones": job.get("speaker_clones", {}),
        })
        yield _sse_event("done", {})

    async def gen():
        # Terminal-event guard (#516): the SSE stream must NEVER close without a
        # terminal event. Any unanticipated exception in the body (e.g. an ASR
        # load that escapes the per-chunk handler) previously dropped the
        # connection, which the frontend can only report as "stream dropped,
        # likely ASR failed" — hiding the real cause. Emit a structured `error`
        # (with the actionable hint from build_failure) then `done`, so the user
        # sees the real failure + a Retry instead of a silent disconnect.
        try:
            async for ev in _gen_body():
                yield ev
        except Exception as e:  # noqa: BLE001 — last-resort stream finalizer
            logger.exception("transcribe stream crashed (job=%s)", job_id)
            from core.failure import build_failure
            f = build_failure(e, stage="transcribe", include_diagnostic=False)
            detail = f["reason"] + (f" — {f['hint']}" if f.get("hint") else "")
            yield _sse_event("error", {"detail": detail, "retryable": True})
            yield _sse_event("done", {})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/dub/transcribe/{job_id}")
async def dub_transcribe(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    _model = await get_model()

    def _transcribe():
        
        asr_audio_target = job.get("vocals_path")
        if not asr_audio_target or not os.path.exists(asr_audio_target):
            asr_audio_target = job.get("audio_path")
            
        import torch

        detected_lang = None

        # Route through services.asr_backend — picks WhisperX / faster-whisper
        # / mlx / pytorch based on what's installed + user preference. Works
        # identically on all platforms; the older mlx-vs-pytorch branching
        # here duplicated the logic in asr_backend.py and skipped WhisperX.
        from services.asr_backend import get_active_asr_backend
        _asr = get_active_asr_backend(asr_pipe=getattr(_model, "_asr_pipe", None))
        try:
            try:
                logger.info("Transcribing full audio via %s ...", _asr.id)
                result = _asr.transcribe(asr_audio_target, word_timestamps=True)
                detected_lang = result.get("language")
            except Exception as e:
                logger.error("ASR backend %s failed: %s", _asr.id, e)
                if getattr(_model, "_asr_pipe", None) is None:
                    raise RuntimeError(
                        f"ASR backend {_asr.id} failed and PyTorch Whisper fallback is not preloaded: {e}"
                    ) from e
                # Last-resort fallback — in-memory pytorch whisper via the TTS
                # model's pipeline when explicitly preloaded.
                audio_np, sr = sf.read(asr_audio_target, dtype="float32")
                if audio_np.ndim > 1: audio_np = audio_np.mean(axis=1)
                bs = 16 if torch.cuda.is_available() else 1
                result = _model._asr_pipe(
                    {"array": audio_np, "sampling_rate": sr},
                    return_timestamps=True, chunk_length_s=15, batch_size=bs,
                )
                detected_lang = (result.get("language") if isinstance(result, dict) else None)
        finally:
            try:
                _asr.unload()
            except Exception as e:
                logger.warning("Failed to unload ASR backend: %s", e)

        job["source_lang"] = (detected_lang or "en").split("_")[0][:2].lower()

        scene_cuts = job.get("scene_cuts") or []
        segments = segment_transcript(result, duration=job.get("duration", 0.0), scene_cuts=scene_cuts)

        # #280: snap segment starts forward to the actual speech onset so the
        # dub doesn't begin seconds before the original speaker does.
        try:
            audio_for_onset, onset_sr = sf.read(asr_audio_target, dtype="float32")
            snap_segment_starts(segments, audio_for_onset, onset_sr)
            if needs_silence_resplit(segments):
                segments = split_segments_at_silence(segments, audio_for_onset, onset_sr)
        except Exception as e:
            logger.warning("onset alignment skipped: %s", e)

        diar_pipe = get_diarization_pipeline()
        if diar_pipe:
            try:
                diar_target = job.get("vocals_path") or job.get("audio_path")
                diarization = diar_pipe(diar_target)
                segments = assign_speakers_from_diarization(segments, diarization)
            except Exception as e:
                logger.error(f"Pyannote diarization failed during inference: {e}. Falling back to heuristic.")
                segments = assign_speakers_heuristic(segments)
        else:
            segments = assign_speakers_heuristic(segments)

        # Previously ran `segment_for_subtitles(segments)` here. Removed 2026-04-21 —
        # that splitter enforces Netflix's 17 CPS reading-speed ceiling which
        # trips on normal speech (15–25 CPS) and recurses to word-level.
        # For dubbing, keep the sentence-level output. Apply subtitle rules at
        # SRT export time only.

        for s in segments:
            s.setdefault("text_original", s.get("text", ""))
        job["full_transcript"] = " ".join(s["text"] for s in segments)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return segments

    try:
        loop = asyncio.get_running_loop()
        try:
            segments_result = await loop.run_in_executor(_gpu_pool, _transcribe)
        except asyncio.CancelledError:
            job["aborted"] = True
            raise
        if job.get("aborted"):
            raise HTTPException(status_code=499, detail="Transcription aborted")
        job["segments"] = segments_result
        source_lang = job.get("source_lang")
        _save_job(job_id, job)
        return {
            "job_id": job_id,
            "segments": segments_result,
            "full_transcript": job.get("full_transcript", ""),
            "source_lang": source_lang,
        }
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
