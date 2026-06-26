"""Batch dubbing queue — POST videos with settings, process sequentially.

This is a lightweight batch orchestrator. Each job is a dub project that
runs through the same ingest→transcribe→translate→generate pipeline as
a manual dub, but driven by the queue instead of the UI.

The queue is in-memory (lives for the process lifetime). Jobs persist to
the SQLite `jobs` table for history, but the queue itself restarts empty
on backend restart — intentional, since GPU jobs can't be safely resumed.
"""
import os
import uuid
import time
import asyncio
import logging
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

from core.config import DATA_DIR
from core import failure

router = APIRouter()
logger = logging.getLogger("omnivoice.batch")

# ── In-memory queue ─────────────────────────────────────────────────────

_queue: asyncio.Queue = None       # Lazily initialised
_worker_task: asyncio.Task = None  # Background consumer
_jobs: dict = {}                   # job_id → status dict


class BatchJobStatus(BaseModel):
    id: str
    batch_group_id: Optional[str] = None
    status: str  # "queued" | "running" | "done" | "failed" | "cancelled"
    filename: str
    langs: List[str]
    voice_id: Optional[str] = None
    timing_strategy: Optional[str] = "concise"
    preserve_bg: bool = True
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    progress: Optional[dict] = None


class LocalBatchRequest(BaseModel):
    input_dir: str
    output_dir: str
    langs: str
    voice_id: Optional[str] = None
    translation_provider: Optional[str] = "google"
    timing_strategy: Optional[str] = "concise"
    preserve_bg: bool = True


def _ensure_queue():
    """Lazy-init the asyncio queue + worker on first use."""
    global _queue, _worker_task
    if _queue is None:
        _queue = asyncio.Queue()
        _worker_task = asyncio.ensure_future(_worker())


async def _worker():
    """Process jobs one at a time from the queue."""
    while True:
        job_id = await _queue.get()
        job = _jobs.get(job_id)
        if not job or job["status"] == "cancelled":
            _queue.task_done()
            continue

        job["status"] = "running"
        job["started_at"] = time.time()
        logger.info("Batch job %s starting: %s", job_id, job["filename"])

        try:
            await _run_batch_pipeline(job_id, job)
            if job["status"] != "cancelled":
                job["status"] = "done"
                job["finished_at"] = time.time()
                logger.info(
                    "Batch job %s completed in %.1fs",
                    job_id, job["finished_at"] - job["started_at"],
                )
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
        except Exception as e:
            job["status"] = "failed"
            # plan-04 (#131): guaranteed non-empty, structured reason.
            job["error"] = failure.build_failure(e, stage="batch", include_diagnostic=False)["reason"]
            job["finished_at"] = time.time()
            logger.error("Batch job %s failed: %s", job_id, e, exc_info=True)
        finally:
            _queue.task_done()


def _set_progress(job, stage, percent=0, **extra):
    """Update a job's progress dict."""
    job["progress"] = {"stage": stage, "percent": percent, **extra}


async def _run_batch_pipeline(job_id: str, job: dict):
    """Full batch dub pipeline: extract → transcribe → translate → generate → mix → export."""
    import subprocess

    loop = asyncio.get_running_loop()
    video_path = job["video_path"]
    langs = job["langs"]
    batch_dir = os.path.join(DATA_DIR, "batch", job_id)
    os.makedirs(batch_dir, exist_ok=True)

    # ── 1. Extract audio ──────────────────────────────────────────────
    _set_progress(job, "extract", 0)
    audio_path = os.path.join(batch_dir, "audio.wav")

    from services.ffmpeg_utils import find_ffmpeg
    ffmpeg = find_ffmpeg()

    def _extract():
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
             audio_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=300, check=True,
        )
        # Get duration
        result = subprocess.run(
            [ffmpeg, "-i", audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30,
        )
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr.decode("utf-8", errors="replace"))
        if match:
            h, m, s, cs = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
        return 0.0

    duration = await loop.run_in_executor(None, _extract)
    job["duration"] = duration
    _set_progress(job, "extract", 100)

    if job["status"] == "cancelled":
        return

    # ── 2. Transcribe ─────────────────────────────────────────────────
    _set_progress(job, "transcribe", 0)

    from services.asr_backend import get_active_asr_backend
    from services.model_manager import _gpu_pool, _cpu_pool
    from services.segmentation import (
        segment_transcript, assign_speakers_heuristic,
    )

    def _transcribe():
        backend = get_active_asr_backend()
        result = backend.transcribe(audio_path, word_timestamps=True)
        detected_lang = result.get("language", "en")
        segments = segment_transcript(result, duration=duration)
        segments = assign_speakers_heuristic(segments)
        for i, s in enumerate(segments):
            s["id"] = f"s{i:05x}"
            s.setdefault("text_original", s.get("text", ""))
        try:
            backend.unload()
        except Exception:
            pass
        return segments, detected_lang

    segments, source_lang = await loop.run_in_executor(_gpu_pool, _transcribe)
    source_lang = (source_lang or "en").split("_")[0][:2].lower()
    job["segments"] = segments
    job["source_lang"] = source_lang
    _set_progress(job, "transcribe", 100, segments_count=len(segments))

    if job["status"] == "cancelled" or not segments:
        if not segments:
            job["error"] = "Transcription produced no segments"
            job["status"] = "failed"
        return

    # ── 2b. Extract speaker clones if voice_id is not set ────────────────
    clones = {}
    if not job.get("voice_id"):
        try:
            from services.speaker_clone import extract_speaker_clones
            clones = await loop.run_in_executor(
                _cpu_pool, extract_speaker_clones,
                audio_path, segments, batch_dir,
            )
            logger.info("Batch speaker clones extracted: %s", list(clones.keys()))
        except Exception as e:
            logger.warning("Batch speaker clone extraction skipped: %s", e)

    # ── 3. Translate + Generate per language ───────────────────────────
    total_langs = len(langs)
    outputs = {}

    for lang_idx, target_lang in enumerate(langs):
        if job["status"] == "cancelled":
            return

        # ── 3a. Translate ─────────────────────────────────────────────
        _set_progress(
            job, "translate",
            percent=int((lang_idx / total_langs) * 100),
            current_lang=target_lang,
        )

        translated_segments = list(segments)  # copy
        if target_lang != source_lang:
            try:
                from api.routers.dub_translate import dub_translate
                from schemas.requests import TranslateRequest, TranslateSegment

                req_segs = [
                    TranslateSegment(id=s["id"], text=s.get("text_original") or s.get("text") or "", target_lang=target_lang)
                    for s in segments
                ]
                translate_req = TranslateRequest(
                    segments=req_segs,
                    target_lang=target_lang,
                    provider=job.get("translation_provider", "google"),
                    source_lang=source_lang,
                    job_id=job_id,
                    quality="fast"
                )

                translate_res = await dub_translate(translate_req)
                translated_map = {str(item["id"]): item["text"] for item in translate_res["translated"]}

                translated_segments = []
                for s in segments:
                    s_copy = dict(s)
                    s_copy["text"] = translated_map.get(str(s["id"]), s.get("text_original") or s.get("text"))
                    translated_segments.append(s_copy)
            except Exception as e:
                logger.exception("Translation failed for %s: %s, using original", target_lang, e)
                translated_segments = segments

        if job["status"] == "cancelled":
            return

        # ── 3b. Generate TTS ──────────────────────────────────────────
        _set_progress(
            job, "generate",
            percent=int((lang_idx / total_langs) * 100),
            current_lang=target_lang,
            current_segment=0,
            total_segments=len(translated_segments),
        )

        from services.model_manager import get_model
        from services.audio_dsp import apply_mastering, normalize_audio
        from services.audio_io import atomic_save_wav
        import torch

        _model = await get_model()
        sr = _model.sampling_rate
        total_samples = int(duration * sr)
        full_audio = torch.zeros(1, total_samples)
        total_segs = len(translated_segments)

        strategy = (job.get("timing_strategy") or "concise").lower()

        for i, seg in enumerate(translated_segments):
            if job["status"] == "cancelled":
                return

            _set_progress(
                job, "generate",
                percent=int(((lang_idx + (i / total_segs)) / total_langs) * 100),
                current_lang=target_lang,
                current_segment=i + 1,
                total_segments=total_segs,
            )

            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_duration = seg_end - seg_start
            seg_text = seg.get("text", "").strip()

            if seg_duration <= 0.05 or not seg_text:
                continue

            dur_for_tts = seg_duration if strategy == "strict_slot" else None

            def _gen(text=seg_text, lang=target_lang, dur=dur_for_tts):
                ref_audio = None
                ref_text = None

                # Use voice_id if provided
                if job.get("voice_id"):
                    from core.db import db_conn
                    from core.config import VOICES_DIR as _VD
                    with db_conn() as conn:
                        row = conn.execute(
                            "SELECT * FROM voice_profiles WHERE id=?",
                            (job["voice_id"],),
                        ).fetchone()
                    if row:
                        if row["is_locked"] and row["locked_audio_path"]:
                            ref_audio = os.path.join(_VD, row["locked_audio_path"])
                        elif row["ref_audio_path"]:
                            ref_audio = os.path.join(_VD, row["ref_audio_path"])
                        ref_text = row.get("ref_text")
                else:
                    # Use speaker clone if available
                    spk = seg.get("speaker_id") or "Speaker 1"
                    if spk in clones:
                        ref_audio = clones[spk].get("ref_audio")
                        ref_text = clones[spk].get("ref_text")

                try:
                    audios = _model.generate(
                        text=text, language=lang,
                        ref_audio=ref_audio, ref_text=ref_text,
                        duration=dur, num_step=16,
                        guidance_scale=2.0, speed=1.0,
                        denoise=True, postprocess_output=True,
                    )
                    audio_out = audios[0]
                    # TODO(#312): this route runs the OmniVoice model directly (not the active
                    # backend), so VoxCPM2 never reaches it. When these routes become
                    # engine-aware, guard with `if not getattr(backend, "applies_own_mastering", False)`.
                    mastered = apply_mastering(
                        audio_out,
                        sample_rate=sr,
                    )
                    return normalize_audio(mastered, target_dBFS=-2.0)
                except Exception as e:
                    logger.warning("TTS failed for seg %d (lang=%s): %s", i, lang, e)
                    return torch.zeros(1, int((dur or seg_duration) * sr))

            try:
                audio_tensor = await loop.run_in_executor(_gpu_pool, _gen)

                # Fit to slot based on timing_strategy
                target_samples_seg = int(seg_duration * sr)
                current_samples = audio_tensor.shape[-1]

                if strategy == "strict_slot" and current_samples > target_samples_seg:
                    try:
                        from services.ffmpeg_utils import _pitch_preserving_stretch
                        audio_tensor = await _pitch_preserving_stretch(
                            audio_tensor, target_samples_seg, sr
                        )
                    except Exception as e:
                        logger.warning("Batch stretch failed for strict_slot: %s", e)
                        audio_tensor = audio_tensor[..., :target_samples_seg]
                elif strategy == "smart_fit" and current_samples > target_samples_seg:
                    ratio = current_samples / target_samples_seg
                    if ratio > 1.0:
                        capped_ratio = min(ratio, 1.25)
                        capped_target = int(current_samples / capped_ratio)
                        try:
                            from services.ffmpeg_utils import _pitch_preserving_stretch
                            audio_tensor = await _pitch_preserving_stretch(
                                audio_tensor, capped_target, sr
                            )
                        except Exception as e:
                            logger.warning("Batch stretch failed for smart_fit: %s", e)

                        current_samples = audio_tensor.shape[-1]
                        if current_samples > target_samples_seg:
                            audio_tensor = audio_tensor[..., :target_samples_seg]

                # Final padding or trimming to ensure exact match with segment timeline
                current_samples = audio_tensor.shape[-1]
                if target_samples_seg > current_samples:
                    audio_tensor = torch.nn.functional.pad(
                        audio_tensor, (0, target_samples_seg - current_samples)
                    )
                elif current_samples > target_samples_seg:
                    audio_tensor = audio_tensor[..., :target_samples_seg]

                # Crossfade
                fade_samples = int(0.015 * sr)
                wl = audio_tensor.shape[-1]
                if wl > fade_samples * 2:
                    ramp_up = torch.linspace(0, 1, fade_samples)
                    ramp_down = torch.linspace(1, 0, fade_samples)
                    audio_tensor[0, :fade_samples] *= ramp_up
                    audio_tensor[0, -fade_samples:] *= ramp_down

                s_idx = int(seg_start * sr)
                e_idx = min(s_idx + wl, total_samples)
                full_audio[:, s_idx:e_idx] += audio_tensor[:, :e_idx - s_idx]

            except Exception as e:
                logger.warning("Batch TTS seg %d failed: %s", i, e)

        # ── 3c. Save dubbed audio track ───────────────────────────────
        # Same assembly pattern as dub_generate.py:390 — `full_audio` is a
        # zero-init tensor that gets +='d from torch.cat-style slices, so
        # it can land non-contiguous + out-of-range. Go through the
        # audited + atomic helper to defend against #48 silent corruption
        # and partial-write truncation simultaneously.
        track_path = os.path.join(batch_dir, f"dubbed_{target_lang}.wav")
        atomic_save_wav(track_path, full_audio, sr)

        # ── 3d. Mix with original video ───────────────────────────────
        _set_progress(
            job, "mix",
            percent=int(((lang_idx + 0.8) / total_langs) * 100),
            current_lang=target_lang,
        )

        output_path = os.path.join(batch_dir, f"output_{target_lang}.mp4")

        def _mix(bg=job.get("preserve_bg", True)):
            if bg:
                # Mix dubbed audio with original background
                subprocess.run(
                    [ffmpeg, "-y",
                     "-i", video_path,
                     "-i", track_path,
                     "-filter_complex",
                     "[0:a]volume=0.15[bg];[1:a]volume=1.0[dub];[bg][dub]amix=inputs=2:duration=first[out]",
                     "-map", "0:v", "-map", "[out]",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", output_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=600, check=True,
                )
            else:
                # Replace audio entirely
                subprocess.run(
                    [ffmpeg, "-y",
                     "-i", video_path,
                     "-i", track_path,
                     "-map", "0:v", "-map", "1:a",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", output_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=600, check=True,
                )

        await loop.run_in_executor(None, _mix)
        outputs[target_lang] = output_path

        # Copy to output directory if specified (Batch Clone Hàng loạt)
        output_dir = job.get("output_dir")
        if output_dir:
            try:
                import shutil
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(job["filename"])[0]
                dest_filename = f"{base_name}_{target_lang}.mp4"
                dest_path = os.path.join(output_dir, dest_filename)
                if os.path.abspath(output_path) != os.path.abspath(dest_path):
                    shutil.copy2(output_path, dest_path)
                    logger.info("Copied batch output to final destination: %s", dest_path)
                    outputs[target_lang] = dest_path
            except Exception as e:
                logger.exception("Failed to copy dubbed video to final destination: %s", e)

    job["outputs"] = outputs
    _set_progress(job, "done", 100)


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/batch/enqueue")
async def enqueue_batch_job(
    video: UploadFile = File(...),
    langs: str = Form("es"),            # comma-separated lang codes
    voice_id: Optional[str] = Form(None),
    timing_strategy: Optional[str] = Form("concise"),
    preserve_bg: bool = Form(True),
):
    """Enqueue a video for batch dubbing.

    The video is saved to disk and a job is added to the queue.
    Returns the job ID for status polling.
    """
    _ensure_queue()

    job_id = str(uuid.uuid4())[:12]
    lang_list = [l.strip() for l in langs.split(",") if l.strip()]
    if not lang_list:
        raise HTTPException(400, "At least one target language is required")

    # Save the uploaded video
    batch_dir = os.path.join(DATA_DIR, "batch")
    os.makedirs(batch_dir, exist_ok=True)
    ext = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    video_path = os.path.join(batch_dir, f"{job_id}{ext}")

    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    job = {
        "id": job_id,
        "status": "queued",
        "filename": video.filename or f"{job_id}{ext}",
        "video_path": video_path,
        "langs": lang_list,
        "voice_id": voice_id,
        "timing_strategy": timing_strategy,
        "preserve_bg": preserve_bg,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "progress": None,
    }
    _jobs[job_id] = job
    await _queue.put(job_id)

    logger.info("Batch job %s enqueued: %s → %s", job_id, video.filename, lang_list)
    return {"job_id": job_id, "status": "queued", "queue_position": _queue.qsize()}


@router.post("/batch/enqueue-local")
async def enqueue_local_batch(req: LocalBatchRequest):
    """Enqueue all video files in a local directory for sequential dubbing."""
    _ensure_queue()

    if not os.path.exists(req.input_dir):
        raise HTTPException(400, f"Thư mục đầu vào không tồn tại: {req.input_dir}")
    if not os.path.isdir(req.input_dir):
        raise HTTPException(400, f"Đường dẫn đầu vào không phải là thư mục: {req.input_dir}")

    # Scan for video files
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    video_files = []
    try:
        for f in os.listdir(req.input_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in video_exts:
                video_files.append(f)
    except Exception as e:
        raise HTTPException(500, f"Không thể đọc thư mục đầu vào: {e}")

    if not video_files:
        raise HTTPException(400, f"Không tìm thấy file video nào trong thư mục: {req.input_dir}")

    # Natural sorting of video files
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    video_files.sort(key=natural_sort_key)

    lang_list = [l.strip() for l in req.langs.split(",") if l.strip()]
    if not lang_list:
        raise HTTPException(400, "Yêu cầu ít nhất một ngôn ngữ đích")

    job_ids = []
    batch_group_id = str(uuid.uuid4())[:12]
    
    for filename in video_files:
        job_id = str(uuid.uuid4())[:12]
        video_path = os.path.join(req.input_dir, filename)
        
        job = {
            "id": job_id,
            "batch_group_id": batch_group_id,
            "status": "queued",
            "filename": filename,
            "video_path": video_path,
            "output_dir": req.output_dir,
            "langs": lang_list,
            "voice_id": req.voice_id,
            "translation_provider": req.translation_provider,
            "timing_strategy": req.timing_strategy,
            "preserve_bg": req.preserve_bg,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "progress": None,
        }
        _jobs[job_id] = job
        await _queue.put(job_id)
        job_ids.append(job_id)

    logger.info("Enqueued %d local batch jobs from %s", len(job_ids), req.input_dir)
    return {
        "batch_group_id": batch_group_id,
        "job_ids": job_ids,
        "count": len(job_ids)
    }


@router.get("/batch/jobs")
def list_batch_jobs(
    status: Optional[str] = None,
    batch_group_id: Optional[str] = None,
    limit: int = 50
):
    """List batch jobs, optionally filtered by status or batch group ID."""
    jobs = list(_jobs.values())
    if batch_group_id:
        jobs = [j for j in jobs if j.get("batch_group_id") == batch_group_id]
    if status:
        if status == "active":
            jobs = [j for j in jobs if j["status"] in ("queued", "running")]
        else:
            jobs = [j for j in jobs if j["status"] == status]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return jobs[:limit]


@router.get("/batch/jobs/{job_id}")
def get_batch_job(job_id: str):
    """Get the status of a specific batch job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/batch/jobs/{job_id}/cancel")
def cancel_batch_job(job_id: str):
    """Cancel a queued or running batch job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("done", "failed", "cancelled"):
        return {"already": job["status"]}
    job["status"] = "cancelled"
    job["finished_at"] = time.time()
    return {"cancelled": True}


@router.delete("/batch/jobs/{job_id}")
def delete_batch_job(job_id: str):
    """Delete a batch job record and its video file."""
    job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("video_path") and os.path.exists(job["video_path"]):
        try:
            os.remove(job["video_path"])
        except Exception:
            pass
    return {"deleted": True}


@router.get("/batch/download/{job_id}/{lang}")
def download_batch_output(job_id: str, lang: str):
    """Download a completed batch job's output video for a given language."""
    from fastapi.responses import FileResponse

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job is {job['status']}, not done")

    outputs = job.get("outputs", {})
    path = outputs.get(lang)
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"No output for language '{lang}'")

    filename = f"{os.path.splitext(job['filename'])[0]}_{lang}.mp4"
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
    )
