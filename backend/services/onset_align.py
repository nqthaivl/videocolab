"""
Speech-onset alignment for transcript segments (issue #280, item 1).

Whisper-family ASR models are prone to stretching a segment's *start* back
over leading non-speech (intro music, room tone, silence). The classic
symptom from the issue report: the speaker starts talking at 0:02–0:03,
but the first transcript segment says ``start=0.0`` — so the dubbed line
plays the moment the video begins and everything feels desynchronised.

``snap_segment_starts`` post-processes segments against the actual audio
(ideally the Demucs-isolated vocals track, which the dub pipeline already
produces): for each segment it scans the waveform inside ``[start, end]``
for the first frame whose RMS rises above an adaptive threshold and moves
``start`` forward to just before that onset.

Design constraints:

* **Forward-only.** A segment start is never moved earlier — that could
  collide with the previous speaker. We only trim leading non-speech.
* **Conservative.** Shifts below ``min_shift_s`` are ignored (word-level
  timestamps are usually within ~100 ms already); a minimum segment
  duration is always preserved; segments whose window looks silent
  (no frame above the absolute floor) are left untouched.
* **Pure NumPy.** No model, no platform-specific code — identical
  behaviour on macOS / Windows / Linux, trivially unit-testable.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger("omnivoice.onset_align")

# Analysis frame for RMS energy. 20 ms is fine-grained enough to localise
# a syllable onset while staying cheap (a 10-min track is ~30k frames).
FRAME_S = 0.02
# Keep this much audio before the detected onset so plosives/breaths that
# sit just under the threshold aren't clipped off.
PRE_ROLL_S = 0.05
# Shifts smaller than this are noise — word-level ASR timestamps are
# usually accurate to ~0.1 s, so don't churn segment data for less.
MIN_SHIFT_S = 0.15
# Never shrink a segment below this duration when shifting its start.
MIN_SEG_DUR_S = 0.30
# A frame must exceed `RELATIVE_THRESHOLD × peak RMS of the window` to
# count as speech onset…
RELATIVE_THRESHOLD = 0.10
# …and the window's peak RMS must exceed this absolute floor, otherwise
# the whole window is treated as silence and left alone (we'd only be
# snapping to noise).
ABS_RMS_FLOOR = 1e-3


def _frame_rms(x: np.ndarray, frame_len: int) -> np.ndarray:
    """RMS per non-overlapping frame; the ragged tail frame is dropped."""
    n = (len(x) // frame_len) * frame_len
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = x[:n].reshape(-1, frame_len).astype(np.float64, copy=False)
    return np.sqrt((frames * frames).mean(axis=1)).astype(np.float32)


def detect_speech_onset(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
) -> float | None:
    """Return the absolute time (s) of the first speech-like frame inside
    ``[start_s, end_s]``, or ``None`` when the window is empty / silent.
    """
    if sr <= 0 or end_s <= start_s:
        return None
    i0 = max(0, int(start_s * sr))
    i1 = min(len(audio), int(end_s * sr))
    if i1 <= i0:
        return None
    window = audio[i0:i1]
    frame_len = max(1, int(FRAME_S * sr))
    rms = _frame_rms(window, frame_len)
    if rms.size == 0:
        return None
    peak = float(rms.max())
    if peak < ABS_RMS_FLOOR:
        return None  # whole window is effectively silent
    threshold = max(RELATIVE_THRESHOLD * peak, ABS_RMS_FLOOR)
    above = np.nonzero(rms >= threshold)[0]
    if above.size == 0:
        return None
    return start_s + float(above[0]) * (frame_len / sr)


# Hysteresis for full-track onset listing: after a frame crosses the
# threshold, the energy must stay *below* it for at least this long before
# the next rise counts as a new onset. Stops syllable-internal dips from
# spamming the timeline with ticks.
MIN_ONSET_GAP_S = 0.15


def detect_speech_onsets(audio: np.ndarray, sr: int) -> list[float]:
    """Return the times (s) of every speech-like onset across the whole track.

    Powers the timeline editor's snap-to-onset ticks (issue #280, item 3):
    frame RMS over the full track, single adaptive threshold
    ``max(RELATIVE_THRESHOLD × peak, ABS_RMS_FLOOR)``, and hysteresis — a
    new onset registers only when the energy rises above the threshold
    after at least ``MIN_ONSET_GAP_S`` below it.

    Pure NumPy, identical behaviour on every platform. Returns ``[]`` for
    empty/silent audio.
    """
    if sr <= 0 or audio is None or len(audio) == 0:
        return []
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_len = max(1, int(FRAME_S * sr))
    rms = _frame_rms(audio, frame_len)
    if rms.size == 0:
        return []
    peak = float(rms.max())
    if peak < ABS_RMS_FLOOR:
        return []  # whole track is effectively silent
    threshold = max(RELATIVE_THRESHOLD * peak, ABS_RMS_FLOOR)
    gap_frames = max(1, int(round(MIN_ONSET_GAP_S / FRAME_S)))
    frame_s = frame_len / sr

    onsets: list[float] = []
    below_run = gap_frames  # armed, so speech at t=0 still counts
    for i, v in enumerate(rms):
        if v >= threshold:
            if below_run >= gap_frames:
                onsets.append(round(i * frame_s, 3))
            below_run = 0
        else:
            below_run += 1
    return onsets


def snap_segment_starts(
    segments: Sequence[dict],
    audio: np.ndarray,
    sr: int,
    *,
    min_shift_s: float = MIN_SHIFT_S,
) -> int:
    """Snap each segment's ``start`` forward to the actual speech onset.

    Mutates the segment dicts in place (the shape the dub pipeline passes
    around). Returns the number of segments adjusted.

    ``audio`` should be mono float; the Demucs vocals track gives the best
    signal but the mixed track still beats nothing.
    """
    if sr <= 0 or audio is None or len(audio) == 0:
        return 0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    adjusted = 0
    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end - start < MIN_SEG_DUR_S + min_shift_s:
            continue  # too short for a meaningful shift
        onset = detect_speech_onset(audio, sr, start, end)
        if onset is None:
            continue
        new_start = max(start, onset - PRE_ROLL_S)
        shift = new_start - start
        if shift < min_shift_s:
            continue
        # Preserve a minimum playable duration.
        new_start = min(new_start, end - MIN_SEG_DUR_S)
        if new_start - start < min_shift_s:
            continue
        seg["start"] = round(new_start, 3)
        adjusted += 1

    if adjusted:
        logger.info("onset-align: snapped %d/%d segment start(s) to speech onset",
                    adjusted, len(segments))
    return adjusted


# Minimum continuous silence inside a segment to split at (seconds).
MIN_SILENCE_SPLIT_S = 0.45
# Segments shorter than this are never split further.
MIN_SPLIT_SEGMENT_S = 2.0
# Each sub-segment after a silence split must be at least this long.
MIN_SPLIT_PART_S = 1.0


def detect_silence_split_points(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    *,
    min_gap_s: float = MIN_SILENCE_SPLIT_S,
) -> list[float]:
    """Return absolute split times at silence midpoints inside ``[start_s, end_s]``."""
    if sr <= 0 or end_s - start_s < MIN_SPLIT_SEGMENT_S:
        return []
    i0 = max(0, int(start_s * sr))
    i1 = min(len(audio), int(end_s * sr))
    if i1 <= i0:
        return []
    window = audio[i0:i1]
    if window.ndim > 1:
        window = window.mean(axis=1)
    frame_len = max(1, int(FRAME_S * sr))
    rms = _frame_rms(window, frame_len)
    if rms.size == 0:
        return []
    peak = float(rms.max())
    if peak < ABS_RMS_FLOOR:
        return []
    silence_threshold = max(RELATIVE_THRESHOLD * peak * 0.35, ABS_RMS_FLOOR)
    gap_frames = max(1, int(round(min_gap_s / FRAME_S)))
    frame_s = frame_len / sr

    split_points: list[float] = []
    silent_run = 0
    silent_start = 0
    for i, v in enumerate(rms):
        if v < silence_threshold:
            if silent_run == 0:
                silent_start = i
            silent_run += 1
        else:
            if silent_run >= gap_frames:
                mid = silent_start + silent_run // 2
                t = start_s + mid * frame_s
                if start_s + MIN_SPLIT_PART_S < t < end_s - MIN_SPLIT_PART_S:
                    split_points.append(round(t, 3))
            silent_run = 0
    if silent_run >= gap_frames:
        mid = silent_start + silent_run // 2
        t = start_s + mid * frame_s
        if start_s + MIN_SPLIT_PART_S < t < end_s - MIN_SPLIT_PART_S:
            split_points.append(round(t, 3))
    return split_points


def _split_text_by_time_fraction(text: str, frac_start: float, frac_end: float) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    frac_start = max(0.0, min(1.0, frac_start))
    frac_end = max(frac_start, min(1.0, frac_end))
    if " " in text:
        words = text.split()
        n = len(words)
        i0 = max(0, int(n * frac_start))
        i1 = min(n, max(i0 + 1, int(n * frac_end)))
        return " ".join(words[i0:i1]).strip()
    n = len(text)
    i0 = max(0, int(n * frac_start))
    i1 = min(n, max(i0 + 1, int(n * frac_end)))
    return text[i0:i1].strip()


def split_segments_at_silence(
    segments: Sequence[dict],
    audio: np.ndarray,
    sr: int,
    *,
    min_gap_s: float = MIN_SILENCE_SPLIT_S,
) -> list[dict]:
    """Split long ASR segments at internal silence gaps (FunASR/Whisper fallback).

    When VAD returns one long utterance, energy valleys in the vocals track
    still mark natural pauses between phrases.
    """
    if sr <= 0 or audio is None or len(audio) == 0:
        return list(segments)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    out: list[dict] = []
    split_count = 0
    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            out.append(seg)
            continue
        text = (seg.get("text") or "").strip()
        if not text or end - start < MIN_SPLIT_SEGMENT_S:
            out.append(seg)
            continue
        points = detect_silence_split_points(audio, sr, start, end, min_gap_s=min_gap_s)
        if not points:
            out.append(seg)
            continue
        bounds = [start] + points + [end]
        dur = end - start
        pieces: list[dict] = []
        for i in range(len(bounds) - 1):
            t0, t1 = bounds[i], bounds[i + 1]
            pieces.append({
                "start": round(t0, 3),
                "end": round(t1, 3),
                "text": _split_text_by_time_fraction(
                    text, (t0 - start) / dur, (t1 - start) / dur
                ),
            })
        # Fold ultra-short splits back into a neighbor so no text is dropped.
        merged: list[dict] = []
        for p in pieces:
            if not p["text"]:
                continue
            if merged and (p["end"] - p["start"] < MIN_SPLIT_PART_S):
                merged[-1]["end"] = p["end"]
                merged[-1]["text"] = f"{merged[-1]['text']} {p['text']}".strip()
            else:
                merged.append(dict(p))
        if len(merged) <= 1:
            out.append(seg)
            continue
        for p in merged:
            sub = dict(seg)
            sub.update(p)
            out.append(sub)
            split_count += 1
    if split_count:
        logger.info("silence-split: produced %d sub-segment(s) from %d segment(s)",
                    split_count, len(segments))
    return out or list(segments)


def needs_silence_resplit(segments: Sequence[dict]) -> bool:
    """Return True when post-ASR silence splitting is worth the extra pass.

    FunASR / WhisperX with good VAD already emit short segments ΓÇö re-scanning
    the waveform for every segment is redundant and slows the transcribe stream
    (especially when there are dozens of segments).  Modes via
    ``OMNIVOICE_SILENCE_SPLIT``:

    * ``auto`` (default) ΓÇö split only when ASR left long monolithic blocks
    * ``1`` / ``true`` ΓÇö always run (legacy behaviour)
    * ``0`` / ``false`` ΓÇö never run
    """
    mode = os.environ.get("OMNIVOICE_SILENCE_SPLIT", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    if not segments:
        return False
    durs: list[float] = []
    for s in segments:
        try:
            d = float(s["end"]) - float(s["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if d > 0:
            durs.append(d)
    if not durs:
        return False
    longest = max(durs)
    avg = sum(durs) / len(durs)
    # ASR already gave phrase-level segments ΓÇö skip the extra RMS pass.
    if longest < MIN_SPLIT_SEGMENT_S:
        return False
    if longest <= 12.0 and avg <= 8.0:
        return False
    return True
