"""Broadcast-grade segmentation for dubbing.

Rules (in priority order):
  1. Never split mid-word. Whitespace or nothing.
  2. Prefer sentence punctuation > clause punctuation (, ; : —) > word boundaries.
  3. Reject any candidate split that leaves either side below the minimum floor.
  4. Fragments below the floor merge into same-speaker neighbor; gap < MERGE_GAP
     prefers previous, else next.
  5. Scene-cut assisted splits apply only when both halves remain viable.
  6. Never merge across a speaker boundary.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


MIN_DUR = 1.5         # seconds — below this, a segment must merge
MIN_CHARS = 12        # characters — below this, a segment must merge (Latin-ish)
MIN_WORDS = 3         # words — below this, a segment is considered a fragment
STITCH_DUR = 2.5      # seconds — pair of short neighbors under this combine even when each is legal
STITCH_GAP = 0.9      # seconds — max silence between two stitch candidates
IDEAL_DUR = 4.5       # seconds — target length for splits
MAX_DUR = 9.0         # seconds — above this, force a split
MAX_CHARS = 140       # characters — above this, force a split
MERGE_GAP = 0.6       # seconds — tolerated silence when folding a fragment backward
MERGE_GAP_ULTRA = 2.0 # seconds — wider gap tolerated for ultra-short (< 0.5s or < 3 chars)
ULTRA_SHORT_DUR = 0.5 # seconds — threshold for "always fold" regardless of neighbor match
ULTRA_SHORT_CHARS = 4 # chars — same tier
SPEAKER_GAP = 1.2     # seconds — heuristic speaker-change gap (no pyannote)

# Sentence-end punctuation across Latin, CJK, Bengali, Arabic, Thai, Armenian, Hindi, etc.
_SENTENCE_END = re.compile(
    r'([.!?。！？।؟…؛܀։՝።။၊।]["\')\]]?)(\s+|$)'
)
_CLAUSE_END = re.compile(r'([,;:—、،؍])(\s+|$)')
_WS = re.compile(r'\s+')


def _word_count(text: str) -> int:
    if not text:
        return 0
    # Latin-like scripts use whitespace; CJK scripts count each glyph as a word.
    tokens = [t for t in text.split() if t]
    if len(tokens) >= MIN_WORDS:
        return len(tokens)
    # For scripts without spaces (CJK), approximate word count as graphemes / 2.
    non_space = sum(1 for ch in text if not ch.isspace())
    approx = max(len(tokens), non_space // 2)
    return approx


def _is_short(seg) -> bool:
    return (
        seg.duration < MIN_DUR
        or seg.char_count < MIN_CHARS
        or _word_count(seg.text) < MIN_WORDS
    )


def _is_ultra_short(seg) -> bool:
    return seg.duration < ULTRA_SHORT_DUR or seg.char_count < ULTRA_SHORT_CHARS


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: str = "Speaker 1"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "speaker_id": self.speaker_id,
        }
        if self.words:
            d["words"] = [{"start": round(w.start, 2), "end": round(w.end, 2), "text": w.text} for w in self.words]
        return d


def _clean(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _best_boundary(text: str, ideal_pos: int) -> int:
    """Return a character offset to split at. Prefer sentence > clause > word.

    Scans the full text for each candidate class and picks the one whose offset
    is closest to `ideal_pos`. Sentence endings always beat clause endings, which
    always beat bare word boundaries.
    """
    if not text:
        return 0
    length = len(text)
    if length <= 1:
        return length

    def _closest(offsets: List[int]) -> Optional[int]:
        if not offsets:
            return None
        return min(offsets, key=lambda o: abs(o - ideal_pos))

    sentence_offsets = [m.end(1) for m in _SENTENCE_END.finditer(text)]
    pick = _closest(sentence_offsets)
    if pick is not None:
        return pick

    clause_offsets = [m.end(1) for m in _CLAUSE_END.finditer(text)]
    pick = _closest(clause_offsets)
    if pick is not None:
        return pick

    # Bare word boundaries: every space position.
    space_offsets = [i for i, ch in enumerate(text) if ch == " "]
    pick = _closest(space_offsets)
    if pick is not None:
        return pick
    return length


def _words_from_whisper(result: dict) -> List[Word]:
    """Extract word-level timing if available, otherwise fall back to chunk-level."""
    words: List[Word] = []
    segs = result.get("segments") if isinstance(result, dict) else None
    if segs:
        for seg in segs:
            for w in seg.get("words", []) or []:
                wt = (w.get("word") or w.get("text") or "").strip()
                if not wt:
                    continue
                ws = float(w.get("start", seg.get("start", 0.0)))
                we = float(w.get("end", seg.get("end", ws + 0.1)))
                if we <= ws:
                    we = ws + 0.05
                words.append(Word(start=ws, end=we, text=wt))
        if words:
            return words

        # Segment-level timings (FunASR sentence_info, etc.) — one token per VAD segment
        # so downstream grouping preserves natural speech boundaries.
        for seg in segs:
            txt = _clean((seg.get("text") or "").strip())
            if not txt:
                continue
            s = float(seg.get("start", 0.0))
            e_raw = seg.get("end")
            e = float(e_raw if e_raw is not None else s + 0.1)
            if e <= s:
                e = s + 0.1
            words.append(Word(start=s, end=e, text=txt))
        if words:
            return words

    # Fallback: chunk-level timings (no per-word granularity)
    for chunk in result.get("chunks", []) or []:
        ts = chunk.get("timestamp") or (0.0, 0.0)
        s = float(ts[0] or 0.0)
        e = float(ts[1] or s + 0.1)
        text = _clean(chunk.get("text", ""))
        if not text or e <= s:
            continue
        # Distribute time evenly across the tokens inside the chunk
        tokens = text.split(" ")
        dur = (e - s) / max(len(tokens), 1)
        t = s
        for tok in tokens:
            words.append(Word(start=t, end=t + dur, text=tok))
            t += dur
    return words


def _build_segments_from_words(words: Sequence[Word]) -> List[Segment]:
    """Greedy grouping of words into IDEAL_DUR sentences, cut at natural boundaries."""
    segments: List[Segment] = []
    if not words:
        return segments

    buf: List[Word] = []
    buf_start = words[0].start

    def flush_buf(force: bool = False) -> None:
        nonlocal buf, buf_start
        if not buf:
            return
        text = _clean(" ".join(w.text for w in buf))
        if not text:
            buf = []
            return
        segments.append(Segment(start=buf_start, end=buf[-1].end, text=text, words=list(buf)))
        buf = []
        if not force:
            buf_start = 0.0

    for i, w in enumerate(words):
        if not buf:
            buf_start = w.start
        buf.append(w)
        buf_dur = buf[-1].end - buf_start
        buf_chars = sum(len(x.text) + 1 for x in buf)
        next_gap = 0.0
        if i + 1 < len(words):
            next_gap = max(0.0, words[i + 1].start - w.end)

        ends_sentence = bool(_SENTENCE_END.search(w.text))
        ends_clause = bool(_CLAUSE_END.search(w.text))

        too_long = buf_dur >= MAX_DUR or buf_chars >= MAX_CHARS
        at_ideal = buf_dur >= IDEAL_DUR and buf_chars >= MIN_CHARS

        # Natural-boundary flush at target length.
        if at_ideal and ends_sentence:
            flush_buf()
        elif too_long and (ends_sentence or ends_clause):
            flush_buf()
        elif too_long and next_gap >= 0.35:
            flush_buf()
        elif too_long:
            # Last-resort split on a word boundary. Choose the word whose
            # cumulative position is closest to IDEAL_DUR from buf_start.
            best_idx = None
            best_score = float("inf")
            for k, bw in enumerate(buf[:-1]):  # must leave ≥1 word on right
                left_dur = bw.end - buf_start
                if left_dur < MIN_DUR:
                    continue
                right_dur = buf[-1].end - buf[k + 1].start
                if right_dur < MIN_DUR:
                    continue
                # Prefer words ending in sentence / clause punctuation.
                boundary_bonus = 0.0
                if _SENTENCE_END.search(bw.text):
                    boundary_bonus = -2.0
                elif _CLAUSE_END.search(bw.text):
                    boundary_bonus = -0.8
                score = abs(left_dur - IDEAL_DUR) + boundary_bonus
                if score < best_score:
                    best_score = score
                    best_idx = k

            if best_idx is not None:
                left_buf = buf[: best_idx + 1]
                right_buf = buf[best_idx + 1 :]
                segments.append(Segment(
                    start=buf_start,
                    end=left_buf[-1].end,
                    text=_clean(" ".join(x.text for x in left_buf)),
                    words=list(left_buf),
                ))
                buf = list(right_buf)
                buf_start = right_buf[0].start
            else:
                flush_buf()

    flush_buf(force=True)
    return segments


def _merge_short(segments: List[Segment]) -> List[Segment]:
    """Fold fragments below the floor into adjacent same-speaker segment.

    Runs multi-pass until no further merges happen. Ultra-short segments
    (< 0.5s or < 4 chars) fold across larger gaps and across speakers when
    no same-speaker neighbor is close — stray tokens like "STR" are never
    allowed to survive as standalone segments.
    """
    if not segments:
        return segments

    for _ in range(64):  # bounded iterations so misuse can't hang
        did_merge = False
        i = 0
        while i < len(segments):
            s = segments[i]
            if not _is_short(s):
                i += 1
                continue

            prev = segments[i - 1] if i > 0 else None
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            gap_tolerance = MERGE_GAP_ULTRA if _is_ultra_short(s) else MERGE_GAP

            prev_same = bool(prev and prev.speaker_id == s.speaker_id)
            next_same = bool(nxt and nxt.speaker_id == s.speaker_id)
            prev_gap = (s.start - prev.end) if prev else float("inf")
            next_gap = (nxt.start - s.end) if nxt else float("inf")

            prev_ok = prev_same and prev_gap <= gap_tolerance
            next_ok = next_same and next_gap <= gap_tolerance

            target = None
            if prev_ok and next_ok:
                target = prev if prev.duration <= nxt.duration else nxt
            elif prev_ok:
                target = prev
            elif next_ok:
                target = nxt
            elif prev_same:
                target = prev
            elif next_same:
                target = nxt
            elif _is_ultra_short(s):
                # Stray token — fold into closest neighbor regardless of speaker.
                if prev and nxt:
                    target = prev if prev_gap <= next_gap else nxt
                else:
                    target = prev or nxt
            elif prev:
                target = prev
            elif nxt:
                target = nxt

            if target is None:
                i += 1
                continue
            if target is prev:
                prev.text = _clean(prev.text + " " + s.text)
                prev.end = max(prev.end, s.end)
                prev.words.extend(s.words)
                segments.pop(i)
                did_merge = True
                continue
            if target is nxt:
                nxt.text = _clean(s.text + " " + nxt.text)
                nxt.start = min(nxt.start, s.start)
                nxt.words = s.words + nxt.words
                segments.pop(i)
                did_merge = True
                continue

            i += 1
        if not did_merge:
            break
    return segments


def _stitch_adjacent_shorts(segments: List[Segment]) -> List[Segment]:
    """Combine adjacent same-speaker segments when both are short and close.

    Catches the case where each segment individually passes MIN_DUR but a
    rapid-fire pair produces a jittery dub. Only stitches when both halves
    live under STITCH_DUR and the gap between them is minimal.
    """
    if len(segments) < 2:
        return segments

    for _ in range(32):
        did = False
        i = 0
        while i + 1 < len(segments):
            a, b = segments[i], segments[i + 1]
            same = a.speaker_id == b.speaker_id
            gap = b.start - a.end
            combined_dur = (b.end - a.start)
            if (
                same
                and gap <= STITCH_GAP
                and a.duration <= STITCH_DUR
                and b.duration <= STITCH_DUR
                and combined_dur <= MAX_DUR
            ):
                a.text = _clean(a.text + " " + b.text)
                a.end = b.end
                a.words.extend(b.words)
                segments.pop(i + 1)
                did = True
                continue
            i += 1
        if not did:
            break
    return segments


def clean_up_segments(segments: List[dict]) -> List[dict]:
    """Public entry: run merge + stitch passes on already-persisted segments.

    Used by the UI's "Clean up segments" action so users can repair jobs
    that were segmented under older, looser rules.
    """
    objs: List[Segment] = []
    for s in segments or []:
        try:
            words_list = []
            for w in s.get("words", []) or []:
                words_list.append(Word(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    text=str(w.get("text", "")),
                ))
            objs.append(Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=_clean(str(s.get("text", ""))),
                speaker_id=str(s.get("speaker_id") or "Speaker 1"),
                id=str(s.get("id") or uuid.uuid4().hex[:8]),
                words=words_list,
            ))
        except (TypeError, ValueError):
            continue
    objs = [s for s in objs if s.end > s.start and s.text]
    objs = _merge_short(objs)
    objs = _stitch_adjacent_shorts(objs)
    objs = _merge_short(objs)
    return [s.to_dict() for s in objs]


def _apply_scene_cuts(segments: List[Segment], scene_cuts: Iterable[float]) -> List[Segment]:
    """Split segments at scene cuts only if both halves remain viable."""
    cuts = sorted(c for c in scene_cuts if c > 0)
    if not cuts:
        return segments

    out: List[Segment] = []
    for s in segments:
        inner_cuts = [c for c in cuts if s.start + MIN_DUR < c < s.end - MIN_DUR]
        if not inner_cuts:
            out.append(s)
            continue

        remaining = s
        for cut in inner_cuts:
            dur_total = remaining.duration
            if dur_total <= 0:
                break
            ratio = (cut - remaining.start) / dur_total
            tentative_split = int(len(remaining.text) * ratio)
            pos = _best_boundary(remaining.text, tentative_split)
            left_text = remaining.text[:pos].strip()
            right_text = remaining.text[pos:].strip()
            # Viability check — refuse the cut if either half would be a fragment.
            if (
                not left_text
                or not right_text
                or len(left_text) < MIN_CHARS
                or len(right_text) < MIN_CHARS
                or (cut - remaining.start) < MIN_DUR
                or (remaining.end - cut) < MIN_DUR
            ):
                continue
            
            # Preserve word timing metadata across scene cuts
            left_words = [w for w in remaining.words if w.end <= cut]
            right_words = [w for w in remaining.words if w.start >= cut]
            straddling = [w for w in remaining.words if w.start < cut < w.end]
            for w in straddling:
                if (cut - w.start) >= (w.end - cut):
                    left_words.append(w)
                else:
                    right_words.append(w)
            left_words.sort(key=lambda w: w.start)
            right_words.sort(key=lambda w: w.start)

            out.append(Segment(
                start=remaining.start, end=cut, text=left_text, speaker_id=remaining.speaker_id,
                words=left_words,
            ))
            remaining = Segment(
                start=cut, end=remaining.end, text=right_text, speaker_id=remaining.speaker_id,
                words=right_words,
            )
        out.append(remaining)
    return out


def _segments_from_asr_timings(whisper_result: dict) -> Optional[List[Segment]]:
    """Use pre-segmented ASR output (FunASR sentence_info, etc.) when no word timings."""
    segs_in = (whisper_result or {}).get("segments") or []
    if not segs_in:
        return None
    if any((s.get("words") or []) for s in segs_in if isinstance(s, dict)):
        return None
    segments: List[Segment] = []
    for s in segs_in:
        if not isinstance(s, dict):
            continue
        txt = _clean(s.get("text", ""))
        if not txt or s.get("start") is None:
            continue
        start = float(s.get("start", 0.0))
        end_raw = s.get("end")
        end = float(end_raw if end_raw is not None else start + 0.1)
        if end <= start:
            end = start + 0.1
        segments.append(Segment(start=start, end=end, text=txt))
    return segments or None


def segment_transcript(
    whisper_result: dict,
    duration: float,
    scene_cuts: Optional[Iterable[float]] = None,
) -> List[dict]:
    """Public entry point: whisper result → clean dub segments (as dicts)."""
    asr_segments = _segments_from_asr_timings(whisper_result)
    if asr_segments:
        segments = _merge_short(asr_segments)
        if scene_cuts:
            segments = _apply_scene_cuts(segments, scene_cuts)
            segments = _merge_short(segments)
        segments = _stitch_adjacent_shorts(segments)
        segments = _merge_short(segments)
        return [s.to_dict() for s in segments]

    words = _words_from_whisper(whisper_result)
    if not words:
        text = _clean((whisper_result or {}).get("text", ""))
        if text:
            return [Segment(start=0.0, end=max(duration, 0.1), text=text).to_dict()]
        return []

    segments = _build_segments_from_words(words)
    segments = _merge_short(segments)
    if scene_cuts:
        segments = _apply_scene_cuts(segments, scene_cuts)
        segments = _merge_short(segments)
    segments = _stitch_adjacent_shorts(segments)
    segments = _merge_short(segments)
    return [s.to_dict() for s in segments]


def _split_segments_by_word_speakers(
    segments: List[dict],
    diarization_or_turns,
    is_diarization: bool
) -> List[dict]:
    new_segments = []
    
    for s in segments:
        words = s.get("words", [])
        
        # 1. Determine overall winner speaker for this segment as a fallback
        start, end = s["start"], s["end"]
        mid = (start + end) / 2.0
        overlap = {}
        
        if is_diarization:
            for turn, _, speaker in diarization_or_turns.itertracks(yield_label=True):
                left = max(start, turn.start)
                right = min(end, turn.end)
                if right > left:
                    overlap[speaker] = overlap.get(speaker, 0.0) + (right - left)
            if overlap:
                winner = max(overlap.items(), key=lambda kv: kv[1])[0]
            else:
                winner = None
                for turn, _, speaker in diarization_or_turns.itertracks(yield_label=True):
                    if turn.start <= mid <= turn.end:
                        winner = speaker
                        break
        else:
            for t in diarization_or_turns:
                left = max(start, t["start"])
                right = min(end, t["end"])
                if right > left:
                    overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + (right - left)
            if overlap:
                winner = max(overlap.items(), key=lambda kv: kv[1])[0]
            else:
                winner = None
                for t in diarization_or_turns:
                    if t["start"] <= mid <= t["end"]:
                        winner = t["speaker"]
                        break
                        
        if winner is not None:
            if is_diarization:
                try:
                    idx = int(winner.split("_")[-1]) + 1
                    seg_speaker = f"Speaker {idx}"
                except ValueError:
                    seg_speaker = winner
            else:
                seg_speaker = winner
        else:
            seg_speaker = s.get("speaker_id") or "Speaker 1"

        # 2. If no words are present, assign the overall speaker and keep as a single segment
        if not words:
            s["speaker_id"] = seg_speaker
            new_segments.append(s)
            continue
            
        # 3. Determine speaker for each word
        word_speakers = []
        for w in words:
            w_start = w.get("start", start)
            w_end = w.get("end", end)
            w_mid = (w_start + w_end) / 2.0
            w_overlap = {}
            
            if is_diarization:
                for turn, _, speaker in diarization_or_turns.itertracks(yield_label=True):
                    left = max(w_start, turn.start)
                    right = min(w_end, turn.end)
                    if right > left:
                        w_overlap[speaker] = w_overlap.get(speaker, 0.0) + (right - left)
                if w_overlap:
                    w_winner = max(w_overlap.items(), key=lambda kv: kv[1])[0]
                else:
                    w_winner = None
                    for turn, _, speaker in diarization_or_turns.itertracks(yield_label=True):
                        if turn.start <= w_mid <= turn.end:
                            w_winner = speaker
                            break
            else:
                for t in diarization_or_turns:
                    left = max(w_start, t["start"])
                    right = min(w_end, t["end"])
                    if right > left:
                        w_overlap[t["speaker"]] = w_overlap.get(t["speaker"], 0.0) + (right - left)
                if w_overlap:
                    w_winner = max(w_overlap.items(), key=lambda kv: kv[1])[0]
                else:
                    w_winner = None
                    for t in diarization_or_turns:
                        if t["start"] <= w_mid <= t["end"]:
                            w_winner = t["speaker"]
                            break
                            
            if w_winner is not None:
                if is_diarization:
                    try:
                        idx = int(w_winner.split("_")[-1]) + 1
                        w_spk = f"Speaker {idx}"
                    except ValueError:
                        w_spk = w_winner
                else:
                    w_spk = w_winner
            else:
                w_spk = seg_speaker
                
            word_speakers.append(w_spk)
            
        # 4. Group contiguous words with the same speaker, also splitting at sentence boundaries
        groups = []
        current_group = []
        current_speaker = None
        for w, spk in zip(words, word_speakers):
            # Check if previous word ended a sentence
            prev_ended_sentence = False
            if current_group:
                prev_text = current_group[-1].get("text", "")
                if _SENTENCE_END.search(prev_text):
                    prev_ended_sentence = True

            if current_speaker is None:
                current_speaker = spk
                current_group = [w]
            elif spk != current_speaker or prev_ended_sentence:
                groups.append((current_speaker, current_group))
                current_speaker = spk
                current_group = [w]
            else:
                current_group.append(w)
        if current_group:
            groups.append((current_speaker, current_group))
            
        # 5. Create sub-segments
        for g_idx, (spk, g_words) in enumerate(groups):
            g_text = _clean(" ".join(str(w.get("text", "")) for w in g_words))
            if not g_text:
                continue
                
            sub_start = max(s["start"], g_words[0].get("start", s["start"]))
            sub_end = min(s["end"], g_words[-1].get("end", s["end"]))
            if sub_end <= sub_start:
                sub_end = sub_start + 0.1
                
            sub_seg = {
                "id": f"{s['id']}_{g_idx}" if len(groups) > 1 else s["id"],
                "start": round(sub_start, 2),
                "end": round(sub_end, 2),
                "text": g_text,
                "text_original": g_text,
                "speaker_id": spk,
                "words": g_words,
            }
            new_segments.append(sub_seg)
            
    return new_segments


def assign_speakers_from_diarization(
    segments: List[dict],
    diarization,
) -> List[dict]:
    """Replace speaker_id based on a pyannote diarization result (overlap-weighted).
    Splits segments if they contain multiple speaker turns.
    """
    if not diarization:
        return assign_speakers_heuristic(segments)
    return _split_segments_by_word_speakers(segments, diarization, is_diarization=True)


def assign_speakers_from_turns(
    segments: List[dict],
    turns: List[dict],
) -> List[dict]:
    """Assign speaker_id by overlap against a list of ``{start, end, speaker}``
    turns produced by an ASR backend that diarizes inline (e.g. FunASR's cam++).
    Splits segments if they contain multiple speaker turns.
    """
    clean = [
        t for t in (turns or [])
        if t.get("speaker") is not None and t.get("start") is not None and t.get("end") is not None
    ]
    if not clean:
        return assign_speakers_heuristic(segments)
    return _split_segments_by_word_speakers(segments, clean, is_diarization=False)


def assign_speakers_heuristic(segments: List[dict]) -> List[dict]:
    """Two-speaker alternation based on silence gaps."""
    current = 1
    last_end = 0.0
    for i, s in enumerate(segments):
        if i > 0 and (s["start"] - last_end) > SPEAKER_GAP:
            current = 2 if current == 1 else 1
        s["speaker_id"] = f"Speaker {current}"
        last_end = s["end"]
    return segments
