"""Silence-based segment splitting and Vietnamese translation validation."""

import numpy as np

from api.routers.dub_translate import _looks_like_target, _resolve_source_lang
from schemas.requests import TranslateRequest, TranslateSegment
from services.onset_align import split_segments_at_silence, needs_silence_resplit


def test_split_segments_at_silence_splits_long_utterance():
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    # Speech 0-3s, silence 3-4s, speech 4-7s
    audio[: sr * 3] = 0.3
    audio[sr * 4 : sr * 7] = 0.35
    segments = [{"start": 0.0, "end": 7.0, "text": "τ¼¼Σ╕ÇσÅÑτ¼¼Σ║îσÅÑτ¼¼Σ╕ëσÅÑτ¼¼σ¢¢σÅÑ"}]
    out = split_segments_at_silence(segments, audio, sr, min_gap_s=0.45)
    assert len(out) >= 2
    assert out[0]["end"] <= out[-1]["start"]


def test_vietnamese_validator_rejects_english_drift():
    assert not _looks_like_target(
        "The parts that come out won't hold their shape when you try this.",
        "vi",
    )
    assert _looks_like_target(
        "Phß║ºn b├ính ra l├▓ sß║╜ kh├┤ng giß╗» ─æ╞░ß╗úc h├¼nh dß║íng khi bß║ín thß╗¡.",
        "vi",
    )


def test_resolve_source_lang_prefers_cjk_over_en_fallback():
    req = TranslateRequest(
        segments=[TranslateSegment(id="1", text="Σ╗èσñ⌐µêæΣ╗¼µ¥ÑσüÜΣ╕ÇΣ╕¬Φ¢ïτ│ò", target_lang="vi")],
        target_lang="vi",
        job_id=None,
        source_lang="en",
    )
    assert _resolve_source_lang(req) == "zh"


def test_needs_silence_resplit_skips_fine_grained_segments(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_SILENCE_SPLIT", raising=False)
    fine = [{"start": 0.0, "end": 5.0, "text": "a"}, {"start": 5.0, "end": 10.0, "text": "b"}]
    assert needs_silence_resplit(fine) is False
    coarse = [{"start": 0.0, "end": 59.0, "text": "long block"}]
    assert needs_silence_resplit(coarse) is True
