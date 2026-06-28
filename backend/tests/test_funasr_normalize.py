"""FunASR output normalisation ΓÇö VAD sentence_info boundaries."""

from services.asr_backend import _normalize_funasr
from services.segmentation import segment_transcript


def test_normalize_funasr_sentence_info_uses_sentence_field():
    res = [{
        "language": "zh",
        "text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>full blob",
        "sentence_info": [
            {"start": 600, "end": 3200, "spk": 0, "sentence": "<|zh|>µ¼óΦ┐Äσñºσ«╢µ¥ÑΣ╜ôΘ¬î"},
            {"start": 3500, "end": 8100, "spk": 0, "sentence": "<|zh|>Φ╛╛µæ⌐ΘÖóµÄ¿σç║τÜäΦ»¡Θƒ│Φ»åσê½µ¿íσ₧ï"},
        ],
    }]
    out = _normalize_funasr(res)
    assert len(out["segments"]) == 2
    assert out["segments"][0]["start"] == 0.6
    assert out["segments"][0]["end"] == 3.2
    assert "µ¼óΦ┐Ä" in out["segments"][0]["text"]
    assert out["segments"][0]["speaker"] == "Speaker 1"
    assert len(out["chunks"]) == 2


def test_segment_transcript_preserves_funasr_vad_boundaries():
    whisper_like = {
        "segments": [
            {
                "text": "µ¼óΦ┐Äσñºσ«╢µ¥ÑΣ╜ôΘ¬îΦ╛╛µæ⌐ΘÖóµÄ¿σç║τÜäΦ»¡Θƒ│Φ»åσê½µ¿íσ₧ïτ¼¼Σ╕ÇσÅÑ",
                "start": 0.6,
                "end": 3.2,
            },
            {
                "text": "Φ┐Öµÿ»τ¼¼Σ║îσÅÑσåàσ«╣Φ╢│σñƒΘò┐Σ╗ÑΣ╛┐Σ╕ìΣ╝ÜΦó½σÉêσ╣╢µêÉσìòΣ╕¬τëçµ«╡",
                "start": 3.5,
                "end": 8.1,
            },
        ],
        "chunks": [],
    }
    segs = segment_transcript(whisper_like, duration=10.0)
    assert len(segs) >= 2
    assert segs[0]["start"] == 0.6
    assert segs[0]["end"] == 3.2
