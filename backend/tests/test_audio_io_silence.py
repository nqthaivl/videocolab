"""Guards for zero-length audio tensors (issue #48 / SRT dubbing)."""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.audio_io import (  # noqa: E402
    atomic_save_wav,
    coerce_non_empty_audio,
    duration_to_samples,
    resolve_timeline_duration,
    silence_tensor,
)


def test_duration_to_samples_never_zero():
    assert duration_to_samples(0.0, 24000) >= 1
    assert duration_to_samples(0.00001, 24000) >= 1


def test_silence_tensor_is_writable():
    wav = silence_tensor(0.0, 24000)
    assert wav.numel() > 0
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "silence.wav")
        atomic_save_wav(target, wav, 24000)
        assert os.path.getsize(target) > 44


def test_coerce_non_empty_audio_fills_empty():
    empty = torch.zeros(1, 0)
    filled = coerce_non_empty_audio(empty, 24000, duration_s=0.05)
    assert filled.numel() > 0


def test_coerce_non_empty_audio_preserves_real():
    real = torch.randn(1, 100)
    assert torch.equal(coerce_non_empty_audio(real, 24000), real)


def test_resolve_timeline_duration_from_cues():
    assert resolve_timeline_duration(0.0, [1.0, 5400.5]) == 5400.5
    assert resolve_timeline_duration(7200.0, [100.0]) == 7200.0


if __name__ == "__main__":
    test_duration_to_samples_never_zero()
    test_silence_tensor_is_writable()
    test_coerce_non_empty_audio_fills_empty()
    test_coerce_non_empty_audio_preserves_real()
    test_resolve_timeline_duration_from_cues()
    print("ok")
