"""HunyuanOCR backend for hard-subtitle extraction (Tencent tencent/HunyuanOCR).

Uses transformers>=5.3 (HiggsAudio TTS) + HunYuanVL patch (services.hunyuan_vl_patch).

Optional vLLM server (OpenAI-compatible):
  set HUNYUAN_OCR_VLLM_URL=http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
from typing import Any

from PIL import Image

logger = logging.getLogger("omnivoice.hunyuan_ocr")

MODEL_ID = os.environ.get("HUNYUAN_OCR_MODEL", "tencent/HunyuanOCR")
VLLM_URL = os.environ.get("HUNYUAN_OCR_VLLM_URL", "").strip().rstrip("/")

_SUBTITLE_PROMPTS: dict[str, str] = {
    "auto": "提取图中的字幕",
    "zh": "提取图中的字幕",
    "zh-cn": "提取图中的字幕",
    "zh-tw": "提取图中的字幕",
    "en": "Extract the subtitles from the image.",
    "vi": "Trích xuất phụ đề trong ảnh.",
    "ja": "画像内の字幕を抽出してください。",
    "ko": "이미지에서 자막을 추출하세요.",
}

_engine_lock = threading.Lock()
_engine: "HunyuanOcrEngine | None" = None


def clean_repeated_substrings(text: str) -> str:
    """Official HunyuanOCR post-processing for repeated generation tails."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[: n - length * (count - 1)]
    return text


def _subtitle_prompt(language: str | None) -> str:
    if not language or language in ("auto", ""):
        return _SUBTITLE_PROMPTS["auto"]
    key = language.lower().replace("_", "-")
    if key.startswith("zh"):
        return _SUBTITLE_PROMPTS["zh"]
    return _SUBTITLE_PROMPTS.get(key, _SUBTITLE_PROMPTS.get(key.split("-")[0][:2], _SUBTITLE_PROMPTS["auto"]))


def _import_hunyuan_classes():
    try:
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        return AutoProcessor, HunYuanVLForConditionalGeneration
    except ImportError:
        from services.hunyuan_vl_patch import ensure_hunyuan_vl

        ensure_hunyuan_vl()
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        return AutoProcessor, HunYuanVLForConditionalGeneration


def is_hunyuan_available() -> tuple[bool, str | None]:
    if VLLM_URL:
        return True, None
    try:
        _import_hunyuan_classes()
        return True, None
    except RuntimeError as exc:
        return False, str(exc)


def is_hunyuan_model_cached() -> bool:
    """True when full HunyuanOCR weights exist (same check as Settings → Models)."""
    try:
        from api.routers.setup.models import KNOWN_MODELS, model_is_installed

        for entry in KNOWN_MODELS:
            if entry.get("repo_id") == MODEL_ID:
                return model_is_installed(entry)
        return False
    except Exception as exc:
        logger.debug("HunyuanOCR cache check failed: %s", exc)
        return False


def is_hunyuan_model_loaded() -> bool:
    """True when weights are already in GPU/RAM (warm engine)."""
    return _engine is not None and _engine._model is not None


class HunyuanOcrEngine:
    """Lazy-loaded HunyuanOCR (Transformers or vLLM HTTP)."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._infer_lock = threading.Lock()
        self._use_vllm = bool(VLLM_URL)

    def _ensure_local_model(self) -> None:
        if self._model is not None:
            return
        import torch

        AutoProcessor, HunYuanVLForConditionalGeneration = _import_hunyuan_classes()
        cached = is_hunyuan_model_cached()
        if cached:
            logger.info("HunyuanOCR weights found in cache — loading into GPU …")
        else:
            logger.info("HunyuanOCR not in cache — downloading %s from HuggingFace …", MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
        dtype = torch.bfloat16
        if torch.cuda.is_available():
            try:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                dtype = torch.float16
        self._model = HunYuanVLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            attn_implementation="eager",
            torch_dtype=dtype,
            device_map="auto",
        )
        self._model.eval()
        self._device = str(next(self._model.parameters()).device)
        logger.info("HunyuanOCR ready on %s", self._device)

    def _ocr_via_vllm(self, img: Image.Image, prompt: str) -> str:
        from openai import OpenAI

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        client = OpenAI(api_key=os.environ.get("HUNYUAN_OCR_VLLM_KEY", "EMPTY"), base_url=VLLM_URL, timeout=120.0)
        resp = client.chat.completions.create(
            model=os.environ.get("HUNYUAN_OCR_VLLM_MODEL", MODEL_ID),
            messages=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
        return clean_repeated_substrings(text)

    def _ocr_via_transformers(self, img: Image.Image, prompt: str) -> str:
        import torch

        self._ensure_local_model()
        assert self._processor is not None and self._model is not None

        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "frame.png"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        rgb = img.convert("RGB")
        texts = [
            self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in [messages]
        ]
        inputs = self._processor(
            text=texts,
            images=rgb,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)
        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated_ids)]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = decoded[0] if decoded else ""
        return clean_repeated_substrings(text.strip())

    def ocr_subtitle(self, img: Image.Image, *, language: str | None = None) -> tuple[str, float]:
        """Return (subtitle_text, pseudo_confidence)."""
        prompt = _subtitle_prompt(language)
        with self._infer_lock:
            try:
                if self._use_vllm:
                    text = self._ocr_via_vllm(img, prompt)
                else:
                    text = self._ocr_via_transformers(img, prompt)
            except Exception as exc:
                logger.warning("HunyuanOCR inference failed: %s", exc)
                raise RuntimeError(f"HunyuanOCR lỗi: {exc}") from exc
        text = " ".join(text.split()).strip()
        return text, (0.92 if text else 0.0)


def get_engine() -> HunyuanOcrEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                ok, err = is_hunyuan_available()
                if not ok:
                    raise RuntimeError(err or "HunyuanOCR không khả dụng.")
                _engine = HunyuanOcrEngine()
    return _engine


def ocr_subtitle_image(img: Image.Image, *, language: str | None = None) -> tuple[str, float]:
    return get_engine().ocr_subtitle(img, language=language)
