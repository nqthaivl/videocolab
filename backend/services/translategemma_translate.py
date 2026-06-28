"""TranslateGemma via HuggingFace transformers (bypasses llama-server).

TranslateGemma's Jinja chat template is incompatible with llama-server init in
recent llama.cpp builds (template verification fails at startup). The official
HF path uses ``apply_chat_template`` with structured messages and works reliably.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("omnivoice.translategemma")

_model = None
_processor = None
_loaded_hf_id: str | None = None


def _iso639(code: str) -> str:
    normalized = (code or "en").strip().lower().replace("_", "-")
    if normalized.startswith("zh") or normalized in ("cmn-hans", "cmn"):
        return "zh"
    return normalized.split("-")[0]


def hf_model_id_for_llama_model(llama_model: str) -> str:
    from api.routers.setup.models import catalog_by_llama_model

    entry = catalog_by_llama_model(llama_model)
    if entry and entry.get("hf_translate_model"):
        return entry["hf_translate_model"]
    if "12b" in llama_model.lower():
        return "google/translategemma-12b-it"
    return "google/translategemma-4b-it"


def unload() -> None:
    global _model, _processor, _loaded_hf_id
    import gc

    _model = None
    _processor = None
    _loaded_hf_id = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _load(hf_id: str):
    global _model, _processor, _loaded_hf_id
    if _model is not None and _processor is not None and _loaded_hf_id == hf_id:
        return _model, _processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Loading TranslateGemma HF model %s", hf_id)
    processor = AutoProcessor.from_pretrained(hf_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(hf_id, **kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")

    _model = model
    _processor = processor
    _loaded_hf_id = hf_id
    return model, processor


def translate_segments(llama_model: str, segments, src_lang: str, target_lang: str) -> list[dict]:
    """Translate segments using the official TranslateGemma chat template."""
    hf_id = hf_model_id_for_llama_model(llama_model)
    try:
        model, processor = _load(hf_id)
    except Exception as exc:
        logger.exception("TranslateGemma model load failed")
        return [
            {"id": seg.id, "text": seg.text, "error": f"Model load error: {exc}"}
            for seg in segments
        ]

    src_iso = _iso639(src_lang)
    tgt_iso = _iso639(target_lang)
    results: list[dict] = []

    for seg in segments:
        if not seg.text or not str(seg.text).strip():
            results.append({"id": seg.id, "text": seg.text})
            continue
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": src_iso,
                            "target_lang_code": tgt_iso,
                            "text": seg.text,
                        }
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            translated = processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not translated:
                results.append({"id": seg.id, "text": seg.text, "error": "empty TranslateGemma response"})
            else:
                results.append({"id": seg.id, "text": translated})
        except Exception as exc:
            logger.warning("TranslateGemma segment %s failed: %s", seg.id, exc)
            results.append({"id": seg.id, "text": seg.text, "error": str(exc)})

    if os.environ.get("OMNIVOICE_UNLOAD_NLLB", "1") == "1":
        unload()

    return results
