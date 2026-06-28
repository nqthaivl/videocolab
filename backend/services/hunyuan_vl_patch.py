"""Register HunYuanVL with transformers 5.x without downgrading the package.

OmniVoice TTS needs transformers>=5.3 (HiggsAudioV2TokenizerModel).
HunyuanOCR needs HunYuanVL classes from a specific transformers commit.
This module vendors those model files under backend/vendor/hunyuan_vl/.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import types
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger("omnivoice.hunyuan_vl_patch")

_COMMIT = "82a06db03535c49aa987719ed0746a76093b1ec4"
_ZIP_URL = f"https://github.com/huggingface/transformers/archive/{_COMMIT}.zip"
_ZIP_PREFIX = f"transformers-{_COMMIT}/src/transformers/models/hunyuan_vl/"
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "hunyuan_vl"

_PATCHED = False
_PATCH_LOCK = threading.Lock()

_MODULE_ORDER = (
    "configuration_hunyuan_vl",
    "image_processing_hunyuan_vl",
    "modeling_hunyuan_vl",
    "processing_hunyuan_vl",
)


def _download_vendor() -> None:
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    if (_VENDOR_DIR / "modeling_hunyuan_vl.py").is_file():
        return
    import tempfile

    zip_path = os.path.join(tempfile.gettempdir(), f"transformers-{_COMMIT}.zip")
    logger.info("Downloading HunYuanVL code patch for transformers 5.x …")
    urlretrieve(_ZIP_URL, zip_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.startswith(_ZIP_PREFIX) or not name.endswith(".py"):
                    continue
                rel = name[len(_ZIP_PREFIX) :]
                if not rel or "/" in rel:
                    continue
                (_VENDOR_DIR / rel).write_bytes(zf.read(name))
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass
    if not (_VENDOR_DIR / "modeling_hunyuan_vl.py").is_file():
        raise RuntimeError("Không tải được HunYuanVL patch — kiểm tra kết nối mạng.")


def _load_vendor_modules() -> None:
    pkg_name = "transformers.models.hunyuan_vl"
    import transformers.models as tf_models

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_VENDOR_DIR)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        setattr(tf_models, "hunyuan_vl", pkg)

    for mod_name in _MODULE_ORDER:
        full = f"{pkg_name}.{mod_name}"
        if full in sys.modules:
            continue
        path = _VENDOR_DIR / f"{mod_name}.py"
        if not path.is_file():
            raise RuntimeError(f"Thiếu file HunYuanVL patch: {path.name}")
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Không load được HunYuanVL module: {mod_name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)


def _register_mappings() -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
    from transformers.models.hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig
    from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor
    from transformers.models.hunyuan_vl.modeling_hunyuan_vl import HunYuanVLForConditionalGeneration
    from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor

    CONFIG_MAPPING.register("hunyuan_vl", HunYuanVLConfig, exist_ok=True)
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING.register(
        HunYuanVLConfig, HunYuanVLForConditionalGeneration, exist_ok=True
    )
    PROCESSOR_MAPPING.register(HunYuanVLConfig, HunYuanVLProcessor, exist_ok=True)
    try:
        IMAGE_PROCESSOR_MAPPING.register(HunYuanVLConfig, HunYuanVLImageProcessor, exist_ok=True)
    except Exception as exc:
        logger.debug("HunYuanVL image processor mapping skipped: %s", exc)

    import transformers

    transformers.HunYuanVLForConditionalGeneration = HunYuanVLForConditionalGeneration
    transformers.HunYuanVLProcessor = HunYuanVLProcessor


def ensure_hunyuan_vl() -> None:
    """Load HunYuanVL into the active transformers install (idempotent)."""
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        try:
            from transformers import HunYuanVLForConditionalGeneration  # noqa: F401

            _PATCHED = True
            return
        except ImportError:
            pass
        _download_vendor()
        _load_vendor_modules()
        _register_mappings()
        _PATCHED = True
        logger.info("HunYuanVL patch registered with transformers %s", _transformers_version())


def _transformers_version() -> str:
    try:
        import transformers

        return getattr(transformers, "__version__", "?")
    except Exception:
        return "?"
