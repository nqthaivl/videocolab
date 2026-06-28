"""Cloud translation API configuration (OpenAI, Google Gemini, DeepSeek, 9Router)."""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger("omnivoice.cloud_translate")

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
NINEROUTER_DEFAULT_MODEL = ""
NINEROUTER_DEFAULT_BASE_URL = "http://localhost:20128/v1"

OPENAI_FALLBACK_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
    "chatgpt-4o-latest",
]

GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

DEEPSEEK_FALLBACK_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
]

NINEROUTER_FALLBACK_MODELS = [
    "google/gemini-2.0-flash",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4",
    "deepseek/deepseek-chat",
    "cc/claude-sonnet-4-5",
]

_OPENAI_SKIP = re.compile(
    r"(embed|whisper|tts|dall-e|davinci|babbage|moderation|realtime|audio|transcri|"
    r"search|image|computer-use|codex-mini)",
    re.I,
)


def _mask(secret: str | None) -> str | None:
    if not secret:
        return None
    return f"…{secret[-4:]}" if len(secret) > 4 else "set"


def openai_translate_state() -> dict:
    api_key = (
        os.environ.get("OPENAI_TRANSLATE_API_KEY")
        or os.environ.get("TRANSLATE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    model = os.environ.get("OPENAI_TRANSLATE_MODEL") or os.environ.get("TRANSLATE_MODEL") or OPENAI_DEFAULT_MODEL
    base_url = os.environ.get("OPENAI_TRANSLATE_BASE_URL") or os.environ.get("TRANSLATE_BASE_URL") or ""
    return {
        "api_key_masked": _mask(api_key),
        "model": model,
        "base_url": base_url,
        "configured": bool(api_key and api_key.strip()),
    }


def gemini_translate_state() -> dict:
    api_key = os.environ.get("GEMINI_TRANSLATE_API_KEY") or ""
    model = os.environ.get("GEMINI_TRANSLATE_MODEL") or GEMINI_DEFAULT_MODEL
    return {
        "api_key_masked": _mask(api_key),
        "model": model,
        "base_url": GEMINI_BASE_URL,
        "configured": bool(api_key.strip()),
    }


def deepseek_translate_state() -> dict:
    api_key = os.environ.get("DEEPSEEK_TRANSLATE_API_KEY") or ""
    model = os.environ.get("DEEPSEEK_TRANSLATE_MODEL") or DEEPSEEK_DEFAULT_MODEL
    base_url = os.environ.get("DEEPSEEK_TRANSLATE_BASE_URL") or ""
    return {
        "api_key_masked": _mask(api_key),
        "model": model,
        "base_url": base_url,
        "configured": bool(api_key.strip()),
    }


def ninerouter_translate_state() -> dict:
    api_key = os.environ.get("NINEROUTER_TRANSLATE_API_KEY") or ""
    model = os.environ.get("NINEROUTER_TRANSLATE_MODEL") or NINEROUTER_DEFAULT_MODEL
    base_url = os.environ.get("NINEROUTER_TRANSLATE_BASE_URL") or NINEROUTER_DEFAULT_BASE_URL
    return {
        "api_key_masked": _mask(api_key),
        "model": model,
        "base_url": base_url,
        "configured": bool(api_key.strip()),
    }


def translate_cloud_state() -> dict:
    return {
        "openai": openai_translate_state(),
        "gemini": gemini_translate_state(),
        "deepseek": deepseek_translate_state(),
        "9router": ninerouter_translate_state(),
    }


def is_cloud_translate_configured(provider: str) -> bool:
    provider = (provider or "").lower()
    if provider == "openai":
        return openai_translate_state()["configured"]
    if provider == "gemini":
        return gemini_translate_state()["configured"]
    if provider == "deepseek":
        return deepseek_translate_state()["configured"]
    if provider in ("9router", "ninerouter"):
        return ninerouter_translate_state()["configured"]
    return False


def resolve_openai_translate() -> tuple[str, str, str] | None:
    """Return (base_url, model, api_key) for OpenAI translation."""
    api_key = (
        os.environ.get("OPENAI_TRANSLATE_API_KEY")
        or os.environ.get("TRANSLATE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key or not str(api_key).strip():
        return None
    model = os.environ.get("OPENAI_TRANSLATE_MODEL") or os.environ.get("TRANSLATE_MODEL") or OPENAI_DEFAULT_MODEL
    base_url = (
        os.environ.get("OPENAI_TRANSLATE_BASE_URL")
        or os.environ.get("TRANSLATE_BASE_URL")
        or OPENAI_DEFAULT_BASE_URL
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1" if base_url else OPENAI_DEFAULT_BASE_URL
    return base_url, model, api_key.strip()


def resolve_gemini_translate() -> tuple[str, str, str] | None:
    """Return (base_url, model, api_key) for Gemini OpenAI-compatible API."""
    api_key = os.environ.get("GEMINI_TRANSLATE_API_KEY")
    if not api_key or not str(api_key).strip():
        return None
    model = os.environ.get("GEMINI_TRANSLATE_MODEL") or GEMINI_DEFAULT_MODEL
    return GEMINI_BASE_URL.rstrip("/"), model, api_key.strip()


def resolve_deepseek_translate() -> tuple[str, str, str] | None:
    """Return (base_url, model, api_key) for DeepSeek OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_TRANSLATE_API_KEY")
    if not api_key or not str(api_key).strip():
        return None
    model = os.environ.get("DEEPSEEK_TRANSLATE_MODEL") or DEEPSEEK_DEFAULT_MODEL
    base_url = (
        os.environ.get("DEEPSEEK_TRANSLATE_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url or DEEPSEEK_DEFAULT_BASE_URL, model, api_key.strip()


def resolve_ninerouter_translate() -> tuple[str, str, str] | None:
    """Return (base_url, model, api_key) for 9Router OpenAI-compatible proxy."""
    api_key = os.environ.get("NINEROUTER_TRANSLATE_API_KEY")
    if not api_key or not str(api_key).strip():
        return None
    model = os.environ.get("NINEROUTER_TRANSLATE_MODEL") or NINEROUTER_DEFAULT_MODEL
    if not model.strip():
        return None
    base_url = (
        os.environ.get("NINEROUTER_TRANSLATE_BASE_URL") or NINEROUTER_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url or NINEROUTER_DEFAULT_BASE_URL, model.strip(), api_key.strip()


def resolve_cloud_translate(provider: str) -> tuple[str, str, str] | None:
    provider = (provider or "").lower()
    if provider == "openai":
        return resolve_openai_translate()
    if provider == "gemini":
        return resolve_gemini_translate()
    if provider == "deepseek":
        return resolve_deepseek_translate()
    if provider in ("9router", "ninerouter"):
        return resolve_ninerouter_translate()
    return None


def _openai_api_key(override: str | None = None) -> str | None:
    key = (
        override
        or os.environ.get("OPENAI_TRANSLATE_API_KEY")
        or os.environ.get("TRANSLATE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    return key.strip() if key and str(key).strip() else None


def _openai_base_url(override: str | None = None) -> str:
    base = (
        override
        or os.environ.get("OPENAI_TRANSLATE_BASE_URL")
        or os.environ.get("TRANSLATE_BASE_URL")
        or OPENAI_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if base and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base or OPENAI_DEFAULT_BASE_URL


def _gemini_api_key(override: str | None = None) -> str | None:
    key = override or os.environ.get("GEMINI_TRANSLATE_API_KEY")
    return key.strip() if key and str(key).strip() else None


def _deepseek_api_key(override: str | None = None) -> str | None:
    key = override or os.environ.get("DEEPSEEK_TRANSLATE_API_KEY")
    return key.strip() if key and str(key).strip() else None


def _deepseek_base_url(override: str | None = None) -> str:
    base = (
        override
        or os.environ.get("DEEPSEEK_TRANSLATE_BASE_URL")
        or DEEPSEEK_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if base and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base or DEEPSEEK_DEFAULT_BASE_URL


def _ninerouter_api_key(override: str | None = None) -> str | None:
    key = override or os.environ.get("NINEROUTER_TRANSLATE_API_KEY")
    return key.strip() if key and str(key).strip() else None


def _ninerouter_base_url(override: str | None = None) -> str:
    base = (
        override
        or os.environ.get("NINEROUTER_TRANSLATE_BASE_URL")
        or NINEROUTER_DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if base and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base or NINEROUTER_DEFAULT_BASE_URL


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_openai_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    if _OPENAI_SKIP.search(mid):
        return False
    return mid.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))


def _sort_openai_models(models: list[str]) -> list[str]:
    priority = {
        "gpt-4.1": 0,
        "gpt-4.1-mini": 1,
        "gpt-4.1-nano": 2,
        "gpt-4o": 3,
        "gpt-4o-mini": 4,
    }

    def key(model_id: str) -> tuple:
        for prefix, rank in priority.items():
            if model_id.startswith(prefix):
                return (rank, model_id)
        return (50, model_id)

    return sorted(set(models), key=key)


def fetch_openai_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """List chat-capable OpenAI models for the settings dropdown."""
    key = _openai_api_key(api_key)
    if not key:
        return {
            "models": list(OPENAI_FALLBACK_MODELS),
            "source": "fallback",
            "error": "Chưa có OpenAI API key — hiển thị danh sách mặc định.",
        }
    url = f"{_openai_base_url(base_url)}/models"
    try:
        payload = _http_get_json(url, headers={"Authorization": f"Bearer {key}"})
        raw = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
        models = _sort_openai_models([m for m in raw if m and _is_openai_chat_model(m)])
        if not models:
            return {
                "models": list(OPENAI_FALLBACK_MODELS),
                "source": "fallback",
                "error": "API không trả về model chat — dùng danh sách mặc định.",
            }
        return {"models": models, "source": "api", "error": None}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        logger.warning("OpenAI models list failed: %s %s", exc.code, detail)
        return {
            "models": list(OPENAI_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách OpenAI ({exc.code}).",
        }
    except Exception as exc:
        logger.warning("OpenAI models list failed: %s", exc)
        return {
            "models": list(OPENAI_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách OpenAI: {exc}",
        }


def _gemini_model_id(name: str) -> str:
    return name.split("/")[-1] if name else ""


def _sort_gemini_models(models: list[str]) -> list[str]:
    def key(model_id: str) -> tuple:
        if model_id.startswith("gemini-2.5"):
            return (0, model_id)
        if model_id.startswith("gemini-2.0"):
            return (1, model_id)
        if model_id.startswith("gemini-1.5"):
            return (2, model_id)
        return (9, model_id)

    return sorted(set(models), key=key)


def fetch_gemini_models(*, api_key: str | None = None) -> dict:
    """List Gemini generateContent models for the settings dropdown."""
    key = _gemini_api_key(api_key)
    if not key:
        return {
            "models": list(GEMINI_FALLBACK_MODELS),
            "source": "fallback",
            "error": "Chưa có Gemini API key — hiển thị danh sách mặc định.",
        }
    url = f"{GEMINI_MODELS_URL}?key={urllib.request.quote(key, safe='')}"
    try:
        payload = _http_get_json(url)
        models: list[str] = []
        for item in payload.get("models", []):
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            model_id = _gemini_model_id(str(item.get("name", "")))
            if not model_id.lower().startswith("gemini"):
                continue
            if "embed" in model_id.lower():
                continue
            models.append(model_id)
        models = _sort_gemini_models(models)
        if not models:
            return {
                "models": list(GEMINI_FALLBACK_MODELS),
                "source": "fallback",
                "error": "API không trả về model Gemini — dùng danh sách mặc định.",
            }
        return {"models": models, "source": "api", "error": None}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        logger.warning("Gemini models list failed: %s %s", exc.code, detail)
        return {
            "models": list(GEMINI_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách Gemini ({exc.code}).",
        }
    except Exception as exc:
        logger.warning("Gemini models list failed: %s", exc)
        return {
            "models": list(GEMINI_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách Gemini: {exc}",
        }


def _is_deepseek_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    return mid.startswith("deepseek")


def _sort_deepseek_models(models: list[str]) -> list[str]:
    priority = {"deepseek-chat": 0, "deepseek-reasoner": 1}

    def key(model_id: str) -> tuple:
        return (priority.get(model_id, 50), model_id)

    return sorted(set(models), key=key)


def fetch_deepseek_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """List chat-capable DeepSeek models for the settings dropdown."""
    key = _deepseek_api_key(api_key)
    if not key:
        return {
            "models": list(DEEPSEEK_FALLBACK_MODELS),
            "source": "fallback",
            "error": "Chưa có DeepSeek API key — hiển thị danh sách mặc định.",
        }
    url = f"{_deepseek_base_url(base_url)}/models"
    try:
        payload = _http_get_json(url, headers={"Authorization": f"Bearer {key}"})
        raw = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
        models = _sort_deepseek_models([m for m in raw if m and _is_deepseek_chat_model(m)])
        if not models:
            return {
                "models": list(DEEPSEEK_FALLBACK_MODELS),
                "source": "fallback",
                "error": "API không trả về model DeepSeek — dùng danh sách mặc định.",
            }
        return {"models": models, "source": "api", "error": None}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        logger.warning("DeepSeek models list failed: %s %s", exc.code, detail)
        return {
            "models": list(DEEPSEEK_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách DeepSeek ({exc.code}).",
        }
    except Exception as exc:
        logger.warning("DeepSeek models list failed: %s", exc)
        return {
            "models": list(DEEPSEEK_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách DeepSeek: {exc}",
        }


def fetch_ninerouter_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """List models exposed by a local 9Router instance (OpenAI-compatible /v1/models)."""
    key = _ninerouter_api_key(api_key)
    if not key:
        return {
            "models": list(NINEROUTER_FALLBACK_MODELS),
            "source": "fallback",
            "error": "Chưa có 9Router API key — hiển thị danh sách mặc định.",
        }
    url = f"{_ninerouter_base_url(base_url)}/models"
    try:
        payload = _http_get_json(url, headers={"Authorization": f"Bearer {key}"})
        raw = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
        models = sorted({m for m in raw if m})
        if not models:
            return {
                "models": list(NINEROUTER_FALLBACK_MODELS),
                "source": "fallback",
                "error": "9Router không trả về model — kiểm tra proxy đang chạy (port 20128).",
            }
        return {"models": models, "source": "api", "error": None}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        logger.warning("9Router models list failed: %s %s", exc.code, detail)
        return {
            "models": list(NINEROUTER_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không lấy được danh sách 9Router ({exc.code}). Proxy đang chạy?",
        }
    except Exception as exc:
        logger.warning("9Router models list failed: %s", exc)
        return {
            "models": list(NINEROUTER_FALLBACK_MODELS),
            "source": "fallback",
            "error": f"Không kết nối 9Router: {exc}. Chạy 9Router tại http://localhost:20128",
        }
