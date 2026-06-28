"""Settings API — HF token save/clear/state endpoints (Phase 1 AUTH-03 backend half).

These endpoints are the backend half of the Wave 2 Settings → API Keys
panel. Threat T-01-03 mitigation: every write endpoint is gated by the
router-level `require_loopback` dep, so non-loopback origins get 403
before the handler runs. Reads are loopback-gated too — the masked
token preview is useful telemetry that we still don't want exposed on
the LAN.

The state endpoint duplicates `/system/hf-token/state` (which lives on
`system.py` for legacy-router compatibility); both return the same shape.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import require_loopback

logger = logging.getLogger("omnivoice.api.settings")

router = APIRouter(
    prefix="/api/settings",
    tags=["settings"],
    dependencies=[Depends(require_loopback)],
)


class _HFTokenBody(BaseModel):
    token: str = Field(..., min_length=1, description="HuggingFace access token")


def _state_response() -> dict:
    """Return the same shape the React panel renders. Never includes raw token."""
    from services import token_resolver

    s = token_resolver.state()
    return {
        "active": s["active"],
        "sources": [asdict(row) for row in s["sources"]],
    }


@router.post("/hf-token")
def save_hf_token(body: _HFTokenBody):
    """Persist a new HF token to the encrypted settings store + the HF
    canonical file (via huggingface_hub.login). Returns the updated
    cascade state."""
    token = body.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="token must be non-empty")
    from services import token_resolver
    try:
        token_resolver.save_app_token(token)
    except Exception:
        logger.exception("save_app_token failed")
        raise HTTPException(status_code=500, detail="Failed to save HF token")
    return _state_response()


@router.delete("/hf-token")
def clear_hf_token(also_clear_hf_cli: bool = Query(False)):
    """Clear the App-source token. Optionally also call huggingface_hub.logout
    to clear the canonical HF file. Returns the updated cascade state."""
    from services import token_resolver
    try:
        token_resolver.clear_app_token(also_clear_hf_cli=also_clear_hf_cli)
    except Exception:
        logger.exception("clear_app_token failed")
        raise HTTPException(status_code=500, detail="Failed to clear HF token")
    return _state_response()


@router.get("/hf-token/state")
def get_hf_token_state():
    """3-source HF token cascade state for the Settings UI."""
    return _state_response()


# ── Performance settings (INST-12) ────────────────────────────────────────
# Threat T-02-04: same loopback guard as the hf-token endpoints via the
# router-level `require_loopback` dep.


_TORCH_COMPILE_KEY = "perf.torch_compile_disabled"


class _TorchCompileBody(BaseModel):
    enabled: bool = Field(..., description="True to set TORCH_COMPILE_DISABLE=1 on engine subprocesses")


def _torch_compile_state() -> dict:
    import sys
    from services import settings_store

    raw = settings_store.get_text(_TORCH_COMPILE_KEY, "0")
    return {"enabled": raw == "1", "platform": sys.platform}


@router.get("/perf/torch-compile-disabled")
def get_torch_compile_disabled():
    """Return the current torch.compile-disabled toggle + the runtime platform.
    UI uses the platform to render the toggle disabled (with an explainer)
    on non-Windows hosts, since the OOM is Windows-specific (issue #65)."""
    return _torch_compile_state()


@router.put("/perf/torch-compile-disabled")
def set_torch_compile_disabled(body: _TorchCompileBody):
    """Persist the toggle. Honoured by `services.engine_env.build_engine_env()`
    which injects TORCH_COMPILE_DISABLE=1 on Windows when enabled."""
    from services import settings_store

    try:
        settings_store.set_text(_TORCH_COMPILE_KEY, "1" if body.enabled else "0")
    except Exception:
        logger.exception("set_torch_compile_disabled failed")
        raise HTTPException(status_code=500, detail="Failed to persist setting")
    return _torch_compile_state()


# ── Dictation refinement (parity program Wave 2.1 / Spec 3 phase 2) ───────


class _RefinementBody(BaseModel):
    auto: bool | None = None
    smart_cleanup: bool | None = None
    self_correction: bool | None = None
    preserve_technical: bool | None = None


def _refinement_state():
    from services.refinement import get_refinement_config
    from services.llm_backend import get_active_llm_backend

    cfg = get_refinement_config()
    # The UI shows whether refinement can actually run (needs an LLM).
    cfg["llm_ready"] = get_active_llm_backend().id != "off"
    return cfg


@router.get("/dictation-refinement")
def get_dictation_refinement():
    """Current refinement config + whether an LLM backend is configured."""
    return _refinement_state()


@router.put("/dictation-refinement")
def set_dictation_refinement(body: _RefinementBody):
    from services.refinement import set_refinement_config

    try:
        set_refinement_config({k: v for k, v in body.model_dump().items() if v is not None})
    except Exception:
        logger.exception("set_dictation_refinement failed")
        raise HTTPException(status_code=500, detail="Failed to persist setting")
    return _refinement_state()


# ── LLM endpoint (parity program Wave 2.4 / §R2 rung 4) ───────────────────
# Focused configuration for the OpenAI-compatible LLM endpoint that powers
# cinematic translate, glossary auto-extract, and dictation refinement.
# Persistence rides the existing TRANSLATE_BASE_URL / TRANSLATE_API_KEY /
# TRANSLATE_MODEL env vars (already in system.py PERSISTENT_KEYS, restored
# at startup) so the resolution path in llm_backend/translator is unchanged.


class _LLMEndpointBody(BaseModel):
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None  # None = leave unchanged; "" = clear


def _mask(secret: str | None) -> str | None:
    if not secret:
        return None
    return f"…{secret[-4:]}" if len(secret) > 4 else "set"


def _llm_endpoint_state():
    from services.llm_backend import OpenAICompatBackend

    ok, reason = OpenAICompatBackend.is_available()
    return {
        "base_url": os.environ.get("TRANSLATE_BASE_URL", ""),
        "model": os.environ.get("TRANSLATE_MODEL", ""),
        "api_key_masked": _mask(
            os.environ.get("TRANSLATE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        ),
        "available": ok,
        "reason": None if ok else reason,
    }


@router.get("/llm-endpoint")
def get_llm_endpoint():
    """Current OpenAI-compatible LLM endpoint config + live availability."""
    return _llm_endpoint_state()


@router.put("/llm-endpoint")
def set_llm_endpoint(body: _LLMEndpointBody):
    """Persist base URL / model / API key for the OpenAI-compatible endpoint.

    Reuses the env-var persistence path (prefs.json, restored at startup):
    base_url -> TRANSLATE_BASE_URL, model -> TRANSLATE_MODEL,
    api_key -> TRANSLATE_API_KEY. A None field is left unchanged; an empty
    string clears it. Ollama ignores the key; vLLM / LM Studio require it.
    """
    from core.prefs import set_ as prefs_set, delete as prefs_delete

    mapping = {
        "TRANSLATE_BASE_URL": body.base_url,
        "TRANSLATE_MODEL": body.model,
        "TRANSLATE_API_KEY": body.api_key,
    }
    for env_key, val in mapping.items():
        if val is None:
            continue  # untouched
        val = val.strip()
        if val:
            os.environ[env_key] = val
            prefs_set(f"env.{env_key}", val)
        else:
            os.environ.pop(env_key, None)
            prefs_delete(f"env.{env_key}")
    # get_active_llm_backend() builds a fresh backend (and its OpenAI client
    # reads env at construction) on every call, so there's no singleton to
    # invalidate — the next translate/refine picks up the new values.
    return _llm_endpoint_state()


@router.post("/llama-server/ensure")
async def ensure_llama_server():
    """Start llama-server with the GGUF matching TRANSLATE_MODEL, if needed."""
    from services.llama_server import ensure_llama_server_for_model, is_translategemma_model, status as llama_status

    model_name = os.environ.get("TRANSLATE_MODEL", "").strip()
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Chưa chọn model llama.cpp. Tải model tại Cấu hình → Model dịch.",
        )
    if is_translategemma_model(model_name):
        return {
            "ok": True,
            "reason": "TranslateGemma dùng HuggingFace trực tiếp (không cần llama-server).",
            **llama_status(),
        }
    ok, reason = await ensure_llama_server_for_model(model_name)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    return {"ok": True, "reason": reason, **llama_status()}


@router.get("/llama-server/status")
def get_llama_server_status():
    from services.llama_server import status as llama_status

    return llama_status()


# ── Cloud translation APIs (OpenAI, Google Gemini, DeepSeek, 9Router) ─────────


class _CloudProviderBody(BaseModel):
    api_key: str | None = None  # None = unchanged; "" = clear
    model: str | None = None
    base_url: str | None = None  # OpenAI only


class _TranslateCloudBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    openai: _CloudProviderBody | None = None
    gemini: _CloudProviderBody | None = None
    deepseek: _CloudProviderBody | None = None
    nine_router: _CloudProviderBody | None = Field(None, alias="9router")


def _persist_env_fields(mapping: dict[str, str | None]) -> None:
    from core.prefs import set_ as prefs_set, delete as prefs_delete

    for env_key, val in mapping.items():
        if val is None:
            continue
        val = val.strip()
        if val:
            os.environ[env_key] = val
            prefs_set(f"env.{env_key}", val)
        else:
            os.environ.pop(env_key, None)
            prefs_delete(f"env.{env_key}")


@router.get("/translate-cloud")
def get_translate_cloud():
    """OpenAI + Gemini translation API settings for the translate dropdown."""
    from services.cloud_translate import translate_cloud_state

    return translate_cloud_state()


@router.put("/translate-cloud")
def set_translate_cloud(body: _TranslateCloudBody):
    """Persist cloud translation API keys and model names."""
    from services.cloud_translate import translate_cloud_state

    if body.openai is not None:
        _persist_env_fields({
            "OPENAI_TRANSLATE_API_KEY": body.openai.api_key,
            "OPENAI_TRANSLATE_MODEL": body.openai.model,
            "OPENAI_TRANSLATE_BASE_URL": body.openai.base_url,
        })
    if body.gemini is not None:
        _persist_env_fields({
            "GEMINI_TRANSLATE_API_KEY": body.gemini.api_key,
            "GEMINI_TRANSLATE_MODEL": body.gemini.model,
        })
    if body.deepseek is not None:
        _persist_env_fields({
            "DEEPSEEK_TRANSLATE_API_KEY": body.deepseek.api_key,
            "DEEPSEEK_TRANSLATE_MODEL": body.deepseek.model,
            "DEEPSEEK_TRANSLATE_BASE_URL": body.deepseek.base_url,
        })
    if body.nine_router is not None:
        _persist_env_fields({
            "NINEROUTER_TRANSLATE_API_KEY": body.nine_router.api_key,
            "NINEROUTER_TRANSLATE_MODEL": body.nine_router.model,
            "NINEROUTER_TRANSLATE_BASE_URL": body.nine_router.base_url,
        })
    return translate_cloud_state()


@router.get("/translate-cloud/models/openai")
def list_openai_translate_models(
    api_key: str | None = Query(None, description="Optional preview key before saving"),
    base_url: str | None = Query(None, description="Optional custom OpenAI-compatible base URL"),
):
    from services.cloud_translate import fetch_openai_models

    return fetch_openai_models(api_key=api_key, base_url=base_url)


@router.get("/translate-cloud/models/gemini")
def list_gemini_translate_models(
    api_key: str | None = Query(None, description="Optional preview key before saving"),
):
    from services.cloud_translate import fetch_gemini_models

    return fetch_gemini_models(api_key=api_key)


@router.get("/translate-cloud/models/deepseek")
def list_deepseek_translate_models(
    api_key: str | None = Query(None, description="Optional preview key before saving"),
    base_url: str | None = Query(None, description="Optional custom DeepSeek base URL"),
):
    from services.cloud_translate import fetch_deepseek_models

    return fetch_deepseek_models(api_key=api_key, base_url=base_url)


@router.get("/translate-cloud/models/9router")
def list_ninerouter_translate_models(
    api_key: str | None = Query(None, description="Optional preview key before saving"),
    base_url: str | None = Query(None, description="Optional custom 9Router base URL"),
):
    from services.cloud_translate import fetch_ninerouter_models

    return fetch_ninerouter_models(api_key=api_key, base_url=base_url)


# ── License acceptance (Phase 3 Plan 03-01 / TTS-05) ──────────────────────
# Frontend ``SupertonicLicenseDialog`` flips the engine-license bit via this
# endpoint. The handler is loopback-gated (router-level dep) and the
# engine_id is allow-listed so an arbitrary string cannot be persisted.
# Threat T-03-04 in the plan frontmatter: this is an honest-acknowledgment
# gate, not a security boundary; the loopback + allow-list keeps the
# attack surface tight regardless.


#: Engines that have an in-tree acceptance dialog. Adding a new engine
#: here means adding a corresponding frontend dialog + a license URLs
#: dict in its constants module. Until that, the API refuses the write.
_LICENSE_ALLOWED_ENGINES: frozenset[str] = frozenset({"supertonic3"})


class _LicenseAcceptBody(BaseModel):
    engine_id: str = Field(..., min_length=1, max_length=64)
    accepted: bool = Field(..., description="True to accept the license terms")


@router.post("/license")
def post_license_acceptance(body: _LicenseAcceptBody) -> dict:
    """Persist a per-engine license-acceptance boolean.

    Returns ``{"ok": True, "engine_id": ..., "accepted": ...}`` so the
    caller can update its UI without a second round-trip. Validation:
    ``engine_id`` must be in the in-tree allow-list ‑‑ refuses arbitrary
    keys so the settings table can't be polluted via this route.
    """
    eid = body.engine_id.strip().lower()
    if eid not in _LICENSE_ALLOWED_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"engine_id {eid!r} is not in the license allow-list "
                f"{sorted(_LICENSE_ALLOWED_ENGINES)}"
            ),
        )
    from services import settings_store
    try:
        settings_store.set_license_accepted(eid, body.accepted)
    except Exception:
        logger.exception("set_license_accepted failed for %s", eid)
        raise HTTPException(status_code=500, detail="Failed to persist license acceptance")
    return {"ok": True, "engine_id": eid, "accepted": bool(body.accepted)}


@router.get("/license/{engine_id}")
def get_license_acceptance(engine_id: str) -> dict:
    """Return ``{"engine_id": ..., "accepted": bool}``.

    Same allow-list as the POST handler so an unknown engine id is a
    400 rather than a silent ``accepted=false`` for a non-existent
    engine.
    """
    eid = engine_id.strip().lower()
    if eid not in _LICENSE_ALLOWED_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"engine_id {eid!r} is not in the license allow-list "
                f"{sorted(_LICENSE_ALLOWED_ENGINES)}"
            ),
        )
    from services import settings_store
    try:
        accepted = settings_store.get_license_accepted(eid)
    except Exception:
        logger.exception("get_license_accepted failed for %s", eid)
        raise HTTPException(status_code=500, detail="Failed to read license acceptance")
    return {"engine_id": eid, "accepted": bool(accepted)}


# ── Storage: configurable models directory (#64) ──────────────────────────
# Where HuggingFace / Torch download model weights. The user's choice is
# persisted durably to the per-user env file as OMNIVOICE_CACHE_DIR, which
# main.py maps to HF_HOME / HF_HUB_CACHE / TORCH_HOME at startup. That env file
# is the *single source of truth*: PUT writes it, GET reads it back — there is
# no second store to diverge from. Takes effect on the next backend restart
# (a storage-location change can't safely move an in-use cache mid-process).
_MODELS_DIR_ENV = "OMNIVOICE_CACHE_DIR"


def _default_models_dir() -> str:
    """huggingface_hub's default cache root, honoring XDG_CACHE_HOME on Linux
    (matches HF so GET reports the *true* default the backend would use)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "huggingface")


def _effective_models_dir() -> str:
    return (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or _default_models_dir()
    )


class _ModelsDirBody(BaseModel):
    path: str = Field(default="", description="Absolute directory; empty clears → default cache")


@router.get("/storage/models-dir")
def get_models_dir():
    """Current models directory: the persisted choice (from the durable env
    file — the same value main.py reads at startup), what's effective in this
    process, and the platform default."""
    from core import user_env

    configured = user_env.get_user_env(_MODELS_DIR_ENV) or None
    return {
        "configured": configured,
        "effective": _effective_models_dir(),
        "default": _default_models_dir(),
        "restart_required": False,
    }


@router.put("/storage/models-dir")
def set_models_dir(body: _ModelsDirBody):
    """Set (or clear, with an empty path) the models download directory.

    Validates the directory is writable, then writes OMNIVOICE_CACHE_DIR to the
    durable per-user env file so main.py applies it on the next launch. The env
    file is the only persisted store, so GET can never diverge from what was
    saved. Returns restart_required=True.
    """
    from core import user_env

    raw = (body.path or "").strip()
    if not raw:
        user_env.unset_user_env(_MODELS_DIR_ENV)
        return {"configured": None, "default": _default_models_dir(), "restart_required": True}

    # Reject control characters / NUL before touching the filesystem: an
    # embedded NUL makes os.makedirs raise ValueError (→ 500). This is also
    # the input-validation barrier for the path before it reaches any fs call
    # (the dir is user-chosen by design — this is a loopback-gated, same-user
    # local file picker, not a cross-privilege boundary).
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        raise HTTPException(status_code=400, detail="Path contains invalid control characters")

    path = os.path.abspath(os.path.expanduser(raw))
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".omnivoice_write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Directory is not writable: {e}") from e
    finally:
        # Best-effort cleanup; a failed remove (concurrent process, perm change)
        # must not leave the request hanging or mask the real error.
        try:
            os.remove(os.path.join(path, ".omnivoice_write_test"))
        except OSError:
            pass

    user_env.set_user_env(_MODELS_DIR_ENV, path)
    return {"configured": path, "effective": _effective_models_dir(), "restart_required": True}


# ── HF mirror endpoint (parity program Wave 4.3 / §R4 c) ──────────────────
# Restricted-network users (e.g. behind the Great Firewall) need to point
# huggingface_hub at a mirror. HF reads HF_ENDPOINT at import time, so a
# change takes effect on the next backend start — persisted to the durable
# per-user env so it survives Tauri/Finder launches that don't inherit a
# shell. Loopback-gated via the router dep.

_HF_ENDPOINT_ENV = "HF_ENDPOINT"

# A few well-known mirrors, surfaced as quick-picks in the UI. hf-mirror.com
# is the community mirror most-used in China; the official endpoint clears it.
_HF_MIRROR_PRESETS = [
    {"label": "Hugging Face (official)", "url": ""},
    {"label": "hf-mirror.com (community, China)", "url": "https://hf-mirror.com"},
]


class _HFMirrorBody(BaseModel):
    url: str = Field("", description="HF_ENDPOINT URL; empty string clears it (official endpoint)")


@router.get("/hf-mirror")
def get_hf_mirror():
    from core import user_env

    configured = user_env.get_user_env(_HF_ENDPOINT_ENV) or ""
    return {
        # The value that will apply after restart (persisted), and what's
        # live in this process (env may differ until then).
        "configured": configured,
        "effective": os.environ.get(_HF_ENDPOINT_ENV, ""),
        "presets": _HF_MIRROR_PRESETS,
    }


@router.put("/hf-mirror")
def set_hf_mirror(body: _HFMirrorBody):
    from core import user_env

    url = (body.url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Mirror URL must start with http(s)://")
    try:
        if url:
            user_env.set_user_env(_HF_ENDPOINT_ENV, url)
            os.environ[_HF_ENDPOINT_ENV] = url  # best-effort for new downloads this session
        else:
            user_env.unset_user_env(_HF_ENDPOINT_ENV)
            os.environ.pop(_HF_ENDPOINT_ENV, None)
    except Exception:
        logger.exception("set_hf_mirror failed")
        raise HTTPException(status_code=500, detail="Failed to persist mirror setting")
    # HF endpoint is read at import time by huggingface_hub, so the override
    # is only guaranteed once the backend restarts.
    return {"configured": url, "restart_required": True, "presets": _HF_MIRROR_PRESETS}
