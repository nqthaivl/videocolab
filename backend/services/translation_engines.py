"""
Translation engine registry + UI-driven install/uninstall.

This is the single source of truth for which translation providers we know
about, what pip package they need, and whether that package is importable
right now. The Engine dropdown in the Dub tab reads list_engines() to
decide which options are ready-to-use vs. "needs install".

Why a registry rather than inline probes in dub_translate.py? The UI wants
to render the availability table BEFORE the user clicks Translate, so we
don't surface a cryptic ModuleNotFoundError for every segment. Having the
registry live next to the dub_translate dispatch also means adding a new
engine is one entry here + one branch in _build_translator.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger("omnivoice.translation_engines")


# Engine ID → registry entry. Keyed by the `provider` string sent from the
# frontend (must match the values of `translateProvider` in the store).
REGISTRY: dict[str, dict] = {
    "argos": {
        "id": "argos",
        "display_name": "Argos (Local, Fast)",
        "pip_package": "argostranslate",
        "probe_module": "argostranslate",
        "category": "offline",
        "needs_key": False,
        "builtin": True,
        "notes": "Pure-CPU offline translator. Downloads a ~50MB language pack on first use per pair.",
    },
    "nllb": {
        "id": "nllb",
        "display_name": "NLLB-200 600M (Local, Heavy)",
        "pip_package": None,          # uses HF transformers — already a core dep
        "probe_module": "transformers",
        "category": "offline",
        "needs_key": False,
        "builtin": True,
        "notes": "Meta's 200-language NMT model (600M). ~2.4GB download. Covers all language pairs.",
    },
    "nllb_1_3b": {
        "id": "nllb_1_3b",
        "display_name": "NLLB-200 1.3B (Local, Best Quality)",
        "pip_package": None,          # uses HF transformers — already a core dep
        "probe_module": "transformers",
        "category": "offline",
        "needs_key": False,
        "builtin": True,
        "notes": "Meta's NLLB-200 distilled 1.3B — significantly better quality than 600M, especially for Asian languages (Chinese↔Vietnamese). ~5.2GB download. Recommended for Zh→Vi.",
    },
    "marian_zh_vi": {
        "id": "marian_zh_vi",
        "display_name": "MarianMT Zh→Vi (Local, Specialized)",
        "pip_package": None,          # uses HF transformers + sentencepiece
        "probe_module": "transformers",
        "category": "offline",
        "needs_key": False,
        "builtin": True,
        "notes": "Helsinki-NLP opus-mt-zh-vi: lightweight (~300MB), specialized only for Chinese→Vietnamese. Much faster than NLLB and significantly better Zh→Vi quality than NLLB-600M.",
    },
    "google": {
        "id": "google",
        "display_name": "Google Translate (Online, Free)",
        "pip_package": "deep_translator",
        "probe_module": "deep_translator",
        "category": "online",
        "needs_key": False,
        "notes": "Free web endpoint via deep_translator. Rate-limited by Google; no API key required.",
    },
    "deepl": {
        "id": "deepl",
        "display_name": "DeepL (Online, Key)",
        "pip_package": "deep_translator",
        "probe_module": "deep_translator",
        "category": "online",
        "needs_key": True,
        "notes": "High-quality EU MT. Free tier: 500K chars/month. Set DEEPL_API_KEY.",
    },
    "microsoft": {
        "id": "microsoft",
        "display_name": "Microsoft Translator (Online, Key)",
        "pip_package": "deep_translator",
        "probe_module": "deep_translator",
        "category": "online",
        "needs_key": True,
        "notes": "Azure Cognitive Services. Free tier: 2M chars/month. Set MICROSOFT_API_KEY.",
    },
    "mymemory": {
        "id": "mymemory",
        "display_name": "MyMemory (Online, No Key)",
        "pip_package": "deep_translator",
        "probe_module": "deep_translator",
        "category": "online",
        "needs_key": False,
        "notes": "Crowdsourced MT. Free, 5K chars/day anonymous; more with an email param.",
    },
    "openai": {
        "id": "openai",
        "display_name": "LLM (OpenAI-compatible)",
        "pip_package": "openai",
        "probe_module": "openai",
        "category": "llm",
        "needs_key": True,
        "notes": (
            "Any OpenAI-compatible endpoint: GPT-4/5 (OpenAI), Claude (via OpenRouter), "
            "Gemini (OpenAI-compat mode), DeepSeek, Qwen, Ollama, LM Studio. "
            "Set TRANSLATE_BASE_URL + TRANSLATE_API_KEY + TRANSLATE_MODEL."
        ),
    },
    "gemini": {
        "id": "gemini",
        "display_name": "Google Gemini (Online, Key)",
        "pip_package": "openai",
        "probe_module": "openai",
        "category": "llm",
        "needs_key": True,
        "notes": (
            "Google Gemini via OpenAI-compatible endpoint. "
            "Set GEMINI_TRANSLATE_API_KEY in Settings → API dịch cloud."
        ),
    },
    "deepseek": {
        "id": "deepseek",
        "display_name": "DeepSeek (Online, Key)",
        "pip_package": "openai",
        "probe_module": "openai",
        "category": "llm",
        "needs_key": True,
        "notes": (
            "DeepSeek via OpenAI-compatible API (https://api.deepseek.com/v1). "
            "Set DEEPSEEK_TRANSLATE_API_KEY in Settings → API dịch cloud."
        ),
    },
    "9router": {
        "id": "9router",
        "display_name": "9Router (Local proxy, Key)",
        "pip_package": "openai",
        "probe_module": "openai",
        "category": "llm",
        "needs_key": True,
        "notes": (
            "Local 9Router proxy (default http://localhost:20128/v1). "
            "Routes to 40+ providers (Claude, Gemini, OpenAI, …). "
            "Set NINEROUTER_TRANSLATE_API_KEY in Settings → API dịch cloud."
        ),
    },
    "llama_cpp": {
        "id": "llama_cpp",
        "display_name": "llama.cpp (local LLM)",
        "pip_package": "openai",
        "probe_module": "openai",
        "category": "llm",
        "needs_key": False,
        "notes": (
            "Local llama-server with OpenAI-compatible API (default http://127.0.0.1:8080/v1). "
            "Download a GGUF translation model in Settings → Models, then select it below."
        ),
    },
}


def is_frozen() -> bool:
    """True when running inside a packaged Tauri / PyInstaller bundle.

    In that case the Python site-packages is read-only and signed, so we
    refuse install/uninstall requests instead of corrupting the bundle.
    """
    return bool(getattr(sys, "frozen", False) or os.environ.get("OMNIVOICE_FROZEN"))


def _probe(entry: dict) -> tuple[bool, str]:
    mod = entry.get("probe_module")
    if not mod:
        return True, "no module required"
    try:
        importlib.import_module(mod)
        return True, "ready"
    except ImportError as e:
        return False, f"import {mod!r} failed: {e}"


def list_engines() -> list[dict]:
    """Return a UI-ready list with per-engine availability stamped in."""
    out = []
    for e in REGISTRY.values():
        installed, reason = _probe(e)
        out.append({
            **e,
            "installed": installed,
            "availability_reason": reason,
        })
    return out


def get_engine(engine_id: str) -> dict | None:
    return REGISTRY.get(engine_id)


def is_installed(engine_id: str) -> bool:
    entry = REGISTRY.get(engine_id)
    if not entry:
        return False
    ok, _ = _probe(entry)
    return ok


def _in_virtualenv() -> bool:
    """True if the current interpreter is inside a venv/virtualenv."""
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or hasattr(sys, "real_prefix")


def _installer_cmd() -> list[str]:
    """Prefer `uv pip` (the dev install's default), fall back to `python -m pip`.

    `python -m pip` ensures we target the same interpreter the server is
    running under — avoids the classic "pip installed into the wrong venv"
    footgun.
    """
    if shutil.which("uv"):
        return ["uv", "pip"]
    return [sys.executable, "-m", "pip"]


async def run_pip(args: list[str], timeout: float = 600.0) -> tuple[int, str]:
    """Run a pip command async and return (rc, combined_output).

    Combines stdout + stderr so the UI can surface a useful tail on failure
    (pip's "ERROR: ..." lines go to stderr).

    When using `uv pip` and running outside a venv (e.g. inside the Docker
    image where Python runs as system), inject `--system` after the
    install/uninstall subcommand. Without it, uv refuses to write to system
    Python with: "No virtual environment found; run `uv venv` to create an
    environment, or pass `--system`...". The `UV_SYSTEM_PYTHON` env var only
    affects `uv venv`, not `uv pip install`.
    """
    base = _installer_cmd()
    using_uv = base[:1] == ["uv"]
    # Pin `uv pip` to the interpreter the backend ACTUALLY runs under. The desktop
    # spawns `<venv>/bin/python -m uvicorn` WITHOUT exporting VIRTUAL_ENV, so bare
    # `uv pip install` finds no venv and 500s with "No virtual environment found"
    # (#529/#527) — and the `--system` branch below never fires, because the
    # running interpreter genuinely IS in a venv (uv just can't auto-discover it).
    # `--python sys.executable` targets the same interpreter _probe()/is_installed()
    # import from, and takes precedence when both flags are present, so the Docker
    # `--system` path is unaffected.
    if using_uv and args and args[0] in ("install", "uninstall") and "--python" not in args:
        args = [args[0], "--python", sys.executable, *args[1:]]
    if using_uv and not _in_virtualenv() and args and args[0] in ("install", "uninstall") and "--system" not in args:
        args = [args[0], "--system", *args[1:]]
    cmd = base + args
    logger.info("pip: %s", " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except NotImplementedError:
        logger.debug("asyncio subprocess not supported, falling back to thread-based subprocess")
        return await _run_pip_thread(cmd, timeout)
    except FileNotFoundError as e:
        return 1, f"installer not found: {e}"
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return 1, f"pip timed out after {timeout:.0f}s"
    out = stdout.decode(errors="replace") if stdout else ""
    return proc.returncode or 0, out


async def _run_pip_thread(cmd: list[str], timeout: float) -> tuple[int, str]:
    """Fallback: run pip in a thread via subprocess.Popen (Windows compat)."""
    loop = asyncio.get_running_loop()

    def _run():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            return 1, f"pip timed out after {timeout:.0f}s"
        out = stdout.decode(errors="replace") if stdout else ""
        return proc.returncode or 0, out

    return await loop.run_in_executor(None, _run)
