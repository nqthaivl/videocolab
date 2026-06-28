"""Manage a local llama-server subprocess for llama.cpp translation.

When the user picks a downloaded GGUF model in the translate dropdown, we
auto-start ``llama-server`` (OpenAI-compatible API on 127.0.0.1:8080) so they
do not have to launch it manually.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("omnivoice.llama_server")

# Pinned llama.cpp release — CPU + CUDA Windows builds.
_LLAMA_CPP_RELEASE = "b9821"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080

_proc: subprocess.Popen | None = None
_loaded_gguf: str | None = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BIN_DIR = _REPO_ROOT / "backend" / "bin"


def _platform_slug() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "windows-x86_64"
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "darwin-arm64"
        return "darwin-x86_64"
    return "linux-x86_64"


def _binary_name() -> str:
    slug = _platform_slug()
    name = "llama-server"
    if slug.startswith("windows"):
        name += ".exe"
    return name


def _release_variant() -> str:
    """Pick CPU or CUDA build when downloading."""
    if os.environ.get("LLAMA_SERVER_VARIANT"):
        return os.environ["LLAMA_SERVER_VARIANT"]
    if _platform_slug().startswith("windows"):
        try:
            import torch

            if torch.cuda.is_available():
                return "win-cuda-12.4-x64"
        except Exception:
            pass
        return "win-cpu-x64"
    if _platform_slug().startswith("linux"):
        try:
            import torch

            if torch.cuda.is_available():
                return "ubuntu-cuda-x64"
        except Exception:
            pass
        return "ubuntu-x64"
    if _platform_slug() == "darwin-arm64":
        return "macos-arm64"
    return "macos-x64"


def _install_dir(variant: str | None = None) -> Path:
    """Directory where a full llama.cpp release bundle is extracted."""
    variant = variant or _release_variant()
    return _BIN_DIR / f"llama-{_LLAMA_CPP_RELEASE}-{variant}"


def _bundle_complete(root: Path) -> bool:
    """True when the launcher can load (b9821+ ships a tiny exe + impl DLL)."""
    exe = root / _binary_name()
    if not exe.is_file():
        return False
    if (root / "llama-server-impl.dll").is_file():
        return True
    # Older releases bundled everything into one large binary.
    try:
        return exe.stat().st_size >= 512 * 1024
    except OSError:
        return False


def _resolve_binary() -> Path | None:
    override = os.environ.get("LLAMA_SERVER_PATH")
    if override:
        path = Path(override)
        if path.is_file():
            return path

    install = _install_dir()
    bundled = install / _binary_name()
    if _bundle_complete(install):
        return bundled

    # Legacy broken install: only the 9 KB launcher was copied without DLLs.
    legacy = _BIN_DIR / _binary_name()
    if legacy.is_file() and _bundle_complete(_BIN_DIR):
        return legacy

    found = shutil.which("llama-server")
    if found:
        return Path(found)

    return None


def _extract_full_zip(zip_path: Path, dest_dir: Path) -> Path | None:
    """Extract the entire release zip — the Windows launcher needs sibling DLLs."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    direct = dest_dir / _binary_name()
    if direct.is_file():
        return direct
    for candidate in dest_dir.rglob(_binary_name()):
        return candidate
    return None


def _download_and_extract(variant: str) -> tuple[Path | None, str]:
    url = (
        f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_CPP_RELEASE}/"
        f"llama-{_LLAMA_CPP_RELEASE}-bin-{variant}.zip"
    )
    dest = _install_dir(variant)
    zip_path = _BIN_DIR / f"llama-{_LLAMA_CPP_RELEASE}-{variant}.zip"
    try:
        logger.info("Downloading llama-server bundle from %s", url)
        _BIN_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        extracted = _extract_full_zip(zip_path, dest)
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass
        if extracted and _bundle_complete(dest):
            return extracted, "downloaded"
    except Exception as exc:
        logger.exception("llama-server download failed (%s)", variant)
        return None, (
            f"Không thể tải llama-server ({variant}): {exc}. "
            f"Tải thủ công từ https://github.com/ggml-org/llama.cpp/releases "
            f"và giải nén vào {dest}."
        )
    return None, f"Không tìm thấy llama-server sau khi giải nén ({variant})."


def ensure_binary() -> tuple[Path | None, str]:
    """Return llama-server binary, downloading the full release bundle when missing."""
    existing = _resolve_binary()
    if existing:
        return existing, "ready"

    if os.environ.get("OMNIVOICE_SKIP_LLAMA_SERVER_DOWNLOAD") == "1":
        return None, (
            "Không tìm thấy llama-server. Cài llama.cpp hoặc đặt biến LLAMA_SERVER_PATH "
            "trỏ tới llama-server.exe (cùng thư mục với các file .dll)."
        )

    # Remove broken legacy install that copied only the 9 KB launcher.
    legacy = _BIN_DIR / _binary_name()
    if legacy.is_file() and not _bundle_complete(_BIN_DIR):
        try:
            legacy.unlink()
        except OSError:
            pass

    variants: list[str] = []
    primary = _release_variant()
    variants.append(primary)
    if primary.startswith("win-cuda"):
        variants.append("win-cpu-x64")
    elif primary.startswith("ubuntu-cuda"):
        variants.append("ubuntu-x64")

    last_reason = "Không thể cài llama-server."
    for variant in variants:
        path, reason = _download_and_extract(variant)
        last_reason = reason
        if path:
            return path, reason

    return None, last_reason


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def _server_healthy(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            req = urllib.request.Request(
                f"http://{host}:{port}{path}",
                headers={"Authorization": "Bearer local"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            continue
    return False


def _gpu_layers() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return os.environ.get("LLAMA_SERVER_NGL", "99")
    except Exception:
        pass
    return os.environ.get("LLAMA_SERVER_NGL", "0")


def is_translategemma_model(llama_model: str) -> bool:
    """True for Google TranslateGemma GGUF models with a structured chat template."""
    from api.routers.setup.models import catalog_by_llama_model

    entry = catalog_by_llama_model(llama_model)
    if entry and entry.get("llama_chat_mode") == "translategemma":
        return True
    return "translategemma" in llama_model.lower()


def server_argv_extras(llama_model: str) -> list[str]:
    """Extra llama-server flags required for certain model families."""
    if is_translategemma_model(llama_model):
        # TranslateGemma's Jinja template rejects the dummy message used at init.
        # Keep Jinja enabled so chat_template_kwargs work at inference time.
        return ["--skip-chat-parsing"]
    return []


def _start_server(binary: Path, gguf_path: str, host: str, port: int, llama_model: str = "") -> None:
    global _proc, _loaded_gguf

    if _proc and _proc.poll() is None and _loaded_gguf == gguf_path:
        if _server_healthy(host, port):
            return

    stop_server()

    cmd = [
        str(binary),
        "-m",
        gguf_path,
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        _gpu_layers(),
        "-c",
        os.environ.get("LLAMA_SERVER_CTX", "4096"),
        *server_argv_extras(llama_model),
    ]
    logger.info("Starting llama-server: %s", " ".join(cmd))
    _proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(binary.parent),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
    )
    _loaded_gguf = gguf_path


def _read_process_error(proc: subprocess.Popen | None, limit: int = 800) -> str:
    if not proc or not proc.stderr:
        return ""
    try:
        raw = proc.stderr.read()
        text = raw.decode(errors="replace").strip()
        if len(text) > limit:
            text = text[-limit:]
        return text
    except Exception:
        return ""


def stop_server() -> None:
    global _proc, _loaded_gguf
    if _proc and _proc.poll() is None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
    _proc = None
    _loaded_gguf = None


def resolve_gguf_for_llama_model(llama_model: str) -> tuple[str | None, str]:
    from api.routers.setup.models import (
        catalog_by_llama_model,
        model_hf_repo,
        resolve_gguf_path,
    )

    entry = catalog_by_llama_model(llama_model)
    if not entry:
        return None, f"Không tìm thấy model trong catalog: {llama_model}"
    gguf_pattern = entry.get("gguf_file")
    if not gguf_pattern:
        return None, f"Model {llama_model} không có cấu hình GGUF."
    path = resolve_gguf_path(model_hf_repo(entry), gguf_pattern)
    if not path:
        return None, (
            f"File GGUF chưa được tải. Vào Cấu hình → Model dịch để tải "
            f"\"{entry.get('label', llama_model)}\"."
        )
    return path, "ready"


async def ensure_llama_server_for_model(
    llama_model: str,
    *,
    host: str | None = None,
    port: int | None = None,
    timeout_s: float = 120.0,
) -> tuple[bool, str]:
    """Ensure llama-server is running with the requested GGUF model loaded."""
    host = host or os.environ.get("LLAMA_SERVER_HOST", _DEFAULT_HOST)
    port = int(port or os.environ.get("LLAMA_SERVER_PORT", _DEFAULT_PORT))

    gguf_path, reason = resolve_gguf_for_llama_model(llama_model)
    if not gguf_path:
        return False, reason

    if _server_healthy(host, port) and _loaded_gguf == gguf_path:
        return True, "already running"

    binary, bin_reason = ensure_binary()
    if not binary:
        return False, bin_reason

    variants_to_try: list[str | None] = [None]
    primary_variant = _release_variant()
    if primary_variant.startswith("win-cuda"):
        variants_to_try.append("win-cpu-x64")
    elif primary_variant.startswith("ubuntu-cuda"):
        variants_to_try.append("ubuntu-x64")

    loop = asyncio.get_running_loop()
    last_reason = "llama-server không khởi động được."

    for variant_override in variants_to_try:
        if variant_override:
            os.environ["LLAMA_SERVER_VARIANT"] = variant_override
            shutil.rmtree(_install_dir(primary_variant), ignore_errors=True)
            binary, bin_reason = ensure_binary()
            if not binary:
                last_reason = bin_reason
                continue

        await loop.run_in_executor(
            None, lambda: _start_server(binary, gguf_path, host, port, llama_model)
        )

        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if _proc and _proc.poll() is not None:
                err = _read_process_error(_proc)
                last_reason = f"llama-server thoát sớm.{(' ' + err) if err else ''}"
                logger.error(last_reason)
                stop_server()
                break
            if _server_healthy(host, port):
                return True, "started"
            await asyncio.sleep(0.5)
        else:
            stop_server()
            last_reason = (
                "llama-server không phản hồi trong thời gian chờ. "
                "Kiểm tra VRAM/RAM hoặc thử model nhẹ hơn."
            )
            continue

        if variant_override is None:
            continue
        logger.warning("Retrying llama-server with CPU build after GPU launch failed")

    return False, last_reason


def status() -> dict:
    host = os.environ.get("LLAMA_SERVER_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("LLAMA_SERVER_PORT", _DEFAULT_PORT))
    binary = _resolve_binary()
    return {
        "running": _server_healthy(host, port),
        "loaded_gguf": _loaded_gguf,
        "binary": str(binary) if binary else None,
        "base_url": _base_url(host, port),
        "pid": _proc.pid if _proc and _proc.poll() is None else None,
    }
