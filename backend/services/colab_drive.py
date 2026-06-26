"""Google Drive export helpers for the Colab GPU runtime."""

from __future__ import annotations

import logging
import os
import re
import shutil
from urllib.parse import quote

logger = logging.getLogger("omnivoice.colab_drive")

SUBFOLDER_NAME = "VideoCloneExports"


def is_colab_runtime() -> bool:
    return os.environ.get("OMNIVOICE_COLAB_RUNTIME") == "1"


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    clean = re.sub(r"[^\w.\- ]+", "_", base).strip()
    return clean or "export.bin"


def resolve_export_dir() -> str | None:
    configured = os.environ.get("OMNIVOICE_DRIVE_EXPORT_DIR", "").strip()
    if configured:
        parent = os.path.dirname(configured)
        if parent and os.path.isdir(parent):
            os.makedirs(configured, exist_ok=True)
            return configured

    my_drive = "/content/drive/MyDrive"
    if os.path.isdir(my_drive):
        export_dir = os.path.join(my_drive, SUBFOLDER_NAME)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir
    return None


def drive_status() -> dict:
    export_dir = resolve_export_dir()
    folder_url = os.environ.get("OMNIVOICE_DRIVE_FOLDER_URL", "").strip()
    if not folder_url:
        folder_url = "https://drive.google.com/drive/my-drive"
    return {
        "colab_runtime": is_colab_runtime(),
        "drive_ready": export_dir is not None,
        "export_dir": export_dir,
        "folder_url": folder_url,
        "folder_label": f"My Drive / {SUBFOLDER_NAME}",
    }


def _build_result(dest_path: str, filename: str, media_type: str, size: int) -> dict:
    status = drive_status()
    folder_url = status["folder_url"]
    search_url = f"https://drive.google.com/drive/search?q={quote(filename)}"
    return {
        "saved": True,
        "destination": "google_drive",
        "path": dest_path,
        "drive_path": f"{status['folder_label']} / {filename}",
        "filename": filename,
        "folder_url": folder_url,
        "file_search_url": search_url,
        "open_url": folder_url,
        "size": size,
        "media_type": media_type,
    }


def save_file_to_drive(source_path: str, filename: str, media_type: str) -> dict:
    export_dir = resolve_export_dir()
    if not export_dir:
        raise ValueError(
            "Google Drive chưa sẵn sàng trên Colab. "
            "Hãy mount Drive trong notebook và chạy lại cell khởi động backend."
        )
    if not os.path.isfile(source_path) or os.path.getsize(source_path) == 0:
        raise ValueError("Tệp xuất rỗng hoặc không tồn tại.")

    safe = _safe_filename(filename)
    dest = os.path.join(export_dir, safe)
    shutil.copy2(source_path, dest)
    size = os.path.getsize(dest)
    logger.info("Colab Drive export wrote %s (%d bytes)", dest, size)
    return _build_result(dest, safe, media_type, size)


def save_bytes_to_drive(data: bytes, filename: str, media_type: str) -> dict:
    export_dir = resolve_export_dir()
    if not export_dir:
        raise ValueError(
            "Google Drive chưa sẵn sàng trên Colab. "
            "Hãy mount Drive trong notebook và chạy lại cell khởi động backend."
        )
    if not data:
        raise ValueError("Nội dung xuất rỗng.")

    safe = _safe_filename(filename)
    dest = os.path.join(export_dir, safe)
    with open(dest, "wb") as handle:
        handle.write(data)
    size = len(data)
    logger.info("Colab Drive export wrote %s (%d bytes)", dest, size)
    return _build_result(dest, safe, media_type, size)
