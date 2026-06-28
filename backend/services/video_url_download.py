"""Download a public video URL to a user-chosen folder via yt-dlp."""
from __future__ import annotations

import logging
import os
import re
import uuid

logger = logging.getLogger("omnivoice.video_url_download")

_SAFE_NAME = re.compile(r"[^\w\s\-_.()\[\]]+", re.UNICODE)


def _sanitize_filename(name: str, fallback: str) -> str:
    cleaned = _SAFE_NAME.sub("_", (name or "").strip()).strip("._ ")
    return cleaned[:120] if cleaned else fallback


def download_video_to_dir(
    url: str,
    output_dir: str,
    *,
    mp4_only: bool = True,
    progress_hook=None,
) -> dict:
    """Blocking yt-dlp download into *output_dir*. Returns paths + metadata."""
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    temp_id = uuid.uuid4().hex[:8]
    outtmpl = os.path.join(output_dir, f"%(title).100s [{temp_id}].%(ext)s")

    ydl_opts: dict = {
        "outtmpl": outtmpl,
        "format": (
            "bv*[vcodec^=avc1][ext=mp4]+ba[acodec^=mp4a][ext=m4a]/"
            "bv*[vcodec^=avc1]+ba[acodec^=mp4a]/"
            "b[vcodec^=avc1][acodec^=mp4a]/"
            "bv*+ba/b"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "socket_timeout": 30,
        "fragment_retries": 10,
        "retries": 10,
        "extractor_retries": 5,
        "skip_unavailable_fragments": True,
    }
    if not mp4_only:
        ydl_opts["writethumbnail"] = True
        ydl_opts["writeinfojson"] = True
    if progress_hook is not None:
        ydl_opts["progress_hooks"] = [progress_hook]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        root, _ext = os.path.splitext(path)
        mp4_path = root + ".mp4"
        video_path = mp4_path if os.path.exists(mp4_path) else path

    title = info.get("title") or os.path.basename(video_path)
    video_id = str(info.get("id") or temp_id)
    safe_title = _sanitize_filename(title, video_id)
    final_name = f"{safe_title}.mp4" if mp4_only else os.path.basename(video_path)
    final_path = os.path.join(output_dir, final_name)

    if os.path.abspath(video_path) != os.path.abspath(final_path):
        try:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.replace(video_path, final_path)
            video_path = final_path
        except OSError as err:
            logger.warning("Could not rename %s -> %s: %s", video_path, final_path, err)

    return {
        "id": video_id,
        "title": title,
        "path": video_path,
        "filename": os.path.basename(video_path),
        "is_folder": False,
    }
