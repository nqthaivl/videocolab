"""Standalone video URL download (yt-dlp) for the Download Video tab."""
from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class DownloadVideoRequest(BaseModel):
    url: str
    output_dir: str = Field(..., description="Absolute path to save the file")
    mp4_only: bool = True


@router.post("/download/video-url")
async def download_video_url(req: DownloadVideoRequest):
    url = (req.url or "").strip()
    output_dir = (req.output_dir or "").strip()

    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL phải bắt đầu bằng http:// hoặc https://")

    if not output_dir or not os.path.isdir(output_dir):
        raise HTTPException(status_code=400, detail="Thư mục lưu không tồn tại hoặc không hợp lệ.")

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="yt-dlp chưa được cài. Chạy `pip install yt-dlp` và khởi động lại backend.",
        )

    loop = asyncio.get_running_loop()
    try:
        from services.video_url_download import download_video_to_dir

        result = await loop.run_in_executor(
            None,
            lambda: download_video_to_dir(url, output_dir, mp4_only=req.mp4_only),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"success": True, **result}
