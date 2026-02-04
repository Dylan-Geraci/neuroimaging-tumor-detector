"""Input validation for file uploads."""

from fastapi import HTTPException, UploadFile
from typing import List
from src.config import settings
from src.logger import logger

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


async def validate_file_upload(file: UploadFile) -> bytes:
    """Validate file size, type, and extension."""
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if file_size_mb > settings.max_file_size_mb:
        logger.warning(f"File too large: {file.filename} ({file_size_mb:.2f}MB)")
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size_mb:.1f}MB). Max: {settings.max_file_size_mb}MB",
        )

    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    if not (
        content_type in ALLOWED_CONTENT_TYPES
        or any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    return contents


async def validate_batch_upload(files: List[UploadFile]) -> None:
    """Validate batch constraints."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max batch size: {settings.max_batch_size}",
        )
