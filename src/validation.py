"""
Input validation utilities.

Validates file uploads and request parameters to prevent security issues.
"""

from typing import List, Tuple
from fastapi import UploadFile, HTTPException, status


# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/tiff",
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

# Maximum file size (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Maximum batch size
MAX_BATCH_SIZE = 50


def validate_file_type(file: UploadFile) -> None:
    """
    Validate that uploaded file is an allowed image type.

    Args:
        file: Uploaded file

    Raises:
        HTTPException: If file type is not allowed
    """
    content_type = file.content_type or ""
    filename = file.filename or ""

    # Check MIME type
    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        # Be lenient if filename has valid extension
        if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {content_type}. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}",
            )

    # Check file extension
    if filename and not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}",
        )


async def validate_file_size(file: UploadFile) -> bytes:
    """
    Validate file size and return file contents.

    Args:
        file: Uploaded file

    Returns:
        File contents as bytes

    Raises:
        HTTPException: If file is too large
    """
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    return contents


def validate_batch_size(files: List[UploadFile]) -> None:
    """
    Validate batch upload size.

    Args:
        files: List of uploaded files

    Raises:
        HTTPException: If batch size exceeds limit
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum batch size: {MAX_BATCH_SIZE}",
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )


async def validate_upload(file: UploadFile) -> bytes:
    """
    Perform complete validation on uploaded file.

    Args:
        file: Uploaded file

    Returns:
        Validated file contents

    Raises:
        HTTPException: If validation fails
    """
    validate_file_type(file)
    contents = await validate_file_size(file)
    return contents


async def validate_batch_upload(files: List[UploadFile]) -> List[Tuple[UploadFile, bytes]]:
    """
    Perform complete validation on batch upload.

    Args:
        files: List of uploaded files

    Returns:
        List of (file, contents) tuples for valid files

    Raises:
        HTTPException: If validation fails
    """
    validate_batch_size(files)

    validated = []
    for file in files:
        try:
            validate_file_type(file)
            contents = await validate_file_size(file)
            validated.append((file, contents))
        except HTTPException:
            # Skip invalid files in batch
            continue

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid image files in batch",
        )

    return validated
