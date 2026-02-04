<<<<<<< HEAD
"""
Authentication and authorization middleware for FastAPI.

Implements API key-based authentication for production environments.
"""

import os
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
=======
"""API Key authentication for protected endpoints."""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from src.config import settings
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


<<<<<<< HEAD
def get_api_key() -> Optional[str]:
    """Get API key from environment variable."""
    return os.getenv("API_KEY")


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    expected_key = get_api_key()

    # If no API key is configured, allow all requests (development mode)
    if not expected_key:
        return "dev-mode"

    # If API key is configured, enforce it
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include X-API-Key header.",
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
=======
async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key. Bypasses in dev mode (no keys configured)."""
    if not settings.auth_enabled:
        return "dev-bypass"

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
>>>>>>> 40c1d5a3f6c3f560c834e5adff95c2c15a0df926
            detail="Invalid API key",
        )

    return api_key
