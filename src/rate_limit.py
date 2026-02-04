"""
Rate limiting middleware to prevent API abuse.

Implements sliding window rate limiting based on client IP address.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import HTTPException, Request, status


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.

    Note: For production with multiple workers, consider using Redis
    or a dedicated rate limiting service.
    """

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        # Dict[IP, List[timestamp]]
        self.requests: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Try X-Forwarded-For first (for proxies/load balancers)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to direct client
        return request.client.host if request.client else "unknown"

    def _clean_old_requests(self, ip: str, current_time: float):
        """Remove requests outside the sliding window."""
        cutoff_time = current_time - self.window_size
        self.requests[ip] = [
            timestamp for timestamp in self.requests[ip]
            if timestamp > cutoff_time
        ]

    async def check_rate_limit(self, request: Request) -> None:
        """
        Check if request exceeds rate limit.

        Args:
            request: FastAPI request object

        Raises:
            HTTPException: If rate limit is exceeded
        """
        ip = self._get_client_ip(request)
        current_time = time.time()

        # Clean old requests
        self._clean_old_requests(ip, current_time)

        # Check if limit exceeded
        if len(self.requests[ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                headers={"Retry-After": "60"},
            )

        # Add current request
        self.requests[ip].append(current_time)

    def get_rate_limit_info(self, request: Request) -> Tuple[int, int]:
        """
        Get current rate limit status for client.

        Args:
            request: FastAPI request object

        Returns:
            Tuple of (requests_made, requests_remaining)
        """
        ip = self._get_client_ip(request)
        current_time = time.time()

        self._clean_old_requests(ip, current_time)

        requests_made = len(self.requests[ip])
        requests_remaining = max(0, self.requests_per_minute - requests_made)

        return requests_made, requests_remaining


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)
