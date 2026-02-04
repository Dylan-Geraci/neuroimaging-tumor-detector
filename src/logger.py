"""
Centralized logging configuration.

Provides structured logging with proper formatting and log levels.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "brain_tumor_api",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file

    Returns:
        Configured logger instance
    """
    # Determine log level
    if level is None:
        level = "INFO"

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger()


def log_request(method: str, path: str, status_code: int, duration_ms: float):
    """
    Log HTTP request details.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
    """
    logger.info(
        f"{method} {path} - {status_code} - {duration_ms:.2f}ms"
    )


def log_error(error: Exception, context: Optional[str] = None):
    """
    Log error with context.

    Args:
        error: Exception that occurred
        context: Optional context description
    """
    if context:
        logger.error(f"{context}: {str(error)}", exc_info=True)
    else:
        logger.error(str(error), exc_info=True)
