"""Centralized logging with JSON formatting for production."""

import logging
import sys
from pythonjsonlogger import jsonlogger
from src.config import settings


def setup_logger(name: str = "brain_tumor_api") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if settings.is_production:
        # JSON for production (log aggregators)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    else:
        # Human-readable for dev
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()
