"""
Centralized application configuration using Pydantic BaseSettings.

Reads from environment variables with sensible defaults.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union
import secrets
import json


class Settings(BaseSettings):
    # Application settings
    model_path: str = "models/best_model.pth"
    host: str = "0.0.0.0"
    port: int = 8000
    num_classes: int = 4
    image_size: int = 224
    cors_origins: Union[str, list[str]] = ["*"]
    database_url: str = "sqlite:///./predictions.db"
    log_level: str = "INFO"
    model_source: str = "local"

    # Security settings
    environment: str = "development"
    secret_key: str = secrets.token_urlsafe(32)
    api_keys: Union[str, List[str]] = []  # Empty = no auth in dev
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 10
    max_file_size_mb: int = 50
    max_batch_size: int = 20

    model_config = {
        "env_prefix": "APP_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Ignore extra environment variables
    }

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins from string or list."""
        if v is None or v == "":
            return ["*"]
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Handle comma-separated values
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["*"]
        return ["*"]

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        """Parse comma-separated API keys from string or list."""
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Handle comma-separated values
            keys = [key.strip() for key in v.split(",") if key.strip()]
            return keys if keys else []
        return []

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_keys) > 0


settings = Settings()

# Override port from environment (Render.com uses PORT env var)
if "PORT" in os.environ:
    settings.port = int(os.environ["PORT"])
