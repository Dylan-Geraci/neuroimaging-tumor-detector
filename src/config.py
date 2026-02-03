"""
Centralized application configuration using Pydantic BaseSettings.

Reads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "models/best_model.pth"
    host: str = "0.0.0.0"
    port: int = 8000
    num_classes: int = 4
    image_size: int = 224
    cors_origins: list[str] = ["*"]
    database_url: str = "sqlite:///./predictions.db"

    model_config = {"env_prefix": "APP_"}


settings = Settings()
