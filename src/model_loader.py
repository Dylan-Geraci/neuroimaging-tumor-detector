"""Model loading from multiple sources."""

import os
import torch
from huggingface_hub import hf_hub_download
from src.config import settings
from src.logger import logger


def load_model_checkpoint(device: str) -> dict:
    """Load model from configured source (local or huggingface)."""
    if settings.model_source == "huggingface":
        logger.info("Downloading from Hugging Face Hub...")
        model_path = hf_hub_download(
            repo_id="YOUR-USERNAME/brain-tumor-classifier",  # TODO: Update this
            filename="best_model.pth",
            cache_dir="./models_cache",
        )
        logger.info(f"Downloaded to {model_path}")
    elif settings.model_source == "local":
        model_path = settings.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Set APP_MODEL_SOURCE=huggingface or place model file"
            )
        logger.info(f"Loading from {model_path}")
    else:
        raise ValueError(f"Invalid model_source: {settings.model_source}")

    return torch.load(model_path, map_location=device)
