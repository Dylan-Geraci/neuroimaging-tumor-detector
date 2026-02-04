"""Download model from Hugging Face to local models/."""

from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path


def main():
    print("Downloading from Hugging Face...")
    model_path = hf_hub_download(
        repo_id="YOUR-USERNAME/brain-tumor-classifier",  # TODO: Update this
        filename="best_model.pth",
    )

    dest = Path("models/best_model.pth")
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(model_path, dest)
    print(f"Downloaded to {dest} ({dest.stat().st_size / 1024**2:.1f}MB)")


if __name__ == "__main__":
    main()
