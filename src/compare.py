"""
Compare multiple model checkpoints on the test set.

Produces a markdown table with accuracy, per-class F1, and parameter count.

Usage:
    python -m src.compare --checkpoints models/best_model.pth models/v2.pth
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.model import create_model, count_parameters
from src.data import get_data_loaders, CLASSES


def evaluate_checkpoint(checkpoint_path: str, test_loader, device: str) -> dict:
    """Evaluate a single checkpoint and return metrics."""
    model = create_model(num_classes=len(CLASSES), pretrained=False, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=Path(checkpoint_path).name, leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    trainable, total = count_parameters(model)

    return {
        "checkpoint": Path(checkpoint_path).name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_per_class": {
            CLASSES[i]: f1_score(y_true, y_pred, labels=[i], average="micro")
            for i in range(len(CLASSES))
        },
        "trainable_params": trainable,
        "total_params": total,
        "epoch": checkpoint.get("epoch", "N/A"),
    }


def format_table(results: list[dict]) -> str:
    """Format results as a markdown table."""
    header = "| Checkpoint | Accuracy | F1 (macro) | " + " | ".join(f"F1 {c}" for c in CLASSES) + " | Params | Epoch |"
    sep = "|" + "---|" * (5 + len(CLASSES))

    rows = [header, sep]
    for r in results:
        per_class = " | ".join(f"{r['f1_per_class'][c]:.4f}" for c in CLASSES)
        rows.append(
            f"| {r['checkpoint']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
            f"{per_class} | {r['trainable_params']:,} | {r['epoch']} |"
        )

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to .pth files")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)

    results = []
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"WARNING: {ckpt} not found, skipping")
            continue
        results.append(evaluate_checkpoint(ckpt, test_loader, device))

    print("\n" + format_table(results) + "\n")


if __name__ == "__main__":
    main()
