"""
Data loading and preprocessing for brain MRI tumor classification.

This module handles:
- Loading grayscale MRI images from disk
- Applying appropriate transforms for medical imaging
- Creating stratified train/validation/test splits
- Providing PyTorch DataLoaders for training
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import numpy as np


# Class labels (matches folder structure)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)


def get_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get image transforms for MRI data.

    Key differences from notebook:
    - Keeps images as single-channel grayscale (medical imaging standard)
    - Uses medical imaging-appropriate augmentations
    - Will compute normalization stats from actual data instead of ImageNet

    Args:
        augment: If True, apply data augmentation (for training)

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Grayscale(num_output_channels=1),  # Keep as single channel
        transforms.Resize((224, 224)),
    ]

    if augment:
        # Medical imaging augmentations - subtle to preserve diagnostic features
        transform_list.extend([
            transforms.RandomRotation(15),  # Brain can be slightly rotated
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight shifts
            transforms.RandomHorizontalFlip(p=0.5),  # Left/right flip is valid for brain
        ])

    transform_list.extend([
        transforms.ToTensor(),
        # For now using basic normalization - will compute dataset stats later if needed
        transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel
    ])

    return transforms.Compose(transform_list)


class BrainMRIDataset(Dataset):
    """
    Custom Dataset for brain MRI images.

    This wraps ImageFolder and applies transforms at load time,
    allowing different transforms for train vs validation.
    """
    def __init__(self, image_folder: datasets.ImageFolder, indices: np.ndarray,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_folder: Base ImageFolder dataset (no transforms)
            indices: Indices to use from the base dataset
            transform: Transforms to apply
        """
        self.base_dataset = image_folder
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get the actual index in base dataset
        real_idx = self.indices[idx]

        # Load image path and label
        img_path, label = self.base_dataset.samples[real_idx]

        # Load image
        image = Image.open(img_path).convert('L')  # Ensure grayscale

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(
    data_dir: str = "data",
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Key improvements:
    - Stratified split ensures balanced classes in train/val
    - Separate test set from Testing folder (completely held out)
    - Different augmentations for train vs val/test

    Args:
        data_dir: Root directory containing Training/ and Testing/ folders
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of worker processes for data loading
        random_state: Random seed for reproducibility

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_path = os.path.join(data_dir, "Training")
    test_path = os.path.join(data_dir, "Testing")

    # Check paths exist
    if not os.path.exists(train_path):
        raise ValueError(f"Training path not found: {train_path}")
    if not os.path.exists(test_path):
        raise ValueError(f"Testing path not found: {test_path}")

    # Load base datasets (no transforms yet)
    train_base = datasets.ImageFolder(train_path)
    test_base = datasets.ImageFolder(test_path)

    # Get labels for stratified split
    labels = np.array([label for _, label in train_base.samples])

    # Stratified train/val split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)
    train_indices, val_indices = next(splitter.split(np.zeros(len(labels)), labels))

    # Create datasets with appropriate transforms
    train_transform = get_transforms(augment=True)
    eval_transform = get_transforms(augment=False)

    train_dataset = BrainMRIDataset(train_base, train_indices, train_transform)
    val_dataset = BrainMRIDataset(train_base, val_indices, eval_transform)

    # Test dataset - use all indices
    test_indices = np.arange(len(test_base.samples))
    test_dataset = BrainMRIDataset(test_base, test_indices, eval_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def print_dataset_info(data_dir: str = "data"):
    """
    Print information about the dataset.
    Useful for understanding class distribution and data size.
    """
    train_path = os.path.join(data_dir, "Training")
    test_path = os.path.join(data_dir, "Testing")

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    for split_name, split_path in [("Training", train_path), ("Testing", test_path)]:
        print(f"\n{split_name} Set:")
        total = 0
        for class_name in sorted(CLASSES):
            class_path = os.path.join(split_path, class_name)
            count = len(list(Path(class_path).glob("*")))
            print(f"  {class_name:12s}: {count:4d} images")
            total += count
        print(f"  {'Total':12s}: {total:4d} images")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Demo: Load data and print info
    print_dataset_info()

    print("\nLoading data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Show a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape: {images.shape}")  # Should be [32, 1, 224, 224]
    print(f"Sample labels: {labels[:8].tolist()}")
