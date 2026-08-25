"""
JetRacer Lane Tracking Dataset Loader
=====================================
Custom PyTorch Dataset that loads camera images + steering labels
from the JetRacer dataset sessions. Includes:
- Data augmentation (color jitter, flip with steering negate, blur, affine)
- Weighted random sampling for imbalanced steering distribution
- Stratified train/val split
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# Project root directory (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class JetRacerDataset(Dataset):
    """Dataset for JetRacer steering prediction.
    
    Loads all sessions from the dataset directory.
    Each sample is (image, steering_normalized).
    
    Args:
        data_entries: List of dicts with 'image_path' and 'steering' keys
        transform: torchvision transform pipeline
        augment_flip: If True, horizontal flip augments steering by negating it
    """

    def __init__(self, data_entries, transform=None, augment_flip=True):
        self.data_entries = data_entries
        self.transform = transform
        self.augment_flip = augment_flip

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        image = Image.open(entry['image_path']).convert('RGB')
        steering = entry['steering']

        # Random horizontal flip with steering negation
        if self.augment_flip and self.transform is not None:
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.hflip(image)
                steering = -steering

        if self.transform:
            image = self.transform(image)

        steering = torch.tensor(steering, dtype=torch.float32)
        return image, steering


def load_all_sessions(dataset_dir):
    """Load all session data from the dataset directory.
    
    Args:
        dataset_dir: Path to dataset_steering directory containing session folders
        
    Returns:
        List of dicts with 'image_path', 'steering', 'session' keys
    """
    entries = []
    session_dirs = sorted(glob.glob(os.path.join(dataset_dir, 'session_*')))

    if not session_dirs:
        raise FileNotFoundError(
            f"No session directories found in {dataset_dir}. "
            f"Expected directories named 'session_*'."
        )

    print(f"Found {len(session_dirs)} sessions in {dataset_dir}")

    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, 'labels.csv')
        if not os.path.exists(csv_path):
            print(f"  WARNING: Skipping {session_dir} - no labels.csv found")
            continue

        df = pd.read_csv(csv_path)
        session_name = os.path.basename(session_dir)

        for _, row in df.iterrows():
            img_path = os.path.join(session_dir, row['image_file'])
            if os.path.exists(img_path):
                entries.append({
                    'image_path': img_path,
                    'steering': float(row['steering_normalized']),
                    'session': session_name,
                })
            else:
                print(f"  WARNING: Image not found: {img_path}")

        print(f"  {session_name}: {len(df)} samples loaded")

    print(f"Total samples loaded: {len(entries)}")
    return entries


def stratified_split(entries, val_ratio=0.2, n_bins=5, seed=42):
    """Split entries into train/val with stratified bins on steering.
    
    Bins the steering values and ensures each bin is proportionally
    represented in both train and val sets.
    
    Args:
        entries: List of data entry dicts
        val_ratio: Fraction for validation
        n_bins: Number of steering bins for stratification
        seed: Random seed
        
    Returns:
        train_entries, val_entries
    """
    rng = np.random.RandomState(seed)
    steerings = np.array([e['steering'] for e in entries])

    # Bin steering values for stratification
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(steerings, bin_edges[1:-1])

    train_entries = []
    val_entries = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_entries = [entries[i] for i in range(len(entries)) if mask[i]]

        if len(bin_entries) == 0:
            continue

        rng.shuffle(bin_entries)
        n_val = max(1, int(len(bin_entries) * val_ratio))

        val_entries.extend(bin_entries[:n_val])
        train_entries.extend(bin_entries[n_val:])

    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")
    return train_entries, val_entries


def compute_sample_weights(entries, n_bins=5):
    """Compute sample weights inversely proportional to steering bin frequency.
    
    This allows WeightedRandomSampler to oversample rare steering values
    (turns) and undersample common ones (straight).
    
    Args:
        entries: List of data entry dicts
        n_bins: Number of bins
        
    Returns:
        torch.Tensor of sample weights
    """
    steerings = np.array([e['steering'] for e in entries])
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(steerings, bin_edges[1:-1])

    # Count samples per bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    bin_counts = np.maximum(bin_counts, 1.0)  # avoid division by zero

    # Weight = 1 / count (inverse frequency)
    bin_weights = 1.0 / bin_counts

    # Assign weight to each sample based on its bin
    sample_weights = torch.tensor(
        [bin_weights[bin_idx] for bin_idx in bin_indices],
        dtype=torch.float64
    )
    return sample_weights


def get_train_transform():
    """Training augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform():
    """Validation transform (no augmentation, just normalize)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_dataloaders(dataset_dir, batch_size=16, val_ratio=0.2, num_workers=2, seed=42):
    """Create train and validation DataLoaders.
    
    Args:
        dataset_dir: Path to dataset_steering directory
        batch_size: Batch size
        val_ratio: Validation split ratio
        num_workers: DataLoader workers
        seed: Random seed
        
    Returns:
        train_loader, val_loader, train_entries, val_entries
    """
    # Load all data
    all_entries = load_all_sessions(dataset_dir)

    # Stratified split
    train_entries, val_entries = stratified_split(
        all_entries, val_ratio=val_ratio, seed=seed
    )

    # Create datasets
    train_dataset = JetRacerDataset(
        train_entries,
        transform=get_train_transform(),
        augment_flip=True,
    )
    val_dataset = JetRacerDataset(
        val_entries,
        transform=get_val_transform(),
        augment_flip=False,
    )

    # Weighted sampling for training
    sample_weights = compute_sample_weights(train_entries)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_entries),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_entries, val_entries


if __name__ == '__main__':
    # Quick test
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering')
    train_loader, val_loader, _, _ = create_dataloaders(dataset_dir, batch_size=8)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    images, steerings = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch steerings: {steerings}")
