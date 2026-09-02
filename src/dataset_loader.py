"""Clean, deployment-matched dataset pipeline for model-only steering."""

import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.preprocessing_config import (
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ROAD_CROP_TOP_FRACTION,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def crop_road_roi(image):
    """Remove ceiling/background while preserving the full road width."""
    width, height = image.size
    top = int(round(height * ROAD_CROP_TOP_FRACTION))
    top = max(0, min(height - 1, top))
    return image.crop((0, top, width, height))


def smooth_steering_labels(values):
    """Suppress isolated joystick releases without shifting corners in time."""
    series = pd.Series(np.asarray(values, dtype=np.float64))
    median = series.rolling(window=5, center=True, min_periods=1).median()
    smoothed = median.rolling(window=3, center=True, min_periods=1).mean()
    return np.clip(smoothed.to_numpy(), -1.0, 1.0)


class JetRacerDataset(Dataset):
    def __init__(
        self,
        data_entries,
        transform=None,
        augment_flip=True,
        augment_recovery=False,
        recovery_probability=0.35,
        max_horizontal_shift=0.10,
        steering_correction_gain=1.8,
        cache_images=False,
    ):
        self.data_entries = data_entries
        self.transform = transform
        self.augment_flip = augment_flip
        self.augment_recovery = augment_recovery
        self.recovery_probability = float(recovery_probability)
        self.max_horizontal_shift = float(max_horizontal_shift)
        self.steering_correction_gain = float(steering_correction_gain)
        self._cached_images = None
        if cache_images:
            self._cached_images = [
                self._load_resized_image(entry['image_path'])
                for entry in self.data_entries
            ]

    @staticmethod
    def _load_resized_image(path):
        with Image.open(path) as source:
            image = crop_road_roi(source.convert('RGB'))
            image = image.resize(
                (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
                resample=Image.Resampling.BILINEAR,
            )
            return np.asarray(image, dtype=np.uint8).copy()

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        if self._cached_images is None:
            image = Image.fromarray(
                self._load_resized_image(entry['image_path']), mode='RGB'
            )
        else:
            image = Image.fromarray(self._cached_images[idx], mode='RGB')
        steering = float(entry['steering'])

        if (
            self.augment_recovery
            and torch.rand(1).item() < self.recovery_probability
        ):
            max_pixels = int(round(image.width * self.max_horizontal_shift))
            shift_pixels = int(
                torch.randint(-max_pixels, max_pixels + 1, (1,)).item()
            )
            image, steering = self._apply_recovery_shift(
                image, steering, shift_pixels
            )

        if self.augment_flip and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            steering = -steering
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(steering, dtype=torch.float32)

    def _apply_recovery_shift(self, image, steering, shift_pixels):
        width, height = image.size
        shift_pixels = int(max(-width + 1, min(width - 1, shift_pixels)))
        if shift_pixels > 0:
            padded = transforms.functional.pad(
                image, [shift_pixels, 0, 0, 0], padding_mode='edge'
            )
            image = transforms.functional.crop(padded, 0, 0, height, width)
        elif shift_pixels < 0:
            amount = -shift_pixels
            padded = transforms.functional.pad(
                image, [0, 0, amount, 0], padding_mode='edge'
            )
            image = transforms.functional.crop(
                padded, 0, amount, height, width
            )
        correction = (
            shift_pixels / float(width) * self.steering_correction_gain
        )
        steering = float(np.clip(steering + correction, -1.0, 1.0))
        return image, steering


def load_all_sessions(
    dataset_dir,
    minimum_moving_throttle=0.05,
    clean_labels=True,
    label_lookahead_frames=0,
):
    """Load moving samples and smooth each session independently."""
    entries = []
    session_dirs = sorted(glob.glob(os.path.join(dataset_dir, 'session_*')))
    if not session_dirs:
        raise FileNotFoundError(
            'No session_* directories found in {}'.format(dataset_dir)
        )

    print('Found {} sessions in {}'.format(len(session_dirs), dataset_dir))
    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, 'labels.csv')
        if not os.path.isfile(csv_path):
            print('  WARNING: no labels.csv in {}'.format(session_dir))
            continue
        frame = pd.read_csv(csv_path)
        required = {'image_file', 'steering_normalized'}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                '{} is missing columns: {}'.format(
                    csv_path, ', '.join(sorted(missing))
                )
            )

        raw_labels = frame['steering_normalized'].astype(float).to_numpy()
        labels = (
            smooth_steering_labels(raw_labels)
            if clean_labels else raw_labels
        )
        if 'throttle_normalized' in frame.columns:
            moving = (
                frame['throttle_normalized'].astype(float).abs().to_numpy()
                >= float(minimum_moving_throttle)
            )
        else:
            moving = np.ones(len(frame), dtype=bool)

        session_name = os.path.basename(session_dir)
        loaded = 0
        lookahead = max(0, int(label_lookahead_frames))
        for position, (_, row) in enumerate(frame.iterrows()):
            if not moving[position]:
                continue
            target_position = position + lookahead
            if target_position >= len(labels):
                continue
            relative = str(row['image_file']).replace('/', os.sep)
            relative = relative.replace('\\', os.sep)
            image_path = os.path.join(session_dir, relative)
            if not os.path.isfile(image_path):
                continue
            entries.append({
                'image_path': image_path,
                'steering': float(labels[target_position]),
                'raw_steering': float(raw_labels[target_position]),
                'session': session_name,
                'sample_index': int(row.get('sample_index', position)),
                'label_lookahead_frames': lookahead,
            })
            loaded += 1
        print('  {}: {} moving samples'.format(session_name, loaded))

    if not entries:
        raise RuntimeError('No usable moving samples were loaded')
    print('Total usable samples: {}'.format(len(entries)))
    return entries


def stratified_split(entries, val_ratio=0.2, n_bins=9, seed=42):
    rng = np.random.RandomState(seed)
    steering = np.asarray([entry['steering'] for entry in entries])
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bins = np.digitize(steering, edges[1:-1])
    train_entries, val_entries = [], []
    for bin_index in range(n_bins):
        members = [
            entries[index] for index in range(len(entries))
            if bins[index] == bin_index
        ]
        if not members:
            continue
        rng.shuffle(members)
        count = max(1, int(round(len(members) * val_ratio)))
        val_entries.extend(members[:count])
        train_entries.extend(members[count:])
    return train_entries, val_entries


def session_split(entries, val_ratio=0.2, seed=42):
    """Hold out complete sessions to prevent adjacent-frame leakage."""
    sessions = {}
    for entry in entries:
        sessions.setdefault(entry['session'], []).append(entry)
    if len(sessions) < 2:
        return stratified_split(entries, val_ratio=val_ratio, seed=seed)

    rng = np.random.RandomState(seed)
    names = sorted(sessions)
    rng.shuffle(names)
    target = max(1, int(round(len(entries) * val_ratio)))
    selected, count = [], 0
    for name in names:
        if len(selected) >= len(names) - 1:
            break
        candidate = count + len(sessions[name])
        if not selected or abs(target - candidate) < abs(target - count):
            selected.append(name)
            count = candidate
    selected = set(selected)
    train_entries = [e for e in entries if e['session'] not in selected]
    val_entries = [e for e in entries if e['session'] in selected]
    print(
        'Train: {} from {} sessions, Val: {} from {} sessions'.format(
            len(train_entries), len(sessions) - len(selected),
            len(val_entries), len(selected),
        )
    )
    print('Validation sessions: ' + ', '.join(sorted(selected)))
    return train_entries, val_entries


def compute_sample_weights(entries, n_bins=11, maximum_weight=3.0):
    """Capped square-root balancing avoids extreme-turn domination."""
    steering = np.asarray([entry['steering'] for entry in entries])
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bins = np.digitize(steering, edges[1:-1])
    counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    bin_weights = np.sqrt(float(np.max(counts)) / counts)
    bin_weights = np.clip(bin_weights, 1.0, float(maximum_weight))
    return torch.tensor(
        [bin_weights[index] for index in bins], dtype=torch.float64
    )


def get_train_transform():
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=0.22, contrast=0.22, saturation=0.18, hue=0.04
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def create_dataloaders(
    dataset_dir,
    batch_size=256,
    val_ratio=0.2,
    num_workers=0,
    seed=42,
    augment_flip=True,
    augment_recovery=True,
    label_lookahead_frames=0,
):
    entries = load_all_sessions(
        dataset_dir, label_lookahead_frames=label_lookahead_frames
    )
    train_entries, val_entries = session_split(
        entries, val_ratio=val_ratio, seed=seed
    )
    train_dataset = JetRacerDataset(
        train_entries,
        transform=get_train_transform(),
        augment_flip=bool(augment_flip),
        augment_recovery=bool(augment_recovery),
        cache_images=True,
    )
    val_dataset = JetRacerDataset(
        val_entries,
        transform=get_val_transform(),
        augment_flip=False,
        augment_recovery=False,
        cache_images=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        compute_sample_weights(train_entries),
        num_samples=len(train_entries),
        replacement=True,
        generator=generator,
    )
    common = {
        'batch_size': int(batch_size),
        'num_workers': int(num_workers),
        'pin_memory': torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset, sampler=sampler, drop_last=False, **common
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader, train_entries, val_entries


if __name__ == '__main__':
    default_dir = os.path.join(
        PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'
    )
    loaders = create_dataloaders(default_dir, batch_size=16)
    images, steering = next(iter(loaders[0]))
    print('Batch:', tuple(images.shape), tuple(steering.shape))
