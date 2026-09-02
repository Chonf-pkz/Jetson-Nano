"""Audit the current model-only driving dataset before training."""

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

from src.dataset_loader import (
    PROJECT_ROOT,
    load_all_sessions,
    session_split,
)


def audit(dataset_dir):
    entries = load_all_sessions(dataset_dir)
    corrupt = []
    sizes = Counter()
    hashes = Counter()
    for entry in entries:
        path = entry['image_path']
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                sizes['{}x{}'.format(image.width, image.height)] += 1
            with open(path, 'rb') as handle:
                hashes[hashlib.md5(handle.read()).hexdigest()] += 1
        except Exception as error:
            corrupt.append({'path': path, 'error': str(error)})

    train_entries, val_entries = session_split(entries, seed=42)
    steering = np.asarray([entry['steering'] for entry in entries])
    raw = np.asarray([entry['raw_steering'] for entry in entries])
    bins = [-1.01, -0.55, -0.12, 0.12, 0.55, 1.01]
    labels = ['hard_left', 'left', 'straight', 'right', 'hard_right']
    counts = np.histogram(steering, bins=bins)[0]
    report = {
        'dataset_dir': os.path.abspath(dataset_dir),
        'usable_samples': len(entries),
        'session_count': len(set(entry['session'] for entry in entries)),
        'image_sizes': dict(sizes),
        'corrupt_images': corrupt,
        'exact_duplicate_files': int(sum(count - 1 for count in hashes.values() if count > 1)),
        'steering': {
            'raw_mean': float(raw.mean()),
            'raw_std': float(raw.std()),
            'clean_mean': float(steering.mean()),
            'clean_std': float(steering.std()),
            'mean_abs_cleaning_delta': float(np.mean(np.abs(raw - steering))),
            'maximum_cleaning_delta': float(np.max(np.abs(raw - steering))),
            'bins': {name: int(count) for name, count in zip(labels, counts)},
        },
        'split': {
            'train_samples': len(train_entries),
            'validation_samples': len(val_entries),
            'train_sessions': sorted(set(entry['session'] for entry in train_entries)),
            'validation_sessions': sorted(set(entry['session'] for entry in val_entries)),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir',
        default=os.path.join(
            PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'
        ),
    )
    parser.add_argument(
        '--output',
        default=os.path.join(PROJECT_ROOT, 'dataset', 'dataset_report.json'),
    )
    args = parser.parse_args()
    report = audit(args.dataset_dir)
    with open(args.output, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
