"""Train the selected model-only configuration on every recorded session."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset_loader import (
    PROJECT_ROOT,
    JetRacerDataset,
    compute_sample_weights,
    get_train_transform,
    load_all_sessions,
)
from src.model import LaneTracker
from src.train import run_epoch, save_checkpoint, set_seed


def train_final(dataset_dir, output_dir='checkpoints/final_model', epochs=3):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entries = load_all_sessions(dataset_dir, label_lookahead_frames=0)
    dataset = JetRacerDataset(
        entries,
        transform=get_train_transform(),
        augment_flip=True,
        augment_recovery=False,
        cache_images=True,
    )
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=256,
        sampler=WeightedRandomSampler(
            compute_sample_weights(entries),
            num_samples=len(entries),
            replacement=True,
            generator=generator,
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = LaneTracker(pretrained=False, dropout=0.15).to(device)
    criterion = nn.SmoothL1Loss(beta=0.08)
    optimizer = optim.AdamW(
        model.parameters(), lr=2.5e-4, weight_decay=1e-4
    )
    history = {
        'train_loss': [],
        'train_mae': [],
        'samples': len(entries),
        'epochs': int(epochs),
        'architecture': 'pilotnet_compact_v1',
        'random_initialization': True,
        'selected_from_validation_candidate': 'candidate_no_recovery',
        'augment_flip': True,
        'augment_recovery': False,
        'label_lookahead_frames': 0,
        'loss': 'huber',
        'dropout': 0.15,
    }
    os.makedirs(output_dir, exist_ok=True)
    print('Final full-data train: {} samples on {}'.format(len(entries), device))
    for epoch in range(int(epochs)):
        started = time.time()
        loss, mae = run_epoch(
            model, loader, criterion, device, optimizer=optimizer
        )
        history['train_loss'].append(loss)
        history['train_mae'].append(mae)
        print(
            'Final epoch {}/{} loss {:.5f} mae {:.4f} | {:.1f}s'.format(
                epoch + 1, epochs, loss, mae, time.time() - started
            ),
            flush=True,
        )

    save_checkpoint(
        model, optimizer, int(epochs) - 1,
        {'train_loss': loss, 'train_mae': mae, 'full_data': True},
        output_dir, 'best_model.pth',
    )
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as handle:
        json.dump(history, handle, indent=2)
    model.cpu()
    model.export_onnx(os.path.join(
        output_dir, 'lane_tracker_ir8_opset13.onnx'
    ))
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir',
        default=os.path.join(
            PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'
        ),
    )
    parser.add_argument('--output-dir', default='checkpoints/final_model')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    train_final(args.dataset_dir, args.output_dir, args.epochs)


if __name__ == '__main__':
    main()
