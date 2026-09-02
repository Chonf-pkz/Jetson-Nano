"""Train the compact model-only steering network from random weights."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset_loader import PROJECT_ROOT, create_dataloaders
from src.model import LaneTracker


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    absolute_error_sum = 0.0
    sample_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        count = int(targets.numel())
        loss_sum += float(loss.item()) * count
        absolute_error_sum += float(
            torch.abs(predictions.detach() - targets).sum().item()
        )
        sample_count += count

    denominator = max(sample_count, 1)
    return loss_sum / denominator, absolute_error_sum / denominator


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, filename):
    torch.save({
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': dict(metrics),
        'architecture': 'pilotnet_compact_v1',
        'random_initialization': True,
    }, os.path.join(output_dir, filename))


def train(
    dataset_dir,
    output_dir='checkpoints',
    batch_size=256,
    epochs=40,
    learning_rate=3e-4,
    weight_decay=1e-4,
    patience=8,
    num_workers=0,
    device=None,
    seed=42,
    loss_name='huber',
    dropout=0.15,
    augment_flip=True,
    augment_recovery=False,
    label_lookahead_frames=0,
):
    set_seed(seed)
    device = torch.device(
        device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    os.makedirs(output_dir, exist_ok=True)
    print('Device: {}'.format(device))

    train_loader, val_loader, train_entries, val_entries = create_dataloaders(
        dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        augment_flip=augment_flip,
        augment_recovery=augment_recovery,
        label_lookahead_frames=label_lookahead_frames,
    )
    model = LaneTracker(pretrained=False, dropout=float(dropout)).to(device)
    trainable, total = model.count_parameters()
    print('Random-init PilotNet: {:,} parameters'.format(total))
    print('Trainable: {:,}'.format(trainable))

    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'huber':
        criterion = nn.SmoothL1Loss(beta=0.08)
    else:
        raise ValueError('Unknown loss: {}'.format(loss_name))
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=learning_rate * 0.03
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [], 'lr': [],
        'train_samples': len(train_entries),
        'val_samples': len(val_entries),
        'architecture': 'pilotnet_compact_v1',
        'random_initialization': True,
        'loss': loss_name,
        'dropout': float(dropout),
        'augment_flip': bool(augment_flip),
        'augment_recovery': bool(augment_recovery),
        'label_lookahead_frames': int(label_lookahead_frames),
    }
    best_mae = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(int(epochs)):
        started = time.time()
        train_loss, train_mae = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_loss, val_mae = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)

        marker = ''
        metrics = {'val_loss': val_loss, 'val_mae': val_mae}
        if val_mae < best_mae - 1e-5:
            best_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, metrics,
                output_dir, 'best_model.pth'
            )
            marker = ' [BEST]'
        else:
            epochs_without_improvement += 1

        print(
            'Epoch {:02d}/{:02d} train loss {:.5f} mae {:.4f} | '
            'val loss {:.5f} mae {:.4f} | {:.1f}s{}'.format(
                epoch + 1, epochs, train_loss, train_mae,
                val_loss, val_mae, time.time() - started, marker,
            ),
            flush=True,
        )
        if epochs_without_improvement >= int(patience):
            print('Early stop after {} non-improving epochs'.format(patience))
            break

    save_checkpoint(
        model, optimizer, epoch,
        {'val_loss': val_loss, 'val_mae': val_mae},
        output_dir, 'last_model.pth'
    )
    history['best_epoch'] = best_epoch + 1
    history['best_val_mae'] = best_mae
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as handle:
        json.dump(history, handle, indent=2)

    try:
        checkpoint = torch.load(
            os.path.join(output_dir, 'best_model.pth'),
            map_location='cpu', weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            os.path.join(output_dir, 'best_model.pth'), map_location='cpu'
        )
    model.cpu()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.export_onnx(os.path.join(
        output_dir, 'lane_tracker_ir8_opset13.onnx'
    ))
    print('Best validation MAE: {:.5f} at epoch {}'.format(
        best_mae, best_epoch + 1
    ))
    return model, history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir',
        default=os.path.join(
            PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'
        ),
    )
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss', choices=['mse', 'huber'], default='huber')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--no-flip', action='store_true')
    parser.add_argument('--recovery', action='store_true')
    parser.add_argument('--label-lookahead-frames', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        loss_name=args.loss,
        dropout=args.dropout,
        augment_flip=not args.no_flip,
        augment_recovery=args.recovery,
        label_lookahead_frames=args.label_lookahead_frames,
    )
