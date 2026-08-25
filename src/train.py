"""
JetRacer Lane Tracking - Training Script
==========================================
Two-phase training:
  Phase 1: Freeze backbone, train regression head only (warm-up)
  Phase 2: Unfreeze all layers, fine-tune with lower learning rate

Features:
- Early stopping with patience
- Best model checkpoint (by validation loss)
- Training history logging
- Automatic ONNX export after training
"""

import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset_loader import create_dataloaders, PROJECT_ROOT
from src.model import LaneTracker


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, steerings in dataloader:
        images = images.to(device)
        steerings = steerings.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, steerings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for images, steerings in dataloader:
        images = images.to(device)
        steerings = steerings.to(device)

        predictions = model(images)
        loss = criterion(predictions, steerings)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    dataset_dir,
    output_dir='checkpoints',
    batch_size=16,
    phase1_epochs=5,
    phase2_epochs=45,
    phase1_lr=1e-3,
    phase2_lr=1e-4,
    weight_decay=1e-5,
    patience=10,
    device=None,
    seed=42,
):
    """Full training pipeline.
    
    Args:
        dataset_dir: Path to dataset_steering directory
        output_dir: Directory to save checkpoints and logs
        batch_size: Training batch size
        phase1_epochs: Epochs for Phase 1 (frozen backbone)
        phase2_epochs: Epochs for Phase 2 (fine-tuning)
        phase1_lr: Learning rate for Phase 1
        phase2_lr: Learning rate for Phase 2
        weight_decay: L2 regularization
        patience: Early stopping patience (Phase 2 only)
        device: torch device (auto-detect if None)
        seed: Random seed
    """
    # Setup
    torch.manual_seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # Data
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    train_loader, val_loader, train_entries, val_entries = create_dataloaders(
        dataset_dir, batch_size=batch_size, seed=seed
    )

    # Model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    model = LaneTracker(pretrained=True, dropout=0.5)
    model.to(device)
    trainable, total = model.count_parameters()
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    criterion = nn.MSELoss()
    history = {
        'train_loss': [],
        'val_loss': [],
        'phase': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    # =====================================================
    # Phase 1: Freeze backbone, train head only
    # =====================================================
    print("\n" + "=" * 60)
    print(f"Phase 1: Training regression head ({phase1_epochs} epochs, lr={phase1_lr})")
    print("=" * 60)

    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=weight_decay,
    )

    for epoch in range(phase1_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['phase'].append(1)
        history['lr'].append(phase1_lr)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_loss, output_dir, 'best_model.pth')
            marker = " [BEST]"

        print(
            f"  Epoch {epoch + 1:3d}/{phase1_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Time: {elapsed:.1f}s{marker}"
        )

    # =====================================================
    # Phase 2: Unfreeze all, fine-tune
    # =====================================================
    print("\n" + "=" * 60)
    print(f"Phase 2: Fine-tuning all layers ({phase2_epochs} epochs, lr={phase2_lr})")
    print("=" * 60)

    model.unfreeze_backbone()
    optimizer = optim.Adam(
        model.parameters(),
        lr=phase2_lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    total_epoch_offset = phase1_epochs

    for epoch in range(phase2_epochs):
        global_epoch = total_epoch_offset + epoch
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['phase'].append(2)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = global_epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, global_epoch, val_loss, output_dir, 'best_model.pth')
            marker = " [BEST]"
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch {global_epoch + 1:3d}/{phase1_epochs + phase2_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s{marker}"
        )

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping! No improvement for {patience} epochs.")
            break

    # =====================================================
    # Save final results
    # =====================================================
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")

    # Save last model
    save_checkpoint(model, optimizer, global_epoch, val_loss, output_dir, 'last_model.pth')

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Export ONNX
    print("\nExporting best model to ONNX...")
    best_checkpoint = torch.load(
        os.path.join(output_dir, 'best_model.pth'),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    onnx_path = os.path.join(output_dir, 'lane_tracker.onnx')
    model.export_onnx(onnx_path)

    return model, history


def save_checkpoint(model, optimizer, epoch, val_loss, output_dir, filename):
    """Save model checkpoint."""
    path = os.path.join(output_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train JetRacer Lane Tracker')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'),
        help='Path to dataset_steering directory',
    )
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--phase1-epochs', type=int, default=5)
    parser.add_argument('--phase2-epochs', type=int, default=45)
    parser.add_argument('--phase1-lr', type=float, default=1e-3)
    parser.add_argument('--phase2-lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
