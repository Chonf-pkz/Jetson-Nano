"""
JetRacer Lane Tracking - Evaluation Script
============================================
Evaluates a trained model on the validation set.
- Computes MAE, MSE, RMSE, R² metrics
- Generates predicted vs actual scatter plot
- Shows sample predictions overlaid on images
"""

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset_loader import create_dataloaders, get_val_transform, load_all_sessions, stratified_split, JetRacerDataset, PROJECT_ROOT
from src.model import load_model, SteeringPostProcessor
from torch.utils.data import DataLoader
from PIL import Image


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """Run model on validation set and collect predictions.
    
    Returns:
        all_preds, all_targets as numpy arrays
    """
    model.eval()
    all_preds = []
    all_targets = []

    for images, steerings in val_loader:
        images = images.to(device)
        predictions = model(images)
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(steerings.numpy())

    return np.array(all_preds), np.array(all_targets)


def compute_metrics(preds, targets):
    """Compute regression metrics."""
    errors = preds - targets
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    # R² score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / max(ss_tot, 1e-8))

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Max Error': np.max(np.abs(errors)),
        'Mean Pred': np.mean(preds),
        'Std Pred': np.std(preds),
    }


def plot_predictions_scatter(preds, targets, output_path):
    """Scatter plot of predicted vs actual steering."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.6, s=30, c='#2196F3', edgecolors='white', linewidth=0.5)
    ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Steering', fontsize=12)
    ax.set_ylabel('Predicted Steering', fontsize=12)
    ax.set_title('Predicted vs Actual Steering', fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Error distribution
    ax = axes[1]
    errors = preds - targets
    ax.hist(errors, bins=30, color='#FF9800', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Error Distribution (MAE={np.mean(np.abs(errors)):.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {output_path}")


def plot_sample_predictions(model, val_entries, device, output_path, n_samples=12):
    """Show sample images with predicted and actual steering values."""
    transform = get_val_transform()
    model.eval()

    # Pick evenly spaced samples
    indices = np.linspace(0, len(val_entries) - 1, n_samples, dtype=int)
    
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        entry = val_entries[idx]
        
        # Load and predict
        image = Image.open(entry['image_path']).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_tensor).item()
        
        actual = entry['steering']

        # Show original image
        axes[i].imshow(image)
        
        # Color based on error
        error = abs(pred - actual)
        color = 'green' if error < 0.15 else ('orange' if error < 0.3 else 'red')
        
        axes[i].set_title(
            f'Pred: {pred:+.3f}\nActual: {actual:+.3f}',
            fontsize=10,
            color=color,
            fontweight='bold',
        )
        axes[i].axis('off')

        # Draw steering arrow
        cx, cy = 112, 200  # Center bottom of 224x224 image
        arrow_len = 60
        # Steering arrow direction
        dx = pred * arrow_len
        dy = -30  # Always point upward
        axes[i].annotate(
            '', xy=(cx + dx, cy + dy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', color='cyan', lw=2.5),
        )

    # Hide empty axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green=Good, Orange=OK, Red=Bad)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {output_path}")


def plot_training_history(history_path, output_path):
    """Plot training and validation loss curves."""
    import json
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    # Mark phase boundary
    phase1_end = sum(1 for p in history['phase'] if p == 1)
    if phase1_end > 0 and phase1_end < len(epochs):
        ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Phase 1→2')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate JetRacer Lane Tracker')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'),
        help='Path to dataset_steering directory',
    )
    parser.add_argument('--output-dir', type=str, default='eval_results')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device=device)

    # Load data
    print("Loading dataset...")
    all_entries = load_all_sessions(args.dataset_dir)
    _, val_entries = stratified_split(all_entries)

    val_dataset = JetRacerDataset(val_entries, transform=get_val_transform(), augment_flip=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Evaluate
    print("\nEvaluating...")
    preds, targets = evaluate_model(model, val_loader, device)
    metrics = compute_metrics(preds, targets)

    print("\n" + "=" * 40)
    print("Evaluation Metrics")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name:15s}: {value:.6f}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_predictions_scatter(
        preds, targets,
        os.path.join(args.output_dir, 'predictions_scatter.png'),
    )
    plot_sample_predictions(
        model, val_entries, device,
        os.path.join(args.output_dir, 'sample_predictions.png'),
    )

    # Plot training history if available
    history_path = os.path.join(os.path.dirname(args.checkpoint), 'training_history.json')
    if os.path.exists(history_path):
        plot_training_history(
            history_path,
            os.path.join(args.output_dir, 'training_history.png'),
        )

    # Test post-processor
    print("\n" + "=" * 40)
    print("Post-processor Demo")
    print("=" * 40)
    pp = SteeringPostProcessor(
        dead_zone=0.05,
        max_steering=0.8,
        solid_lane_limit=0.6,
        dashed_lane_limit=0.9,
    )
    
    print("\n  Solid lane limits:")
    for pred in preds[:10]:
        processed = pp.process(pred, lane_type='solid')
        print(f"    Raw: {pred:+.4f} → Processed: {processed:+.4f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
