"""
JetRacer Lane Tracking Model
=============================
ResNet-18 based steering regression model optimized for Jetson Nano.
- Pretrained ImageNet backbone
- Custom regression head with dropout
- tanh output activation for [-1, 1] range
- ONNX export for TensorRT deployment
"""

import torch
import torch.nn as nn
from torchvision import models


class LaneTracker(nn.Module):
    """ResNet-18 based lane tracking model.
    
    Takes a 224x224 RGB image and outputs a steering value in [-1, 1].
    
    Args:
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before final FC layer
    """

    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()

        # Load ResNet-18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        backbone = models.resnet18(weights=weights)

        # Remove the original FC layer, keep everything else
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom regression head
        # ResNet-18 outputs 512-dim features after global avg pool
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Steering prediction of shape (B,) in range [-1, 1]
        """
        features = self.features(x)
        out = self.regressor(features)
        out = torch.tanh(out)
        return out.squeeze(-1)

    def freeze_backbone(self):
        """Freeze all backbone (feature extractor) parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("Backbone frozen - only training regression head")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - fine-tuning all layers")

    def count_parameters(self):
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def export_onnx(self, filepath, input_size=(1, 3, 224, 224)):
        """Export model to ONNX format for TensorRT on Jetson Nano.
        
        Args:
            filepath: Output .onnx file path
            input_size: Input tensor shape (batch, channels, height, width)
        """
        self.eval()
        dummy_input = torch.randn(*input_size)

        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            input_names=['input'],
            output_names=['steering'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'steering': {0: 'batch_size'},
            },
            opset_version=11,  # Compatible with Jetson Nano TensorRT
        )
        print(f"Model exported to ONNX: {filepath}")


class SteeringPostProcessor:
    """Post-processing for steering output with threshold and smoothing.
    
    Applies configurable rules to the raw model output:
    - Dead zone: Small steering values below threshold → 0 (go straight)
    - Clipping: Limit max steering angle
    - Smoothing: Exponential moving average to reduce jitter
    
    This helps the car:
    - Stay in lane without oscillating
    - Not cross solid lane markings
    - Allow crossing dashed lane markings when needed (wider threshold)
    
    Args:
        dead_zone: Steering values with |value| < dead_zone are set to 0
        max_steering: Maximum allowed steering magnitude
        smoothing_alpha: EMA smoothing factor (0=full smooth, 1=no smooth)
        solid_lane_limit: Max steering to avoid crossing solid lanes
        dashed_lane_limit: Max steering allowed near dashed lanes (more permissive)
    """

    def __init__(
        self,
        dead_zone=0.05,
        max_steering=0.8,
        smoothing_alpha=0.7,
        solid_lane_limit=0.6,
        dashed_lane_limit=0.9,
    ):
        self.dead_zone = dead_zone
        self.max_steering = max_steering
        self.smoothing_alpha = smoothing_alpha
        self.solid_lane_limit = solid_lane_limit
        self.dashed_lane_limit = dashed_lane_limit
        self._prev_steering = 0.0

    def process(self, raw_steering, lane_type='solid'):
        """Process raw steering output.
        
        Args:
            raw_steering: Raw model output in [-1, 1]
            lane_type: 'solid' or 'dashed' — determines max steering limit
            
        Returns:
            Processed steering value
        """
        steering = float(raw_steering)

        # 1. Dead zone — ignore tiny steering noise
        if abs(steering) < self.dead_zone:
            steering = 0.0

        # 2. Apply lane-type-specific limit
        if lane_type == 'solid':
            limit = self.solid_lane_limit
        else:  # dashed
            limit = self.dashed_lane_limit

        # Clip to the more restrictive of max_steering and lane limit
        effective_limit = min(self.max_steering, limit)
        steering = max(-effective_limit, min(effective_limit, steering))

        # 3. Exponential moving average smoothing
        steering = (
            self.smoothing_alpha * steering
            + (1 - self.smoothing_alpha) * self._prev_steering
        )
        self._prev_steering = steering

        return steering

    def reset(self):
        """Reset smoothing state."""
        self._prev_steering = 0.0


def load_model(checkpoint_path, device='cpu'):
    """Load a trained LaneTracker model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded LaneTracker model in eval mode
    """
    model = LaneTracker(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


if __name__ == '__main__':
    # Quick test
    model = LaneTracker(pretrained=True)
    trainable, total = model.count_parameters()
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Test forward pass
    dummy = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.detach()}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test post-processor
    pp = SteeringPostProcessor()
    test_values = [0.02, -0.03, 0.5, -0.7, 1.0, -1.0]
    for v in test_values:
        processed = pp.process(v, lane_type='solid')
        print(f"  Raw: {v:+.2f} → Processed (solid): {processed:+.4f}")
    
    pp.reset()
    for v in test_values:
        processed = pp.process(v, lane_type='dashed')
        print(f"  Raw: {v:+.2f} → Processed (dashed): {processed:+.4f}")
