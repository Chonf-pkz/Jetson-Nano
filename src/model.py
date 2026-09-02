"""Compact PilotNet-style steering model trained from random initialization."""

import torch
import torch.nn as nn

from src.preprocessing_config import MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH


class LaneTracker(nn.Module):
    """Fast end-to-end road steering network for Jetson Nano CPU inference."""

    def __init__(self, pretrained=False, dropout=0.20):
        super().__init__()
        # ``pretrained`` remains in the API to reject accidental checkpoint
        # reuse cleanly; this architecture always starts from random weights.
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(inplace=True),
            nn.Linear(100, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        features = self.features(inputs)
        steering = self.regressor(features)
        return torch.tanh(steering).squeeze(-1)

    def freeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = True

    def count_parameters(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return trainable, total

    def export_onnx(self, filepath, input_size=None):
        import onnx

        if input_size is None:
            input_size = (1, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
        self.eval()
        self.cpu()
        dummy = torch.randn(*input_size, dtype=torch.float32)
        torch.onnx.export(
            self,
            dummy,
            filepath,
            input_names=['input'],
            output_names=['steering'],
            opset_version=13,
            dynamo=False,
            external_data=False,
            do_constant_folding=True,
            dynamic_axes=None,
        )
        model = onnx.load(filepath)
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, filepath)
        print('Exported ONNX IR8/opset13: {}'.format(filepath))


class SteeringPostProcessor:
    def __init__(
        self,
        dead_zone=0.04,
        max_steering=0.80,
        smoothing_alpha=0.35,
        solid_lane_limit=0.80,
        dashed_lane_limit=0.80,
        steering_gain=1.0,
    ):
        self.dead_zone = float(dead_zone)
        self.max_steering = float(max_steering)
        self.smoothing_alpha = float(smoothing_alpha)
        self.solid_lane_limit = float(solid_lane_limit)
        self.dashed_lane_limit = float(dashed_lane_limit)
        self.steering_gain = float(steering_gain)
        self._previous = 0.0

    def process(self, raw_steering, lane_type='solid'):
        steering = float(raw_steering) * self.steering_gain
        if abs(steering) < self.dead_zone:
            steering = 0.0
        limit = min(self.max_steering, self.solid_lane_limit)
        steering = max(-limit, min(limit, steering))
        steering = (
            self.smoothing_alpha * steering
            + (1.0 - self.smoothing_alpha) * self._previous
        )
        self._previous = steering
        return steering

    def reset(self):
        self._previous = 0.0


def load_model(checkpoint_path, device='cpu'):
    model = LaneTracker(pretrained=False)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    network = LaneTracker()
    sample = torch.randn(4, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
    output = network(sample)
    print('Output:', tuple(output.shape))
    print('Parameters:', network.count_parameters())
