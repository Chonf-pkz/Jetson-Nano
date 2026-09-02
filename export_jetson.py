from pathlib import Path
import torch

from src.model import LaneTracker


checkpoint_path = Path("checkpoints/best_model.pth")
output_path = Path(
    "checkpoints/lane_tracker_ir8_opset13.onnx"
)

if not checkpoint_path.is_file():
    raise FileNotFoundError(
        "Không tìm thấy checkpoint: {}".format(
            checkpoint_path.resolve()
        )
    )

model = LaneTracker(pretrained=False)

try:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True
    )
except TypeError:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu"
    )

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()
model.cpu()

model.export_onnx(str(output_path))

print("✅ File hoàn thành:")
print(output_path.resolve())