"""Shared image contract for training, ONNX export, and Jetson inference."""

MODEL_INPUT_HEIGHT = 66
MODEL_INPUT_WIDTH = 200
ROAD_CROP_TOP_FRACTION = 0.32
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
