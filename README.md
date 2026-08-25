# JetRacer Lane Tracking

ResNet-18 based autonomous lane tracking for NVIDIA JetRacer on Jetson Nano.

## Project Structure

```
JetRacer/
├── src/
│   ├── model.py              # LaneTracker model (ResNet-18 + regression head)
│   ├── dataset_loader.py     # Dataset loading, augmentation, sampling
│   ├── train.py              # Two-phase training pipeline
│   ├── evaluate.py           # Metrics, visualizations
│   └── inference_jetson.py   # Jetson Nano deployment (PyTorch/ONNX)
├── dataset/                  # Training data (sessions)
├── checkpoints/              # Model weights & ONNX exports
├── eval_results/             # Evaluation plots & metrics
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Train with default settings (5 epochs warmup + 45 epochs fine-tune)
python -m src.train --dataset-dir dataset/track_lane_dataset/dataset_steering

# Custom settings
python -m src.train --batch-size 32 --phase2-epochs 60 --patience 15
```

## Evaluation

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pth
```

## Inference on Jetson Nano

```bash
# PyTorch backend
python -m src.inference_jetson --mode pytorch --checkpoint checkpoints/best_model.pth

# ONNX/TensorRT backend (faster)
python -m src.inference_jetson --mode onnx --model checkpoints/lane_tracker.onnx --csi --jetracer
```

## Model

- **Backbone**: ResNet-18 (ImageNet pretrained)
- **Output**: Steering value in [-1, 1] (tanh activation)
- **Input**: 224×224 RGB image
- **Post-processing**: Dead zone, lane-type limits, EMA smoothing
