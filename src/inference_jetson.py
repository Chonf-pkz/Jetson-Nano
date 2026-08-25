"""
JetRacer Lane Tracking - Jetson Nano Inference
================================================
Inference script for deploying the trained lane tracker on Jetson Nano.
Supports both PyTorch and TensorRT (ONNX) backends.

Features:
- Camera capture via CSI or USB
- Steering post-processing with lane-type thresholds
- Configurable dead zone, max steering, smoothing
- Integration with JetRacer NvidiaRacecar API

Usage on Jetson Nano:
    python -m src.inference_jetson --mode pytorch --checkpoint best_model.pth
    python -m src.inference_jetson --mode onnx --model lane_tracker.onnx
"""

import os
import time
import argparse

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src.model import LaneTracker, SteeringPostProcessor


# ============================================================
# Image preprocessing (same as training validation transform)
# ============================================================
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_frame(frame_bgr):
    """Convert OpenCV BGR frame to model input tensor.
    
    Args:
        frame_bgr: numpy array (H, W, 3) in BGR format (from OpenCV)
        
    Returns:
        torch.Tensor of shape (1, 3, 224, 224)
    """
    # BGR → RGB → PIL
    frame_rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(frame_rgb)
    tensor = PREPROCESS(image).unsqueeze(0)
    return tensor


# ============================================================
# PyTorch inference backend
# ============================================================
class PyTorchBackend:
    """PyTorch inference backend."""

    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        self.model = LaneTracker(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"PyTorch model loaded from {checkpoint_path}")

    @torch.no_grad()
    def predict(self, input_tensor):
        """Run inference.
        
        Args:
            input_tensor: (1, 3, 224, 224) tensor
            
        Returns:
            Steering value as float
        """
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        return output.item()


# ============================================================
# ONNX/TensorRT inference backend (for Jetson Nano)
# ============================================================
class ONNXBackend:
    """ONNX Runtime inference backend.
    
    On Jetson Nano with onnxruntime-gpu, this uses TensorRT EP
    for maximum performance (~30+ FPS).
    """

    def __init__(self, onnx_path):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not found. Install with:\n"
                "  pip install onnxruntime  # CPU\n"
                "  pip install onnxruntime-gpu  # GPU (Jetson)"
            )

        # Prefer TensorRT > CUDA > CPU
        providers = []
        available = ort.get_available_providers()
        if 'TensorrtExecutionProvider' in available:
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX model loaded from {onnx_path}")
        print(f"  Using providers: {self.session.get_providers()}")

    def predict(self, input_tensor):
        """Run inference.
        
        Args:
            input_tensor: (1, 3, 224, 224) tensor
            
        Returns:
            Steering value as float
        """
        input_np = input_tensor.numpy()
        outputs = self.session.run(None, {self.input_name: input_np})
        return float(outputs[0].item())


# ============================================================
# Main inference loop
# ============================================================
def run_inference(
    backend,
    camera_id=0,
    use_csi=False,
    display=True,
    steering_config=None,
    use_jetracer=False,
    throttle=0.4,
):
    """Main inference loop with camera.
    
    Args:
        backend: Inference backend (PyTorchBackend or ONNXBackend)
        camera_id: Camera device ID
        use_csi: Use CSI camera (Jetson) instead of USB
        display: Show preview window
        steering_config: Dict of SteeringPostProcessor kwargs
        use_jetracer: Control JetRacer motors
        throttle: Base throttle value for JetRacer
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not found. Install with: pip install opencv-python")

    # Setup camera
    if use_csi:
        # CSI camera pipeline for Jetson Nano
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            "video/x-raw(memory:NVMM), width=224, height=224, "
            "format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=224, height=224, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    # Setup steering post-processor
    if steering_config is None:
        steering_config = {}
    post_processor = SteeringPostProcessor(**steering_config)

    # Setup JetRacer (optional)
    car = None
    if use_jetracer:
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            car = NvidiaRacecar()
            car.throttle = 0.0
            car.steering = 0.0
            print("JetRacer connected!")
        except Exception as e:
            print(f"WARNING: Could not connect to JetRacer: {e}")
            car = None

    print("\nStarting inference... Press 'q' to quit.")
    fps_history = []

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Preprocess and predict
            input_tensor = preprocess_frame(frame)
            raw_steering = backend.predict(input_tensor)

            # Post-process steering
            # You can change lane_type based on your track setup
            steering = post_processor.process(raw_steering, lane_type='solid')

            # Apply to JetRacer
            if car is not None:
                car.steering = steering
                car.throttle = throttle

            # FPS calculation
            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 1e-6)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            # Display
            if display:
                # Draw steering info on frame
                h, w = frame.shape[:2]
                
                # Steering bar
                bar_y = h - 30
                bar_center = w // 2
                bar_width = int(steering * (w // 2))
                color = (0, 255, 0) if abs(steering) < 0.3 else (0, 165, 255) if abs(steering) < 0.6 else (0, 0, 255)
                cv2.rectangle(frame, (bar_center, bar_y), (bar_center + bar_width, bar_y + 20), color, -1)
                cv2.line(frame, (bar_center, bar_y), (bar_center, bar_y + 20), (255, 255, 255), 2)

                # Text overlay
                cv2.putText(frame, f"Steer: {steering:+.3f} (raw: {raw_steering:+.3f})",
                           (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                           (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                cv2.imshow('JetRacer Lane Tracker', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Cleanup
        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
        cap.release()
        if display:
            cv2.destroyAllWindows()
        print(f"\nAverage FPS: {np.mean(fps_history):.1f}")


def main():
    parser = argparse.ArgumentParser(description='JetRacer Lane Tracker Inference')
    parser.add_argument(
        '--mode', type=str, choices=['pytorch', 'onnx'], default='pytorch',
        help='Inference backend',
    )
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoints/best_model.pth',
        help='Path to PyTorch checkpoint',
    )
    parser.add_argument(
        '--model', type=str, default='checkpoints/lane_tracker.onnx',
        help='Path to ONNX model',
    )
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--csi', action='store_true', help='Use CSI camera')
    parser.add_argument('--no-display', action='store_true', help='Disable preview')
    parser.add_argument('--jetracer', action='store_true', help='Control JetRacer')
    parser.add_argument('--throttle', type=float, default=0.4, help='Base throttle')
    
    # Steering post-processor settings
    parser.add_argument('--dead-zone', type=float, default=0.05)
    parser.add_argument('--max-steering', type=float, default=0.8)
    parser.add_argument('--smoothing', type=float, default=0.7)
    parser.add_argument('--solid-limit', type=float, default=0.6,
                        help='Max steering near solid lane markings')
    parser.add_argument('--dashed-limit', type=float, default=0.9,
                        help='Max steering near dashed lane markings (more permissive)')
    
    args = parser.parse_args()

    # Create backend
    if args.mode == 'pytorch':
        backend = PyTorchBackend(args.checkpoint)
    else:
        backend = ONNXBackend(args.model)

    # Steering config
    steering_config = {
        'dead_zone': args.dead_zone,
        'max_steering': args.max_steering,
        'smoothing_alpha': args.smoothing,
        'solid_lane_limit': args.solid_limit,
        'dashed_lane_limit': args.dashed_limit,
    }

    run_inference(
        backend=backend,
        camera_id=args.camera,
        use_csi=args.csi,
        display=not args.no_display,
        steering_config=steering_config,
        use_jetracer=args.jetracer,
        throttle=args.throttle,
    )


if __name__ == '__main__':
    main()
