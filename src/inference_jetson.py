"""
JetRacer Lane Tracking - Jetson Nano Inference
================================================
Inference script for deploying the trained lane tracker on Jetson Nano.
Supports both PyTorch and TensorRT (ONNX) backends.

Features:
- Camera capture via CSI or USB
- Steering post-processing with lane-type thresholds
- Optional adaptive steering and throttle for corners
- Fast deceleration and cautious acceleration based on curve/instability
- Integration with JetRacer NvidiaRacecar API

Usage on Jetson Nano:
    python -m src.inference_jetson --mode pytorch --checkpoint best_model.pth
    python -m src.inference_jetson --mode onnx --model lane_tracker.onnx
"""

import os
import platform
import time
import argparse

if platform.machine().lower() in ('aarch64', 'arm64'):
    os.environ.setdefault('OPENBLAS_CORETYPE', 'ARMV8')

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

from src.adaptive_controller import AdaptiveDriveController
from src.preprocessing_config import (
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ROAD_CROP_TOP_FRACTION,
)
from src.model import LaneTracker, SteeringPostProcessor


# ============================================================
# Image preprocessing (same as training validation transform)
# ============================================================
PREPROCESS = transforms.Compose([
    transforms.Resize((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    ),
])


def preprocess_frame(frame_bgr):
    """Convert OpenCV BGR frame to model input tensor.
    
    Args:
        frame_bgr: numpy array (H, W, 3) in BGR format (from OpenCV)
        
    Returns:
        torch.Tensor of shape (1, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
    """
    crop_top = int(round(frame_bgr.shape[0] * ROAD_CROP_TOP_FRACTION))
    frame_bgr = frame_bgr[crop_top:, :, :]
    # BGR → RGB → PIL
    frame_rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(frame_rgb)
    tensor = PREPROCESS(image).unsqueeze(0)
    return tensor


def preprocess_frame_onnx(frame_bgr):
    """Deployment preprocessing matched to the PIL training resize."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    crop_top = int(round(image.height * ROAD_CROP_TOP_FRACTION))
    road = image.crop((0, crop_top, image.width, image.height))
    resampling = getattr(Image, 'Resampling', Image)
    resized = np.asarray(
        road.resize(
            (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
            resample=resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    rgb = resized * (1.0 / 255.0)
    rgb -= np.array(NORMALIZE_MEAN, dtype=np.float32)
    rgb /= np.array(NORMALIZE_STD, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])


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
            input_tensor: deployment-sized input tensor
            
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
            input_tensor: deployment-sized input tensor
            
        Returns:
            Steering value as float
        """
        if hasattr(input_tensor, 'detach'):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = np.asarray(input_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: input_np})
        return float(outputs[0].item())

    def predict_frame(self, frame_bgr):
        """Preprocess a BGR frame without PIL/PyTorch and run ONNX."""
        return self.predict(preprocess_frame_onnx(frame_bgr))


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
    adaptive_control=False,
    adaptive_config=None,
    record=False,
    video_out='inference_record.avi',
):
    """Main inference loop with camera.
    
    Args:
        backend: Inference backend (PyTorchBackend or ONNXBackend)
        camera_id: Camera device ID
        use_csi: Use CSI camera (Jetson) instead of USB
        display: Show preview window
        steering_config: Dict of SteeringPostProcessor kwargs
        use_jetracer: Control JetRacer motors
        throttle: Fixed throttle, or maximum straight throttle in adaptive mode
        adaptive_control: Dynamically control steering and throttle
        adaptive_config: Dict of AdaptiveDriveController kwargs
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
            "video/x-raw(memory:NVMM), width=640, height=360, "
            "format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=360, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    # Setup steering post-processor
    if steering_config is None:
        steering_config = {}
    post_processor = SteeringPostProcessor(**steering_config)

    drive_controller = None
    if adaptive_control:
        if adaptive_config is None:
            adaptive_config = {}
        adaptive_config = dict(adaptive_config)
        adaptive_config.setdefault('max_throttle', throttle)
        adaptive_config.setdefault(
            'max_steering', steering_config.get('max_steering', 0.9)
        )
        adaptive_config.setdefault(
            'steering_gain', steering_config.get('steering_gain', 1.0)
        )
        drive_controller = AdaptiveDriveController(**adaptive_config)
        print(
            "Adaptive control enabled: throttle "
            f"{drive_controller.min_throttle:.2f}.."
            f"{drive_controller.max_throttle:.2f}"
        )

    # Setup JetRacer (optional)
    car = None
    if use_jetracer:
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
        except ModuleNotFoundError:
            from src.jetracer_fallback import NvidiaRacecar
        try:
            car = NvidiaRacecar()
            car.throttle = 0.0
            car.steering = 0.0
            print("JetRacer connected!")
        except Exception as e:
            print(f"WARNING: Could not connect to JetRacer: {e}")
            car = None

    # Setup video writer
    video_writer = None
    if record:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width == 0 or frame_height == 0:
            frame_width, frame_height = 640, 360
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_out, fourcc, 30.0, (frame_width, frame_height))
        print(f"Recording video to {video_out}")

    print("\nStarting inference... Press 'q' to quit.")
    fps_history = []
    last_control_time = time.monotonic()
    previous_throttle = (
        drive_controller.min_throttle
        if drive_controller is not None
        else throttle
    )

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Preprocess and predict
            if hasattr(backend, 'predict_frame'):
                raw_steering = backend.predict_frame(frame)
            else:
                input_tensor = preprocess_frame(frame)
                raw_steering = backend.predict(input_tensor)

            control_time = time.monotonic()
            control_dt = control_time - last_control_time
            last_control_time = control_time

            if drive_controller is not None:
                command = drive_controller.update(
                    raw_steering,
                    dt=control_dt,
                )
                steering = command.steering
                applied_throttle = command.throttle
                previous_throttle = applied_throttle
            else:
                # Legacy fixed-speed mode.
                steering = post_processor.process(
                    raw_steering, lane_type='solid'
                )
                applied_throttle = throttle
                command = None

            # Apply to JetRacer
            if car is not None:
                car.steering = steering
                car.throttle = applied_throttle

            # FPS calculation
            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 1e-6)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            # Display / Record
            if display or record:
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
                if command is not None:
                    cv2.putText(
                        frame,
                        f"Throttle: {applied_throttle:.3f} "
                        f"(target {command.target_throttle:.3f})",
                        (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )
                if display:
                    cv2.imshow('JetRacer Lane Tracker', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                if record and video_writer is not None:
                    video_writer.write(frame)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Cleanup
        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        if fps_history:
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
    parser.add_argument(
        '--throttle', type=float, default=0.4,
        help='Fixed throttle, or maximum straight throttle in adaptive mode',
    )
    parser.add_argument(
        '--adaptive-control', action='store_true',
        help='Adapt steering and throttle to curve demand',
    )
    parser.add_argument(
        '--min-throttle', type=float, default=0.18,
        help='Minimum throttle for a tight/uncertain corner',
    )
    parser.add_argument(
        '--curve-full-scale', type=float, default=0.75,
        help='Raw steering magnitude treated as a maximum-demand corner',
    )
    parser.add_argument(
        '--acceleration-rate', type=float, default=0.18,
        help='Maximum throttle increase per second',
    )
    parser.add_argument(
        '--deceleration-rate', type=float, default=1.2,
        help='Maximum throttle decrease per second',
    )
    # Steering post-processor settings
    parser.add_argument('--dead-zone', type=float, default=0.05)
    parser.add_argument('--max-steering', type=float, default=0.8)
    parser.add_argument('--smoothing', type=float, default=0.7)
    parser.add_argument('--solid-limit', type=float, default=0.6,
                        help='Max steering near solid lane markings')
    parser.add_argument('--dashed-limit', type=float, default=0.9,
                        help='Max steering near dashed lane markings (more permissive)')
    parser.add_argument('--steering-gain', type=float, default=1.0,
                        help='Multiplier for raw steering (helps on tight corners)')
    parser.add_argument('--record', action='store_true', help='Record video to file')
    parser.add_argument('--video-out', type=str, default='inference_record.avi',
                        help='Output video file name')
    
    args = parser.parse_args()

    if not 0.0 <= args.min_throttle <= args.throttle <= 1.0:
        parser.error(
            'Expected 0 <= --min-throttle <= --throttle <= 1'
        )
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
        'steering_gain': args.steering_gain,
    }
    adaptive_config = {
        'min_throttle': args.min_throttle,
        'max_throttle': args.throttle,
        'max_steering': args.max_steering,
        'steering_gain': args.steering_gain,
        'dead_zone': args.dead_zone,
        'curve_full_scale': args.curve_full_scale,
        'acceleration_rate': args.acceleration_rate,
        'deceleration_rate': args.deceleration_rate,
    }
    run_inference(
        backend=backend,
        camera_id=args.camera,
        use_csi=args.csi,
        display=not args.no_display,
        steering_config=steering_config,
        use_jetracer=args.jetracer,
        throttle=args.throttle,
        adaptive_control=args.adaptive_control,
        adaptive_config=adaptive_config,
        record=args.record,
        video_out=args.video_out,
    )


if __name__ == '__main__':
    main()
