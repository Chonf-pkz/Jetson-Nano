"""Generate the unified JetRacer live control and inference notebook."""

from pathlib import Path

import nbformat as nbf


OUTPUT_PATH = Path(__file__).resolve().parent / "live_inference.ipynb"


INTRO = r'''# JetRacer Live Control Center

Notebook hợp nhất điều khiển tay cầm, tự lái, camera và recorder.

Chạy các cell theo thứ tự từ trên xuống. Ba cell đầu chỉ setup phần cứng và
model; **cell cuối cùng** hiển thị toàn bộ dashboard tương tác.

Quy tắc an toàn:

- Xe luôn khởi động ở chế độ **STOP**.
- Dataset huấn luyện chỉ được ghi trong chế độ **MANUAL** từ frame camera raw.
- Chế độ **AUTO** có thể ghi video overlay và telemetry để đánh giá, nhưng
  không dùng lệnh tự sinh làm nhãn huấn luyện.
- Chuyển chế độ sẽ tự dừng phiên record hiện tại để không trộn nguồn nhãn.
'''


SETUP = r'''# ============================================================
# 1. SETUP IMPORTS, PROJECT PATHS, AND CLEAN OLD CALLBACKS
# ============================================================
import os

LIVE_BUILD = "MODEL_ONLY_NO_LANE_GATE_V4"

# JetPack 4/OpenBLAS trên Nano có thể chọn nhầm CPU kernel.
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

import sys
import csv
import cv2
import glob
import queue
import time
import threading
import traceback
import numpy as np
import traitlets
import ipywidgets.widgets as widgets

from pathlib import Path
from datetime import datetime
from IPython.display import display


# Giữ lại xe để không tạo thêm một đối tượng ServoKit không cần thiết.
_previous_car = globals().get("car")
_previous_camera = globals().get("camera")

# Nếu cell output đã chạy trước đó, dừng dashboard đúng cách.
_old_shutdown = globals().get("shutdown_dashboard")
if callable(_old_shutdown):
    try:
        _old_shutdown(release_camera=True)
    except Exception:
        pass

# Tương thích với các bản notebook cũ chưa có shutdown_dashboard.
_old_record_event = globals().get("_record_stop_event")
if _old_record_event is not None:
    try:
        _old_record_event.set()
    except Exception:
        pass

for _old_link_name in ("steering_link", "throttle_link"):
    _old_link = globals().get(_old_link_name)
    if _old_link is not None:
        try:
            _old_link.unlink()
        except Exception:
            pass

if _previous_car is not None:
    try:
        _previous_car.throttle = 0.0
        _previous_car.steering = 0.0
    except Exception:
        pass

if _previous_camera is not None:
    for _callback_name in (
        "_on_camera_frame",
        "_update_camera_preview",
        "update_camera",
    ):
        _old_callback = globals().get(_callback_name)
        if _old_callback is not None:
            try:
                _previous_camera.unobserve(_old_callback, names="value")
            except Exception:
                pass
    try:
        _previous_camera.running = False
    except Exception:
        pass
    try:
        _previous_camera.cap.release()
    except Exception:
        pass
    time.sleep(2.0)


# ------------------------------------------------------------
# Khôi phục package JetRacer trên các image JetPack cũ.
# ------------------------------------------------------------
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ModuleNotFoundError as original_error:
    candidate_files = [
        Path("/home/jetson/jetracer/jetracer/nvidia_racecar.py"),
        Path("/home/jetson/ws/jetracer/jetracer/nvidia_racecar.py"),
        Path(
            "/usr/local/lib/python3.6/dist-packages/"
            "jetracer/nvidia_racecar.py"
        ),
        Path(
            "/usr/lib/python3/dist-packages/"
            "jetracer/nvidia_racecar.py"
        ),
        Path(
            "/home/jetson/.local/lib/python3.6/"
            "site-packages/jetracer/nvidia_racecar.py"
        ),
    ]
    candidate_files += [
        Path(path)
        for path in glob.glob(
            "/usr/local/lib/python3.6/dist-packages/"
            "jetracer*.egg/jetracer/nvidia_racecar.py"
        )
    ]
    jetracer_file = next(
        (path for path in candidate_files if path.exists()),
        None,
    )
    if jetracer_file is None:
        try:
            from src.jetracer_fallback import NvidiaRacecar
            print("⚠️ Dùng driver JetRacer dự phòng trong src/")
        except (ModuleNotFoundError, ImportError) as fallback_error:
            raise ModuleNotFoundError(
                "Thiếu driver xe. Hãy upload src/jetracer_fallback.py và "
                "cài adafruit-circuitpython-servokit. Chi tiết: {}"
                .format(fallback_error)
            ) from original_error
    else:
        jetracer_root = str(jetracer_file.parent.parent)
        if jetracer_root not in sys.path:
            sys.path.insert(0, jetracer_root)
        for module_name in list(sys.modules):
            if module_name == "jetracer" or module_name.startswith("jetracer."):
                del sys.modules[module_name]
        from jetracer.nvidia_racecar import NvidiaRacecar

from jetcam.csi_camera import CSICamera


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").is_dir():
    raise RuntimeError(
        "Không thấy thư mục src tại: {}\n"
        "Hãy mở notebook từ thư mục gốc của project."
        .format(PROJECT_ROOT)
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_jetson import ONNXBackend
from src.adaptive_controller import AdaptiveDriveController

print("✅ Setup imports hoàn tất")
print("Python kernel:", sys.executable)
print("Project:", PROJECT_ROOT)
print("Controller: MODEL ONLY (không geometry)")
print("Build:", LIVE_BUILD)
'''


HARDWARE = r'''# ============================================================
# 2. INITIALIZE CAR AND CAMERA (NO UI YET)
# ============================================================
INITIAL_THROTTLE_GAIN = 1.00
INITIAL_STEERING_GAIN = -1.00
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360

car_ready = (
    _previous_car is not None
    and hasattr(_previous_car, "throttle_motor")
    and hasattr(_previous_car, "steering_motor")
)

if car_ready:
    car = _previous_car
    print("✅ Tái sử dụng đối tượng xe")
else:
    car = NvidiaRacecar()
    print("✅ Đã khởi tạo xe")

car.throttle = 0.0
car.steering = 0.0
car.throttle_gain = INITIAL_THROTTLE_GAIN
car.steering_gain = INITIAL_STEERING_GAIN

try:
    car.steering_motor.set_pulse_width_range(500, 2500)
    print("✅ PWM lái: 500–2500 µs")
except Exception as error:
    print("⚠️ Không thể đặt PWM lái:", error)

camera = CSICamera(
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    capture_width=1280,
    capture_height=720,
    capture_fps=30,
)
camera.running = False


def _lock_camera_controls_best_effort(capture):
    """Disable exposure/WB automation when the CSI backend exposes controls."""
    controls = []
    for name, auto_property_name, manual_value, value_property_name in (
        (
            "exposure", "CAP_PROP_AUTO_EXPOSURE", 0.25,
            "CAP_PROP_EXPOSURE",
        ),
        (
            "white balance", "CAP_PROP_AUTO_WB", 0.0,
            "CAP_PROP_WB_TEMPERATURE",
        ),
    ):
        auto_property = getattr(cv2, auto_property_name, None)
        value_property = getattr(cv2, value_property_name, None)
        if auto_property is None or value_property is None:
            controls.append((name, False, 0.0))
            continue
        current_value = capture.get(value_property)
        auto_locked = bool(capture.set(auto_property, manual_value))
        value_locked = False
        if auto_locked and np.isfinite(current_value) and current_value != 0.0:
            value_locked = bool(capture.set(value_property, current_value))
        controls.append((name, auto_locked and value_locked, current_value))
    return controls


camera_control_status = _lock_camera_controls_best_effort(camera.cap)

print("✅ Camera CSI đã khởi tạo; xe đang STOP")
print("Camera output: {}x{}; model crop phần đường và resize 200x66".format(
    CAMERA_WIDTH, CAMERA_HEIGHT
))
for control_name, locked, control_value in camera_control_status:
    if locked:
        print("✅ Đã khóa {} tại {}".format(control_name, control_value))
    else:
        print(
            "⚠️ Backend CSI không cho khóa {} qua OpenCV; "
            "cần cấu hình nvarguscamerasrc sau khi đo giá trị cố định."
            .format(control_name)
        )
print("Throttle gain:", float(car.throttle_gain))
print("Steering gain:", float(car.steering_gain))
'''


MODEL = r'''# ============================================================
# 3. LOAD NEW MODEL-ONLY CONTROLLER
# ============================================================
model_path = (
    PROJECT_ROOT
    / "checkpoints"
    / "lane_tracker_ir8_opset13.onnx"
)
if not model_path.is_file():
    raise FileNotFoundError("Không tìm thấy model: {}".format(model_path))

backend = ONNXBackend(str(model_path))
if not hasattr(backend, "predict_frame"):
    raise RuntimeError(
        "src/inference_jetson.py trên Jetson đang là bản cũ. "
        "Hãy cập nhật file rồi Restart Kernel."
    )

drive_controller = AdaptiveDriveController(
    min_throttle=0.0,
    max_throttle=0.0,
    max_steering=0.85,
    steering_gain=1.10,
    steering_exponent=0.88,
    dead_zone=0.04,
    anticipation_gain=0.30,
    curve_full_scale=0.65,
    straight_time_constant=0.14,
    corner_time_constant=0.07,
    steering_rate_limit=2.40,
    acceleration_rate=0.18,
    deceleration_rate=1.20,
    instability_weight=0.40,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "track_lane_dataset"
    / "dataset_steering"
)
RECORDS_ROOT = PROJECT_ROOT / "records"
SNAPSHOTS_ROOT = RECORDS_ROOT / "snapshots"

DATASET_ROOT.mkdir(parents=True, exist_ok=True)
RECORDS_ROOT.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_ROOT.mkdir(parents=True, exist_ok=True)

print("✅ Model-only controller đã sẵn sàng")
print("Model:", model_path)
print("Dataset:", DATASET_ROOT)
print("Validation records:", RECORDS_ROOT)
'''


DASHBOARD = r'''# ============================================================
# 4. OUTPUT: INTERACTIVE CAMERA / MANUAL / AUTO / RECORD
# ============================================================
# Chạy lại riêng cell này vẫn an toàn: dashboard cũ được tháo callback trước.
_previous_dashboard_shutdown = globals().get("shutdown_dashboard")
if callable(_previous_dashboard_shutdown):
    try:
        _previous_dashboard_shutdown(release_camera=False)
    except Exception:
        pass


MODE_STOP = "STOP"
MODE_MANUAL = "MANUAL"
MODE_AUTO = "AUTO"

PREVIEW_FPS = 12.0
PREVIEW_JPEG_QUALITY = 65
DATASET_JPEG_QUALITY = 90
STEERING_MAX_DEG = 30.0
MODEL_INSTABILITY_STOP = 0.95
MODEL_INSTABILITY_STOP_FRAMES = 8
MODEL_INSTABILITY_RESUME = 0.60
MODEL_INSTABILITY_RESUME_FRAMES = 10
LOW_CONTROL_FPS = 5.0

_state_lock = threading.RLock()
_state = {
    "mode": MODE_STOP,
    "gamepad_connected": False,
    "last_preview_time": 0.0,
    "last_control_time": None,
    "control_fps": 0.0,
    "frame_sequence": 0,
    "instability_count": 0,
    "stability_count": 0,
    "safety_stop": False,
    "last_raw_frame": None,
    "last_display_frame": None,
    "telemetry": {},
    "shutdown": False,
}

_record = {
    "active": False,
    "dataset": False,
    "video": False,
    "log": False,
    "session_dir": None,
    "images_dir": None,
    "csv_file": None,
    "csv_writer": None,
    "video_writer": None,
    "video_frame_index": 0,
    "video_fps": 10.0,
    "sample_index": 0,
    "written_count": 0,
    "start_time": 0.0,
    "next_sample_time": 0.0,
    "interval": 0.1,
    "mode": MODE_STOP,
    "queue": None,
    "thread": None,
    "stop_event": None,
    "dropped_frames": 0,
    "worker_error": "",
}

_gamepad_observers = []


# ------------------------------------------------------------
# Widgets
# ------------------------------------------------------------
camera_view = widgets.Image(
    format="jpeg", width=CAMERA_WIDTH, height=CAMERA_HEIGHT
)
mode_status = widgets.HTML()
camera_status = widgets.HTML(value="<b>📷 Camera:</b> đang khởi động...")
telemetry_status = widgets.HTML()
record_status = widgets.HTML(value="<b>⚪ Record:</b> chưa ghi")
gamepad_status = widgets.HTML(value="<b>🎮 Tay cầm:</b> chưa kích hoạt")

btn_stop = widgets.Button(
    description="🛑 DỪNG KHẨN CẤP", button_style="danger",
    layout=widgets.Layout(width="210px", height="44px"),
)
btn_manual = widgets.Button(
    description="🎮 MANUAL", button_style="info",
    layout=widgets.Layout(width="150px", height="44px"),
)
btn_auto = widgets.Button(
    description="🤖 AUTO", button_style="success",
    layout=widgets.Layout(width="150px", height="44px"),
)
btn_shutdown = widgets.Button(
    description="Tắt dashboard/camera", layout=widgets.Layout(width="210px")
)

gamepad = widgets.Controller(index=0)
btn_gamepad_connect = widgets.Button(
    description="Kích hoạt tay cầm", button_style="success",
    layout=widgets.Layout(width="210px"),
)
manual_max_throttle = widgets.FloatSlider(
    value=0.35, min=0.0, max=1.00, step=0.01,
    description="Ga tay tối đa", continuous_update=False,
    layout=widgets.Layout(width="430px"),
)
manual_steering_scale = widgets.FloatSlider(
    value=1.0, min=0.2, max=1.0, step=0.05,
    description="Biên lái tay", continuous_update=False,
    layout=widgets.Layout(width="430px"),
)
auto_max_throttle = widgets.FloatSlider(
    value=0.0, min=0.0, max=1.00, step=0.01,
    description="Auto max throttle", continuous_update=False,
    layout=widgets.Layout(width="430px"),
)
auto_min_throttle = widgets.FloatSlider(
    value=0.22, min=0.0, max=1.00, step=0.01,
    description="Auto min throttle", continuous_update=False,
    layout=widgets.Layout(width="430px"),
)
record_dataset = widgets.Checkbox(
    value=True, description="Dataset raw + labels.csv (chỉ MANUAL)", indent=False
)
record_video = widgets.Checkbox(
    value=False, description="Video overlay", indent=False
)
record_log = widgets.Checkbox(
    value=True, description="Telemetry CSV", indent=False
)
record_fps = widgets.FloatSlider(
    value=10.0, min=1.0, max=20.0, step=1.0,
    description="Record FPS", continuous_update=False,
    layout=widgets.Layout(width="430px"),
)
btn_record_start = widgets.Button(
    description="🔴 BẮT ĐẦU RECORD", button_style="success",
    layout=widgets.Layout(width="210px", height="42px"),
)
btn_record_stop = widgets.Button(
    description="⏹ DỪNG RECORD", button_style="danger", disabled=True,
    layout=widgets.Layout(width="190px", height="42px"),
)
btn_snapshot = widgets.Button(
    description="📸 LƯU FRAME RAW",
    layout=widgets.Layout(width="190px", height="42px"),
)
event_output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #ddd", height="150px", overflow="auto"
    )
)


def _clip(value, lower=-1.0, upper=1.0):
    return max(lower, min(upper, float(value)))


def _fmt(value, digits=3):
    if value is None:
        return ""
    try:
        return ("{:." + str(digits) + "f}").format(float(value))
    except Exception:
        return ""


def _log(message):
    with event_output:
        print("{}  {}".format(datetime.now().strftime("%H:%M:%S"), message))


def _set_car(steering, throttle):
    steering = _clip(steering)
    throttle = _clip(throttle)
    car.steering = steering
    car.throttle = throttle
    return steering, throttle


def _update_mode_status(extra=""):
    mode = _state["mode"]
    colors = {MODE_STOP: "red", MODE_MANUAL: "#0b70c9", MODE_AUTO: "green"}
    text = "<b style='color:{};font-size:18px'>MODE: {}</b>".format(
        colors.get(mode, "black"), mode
    )
    if extra:
        text += " — " + str(extra)
    mode_status.value = text


# ------------------------------------------------------------
# Gamepad
# ------------------------------------------------------------
def _unbind_gamepad():
    global _gamepad_observers
    for axis, callback in _gamepad_observers:
        try:
            axis.unobserve(callback, names="value")
        except Exception:
            pass
    _gamepad_observers = []
    _state["gamepad_connected"] = False


def _axis_value(index):
    if len(gamepad.axes) <= index:
        return 0.0
    return float(gamepad.axes[index].value)


def _apply_manual_axes(change=None):
    if _state["mode"] != MODE_MANUAL:
        return
    steering_axis = _axis_value(2)
    throttle_axis = -_axis_value(1)
    if abs(steering_axis) < 0.05:
        steering_axis = 0.0
    if abs(throttle_axis) < 0.08:
        throttle_axis = 0.0
    steering = _clip(steering_axis * float(manual_steering_scale.value))
    throttle = _clip(throttle_axis * float(manual_max_throttle.value))
    _set_car(steering, throttle)


def _connect_gamepad(button=None):
    global _gamepad_observers
    _unbind_gamepad()
    if len(gamepad.axes) < 3:
        gamepad_status.value = (
            "<b style='color:red'>🎮 Chưa thấy đủ trục tay cầm.</b> "
            "Xoay hai cần rồi bấm lại."
        )
        _log("Không thể kích hoạt tay cầm: cần ít nhất 3 axes")
        return
    gamepad.axes[2].observe(_apply_manual_axes, names="value")
    gamepad.axes[1].observe(_apply_manual_axes, names="value")
    _gamepad_observers = [
        (gamepad.axes[2], _apply_manual_axes),
        (gamepad.axes[1], _apply_manual_axes),
    ]
    _state["gamepad_connected"] = True
    gamepad_status.value = (
        "<b style='color:green'>🎮 Tay cầm đã kích hoạt</b> — "
        "cần trái: tiến/lùi, cần phải: bẻ lái"
    )
    _log("Đã kích hoạt tay cầm")


def _on_gamepad_connection(change):
    if bool(change.get("new")):
        return
    _unbind_gamepad()
    gamepad_status.value = (
        "<b style='color:red'>🎮 Tay cầm đã ngắt kết nối</b>"
    )
    if _state["mode"] == MODE_MANUAL:
        _set_mode(MODE_STOP, "mất kết nối tay cầm")


# ------------------------------------------------------------
# Recorder: raw training images are only allowed in MANUAL mode.
# ------------------------------------------------------------
CSV_FIELDS = [
    "sample_index", "frame_id", "timestamp", "frame_monotonic_seconds",
    "control_applied_monotonic_seconds", "control_latency_seconds",
    "elapsed_seconds", "video_frame_index", "video_time_seconds",
    "video_log_offset_seconds", "record_dropped_frames", "image_file",
    "steering_normalized", "steering_angle_est_deg", "steering_gain",
    "throttle_normalized", "throttle_gain", "control_mode",
    "raw_model_steering", "filtered_model_steering",
    "model_instability", "curve_demand", "target_throttle",
    "control_fps", "safety_stop",
]


def _close_record_files():
    video_writer = _record.get("video_writer")
    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass
    csv_file = _record.get("csv_file")
    if csv_file is not None:
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass
    _record["video_writer"] = None
    _record["csv_writer"] = None
    _record["csv_file"] = None


def _write_record_packet(packet):
    """Write one atomic video/log sample owned by the recorder thread."""
    sample_index = int(packet["sample_index"])
    frame_id = int(packet["frame_id"])
    frame_time = float(packet["frame_time"])
    elapsed = frame_time - float(_record["start_time"])
    telemetry = packet["telemetry"]
    image_file = ""

    raw_frame = packet.get("raw_frame")
    if _record["dataset"] and raw_frame is not None:
        image_name = "frame_{:06d}.jpg".format(sample_index)
        image_path = _record["images_dir"] / image_name
        saved = cv2.imwrite(
            str(image_path), raw_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), DATASET_JPEG_QUALITY],
        )
        if not saved:
            raise RuntimeError("Không lưu được {}".format(image_path))
        image_file = "images/" + image_name

    video_frame_index = ""
    video_time_seconds = ""
    video_log_offset_seconds = ""
    video_writer = _record.get("video_writer")
    display_frame = packet.get("display_frame")
    if video_writer is not None and display_frame is not None:
        video_frame_index = int(_record["video_frame_index"])
        video_time = video_frame_index / max(
            float(_record["video_fps"]), 1e-6
        )
        sync_frame = display_frame.copy()
        control_latency = packet["control_applied_time"] - frame_time
        sync_text = "SYNC F{} S{} V{} T{:.3f} L{:.3f}".format(
            frame_id, sample_index, video_frame_index, elapsed,
            control_latency,
        )
        cv2.rectangle(
            sync_frame,
            (0, max(0, sync_frame.shape[0] - 24)),
            (sync_frame.shape[1], sync_frame.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            sync_frame, sync_text, (8, sync_frame.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
        )
        video_writer.write(sync_frame)
        _record["video_frame_index"] = video_frame_index + 1
        video_time_seconds = _fmt(video_time, 6)
        video_log_offset_seconds = _fmt(elapsed - video_time, 6)

    row = {
        "sample_index": sample_index,
        "frame_id": frame_id,
        "timestamp": packet["timestamp"],
        "frame_monotonic_seconds": _fmt(frame_time, 6),
        "control_applied_monotonic_seconds": _fmt(
            packet["control_applied_time"], 6
        ),
        "control_latency_seconds": _fmt(
            packet["control_applied_time"] - frame_time, 6
        ),
        "elapsed_seconds": _fmt(elapsed, 6),
        "video_frame_index": video_frame_index,
        "video_time_seconds": video_time_seconds,
        "video_log_offset_seconds": video_log_offset_seconds,
        "record_dropped_frames": int(packet["dropped_frames"]),
        "image_file": image_file,
        "steering_normalized": _fmt(packet["steering"], 6),
        "steering_angle_est_deg": _fmt(
            packet["steering"] * STEERING_MAX_DEG, 3
        ),
        "steering_gain": _fmt(packet["steering_gain"], 3),
        "throttle_normalized": _fmt(packet["throttle"], 6),
        "throttle_gain": _fmt(packet["throttle_gain"], 3),
        "control_mode": packet["control_mode"],
        "raw_model_steering": _fmt(telemetry.get("raw_model_steering"), 6),
        "filtered_model_steering": _fmt(
            telemetry.get("filtered_model_steering"), 6
        ),
        "model_instability": _fmt(
            telemetry.get("model_instability"), 6
        ),
        "curve_demand": _fmt(telemetry.get("curve_demand"), 6),
        "target_throttle": _fmt(telemetry.get("target_throttle"), 6),
        "control_fps": _fmt(telemetry.get("control_fps"), 3),
        "safety_stop": str(bool(telemetry.get("safety_stop", False))),
    }
    csv_writer = _record.get("csv_writer")
    if csv_writer is not None:
        csv_writer.writerow(row)
        _record["csv_file"].flush()
    _record["written_count"] = int(_record["written_count"]) + 1


def _record_worker(record_queue, stop_event):
    try:
        while not stop_event.is_set() or not record_queue.empty():
            try:
                packet = record_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            try:
                _write_record_packet(packet)
            finally:
                record_queue.task_done()
    except Exception as error:
        _record["worker_error"] = str(error)
        _record["active"] = False
    finally:
        _close_record_files()


def _stop_recording_internal(reason="Đã dừng record"):
    record_thread = None
    with _state_lock:
        record_thread = _record.get("thread")
        if not _record["active"] and record_thread is None:
            return
        _record["active"] = False
        stop_event = _record.get("stop_event")
        if stop_event is not None:
            stop_event.set()
        session_dir = _record["session_dir"]
    if (
        record_thread is not None
        and record_thread is not threading.current_thread()
    ):
        record_thread.join(timeout=5.0)
    with _state_lock:
        if record_thread is not None and record_thread.is_alive():
            record_status.value = (
                "<b>⏳ Đang hoàn tất record nền…</b> — {}".format(reason)
            )
            return
        _close_record_files()
        count = int(_record["written_count"])
        dropped = int(_record["dropped_frames"])
        worker_error = _record.get("worker_error") or ""
        _record["thread"] = None
        _record["queue"] = None
        _record["stop_event"] = None
        btn_record_start.disabled = False
        btn_record_stop.disabled = True
        record_status.value = (
            "<b>⏹ Đã dừng record</b> — {} mẫu, bỏ {} — {}".format(
                count, dropped, reason
            )
        )
        if worker_error:
            record_status.value += " — lỗi: {}".format(worker_error)
        _log("{}; {} mẫu; bỏ {}; {}".format(
            reason, count, dropped, session_dir
        ))


def _start_recording_impl(button=None):
    with _state_lock:
        old_thread = _record.get("thread")
        if _record["active"] or (
            old_thread is not None and old_thread.is_alive()
        ):
            _log("Record đang hoạt động")
            return
        save_dataset = bool(record_dataset.value)
        save_video = bool(record_video.value)
        save_log = bool(record_log.value)
        if not (save_dataset or save_video or save_log):
            _log("Hãy chọn ít nhất một loại dữ liệu cần ghi")
            return
        if save_dataset and _state["mode"] != MODE_MANUAL:
            record_status.value = (
                "<b style='color:red'>❌ Dataset chỉ được ghi ở MANUAL.</b> "
                "AUTO chỉ được ghi video/log đánh giá."
            )
            _log("Từ chối ghi dataset vì mode không phải MANUAL")
            return
        if save_dataset and not _state["gamepad_connected"]:
            record_status.value = (
                "<b style='color:red'>❌ Chưa kích hoạt tay cầm.</b> "
                "Không thể tạo nhãn lái tay đáng tin cậy."
            )
            _log("Từ chối ghi dataset vì tay cầm chưa kết nối")
            return
        frame = getattr(camera, "value", None)
        if frame is None:
            _log("Camera chưa có frame; chưa thể record")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if save_dataset:
            session_dir = DATASET_ROOT / ("session_" + stamp)
        else:
            session_dir = RECORDS_ROOT / ("run_" + stamp)
        session_dir.mkdir(parents=True, exist_ok=False)

        images_dir = None
        if save_dataset:
            images_dir = session_dir / "images"
            images_dir.mkdir(parents=False, exist_ok=False)

        csv_file = None
        csv_writer = None
        if save_dataset or save_log:
            csv_name = "labels.csv" if save_dataset else "telemetry.csv"
            csv_file = (session_dir / csv_name).open("w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            csv_file.flush()

        video_writer = None
        requested_fps = float(record_fps.value)
        fps = requested_fps
        if _state["mode"] == MODE_AUTO:
            observed_fps = float(_state.get("control_fps") or 0.0)
            if observed_fps <= 0.0:
                observed_fps = 3.0
            fps = min(requested_fps, max(1.0, observed_fps))
        if save_video:
            video_path = session_dir / "validation_overlay.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            frame_height, frame_width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, fps,
                (int(frame_width), int(frame_height))
            )
            if not video_writer.isOpened():
                if csv_file is not None:
                    csv_file.close()
                raise RuntimeError(
                    "Không mở được VideoWriter: {}".format(video_path)
                )

        record_start_time = time.monotonic()
        record_queue = queue.Queue(maxsize=8)
        record_stop_event = threading.Event()
        record_thread = threading.Thread(
            target=_record_worker,
            args=(record_queue, record_stop_event),
            name="jetracer-recorder",
            daemon=True,
        )
        _record.update({
            "active": True, "dataset": save_dataset, "video": save_video,
            "log": save_log, "session_dir": session_dir,
            "images_dir": images_dir, "csv_file": csv_file,
            "csv_writer": csv_writer, "video_writer": video_writer,
            "video_frame_index": 0, "video_fps": fps,
            "sample_index": 0, "written_count": 0,
            "start_time": record_start_time,
            "next_sample_time": record_start_time,
            "interval": 1.0 / max(fps, 1.0), "mode": _state["mode"],
            "queue": record_queue, "thread": record_thread,
            "stop_event": record_stop_event, "dropped_frames": 0,
            "worker_error": "",
        })
        record_thread.start()
        btn_record_start.disabled = True
        btn_record_stop.disabled = False
        record_status.value = (
            "<b style='color:red'>🔴 ĐANG RECORD</b> — {} — {:.1f} FPS".format(
                _state["mode"], fps
            )
        )
        _log("Bắt đầu record: {}".format(session_dir))


def _start_recording(button=None):
    try:
        _start_recording_impl(button)
    except Exception as error:
        _record["active"] = False
        _close_record_files()
        btn_record_start.disabled = False
        btn_record_stop.disabled = True
        record_status.value = (
            "<b style='color:red'>❌ Không thể bắt đầu record:</b> {}"
            .format(error)
        )
        _log("Lỗi bắt đầu record: {}".format(error))


def _record_frame_if_due(
    raw_frame, display_frame, frame_id, frame_time,
    frame_timestamp, control_applied_time,
):
    """Queue one matched camera/control/video sample without blocking AUTO."""
    with _state_lock:
        if not _record["active"]:
            return
        if frame_time + 1e-9 < float(_record["next_sample_time"]):
            return

        interval = float(_record["interval"])
        next_sample_time = float(_record["next_sample_time"])
        while next_sample_time <= frame_time:
            next_sample_time += interval
        _record["next_sample_time"] = next_sample_time

        sample_index = int(_record["sample_index"])
        packet = {
            "sample_index": sample_index,
            "frame_id": int(frame_id),
            "frame_time": float(frame_time),
            "timestamp": frame_timestamp,
            "control_applied_time": float(control_applied_time),
            "raw_frame": raw_frame.copy() if _record["dataset"] else None,
            "display_frame": (
                display_frame.copy() if _record["video"] else None
            ),
            "telemetry": dict(_state.get("telemetry") or {}),
            "steering": float(car.steering),
            "throttle": float(car.throttle),
            "steering_gain": float(car.steering_gain),
            "throttle_gain": float(car.throttle_gain),
            "control_mode": _state["mode"],
            "dropped_frames": int(_record["dropped_frames"]),
        }
        record_queue = _record.get("queue")
        if record_queue is None:
            return
        try:
            record_queue.put_nowait(packet)
        except queue.Full:
            # Discard the oldest queued sample so saved video stays near live
            # time.  The dropped counter is written into subsequent log rows.
            try:
                record_queue.get_nowait()
                record_queue.task_done()
                _record["dropped_frames"] += 1
            except queue.Empty:
                pass
            packet["dropped_frames"] = int(_record["dropped_frames"])
            record_queue.put_nowait(packet)
        _record["sample_index"] = sample_index + 1
        record_status.value = (
            "<b style='color:red'>🔴 ĐANG RECORD</b> — hàng đợi {} — "
            "bỏ {} frame".format(
                record_queue.qsize(), _record["dropped_frames"]
            )
        )


def _stop_recording(button=None):
    _stop_recording_internal("Người dùng dừng")


def _save_snapshot(button=None):
    frame = _state.get("last_raw_frame")
    if frame is None:
        _log("Chưa có frame camera để lưu")
        return
    path = SNAPSHOTS_ROOT / "snapshot_{}.jpg".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    saved = cv2.imwrite(
        str(path), frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), DATASET_JPEG_QUALITY],
    )
    if saved:
        _log("Đã lưu frame raw: {}".format(path))
    else:
        _log("Không lưu được frame: {}".format(path))


# ------------------------------------------------------------
# Mode state machine
# ------------------------------------------------------------
def _set_mode(new_mode, reason=""):
    if new_mode not in (MODE_STOP, MODE_MANUAL, MODE_AUTO):
        raise ValueError("Mode không hợp lệ: {}".format(new_mode))
    if _record["active"] and new_mode != _state["mode"]:
        _stop_recording_internal("Tự dừng khi chuyển mode")
    _set_car(0.0, 0.0)
    drive_controller.reset(initial_throttle=0.0)
    _state["instability_count"] = 0
    _state["stability_count"] = 0
    _state["safety_stop"] = False
    _state["last_control_time"] = None
    _state["control_fps"] = 0.0
    _state["mode"] = new_mode
    if new_mode == MODE_MANUAL:
        if _state["gamepad_connected"]:
            _apply_manual_axes()
            extra = "tay cầm đang điều khiển"
        else:
            extra = "hãy kích hoạt tay cầm"
    elif new_mode == MODE_AUTO:
        if float(auto_max_throttle.value) <= 0.0:
            extra = "đang đứng yên; tăng Auto max throttle để chạy"
        else:
            extra = "model ONNX điều khiển góc lái và tốc độ"
    else:
        extra = reason or "xe đã dừng"
    _update_mode_status(extra)
    _log("Chuyển mode: {}{}".format(
        new_mode, " — " + reason if reason else ""
    ))


def _on_stop(button=None):
    _set_mode(MODE_STOP, "DỪNG KHẨN CẤP")


def _on_manual(button=None):
    _set_mode(MODE_MANUAL)


def _on_auto(button=None):
    _set_mode(MODE_AUTO)


# ------------------------------------------------------------
# Camera callback: the only automatic writer to the vehicle.
# ------------------------------------------------------------
def _draw_common_overlay(frame, mode, steering, throttle):
    output = frame.copy()
    mode_color = (
        (0, 0, 255) if mode == MODE_STOP
        else (255, 180, 0) if mode == MODE_MANUAL
        else (0, 255, 0)
    )
    cv2.putText(
        output, "MODE: {}".format(mode), (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 1,
    )
    cv2.putText(
        output, "MODEL ONLY / NO LANE GATE", (8, output.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
    )
    cv2.putText(
        output, "Steer {:+.3f}  Thr {:+.3f}".format(steering, throttle),
        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1,
    )
    if _record["active"]:
        cv2.circle(output, (212, 12), 6, (0, 0, 255), -1)
    return output


def _on_camera_frame(change):
    if _state["shutdown"]:
        return
    raw_frame = change.get("new")
    if raw_frame is None:
        return
    raw_frame = raw_frame.copy()
    _state["last_raw_frame"] = raw_frame
    mode = _state["mode"]
    frame_now = time.monotonic()
    frame_timestamp = datetime.now().isoformat(timespec="milliseconds")
    _state["frame_sequence"] += 1
    frame_id = int(_state["frame_sequence"])
    previous_control_time = _state["last_control_time"]
    if previous_control_time is None:
        control_dt = 1.0 / 30.0
    else:
        control_dt = max(1.0 / 120.0, frame_now - previous_control_time)
        instantaneous_fps = 1.0 / control_dt
        if _state["control_fps"] <= 0.0:
            _state["control_fps"] = instantaneous_fps
        else:
            _state["control_fps"] = (
                0.80 * _state["control_fps"] + 0.20 * instantaneous_fps
            )
    _state["last_control_time"] = frame_now
    control_fps = float(_state["control_fps"])
    telemetry = {
        "raw_model_steering": None,
        "filtered_model_steering": None,
        "model_instability": 0.0,
        "curve_demand": None, "target_throttle": None,
        "control_fps": control_fps,
        "safety_stop": False,
    }

    try:
        if mode == MODE_AUTO:
            raw_steering = float(backend.predict_frame(raw_frame))
            max_throttle = float(auto_max_throttle.value)
            min_throttle = min(float(auto_min_throttle.value), max_throttle)
            drive_controller.set_throttle_limits(min_throttle, max_throttle)
            command = drive_controller.update(
                raw_steering,
                dt=control_dt,
            )
            low_control_fps = (
                control_fps > 0.0 and control_fps < LOW_CONTROL_FPS
            )
            if command.instability >= MODEL_INSTABILITY_STOP:
                _state["instability_count"] += 1
                _state["stability_count"] = 0
            elif command.instability <= MODEL_INSTABILITY_RESUME:
                _state["instability_count"] = max(
                    0, _state["instability_count"] - 2
                )
                _state["stability_count"] += 1
            else:
                _state["instability_count"] = max(
                    0, _state["instability_count"] - 1
                )
                _state["stability_count"] = 0
            if (
                not _state["safety_stop"]
                and _state["instability_count"]
                >= MODEL_INSTABILITY_STOP_FRAMES
            ):
                _state["safety_stop"] = True
            elif (
                _state["safety_stop"]
                and _state["stability_count"]
                >= MODEL_INSTABILITY_RESUME_FRAMES
            ):
                _state["safety_stop"] = False
                _state["instability_count"] = 0
            safety_stop = bool(_state["safety_stop"])
            if safety_stop:
                applied_throttle = 0.0
                steering_request = 0.0
            else:
                applied_throttle = command.throttle
                steering_request = command.steering
                if low_control_fps:
                    applied_throttle = min(applied_throttle, min_throttle)

            steering, throttle = _set_car(steering_request, applied_throttle)
            control_applied_time = time.monotonic()
            display_frame = _draw_common_overlay(
                raw_frame, mode, steering, throttle
            )
            cv2.putText(
                display_frame,
                "Model {:+.2f}  Applied {:+.2f}  FPS {:.1f}".format(
                    raw_steering, command.steering, control_fps,
                ),
                (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1,
            )
            cv2.putText(
                display_frame,
                "Curve {:.2f}  Unstable {:.2f}  Target thr {:.2f}".format(
                    command.curve_demand, command.instability,
                    command.target_throttle,
                ),
                (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1,
            )
            if safety_stop:
                cv2.putText(
                    display_frame, "MODEL UNSTABLE - THROTTLE STOPPED", (8, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 255), 1,
                )
            elif low_control_fps:
                cv2.putText(
                    display_frame, "LOW CONTROL FPS - SPEED LIMITED", (8, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1,
                )
            telemetry.update({
                "raw_model_steering": raw_steering,
                "filtered_model_steering": command.steering,
                "model_instability": command.instability,
                "curve_demand": command.curve_demand,
                "target_throttle": command.target_throttle,
                "safety_stop": safety_stop,
            })
        elif mode == MODE_MANUAL:
            steering = float(car.steering)
            throttle = float(car.throttle)
            control_applied_time = time.monotonic()
            display_frame = _draw_common_overlay(
                raw_frame, mode, steering, throttle
            )
        else:
            steering, throttle = _set_car(0.0, 0.0)
            control_applied_time = time.monotonic()
            display_frame = _draw_common_overlay(
                raw_frame, mode, steering, throttle
            )

        telemetry["steering"] = steering
        telemetry["throttle"] = throttle
        _state["telemetry"] = telemetry
        _state["last_display_frame"] = display_frame

        try:
            _record_frame_if_due(
                raw_frame, display_frame, frame_id, frame_now,
                frame_timestamp, control_applied_time,
            )
        except Exception as record_error:
            _stop_recording_internal("Lỗi recorder: {}".format(record_error))

        now = time.monotonic()
        if now - _state["last_preview_time"] >= 1.0 / PREVIEW_FPS:
            success, jpeg = cv2.imencode(
                ".jpg", display_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY],
            )
            if success:
                camera_view.value = jpeg.tobytes()
                _state["last_preview_time"] = now
            motor_value = throttle * float(car.throttle_gain)
            instability_text = (
                _fmt(telemetry.get("model_instability"), 2) or "--"
            )
            telemetry_status.value = (
                "<b>Steering:</b> {:+.3f} &nbsp; "
                "<b>Throttle:</b> {:+.3f} &nbsp; "
                "<b>Motor:</b> {:+.3f} &nbsp; "
                "<b>Model unstable:</b> {} &nbsp; <b>FPS:</b> {:.1f}".format(
                    steering, throttle, motor_value,
                    instability_text,
                    control_fps,
                )
            )
    except Exception as error:
        _set_car(0.0, 0.0)
        drive_controller.reset(initial_throttle=0.0)
        _state["mode"] = MODE_STOP
        _update_mode_status("Lỗi inference; xe đã dừng")
        camera_status.value = (
            "<b style='color:red'>❌ Lỗi camera/inference:</b> {}".format(error)
        )
        _log("Lỗi callback: {}".format(error))
        with event_output:
            traceback.print_exc()


# ------------------------------------------------------------
# Shutdown and callbacks
# ------------------------------------------------------------
def shutdown_dashboard(button=None, release_camera=True):
    _stop_recording_internal("Dashboard shutdown")
    _state["mode"] = MODE_STOP
    _state["shutdown"] = True
    _set_car(0.0, 0.0)
    drive_controller.reset(initial_throttle=0.0)
    _unbind_gamepad()
    try:
        camera.unobserve(_on_camera_frame, names="value")
    except Exception:
        pass
    try:
        camera.running = False
    except Exception:
        pass
    if release_camera:
        try:
            camera.cap.release()
        except Exception:
            pass
    for button_widget in (
        btn_stop, btn_manual, btn_auto, btn_record_start, btn_record_stop,
        btn_snapshot, btn_gamepad_connect, btn_shutdown,
    ):
        button_widget.disabled = True
    _update_mode_status("dashboard đã tắt")


btn_gamepad_connect.on_click(_connect_gamepad)
gamepad.observe(_on_gamepad_connection, names="connected")
manual_max_throttle.observe(_apply_manual_axes, names="value")
manual_steering_scale.observe(_apply_manual_axes, names="value")
btn_stop.on_click(_on_stop)
btn_manual.on_click(_on_manual)
btn_auto.on_click(_on_auto)
btn_record_start.on_click(_start_recording)
btn_record_stop.on_click(_stop_recording)
btn_snapshot.on_click(_save_snapshot)
btn_shutdown.on_click(shutdown_dashboard)


# ------------------------------------------------------------
# Final dashboard layout: this is the only display cell.
# ------------------------------------------------------------
manual_panel = widgets.VBox([
    widgets.HTML(
        "<h3>🎮 Điều khiển tay cầm</h3>"
        "<p>Hiển thị tay cầm, xoay hai cần, bấm <b>Kích hoạt</b>, "
        "sau đó chọn <b>MANUAL</b>.</p>"
    ),
    gamepad, gamepad_status, btn_gamepad_connect,
    manual_max_throttle, manual_steering_scale,
])
auto_panel = widgets.VBox([
    widgets.HTML(
        "<h3>🤖 Tự lái</h3>"
        "<p>Chỉ model ONNX + adaptive throttle, không dùng geometric. "
        "Auto bắt đầu với max throttle = 0 để xe chưa chạy.</p>"
    ),
    auto_max_throttle, auto_min_throttle,
    widgets.HTML(
        "<p><b>Safety:</b> chỉ cắt ga khi model dao động rất mạnh {} frame; "
        "tự chạy lại sau {} frame ổn định; "
        "control dưới {:.0f} FPS sẽ khóa ga ở mức tối thiểu.</p>".format(
            MODEL_INSTABILITY_STOP_FRAMES,
            MODEL_INSTABILITY_RESUME_FRAMES,
            LOW_CONTROL_FPS,
        )
    ),
])
record_panel = widgets.VBox([
    widgets.HTML(
        "<h3>💾 Record</h3>"
        "<p><b>Dataset</b> lưu frame raw + labels.csv và chỉ hoạt động ở "
        "MANUAL. AUTO dùng video/log để validation.</p>"
    ),
    record_dataset, record_video, record_log, record_fps,
    widgets.HBox([btn_record_start, btn_record_stop, btn_snapshot]),
    record_status,
])
safety_panel = widgets.VBox([
    widgets.HTML(
        "<h3>🛡️ An toàn và phiên làm việc</h3>"
        "<p>Dừng khẩn cấp chỉ đưa xe về STOP. Nút tắt dashboard còn "
        "gỡ callback và giải phóng camera để notebook khác có thể dùng.</p>"
    ),
    btn_shutdown,
    widgets.HTML(
        "<p><b>Dataset:</b> {}</p><p><b>Records:</b> {}</p>".format(
            DATASET_ROOT, RECORDS_ROOT
        )
    ),
])

tabs = widgets.Tab(children=[manual_panel, auto_panel, record_panel, safety_panel])
tabs.set_title(0, "Manual")
tabs.set_title(1, "Auto")
tabs.set_title(2, "Record")
tabs.set_title(3, "Safety")

camera_panel = widgets.VBox([
    widgets.HTML("<h3>📷 Camera live</h3>"),
    camera_view, camera_status, telemetry_status,
])
dashboard = widgets.VBox([
    widgets.HTML(
        "<h2>JetRacer Live Control Center</h2>"
        "<p><b>{}</b> — model-only, không dừng theo lane/ô xanh.</p>"
        .format(LIVE_BUILD)
    ),
    mode_status,
    widgets.HBox([btn_stop, btn_manual, btn_auto]),
    widgets.HBox([camera_panel, tabs]),
    widgets.HTML("<h3>📋 Event log</h3>"),
    event_output,
])

_state["shutdown"] = False
_set_mode(MODE_STOP, "khởi động an toàn")
camera.observe(_on_camera_frame, names="value")
camera.running = True
camera_status.value = (
    "<b style='color:green'>📷 Camera đang hoạt động</b> — "
    "preview khoảng {} FPS".format(int(PREVIEW_FPS))
)
display(dashboard)
_log("Dashboard đã sẵn sàng; xe đang STOP")
'''


def main():
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(INTRO),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(HARDWARE),
        nbf.v4.new_code_cell(MODEL),
        nbf.v4.new_code_cell(DASHBOARD),
    ]
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.6"},
    }
    nbf.write(notebook, str(OUTPUT_PATH))
    print("Generated {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
