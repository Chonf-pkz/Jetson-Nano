import ast
import csv
import json
import tempfile
import threading
import time
import traceback
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import ipywidgets.widgets as widgets
import numpy as np
import traitlets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "live_inference.ipynb"


class LiveInferenceNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.cells
            if cell["cell_type"] == "code"
        ]
        cls.output_cell = cls.code_cells[-1]
        cls.model_cell = cls.code_cells[-2]

    def test_setup_cells_precede_single_output_dashboard(self):
        self.assertEqual(len(self.cells), 5)
        self.assertEqual(
            [cell["cell_type"] for cell in self.cells],
            ["markdown", "code", "code", "code", "code"],
        )
        for setup_source in self.code_cells[:-1]:
            self.assertNotIn("display(", setup_source)
        self.assertEqual(self.output_cell.count("display(dashboard)"), 1)

    def test_setup_uses_model_only_controller(self):
        setup_cell = self.code_cells[0]
        self.assertIn('LIVE_BUILD = "MODEL_ONLY_NO_LANE_GATE_V4"', setup_cell)
        self.assertIn("from src.inference_jetson import ONNXBackend", setup_cell)
        self.assertIn("from src.adaptive_controller import AdaptiveDriveController", setup_cell)
        self.assertIn("from src.jetracer_fallback import NvidiaRacecar", setup_cell)
        self.assertNotIn("lane_geometry", setup_cell)
        self.assertNotIn("PurePursuit", "\n".join(self.code_cells))

    def test_all_code_cells_parse_and_have_no_saved_outputs(self):
        for index, source in enumerate(self.code_cells):
            ast.parse(source, filename="live_inference_cell_{}".format(index))
        for cell in self.cells:
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [])

    def test_dashboard_has_exclusive_modes_and_safe_start(self):
        self.assertIn('MODE_STOP = "STOP"', self.output_cell)
        self.assertIn('MODE_MANUAL = "MANUAL"', self.output_cell)
        self.assertIn('MODE_AUTO = "AUTO"', self.output_cell)
        self.assertIn('value=0.0, min=0.0, max=1.00', self.output_cell)
        self.assertIn('INITIAL_THROTTLE_GAIN = 1.00', self.code_cells[1])
        self.assertIn('_set_mode(MODE_STOP, "khởi động an toàn")', self.output_cell)
        self.assertEqual(
            self.output_cell.count(
                'camera.observe(_on_camera_frame, names="value")'
            ),
            1,
        )

    def test_training_dataset_is_manual_and_uses_raw_frames(self):
        self.assertIn(
            'if save_dataset and _state["mode"] != MODE_MANUAL:',
            self.output_cell,
        )
        self.assertIn(
            'if save_dataset and not _state["gamepad_connected"]:',
            self.output_cell,
        )
        self.assertIn('"control_mode"', self.output_cell)
        self.assertIn('str(image_path), raw_frame,', self.output_cell)
        self.assertIn('video_writer.write(sync_frame)', self.output_cell)
        self.assertIn('target=_record_worker', self.output_cell)

    def test_gamepad_disconnect_stops_manual_mode(self):
        self.assertIn(
            'gamepad.observe(_on_gamepad_connection, names="connected")',
            self.output_cell,
        )
        self.assertIn(
            '_set_mode(MODE_STOP, "mất kết nối tay cầm")',
            self.output_cell,
        )

    def test_auto_control_is_model_only_and_rate_limited(self):
        self.assertIn("max_steering=0.85", self.model_cell)
        self.assertIn("steering_gain=1.10", self.model_cell)
        self.assertIn("curve_full_scale=0.65", self.model_cell)
        self.assertIn("backend.predict_frame(raw_frame)", self.output_cell)
        self.assertIn("drive_controller.update(\n                raw_steering", self.output_cell)
        self.assertIn("MODEL_INSTABILITY_STOP = 0.95", self.output_cell)
        self.assertIn("MODEL_INSTABILITY_STOP_FRAMES = 8", self.output_cell)
        self.assertIn("MODEL_INSTABILITY_RESUME_FRAMES = 10", self.output_cell)
        self.assertIn("MODEL UNSTABLE - THROTTLE STOPPED", self.output_cell)
        self.assertIn("LOW_CONTROL_FPS = 5.0", self.output_cell)
        combined = "\n".join(self.code_cells)
        self.assertIn("MODEL ONLY / NO LANE GATE", self.output_cell)
        self.assertNotIn("lane_estimator", combined)
        self.assertNotIn("pure_pursuit", combined)
        self.assertNotIn("geometry_steering", combined)

    def test_camera_and_recorder_preserve_full_resolution(self):
        hardware_cell = self.code_cells[1]
        self.assertIn("CAMERA_WIDTH = 640", hardware_cell)
        self.assertIn("CAMERA_HEIGHT = 360", hardware_cell)
        self.assertIn("_lock_camera_controls_best_effort", hardware_cell)
        self.assertIn("frame_height, frame_width = frame.shape[:2]", self.output_cell)
        self.assertNotIn("fourcc, fps, (224, 224)", self.output_cell)
        self.assertNotIn(
            'while _record["video_frame_index"] < target_video_frames',
            self.output_cell,
        )

    def test_model_only_diagnostics_are_recorded(self):
        for field in (
            '"raw_model_steering"',
            '"filtered_model_steering"',
            '"model_instability"',
            '"curve_demand"',
            '"safety_stop"',
        ):
            self.assertIn(field, self.output_cell)

    def test_dashboard_smoke_records_raw_manual_frame(self):
        class FakeCar:
            steering = 0.0
            throttle = 0.0
            steering_gain = -1.0
            throttle_gain = 0.9

        class FakeCapture:
            def release(self):
                return None

        class FakeCamera(traitlets.HasTraits):
            value = traitlets.Any(allow_none=True)
            running = traitlets.Bool(default_value=False)

            def __init__(self, frame):
                super().__init__()
                self.value = frame
                self.cap = FakeCapture()

        class FakeDriveController:
            def reset(self, initial_throttle=0.0):
                return None

        frame = np.full((224, 224, 3), 80, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            dataset_root = temporary_root / "dataset"
            records_root = temporary_root / "records"
            snapshots_root = records_root / "snapshots"
            dataset_root.mkdir(parents=True)
            snapshots_root.mkdir(parents=True)

            namespace = {
                "__name__": "live_notebook_smoke",
                "threading": threading,
                "time": time,
                "traceback": traceback,
                "csv": csv,
                "queue": __import__("queue"),
                "cv2": cv2,
                "np": np,
                "widgets": widgets,
                "datetime": datetime,
                "car": FakeCar(),
                "camera": FakeCamera(frame),
                "drive_controller": FakeDriveController(),
                "CAMERA_WIDTH": 224,
                "CAMERA_HEIGHT": 224,
                "LIVE_BUILD": "MODEL_ONLY_NO_LANE_GATE_V4",
                "DATASET_ROOT": dataset_root,
                "RECORDS_ROOT": records_root,
                "SNAPSHOTS_ROOT": snapshots_root,
                "display": lambda value: None,
            }

            exec(compile(self.output_cell, "live_output", "exec"), namespace)
            try:
                self.assertEqual(namespace["_state"]["mode"], "STOP")
                self.assertEqual(namespace["car"].throttle, 0.0)

                namespace["_state"]["gamepad_connected"] = True
                namespace["_set_mode"]("MANUAL")
                namespace["car"].steering = 0.45
                namespace["car"].throttle = 0.25
                namespace["record_video"].value = True
                namespace["_start_recording"]()
                self.assertTrue(namespace["_record"]["active"])

                namespace["_on_camera_frame"]({"new": frame.copy()})
                namespace["_stop_recording"]()

                sessions = list(dataset_root.glob("session_*"))
                self.assertEqual(len(sessions), 1)
                with (sessions[0] / "labels.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["control_mode"], "MANUAL")
                self.assertEqual(rows[0]["frame_id"], "1")
                self.assertEqual(rows[0]["video_frame_index"], "0")
                self.assertEqual(rows[0]["record_dropped_frames"], "0")
                self.assertGreaterEqual(
                    float(rows[0]["control_latency_seconds"]), 0.0
                )
                self.assertAlmostEqual(
                    float(rows[0]["steering_normalized"]), 0.45
                )

                video_path = sessions[0] / "validation_overlay.avi"
                self.assertTrue(video_path.is_file())
                video = cv2.VideoCapture(str(video_path))
                try:
                    ok, video_frame = video.read()
                    self.assertTrue(ok)
                    self.assertIsNotNone(video_frame)
                finally:
                    video.release()

                saved_frame = cv2.imread(
                    str(sessions[0] / rows[0]["image_file"])
                )
                self.assertIsNotNone(saved_frame)
                self.assertAlmostEqual(float(saved_frame.mean()), 80.0, delta=2.0)
            finally:
                namespace["shutdown_dashboard"](release_camera=False)


if __name__ == "__main__":
    unittest.main()
