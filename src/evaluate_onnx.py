"""Dependency-light evaluation of the exact ONNX model deployed to Jetson."""

import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.preprocessing_config import (
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ROAD_CROP_TOP_FRACTION,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rolling(values, window, reducer):
    radius = window // 2
    output = []
    for index in range(len(values)):
        part = values[max(0, index - radius):index + radius + 1]
        output.append(reducer(part))
    return np.asarray(output, dtype=np.float32)


def smooth_labels(values):
    median = _rolling(values, 5, np.median)
    return np.clip(_rolling(median, 3, np.mean), -1.0, 1.0)


def load_entries(dataset_dir, label_lookahead_frames=0):
    sessions = {}
    for session_dir in sorted(glob.glob(os.path.join(dataset_dir, 'session_*'))):
        with open(os.path.join(session_dir, 'labels.csv'), newline='') as handle:
            rows = list(csv.DictReader(handle))
        raw = np.asarray(
            [float(row['steering_normalized']) for row in rows],
            dtype=np.float32,
        )
        cleaned = smooth_labels(raw)
        entries = []
        lookahead = max(0, int(label_lookahead_frames))
        for position, row in enumerate(rows):
            if abs(float(row.get('throttle_normalized') or 0.0)) < 0.05:
                continue
            target_position = position + lookahead
            if target_position >= len(cleaned):
                continue
            path = os.path.join(
                session_dir, row['image_file'].replace('/', os.sep)
            )
            if os.path.isfile(path):
                entries.append({
                    'path': path,
                    'steering': float(cleaned[target_position]),
                    'sample_index': int(row['sample_index']),
                })
        sessions[os.path.basename(session_dir)] = entries
    return sessions


def choose_validation_session(sessions, seed=42, ratio=0.2):
    names = sorted(sessions)
    rng = np.random.RandomState(seed)
    rng.shuffle(names)
    target = max(1, int(round(sum(map(len, sessions.values())) * ratio)))
    selected = []
    count = 0
    for name in names:
        if len(selected) >= len(names) - 1:
            break
        candidate = count + len(sessions[name])
        if not selected or abs(target - candidate) < abs(target - count):
            selected.append(name)
            count = candidate
    return selected


def preprocess(path):
    with Image.open(path) as source:
        image = source.convert('RGB')
        top = int(round(image.height * ROAD_CROP_TOP_FRACTION))
        road = image.crop((0, top, image.width, image.height))
        resampling = getattr(Image, 'Resampling', Image)
        image = np.asarray(
            road.resize(
                (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
                resample=resampling.BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0
    image -= np.asarray(NORMALIZE_MEAN, dtype=np.float32)
    image /= np.asarray(NORMALIZE_STD, dtype=np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])


def compute_metrics(predictions, targets):
    error = predictions - targets
    metrics = {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'maximum_error': float(np.max(np.abs(error))),
        'prediction_mean': float(predictions.mean()),
        'prediction_std': float(predictions.std()),
        'target_mean': float(targets.mean()),
        'target_std': float(targets.std()),
        'temporal_prediction_delta_mean': float(
            np.mean(np.abs(np.diff(predictions)))
        ),
    }
    masks = {
        'left': targets < -0.12,
        'straight': np.abs(targets) <= 0.12,
        'right': targets > 0.12,
        'hard_turn': np.abs(targets) >= 0.55,
    }
    for name, mask in masks.items():
        metrics[name + '_count'] = int(mask.sum())
        metrics[name + '_mae'] = (
            float(np.mean(np.abs(error[mask]))) if mask.any() else None
        )
    turning = np.abs(targets) > 0.12
    metrics['wrong_turn_sign_rate'] = float(np.mean(
        np.sign(predictions[turning]) != np.sign(targets[turning])
    ))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', default=os.path.join(
            PROJECT_ROOT, 'checkpoints', 'lane_tracker_ir8_opset13.onnx'
        )
    )
    parser.add_argument(
        '--dataset-dir', default=os.path.join(
            PROJECT_ROOT, 'dataset', 'track_lane_dataset', 'dataset_steering'
        )
    )
    parser.add_argument(
        '--output', default=os.path.join(
            PROJECT_ROOT, 'eval_results', 'onnx_metrics.json'
        )
    )
    parser.add_argument('--label-lookahead-frames', type=int, default=0)
    args = parser.parse_args()

    sessions = load_entries(
        args.dataset_dir,
        label_lookahead_frames=args.label_lookahead_frames,
    )
    validation_sessions = choose_validation_session(sessions)
    entries = [entry for name in validation_sessions for entry in sessions[name]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        args.model, sess_options=options, providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    predictions = []
    targets = []
    rows = []
    for entry in entries:
        prediction = float(session.run(
            None, {input_name: preprocess(entry['path'])}
        )[0].item())
        predictions.append(prediction)
        targets.append(entry['steering'])
        rows.append((entry['sample_index'], prediction, entry['steering']))

    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    report = {
        'backend': 'onnxruntime',
        'model': os.path.abspath(args.model),
        'validation_sessions': validation_sessions,
        'samples': len(entries),
        'label_lookahead_frames': int(args.label_lookahead_frames),
        'metrics': compute_metrics(predictions, targets),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(report, handle, indent=2)
    predictions_path = os.path.splitext(args.output)[0] + '_predictions.csv'
    with open(predictions_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_index', 'prediction', 'target'])
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
