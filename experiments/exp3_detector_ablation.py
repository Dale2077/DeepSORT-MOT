"""
Experiment 3: Detector Ablation

Compares the impact of detection quality on tracking performance by
running DeepSORT with different detectors:
  - MOT17 public detections (SDP)
  - YOLOv8n (nano)
  - YOLOv8s (small)
  - YOLOv8m (medium)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tabulate import tabulate

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector.base import build_detector
from src.tracker.deepsort import DeepSORTTracker
from src.reid.feature_extractor import ReIDExtractor
from src.utils.io import load_sequences, save_tracks, get_image_path
from src.utils.metrics import MOTEvaluator


def run_sequence(tracker, detector, seq_info, reid):
    """Run DeepSORT on a single sequence with a given detector."""
    detector.load(seq_info["det_file"])
    tracks_per_frame = {}
    det_elapsed = 0.0
    track_elapsed = 0.0

    for frame_id in range(1, seq_info["seq_length"] + 1):
        img_path = get_image_path(seq_info["img_dir"], frame_id, seq_info.get("im_ext", ".jpg"))
        image = cv2.imread(img_path) if os.path.isfile(img_path) else None

        t0 = time.perf_counter()
        detections = detector.detect(frame_id, image)
        det_elapsed += time.perf_counter() - t0

        t0 = time.perf_counter()
        features = None
        if reid is not None and image is not None and len(detections) > 0:
            bboxes = np.array([d.tlbr for d in detections])
            features = reid.extract(image, bboxes)

        outputs = tracker.update(detections, features=features)
        track_elapsed += time.perf_counter() - t0
        tracks_per_frame[frame_id] = outputs

    total_elapsed = det_elapsed + track_elapsed
    return tracks_per_frame, total_elapsed, det_elapsed, track_elapsed


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Detector Ablation")
    parser.add_argument("--data-root", default="data/MOT17")
    parser.add_argument("--output-dir", default="outputs/exp3_detector_ablation")
    parser.add_argument("--reid-model", default="osnet_x0_25")
    parser.add_argument("--reid-weights", default=None, help="Optional Re-ID checkpoint path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = load_sequences(args.data_root, split="train")
    if not sequences:
        logger.error("No sequences found.")
        return

    reid = ReIDExtractor(model_name=args.reid_model, weights=args.reid_weights)
    evaluator = MOTEvaluator()

    # Detectors to compare. YOLOv8 uses Ultralytics' default conf=0.25 rather
    # than 0.5 — at 0.5 ~7k extra pedestrians are dropped on MOT17-train and
    # the tracker is unfairly penalised with inflated FN.
    detector_configs = [
        {"name": "mot17_det", "confidence_threshold": 0.5, "nms_threshold": 0.4, "label": "MOT17-DET"},
        {"name": "yolov8n", "confidence_threshold": 0.25, "nms_threshold": 0.45, "label": "YOLOv8-Nano"},
        {"name": "yolov8s", "confidence_threshold": 0.25, "nms_threshold": 0.45, "label": "YOLOv8-Small"},
        {"name": "yolov8m", "confidence_threshold": 0.25, "nms_threshold": 0.45, "label": "YOLOv8-Medium"},
    ]

    all_results = {}

    for det_cfg in detector_configs:
        det_label = det_cfg.pop("label")
        logger.info(f"\n{'='*60}\nDetector: {det_label}\n{'='*60}")

        seq_metrics = []
        for seq_info in sequences:
            tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2)

            try:
                detector = build_detector(det_cfg)
            except Exception as e:
                logger.warning(f"Failed to build detector {det_cfg['name']}: {e}")
                break

            try:
                tracks, total_t, det_t, track_t = run_sequence(
                    tracker, detector, seq_info, reid
                )
            except Exception as e:
                logger.warning(f"Failed on {seq_info['name']}: {e}")
                continue

            track_file = os.path.join(args.output_dir, det_label.replace(" ", "_"), f"{seq_info['name']}.txt")
            save_tracks(tracks, track_file)

            fps = seq_info["seq_length"] / max(total_t, 1e-6)
            metrics = {"FPS": fps, "Det_Time": det_t, "Track_Time": track_t}

            if os.path.isfile(seq_info["gt_file"]):
                eval_metrics = evaluator.evaluate(track_file, seq_info["gt_file"], seq_info["name"])
                metrics.update(eval_metrics)

            seq_metrics.append(metrics)
            logger.info(f"  {seq_info['name']}: MOTA={metrics.get('MOTA', 'N/A'):.1f}, FPS={fps:.1f}")

        det_cfg["label"] = det_label  # Restore

        if seq_metrics:
            avg = {
                "Detector": det_label,
                "MOTA": np.mean([m.get("MOTA", 0) for m in seq_metrics]),
                "MOTP": np.mean([m.get("MOTP", 0) for m in seq_metrics]),
                "IDF1": np.mean([m.get("IDF1", 0) for m in seq_metrics]),
                "FP": sum(m.get("FP", 0) for m in seq_metrics),
                "FN": sum(m.get("FN", 0) for m in seq_metrics),
                "IDSW": sum(m.get("IDSW", 0) for m in seq_metrics),
                "FPS": np.mean([m.get("FPS", 0) for m in seq_metrics]),
            }
            all_results[det_label] = avg

    # === Summary ===
    logger.info(f"\n{'='*80}\nEXPERIMENT 3: DETECTOR ABLATION SUMMARY\n{'='*80}")

    if all_results:
        rows = list(all_results.values())
        table = tabulate(rows, headers="keys", floatfmt=".1f", tablefmt="grid")
        print(table)

        summary_file = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write("Experiment 3: Detector Ablation (with DeepSORT tracker)\n\n")
            f.write(table + "\n")
        logger.info(f"Summary saved to {summary_file}")

        # Generate charts
        from src.utils.plot_results import generate_all_exp3_charts
        generate_all_exp3_charts(all_results, args.output_dir)


if __name__ == "__main__":
    main()
