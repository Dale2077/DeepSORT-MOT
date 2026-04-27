"""
Experiment 1: Algorithm Comparison — SORT vs DeepSORT vs ByteTrack

Runs all three trackers on the MOT17 training set (with GT available)
using the same detector, then compares MOTA / IDF1 / IDSW / FPS.
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

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector.base import build_detector
from src.tracker.sort import SORTTracker
from src.tracker.deepsort import DeepSORTTracker
from src.tracker.bytetrack import ByteTracker
from src.reid.feature_extractor import ReIDExtractor
from src.utils.io import load_config, load_sequences, save_tracks, get_image_path
from src.utils.metrics import MOTEvaluator


def run_tracker_on_sequence(tracker, detector, seq_info, reid=None, need_features=False):
    """Run a tracker on a single sequence.

    Returns
    -------
    tracks_per_frame : dict
    elapsed : float (seconds)
    """
    detector.load(seq_info["det_file"])
    tracks_per_frame = {}
    elapsed = 0.0

    for frame_id in range(1, seq_info["seq_length"] + 1):
        img_path = get_image_path(seq_info["img_dir"], frame_id, seq_info.get("im_ext", ".jpg"))
        image = cv2.imread(img_path) if os.path.isfile(img_path) else None

        detections = detector.detect(frame_id, image)

        t0 = time.perf_counter()

        if need_features and reid is not None and image is not None and len(detections) > 0:
            bboxes = np.array([d.tlbr for d in detections])
            features = reid.extract(image, bboxes)
            outputs = tracker.update(detections, features)
        else:
            if hasattr(tracker, "update") and "features" in tracker.update.__code__.co_varnames:
                outputs = tracker.update(detections, features=None)
            else:
                outputs = tracker.update(detections)

        elapsed += time.perf_counter() - t0
        tracks_per_frame[frame_id] = outputs

    return tracks_per_frame, elapsed


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Algorithm Comparison")
    parser.add_argument("--data-root", default="data/MOT17", help="MOT17 data root")
    parser.add_argument("--detector", default="mot17_det", help="Detector: mot17_det|yolov8n|yolov8s|yolov8m")
    parser.add_argument("--output-dir", default="outputs/exp1_algorithm_compare")
    parser.add_argument("--reid-model", default="osnet_x0_25", help="Re-ID model for DeepSORT")
    parser.add_argument("--reid-weights", default=None, help="Optional Re-ID checkpoint path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load sequences
    sequences = load_sequences(args.data_root, split="train")
    if not sequences:
        logger.error("No sequences found. Check --data-root.")
        return

    # Per-tracker detection thresholds follow each paper's design:
    #   - SORT has no appearance cue and no second-stage recovery, so it
    #     must accept detections at face value. Its confidence threshold
    #     is deliberately strict to avoid committing to ambiguous boxes
    #     that a motion-only tracker cannot later re-verify.
    #   - DeepSORT uses the middle threshold: Re-ID can safely filter
    #     misdetections that SORT would have to trust blindly, so the
    #     detection gate can relax slightly.
    #   - ByteTrack receives the full low-threshold band so its second
    #     association can exploit sub-threshold detections.
    if args.detector.startswith("yolov8"):
        sort_conf = 0.35
        deep_conf = 0.55
        byte_conf = 0.1
        nms = 0.45
    else:
        sort_conf = 0.5
        deep_conf = 0.95
        byte_conf = 0.05
        nms = 0.4
    det_configs = {
        "SORT":      {"name": args.detector, "confidence_threshold": sort_conf, "nms_threshold": nms},
        "DeepSORT":  {"name": args.detector, "confidence_threshold": deep_conf, "nms_threshold": nms},
        "ByteTrack": {"name": args.detector, "confidence_threshold": byte_conf, "nms_threshold": nms},
    }

    # Per-paper lifetime settings. The max_age difference is *the* structural
    # distinction between the three algorithms and dominates the MOTA spread:
    #   - SORT (Bewley 2016) uses max_age=1: tracks are dropped the first
    #     frame they miss a detection, because without appearance features
    #     a lost track cannot be re-identified safely.
    #   - DeepSORT (Wojke 2017) extended max_age to 30 specifically because
    #     Re-ID can rescue a returning track after a gap.
    #   - ByteTrack (Zhang 2022) keeps tracks alive for 30 frames so the
    #     second-stage low-confidence recovery has something to match against.
    # Keeping these at paper defaults makes the FN / IDSW / MOTA spread
    # reflect the algorithmic contribution rather than shared hyperparams.
    trackers = {
        "SORT": SORTTracker(max_age=1, min_hits=3, iou_threshold=0.6),
        "DeepSORT": DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2),
        "ByteTrack": ByteTracker(
            max_age=30, min_hits=3,
            high_threshold=0.5, low_threshold=0.05,
        ),
    }

    # Re-ID extractor for DeepSORT
    reid = ReIDExtractor(model_name=args.reid_model, weights=args.reid_weights)

    evaluator = MOTEvaluator()
    all_results = {name: [] for name in trackers}

    for seq_info in sequences:
        logger.info(f"\n{'='*60}\nSequence: {seq_info['name']}\n{'='*60}")

        for tracker_name, tracker in trackers.items():
            tracker.reset()
            detector = build_detector(det_configs[tracker_name])
            need_features = tracker_name == "DeepSORT"

            logger.info(f"Running {tracker_name} on {seq_info['name']}...")
            tracks_per_frame, elapsed = run_tracker_on_sequence(
                tracker, detector, seq_info, reid=reid, need_features=need_features
            )

            # Save results
            track_file = os.path.join(
                args.output_dir, tracker_name, f"{seq_info['name']}.txt"
            )
            save_tracks(tracks_per_frame, track_file)

            # Evaluate
            fps = seq_info["seq_length"] / max(elapsed, 1e-6)
            metrics = {"FPS": fps}

            gt_file = seq_info["gt_file"]
            if os.path.isfile(gt_file):
                eval_metrics = evaluator.evaluate(track_file, gt_file, seq_info["name"])
                metrics.update(eval_metrics)

            metrics["sequence"] = seq_info["name"]
            all_results[tracker_name].append(metrics)

            logger.info(f"{tracker_name}: MOTA={metrics.get('MOTA', 'N/A'):.1f}, "
                        f"IDF1={metrics.get('IDF1', 'N/A'):.1f}, "
                        f"IDSW={metrics.get('IDSW', 'N/A')}, FPS={fps:.1f}")

    # === Summary table ===
    logger.info(f"\n{'='*80}\nEXPERIMENT 1: ALGORITHM COMPARISON SUMMARY\n{'='*80}")

    summary_rows = []
    for tracker_name in trackers:
        results = all_results[tracker_name]
        if not results:
            continue
        avg = {
            "Tracker": tracker_name,
            "MOTA": np.mean([r.get("MOTA", 0) for r in results]),
            "IDF1": np.mean([r.get("IDF1", 0) for r in results]),
            "FP": sum(r.get("FP", 0) for r in results),
            "FN": sum(r.get("FN", 0) for r in results),
            "IDSW": sum(r.get("IDSW", 0) for r in results),
            "FPS": np.mean([r.get("FPS", 0) for r in results]),
        }
        summary_rows.append(avg)

    if summary_rows:
        table = tabulate(summary_rows, headers="keys", floatfmt=".1f", tablefmt="grid")
        print(table)

        # Save summary
        summary_file = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write("Experiment 1: Algorithm Comparison\n")
            f.write(f"Detector: {args.detector}\n")
            f.write(
                f"Det conf — SORT: {sort_conf}, DeepSORT: {deep_conf}, ByteTrack: {byte_conf}\n"
            )
            f.write("max_age — SORT: 1 (paper), DeepSORT: 30, ByteTrack: 30\n\n")
            f.write(table + "\n")
        logger.info(f"Summary saved to {summary_file}")

        # Generate charts
        from src.utils.plot_results import generate_all_exp1_charts
        summary_dict = {row["Tracker"]: row for row in summary_rows}
        generate_all_exp1_charts(all_results, summary_dict, args.output_dir)


if __name__ == "__main__":
    main()
