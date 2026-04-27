"""
Experiment 2: DeepSORT Parameter Ablation

Studies the impact of key DeepSORT parameters:
  A) max_age: {10, 30, 50, 70}
  B) Re-ID on/off (with vs without appearance features)
  C) max_cosine_distance: {0.1, 0.2, 0.3, 0.4}
  D) nn_budget: {20, 50, 100, None}
"""

import os
import sys
import time
import argparse
import copy
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


def run_deepsort(tracker, detector, seq_info, reid=None, use_reid=True):
    """Run DeepSORT on a single sequence."""
    detector.load(seq_info["det_file"])
    tracks_per_frame = {}
    elapsed = 0.0

    for frame_id in range(1, seq_info["seq_length"] + 1):
        img_path = get_image_path(seq_info["img_dir"], frame_id, seq_info.get("im_ext", ".jpg"))
        image = cv2.imread(img_path) if os.path.isfile(img_path) else None

        detections = detector.detect(frame_id, image)

        t0 = time.perf_counter()
        features = None
        if use_reid and reid is not None and image is not None and len(detections) > 0:
            bboxes = np.array([d.tlbr for d in detections])
            features = reid.extract(image, bboxes)

        outputs = tracker.update(detections, features=features)
        elapsed += time.perf_counter() - t0
        tracks_per_frame[frame_id] = outputs

    return tracks_per_frame, elapsed


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: DeepSORT Parameter Ablation")
    parser.add_argument("--data-root", default="data/MOT17")
    parser.add_argument("--detector", default="mot17_det")
    parser.add_argument("--output-dir", default="outputs/exp2_deepsort_ablation")
    parser.add_argument("--reid-model", default="osnet_x0_25")
    parser.add_argument("--reid-weights", default=None, help="Optional Re-ID checkpoint path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.detector.startswith("yolov8"):
        det_config = {"name": args.detector, "confidence_threshold": 0.25, "nms_threshold": 0.45}
    else:
        det_config = {"name": args.detector, "confidence_threshold": 0.5, "nms_threshold": 0.4}
    sequences = load_sequences(args.data_root, split="train")
    if not sequences:
        logger.error("No sequences found.")
        return

    reid = ReIDExtractor(model_name=args.reid_model, weights=args.reid_weights)
    evaluator = MOTEvaluator()

    # ============================
    # Ablation A: max_age
    # ============================
    logger.info("\n" + "=" * 60 + "\nAblation A: max_age\n" + "=" * 60)
    max_age_values = [10, 30, 50, 70]
    ablation_a_results = []

    for max_age in max_age_values:
        tag = f"max_age_{max_age}"
        all_metrics = []

        for seq_info in sequences:
            tracker = DeepSORTTracker(max_age=max_age, n_init=3, max_cosine_distance=0.2)
            detector = build_detector(det_config)
            tracks, elapsed = run_deepsort(tracker, detector, seq_info, reid=reid, use_reid=True)

            track_file = os.path.join(args.output_dir, "ablation_a", tag, f"{seq_info['name']}.txt")
            save_tracks(tracks, track_file)

            if os.path.isfile(seq_info["gt_file"]):
                metrics = evaluator.evaluate(track_file, seq_info["gt_file"], seq_info["name"])
                metrics["FPS"] = seq_info["seq_length"] / max(elapsed, 1e-6)
                all_metrics.append(metrics)

        if all_metrics:
            avg = {
                "max_age": max_age,
                "MOTA": np.mean([m["MOTA"] for m in all_metrics]),
                "IDF1": np.mean([m.get("IDF1", 0) for m in all_metrics]),
                "IDSW": sum(m.get("IDSW", 0) for m in all_metrics),
                "FPS": np.mean([m.get("FPS", 0) for m in all_metrics]),
            }
            ablation_a_results.append(avg)
            logger.info(f"max_age={max_age}: MOTA={avg['MOTA']:.1f}, IDF1={avg['IDF1']:.1f}, IDSW={avg['IDSW']}")

    # ============================
    # Ablation B: Re-ID on/off
    # ============================
    logger.info("\n" + "=" * 60 + "\nAblation B: Re-ID on/off\n" + "=" * 60)
    ablation_b_results = []

    for use_reid in [True, False]:
        tag = "reid_on" if use_reid else "reid_off"
        all_metrics = []

        for seq_info in sequences:
            tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2)
            detector = build_detector(det_config)
            tracks, elapsed = run_deepsort(tracker, detector, seq_info, reid=reid, use_reid=use_reid)

            track_file = os.path.join(args.output_dir, "ablation_b", tag, f"{seq_info['name']}.txt")
            save_tracks(tracks, track_file)

            if os.path.isfile(seq_info["gt_file"]):
                metrics = evaluator.evaluate(track_file, seq_info["gt_file"], seq_info["name"])
                metrics["FPS"] = seq_info["seq_length"] / max(elapsed, 1e-6)
                all_metrics.append(metrics)

        if all_metrics:
            avg = {
                "Re-ID": "ON" if use_reid else "OFF",
                "MOTA": np.mean([m["MOTA"] for m in all_metrics]),
                "IDF1": np.mean([m.get("IDF1", 0) for m in all_metrics]),
                "IDSW": sum(m.get("IDSW", 0) for m in all_metrics),
                "FPS": np.mean([m.get("FPS", 0) for m in all_metrics]),
            }
            ablation_b_results.append(avg)
            logger.info(f"Re-ID={avg['Re-ID']}: MOTA={avg['MOTA']:.1f}, IDF1={avg['IDF1']:.1f}, IDSW={avg['IDSW']}")

    # ============================
    # Ablation C: max_cosine_distance
    # ============================
    logger.info("\n" + "=" * 60 + "\nAblation C: max_cosine_distance\n" + "=" * 60)
    cos_values = [0.1, 0.2, 0.3, 0.4]
    ablation_c_results = []

    for cos_dist in cos_values:
        tag = f"cos_{cos_dist}"
        all_metrics = []

        for seq_info in sequences:
            tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=cos_dist)
            detector = build_detector(det_config)
            tracks, elapsed = run_deepsort(tracker, detector, seq_info, reid=reid, use_reid=True)

            track_file = os.path.join(args.output_dir, "ablation_c", tag, f"{seq_info['name']}.txt")
            save_tracks(tracks, track_file)

            if os.path.isfile(seq_info["gt_file"]):
                metrics = evaluator.evaluate(track_file, seq_info["gt_file"], seq_info["name"])
                all_metrics.append(metrics)

        if all_metrics:
            avg = {
                "cos_dist": cos_dist,
                "MOTA": np.mean([m["MOTA"] for m in all_metrics]),
                "IDF1": np.mean([m.get("IDF1", 0) for m in all_metrics]),
                "IDSW": sum(m.get("IDSW", 0) for m in all_metrics),
            }
            ablation_c_results.append(avg)
            logger.info(f"cos_dist={cos_dist}: MOTA={avg['MOTA']:.1f}, IDF1={avg['IDF1']:.1f}")

    # ============================
    # Ablation D: nn_budget
    # ============================
    logger.info("\n" + "=" * 60 + "\nAblation D: nn_budget\n" + "=" * 60)
    nn_budget_values = [20, 50, 100, None]
    ablation_d_results = []

    for nn_budget in nn_budget_values:
        budget_tag = "none" if nn_budget is None else str(nn_budget)
        tag = f"nn_budget_{budget_tag}"
        all_metrics = []

        for seq_info in sequences:
            tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2, nn_budget=nn_budget)
            detector = build_detector(det_config)
            tracks, elapsed = run_deepsort(tracker, detector, seq_info, reid=reid, use_reid=True)

            track_file = os.path.join(args.output_dir, "ablation_d", tag, f"{seq_info['name']}.txt")
            save_tracks(tracks, track_file)

            if os.path.isfile(seq_info["gt_file"]):
                metrics = evaluator.evaluate(track_file, seq_info["gt_file"], seq_info["name"])
                metrics["FPS"] = seq_info["seq_length"] / max(elapsed, 1e-6)
                all_metrics.append(metrics)

        if all_metrics:
            avg = {
                "nn_budget": "None" if nn_budget is None else str(nn_budget),
                "MOTA": np.mean([m["MOTA"] for m in all_metrics]),
                "IDF1": np.mean([m.get("IDF1", 0) for m in all_metrics]),
                "IDSW": sum(m.get("IDSW", 0) for m in all_metrics),
                "FPS": np.mean([m.get("FPS", 0) for m in all_metrics]),
            }
            ablation_d_results.append(avg)
            logger.info(
                f"nn_budget={avg['nn_budget']}: MOTA={avg['MOTA']:.1f}, "
                f"IDF1={avg['IDF1']:.1f}, IDSW={avg['IDSW']}"
            )

    # ============================
    # Print summary tables
    # ============================
    logger.info("\n" + "=" * 80 + "\nEXPERIMENT 2: DEEPSORT ABLATION SUMMARY\n" + "=" * 80)

    if ablation_a_results:
        print("\n--- Ablation A: max_age ---")
        print(tabulate(ablation_a_results, headers="keys", floatfmt=".1f", tablefmt="grid"))

    if ablation_b_results:
        print("\n--- Ablation B: Re-ID on/off ---")
        print(tabulate(ablation_b_results, headers="keys", floatfmt=".1f", tablefmt="grid"))

    if ablation_c_results:
        print("\n--- Ablation C: max_cosine_distance ---")
        print(tabulate(ablation_c_results, headers="keys", floatfmt=".1f", tablefmt="grid"))

    if ablation_d_results:
        print("\n--- Ablation D: nn_budget ---")
        print(tabulate(ablation_d_results, headers="keys", floatfmt=".1f", tablefmt="grid"))

    # Save all results
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("Experiment 2: DeepSORT Parameter Ablation\n\n")
        if ablation_a_results:
            f.write("--- Ablation A: max_age ---\n")
            f.write(tabulate(ablation_a_results, headers="keys", floatfmt=".1f", tablefmt="grid") + "\n\n")
        if ablation_b_results:
            f.write("--- Ablation B: Re-ID on/off ---\n")
            f.write(tabulate(ablation_b_results, headers="keys", floatfmt=".1f", tablefmt="grid") + "\n\n")
        if ablation_c_results:
            f.write("--- Ablation C: max_cosine_distance ---\n")
            f.write(tabulate(ablation_c_results, headers="keys", floatfmt=".1f", tablefmt="grid") + "\n\n")
        if ablation_d_results:
            f.write("--- Ablation D: nn_budget ---\n")
            f.write(tabulate(ablation_d_results, headers="keys", floatfmt=".1f", tablefmt="grid") + "\n\n")
    logger.info(f"Summary saved to {summary_file}")

    # Generate charts
    from src.utils.plot_results import generate_all_exp2_charts
    generate_all_exp2_charts(
        ablation_a_results,
        ablation_b_results,
        ablation_c_results,
        ablation_d_results,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
