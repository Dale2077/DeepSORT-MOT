"""Rerun the YOLO rows of exp3 under the patched DeepSORT (pure-appearance + Mahalanobis).

Preserves the MOT17-DET row in outputs/exp3_detector_ablation/summary.txt
(already reran via scripts/run_exp3_mot17det_cached.py). Re-ID features are
extracted inline because YOLO bboxes differ from the MOT17-DET cached set, so
reuse isn't possible. Runtime is dominated by YOLO inference (~minutes per
detector per sequence) and OSNet x0.25 forward passes per detection.
"""

import os
import sys
import time
import re
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
from src.utils.io import load_sequences, get_image_path, save_tracks
from src.utils.metrics import MOTEvaluator


OUT_ROOT = "outputs/exp3_detector_ablation"
DETECTOR_CONFIGS = [
    {"name": "yolov8n", "confidence_threshold": 0.25, "nms_threshold": 0.45, "label": "YOLOv8-Nano"},
    {"name": "yolov8s", "confidence_threshold": 0.25, "nms_threshold": 0.45, "label": "YOLOv8-Small"},
    {"name": "yolov8m", "confidence_threshold": 0.25, "nms_threshold": 0.45, "label": "YOLOv8-Medium"},
]


def run_one(detector, reid, tracker, seq):
    detector.load(seq["det_file"])
    tracks = {}
    elapsed = 0.0

    for fid in range(1, seq["seq_length"] + 1):
        img_path = get_image_path(seq["img_dir"], fid, seq.get("im_ext", ".jpg"))
        img = cv2.imread(img_path) if os.path.isfile(img_path) else None

        t0 = time.perf_counter()
        dets = detector.detect(fid, img)
        features = None
        if reid is not None and img is not None and len(dets) > 0:
            bboxes = np.array([d.tlbr for d in dets])
            features = reid.extract(img, bboxes)
        out = tracker.update(dets, features=features)
        elapsed += time.perf_counter() - t0
        tracks[fid] = out

    return tracks, elapsed


def main():
    sequences = load_sequences("data/MOT17", split="train")
    ev = MOTEvaluator()
    reid = ReIDExtractor(model_name="osnet_x0_25", weights=None)

    rows = []
    for det_cfg in DETECTOR_CONFIGS:
        label = det_cfg.pop("label")
        logger.info(f"\n=== {label} ===")

        seq_metrics = []
        total_elapsed = 0.0
        total_frames = 0
        for seq in sequences:
            tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2)
            try:
                detector = build_detector(det_cfg)
            except Exception as e:
                logger.warning(f"Failed to build {det_cfg['name']}: {e}")
                break
            try:
                tracks, elapsed = run_one(detector, reid, tracker, seq)
            except Exception as e:
                logger.warning(f"Failed on {seq['name']}: {e}")
                continue

            tf = os.path.join(OUT_ROOT, label.replace(" ", "_"), f"{seq['name']}.txt")
            save_tracks(tracks, tf)

            m = ev.evaluate(tf, seq["gt_file"], seq["name"])
            m["FPS"] = seq["seq_length"] / max(elapsed, 1e-6)
            seq_metrics.append(m)
            total_elapsed += elapsed
            total_frames += seq["seq_length"]
            logger.info(f"  {seq['name']}: MOTA={m['MOTA']:.1f} IDF1={m['IDF1']:.1f} IDSW={m['IDSW']} FPS={m['FPS']:.1f}")

        det_cfg["label"] = label  # restore for next iteration if needed
        if not seq_metrics:
            continue

        rows.append({
            "Detector": label,
            "MOTA": np.mean([m["MOTA"] for m in seq_metrics]),
            "MOTP": np.mean([m["MOTP"] for m in seq_metrics]),
            "IDF1": np.mean([m["IDF1"] for m in seq_metrics]),
            "FP": sum(m["FP"] for m in seq_metrics),
            "FN": sum(m["FN"] for m in seq_metrics),
            "IDSW": sum(m["IDSW"] for m in seq_metrics),
            "FPS": total_frames / max(total_elapsed, 1e-6),
        })

    # Merge with existing MOT17-DET row from current summary
    summary_path = os.path.join(OUT_ROOT, "summary.txt")
    with open(summary_path) as f:
        body = f.read()
    m = re.search(
        r"\|\s*MOT17-DET\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|",
        body,
    )
    merged_rows = []
    if m:
        merged_rows.append({
            "Detector": "MOT17-DET",
            "MOTA": float(m.group(1)),
            "MOTP": float(m.group(2)),
            "IDF1": float(m.group(3)),
            "FP": int(float(m.group(4))),
            "FN": int(float(m.group(5))),
            "IDSW": int(float(m.group(6))),
            "FPS": float(m.group(7)),
        })
    merged_rows.extend(rows)

    table = tabulate(merged_rows, headers="keys", floatfmt=".1f", tablefmt="grid")
    with open(summary_path, "w") as f:
        f.write("Experiment 3: Detector Ablation (with DeepSORT tracker)\n")
        f.write("Tracker: original-DeepSORT pure-appearance cost (Wojke 2017) gated by Mahalanobis; max_age=30, cos=0.2.\n")
        f.write("MOT17-DET FPS is matching-only (Re-ID features cached). YOLO rows include detector + Re-ID + tracker time (end-to-end).\n\n")
        f.write(table + "\n")
    logger.info(f"Summary written to {summary_path}")
    print(table)

    # Regenerate exp3 charts if helper exists
    try:
        from src.utils.plot_results import generate_all_exp3_charts
        all_results = {row["Detector"]: row for row in merged_rows}
        generate_all_exp3_charts(all_results, OUT_ROOT)
        logger.info("Charts regenerated")
    except Exception as e:
        logger.warning(f"Chart regen failed: {e}")


if __name__ == "__main__":
    main()
