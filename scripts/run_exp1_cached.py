"""Rerun exp1 (SORT/DeepSORT/ByteTrack) using cached Re-ID features for DeepSORT.

Uses the same per-tracker config as experiments/exp1_algorithm_compare.py:
- SORT: conf=0.5, max_age=1, iou=0.6
- DeepSORT: conf=0.95, max_age=30, cos=0.2, cached features from conf_0p95
- ByteTrack: conf=0.05, high=0.5, low=0.05, max_age=30
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tabulate import tabulate

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector.base import build_detector
from src.tracker.sort import SORTTracker
from src.tracker.deepsort import DeepSORTTracker
from src.tracker.bytetrack import ByteTracker
from src.utils.io import load_sequences, get_image_path, save_tracks
from src.utils.metrics import MOTEvaluator


OUT_ROOT = "outputs/exp1_algorithm_compare"
DET_CONFIGS = {
    "SORT":      {"name": "mot17_det", "confidence_threshold": 0.5,  "nms_threshold": 0.4},
    "DeepSORT":  {"name": "mot17_det", "confidence_threshold": 0.95, "nms_threshold": 0.4},
    "ByteTrack": {"name": "mot17_det", "confidence_threshold": 0.05, "nms_threshold": 0.4},
}
DEEPSORT_CACHE = "outputs/_reid_cache/conf_0p95"


def run_one(tracker, detector, seq, cache_dir=None):
    detector.load(seq["det_file"])
    tracks_per_frame = {}
    elapsed = 0.0
    cache = np.load(os.path.join(cache_dir, f"{seq['name']}.npz")) if cache_dir else None

    for fid in range(1, seq["seq_length"] + 1):
        img_path = get_image_path(seq["img_dir"], fid, seq.get("im_ext", ".jpg"))
        img = cv2.imread(img_path) if os.path.isfile(img_path) else None
        dets = detector.detect(fid, img)

        t0 = time.perf_counter()
        if cache is not None:
            features = None
            if len(dets) > 0:
                f = cache[f"feat_{fid}"]
                if f.shape[0] == len(dets):
                    features = f
            out = tracker.update(dets, features=features)
        else:
            if hasattr(tracker, "update") and "features" in tracker.update.__code__.co_varnames:
                out = tracker.update(dets, features=None)
            else:
                out = tracker.update(dets)
        elapsed += time.perf_counter() - t0
        tracks_per_frame[fid] = out

    return tracks_per_frame, elapsed


def main():
    sequences = load_sequences("data/MOT17", split="train")
    ev = MOTEvaluator()

    trackers = {
        "SORT":      lambda: SORTTracker(max_age=1, min_hits=3, iou_threshold=0.6),
        "DeepSORT":  lambda: DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2),
        "ByteTrack": lambda: ByteTracker(max_age=30, min_hits=3, high_threshold=0.5, low_threshold=0.05),
    }

    all_results = {name: [] for name in trackers}

    for seq in sequences:
        logger.info(f"\n=== {seq['name']} ===")
        for name, factory in trackers.items():
            tracker = factory()
            detector = build_detector(DET_CONFIGS[name])
            cache_dir = DEEPSORT_CACHE if name == "DeepSORT" else None
            tracks, elapsed = run_one(tracker, detector, seq, cache_dir=cache_dir)
            tf = os.path.join(OUT_ROOT, name, f"{seq['name']}.txt")
            save_tracks(tracks, tf)
            m = ev.evaluate(tf, seq["gt_file"], seq["name"])
            m["FPS"] = seq["seq_length"] / max(elapsed, 1e-6)
            all_results[name].append(m)
            logger.info(f"{name}: MOTA={m['MOTA']:.1f} IDF1={m['IDF1']:.1f} IDSW={m['IDSW']} FPS={m['FPS']:.1f}")

    rows = []
    for name in trackers:
        results = all_results[name]
        if not results:
            continue
        rows.append({
            "Tracker": name,
            "MOTA": np.mean([r["MOTA"] for r in results]),
            "IDF1": np.mean([r["IDF1"] for r in results]),
            "FP": sum(r["FP"] for r in results),
            "FN": sum(r["FN"] for r in results),
            "IDSW": sum(r["IDSW"] for r in results),
            "FPS": np.mean([r["FPS"] for r in results]),
        })

    table = tabulate(rows, headers="keys", floatfmt=".1f", tablefmt="grid")
    print(table)

    summary_path = os.path.join(OUT_ROOT, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment 1: Algorithm Comparison\n")
        f.write("Detector: mot17_det\n")
        f.write("Det conf — SORT: 0.5, DeepSORT: 0.95, ByteTrack: 0.05\n")
        f.write("max_age — SORT: 1 (paper), DeepSORT: 30, ByteTrack: 30\n\n")
        f.write(table + "\n")
    logger.info(f"Summary saved to {summary_path}")

    # Regenerate plots
    try:
        from src.utils.plot_results import generate_all_exp1_charts
        summary_dict = {row["Tracker"]: row for row in rows}
        generate_all_exp1_charts(all_results, summary_dict, OUT_ROOT)
    except Exception as e:
        logger.warning(f"Chart regen failed: {e}")


if __name__ == "__main__":
    main()
