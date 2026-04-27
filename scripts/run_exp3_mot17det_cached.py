"""Rerun exp3's MOT17-DET baseline row under the patched DeepSORT using cached features.

Leaves YOLOv8 rows untouched (those take hours to re-run). Overwrites only the
MOT17-DET row in outputs/exp3_detector_ablation/summary.txt.
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
from src.utils.io import load_sequences, get_image_path, save_tracks
from src.utils.metrics import MOTEvaluator


CACHE_DIR = "outputs/_reid_cache/conf_0p5"
OUT_ROOT = "outputs/exp3_detector_ablation"
DET_CFG = {"name": "mot17_det", "confidence_threshold": 0.5, "nms_threshold": 0.4}


def main():
    sequences = load_sequences("data/MOT17", split="train")
    ev = MOTEvaluator()

    metrics = []
    total_elapsed = 0.0
    total_frames = 0
    for seq in sequences:
        tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2)
        detector = build_detector(DET_CFG)
        detector.load(seq["det_file"])
        cache = np.load(os.path.join(CACHE_DIR, f"{seq['name']}.npz"))

        tracks = {}
        elapsed = 0.0
        for fid in range(1, seq["seq_length"] + 1):
            img_path = get_image_path(seq["img_dir"], fid, seq.get("im_ext", ".jpg"))
            img = cv2.imread(img_path) if os.path.isfile(img_path) else None
            dets = detector.detect(fid, img)
            features = None
            if len(dets) > 0:
                f = cache[f"feat_{fid}"]
                if f.shape[0] == len(dets):
                    features = f

            t0 = time.perf_counter()
            out = tracker.update(dets, features=features)
            elapsed += time.perf_counter() - t0
            tracks[fid] = out

        total_elapsed += elapsed
        total_frames += seq["seq_length"]
        tf = os.path.join(OUT_ROOT, "MOT17-DET", f"{seq['name']}.txt")
        save_tracks(tracks, tf)
        m = ev.evaluate(tf, seq["gt_file"], seq["name"])
        metrics.append(m)

    agg = {
        "Detector": "MOT17-DET",
        "MOTA": np.mean([m["MOTA"] for m in metrics]),
        "MOTP": np.mean([m["MOTP"] for m in metrics]),
        "IDF1": np.mean([m["IDF1"] for m in metrics]),
        "FP": sum(m["FP"] for m in metrics),
        "FN": sum(m["FN"] for m in metrics),
        "IDSW": sum(m["IDSW"] for m in metrics),
        "FPS": total_frames / max(total_elapsed, 1e-6),
    }
    logger.info(f"New MOT17-DET row: MOTA={agg['MOTA']:.1f} IDF1={agg['IDF1']:.1f} IDSW={agg['IDSW']}")

    # Rewrite summary.txt, preserving YOLO rows
    summary_path = os.path.join(OUT_ROOT, "summary.txt")
    rows = [agg]
    # Keep existing YOLO rows from current summary
    with open(summary_path) as f:
        body = f.read()
    for label in ["YOLOv8-Nano", "YOLOv8-Small", "YOLOv8-Medium"]:
        m = re.search(rf"\|\s*{label}\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|", body)
        if m:
            rows.append({
                "Detector": label,
                "MOTA": float(m.group(1)),
                "MOTP": float(m.group(2)),
                "IDF1": float(m.group(3)),
                "FP": int(float(m.group(4))),
                "FN": int(float(m.group(5))),
                "IDSW": int(float(m.group(6))),
                "FPS": float(m.group(7)),
            })

    table = tabulate(rows, headers="keys", floatfmt=".1f", tablefmt="grid")
    with open(summary_path, "w") as f:
        f.write("Experiment 3: Detector Ablation (with DeepSORT tracker)\n\n")
        f.write(table + "\n")
    logger.info(f"Summary written to {summary_path}")
    print(table)


if __name__ == "__main__":
    main()
