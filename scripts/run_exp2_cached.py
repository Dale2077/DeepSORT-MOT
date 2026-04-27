"""Run exp2 DeepSORT ablations using cached Re-ID features.

Reads features from outputs/_reid_cache/conf_0p5/<seq>.npz and replays the
four ablations (max_age, Re-ID on/off, max_cosine_distance, nn_budget).
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
from src.tracker.deepsort import DeepSORTTracker
from src.utils.io import load_sequences, get_image_path, save_tracks
from src.utils.metrics import MOTEvaluator


CACHE_DIR = "outputs/_reid_cache/conf_0p5"
OUT_ROOT = "outputs/exp2_deepsort_ablation"
DET_CFG = {"name": "mot17_det", "confidence_threshold": 0.5, "nms_threshold": 0.4}


def run_ablation(sequences, evaluator, tag_root, tracker_factory, use_reid=True):
    all_metrics = []
    total_elapsed = 0.0
    total_frames = 0

    for seq in sequences:
        tracker = tracker_factory()
        detector = build_detector(DET_CFG)
        detector.load(seq["det_file"])
        cache = np.load(os.path.join(CACHE_DIR, f"{seq['name']}.npz"))

        tracks_per_frame = {}
        elapsed = 0.0
        for fid in range(1, seq["seq_length"] + 1):
            img_path = get_image_path(seq["img_dir"], fid, seq.get("im_ext", ".jpg"))
            img = cv2.imread(img_path) if os.path.isfile(img_path) else None
            dets = detector.detect(fid, img)

            features = None
            if use_reid and len(dets) > 0:
                feats = cache[f"feat_{fid}"]
                if feats.shape[0] == len(dets):
                    features = feats

            t0 = time.perf_counter()
            out = tracker.update(dets, features=features)
            elapsed += time.perf_counter() - t0
            tracks_per_frame[fid] = out

        total_elapsed += elapsed
        total_frames += seq["seq_length"]

        tf = os.path.join(tag_root, f"{seq['name']}.txt")
        save_tracks(tracks_per_frame, tf)

        m = evaluator.evaluate(tf, seq["gt_file"], seq["name"])
        m["FPS"] = seq["seq_length"] / max(elapsed, 1e-6)
        all_metrics.append(m)

    if not all_metrics:
        return None
    return {
        "MOTA": np.mean([m["MOTA"] for m in all_metrics]),
        "IDF1": np.mean([m["IDF1"] for m in all_metrics]),
        "FP": sum(m["FP"] for m in all_metrics),
        "FN": sum(m["FN"] for m in all_metrics),
        "IDSW": sum(m["IDSW"] for m in all_metrics),
        "FPS": total_frames / max(total_elapsed, 1e-6),
    }


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    sequences = load_sequences("data/MOT17", split="train")
    ev = MOTEvaluator()

    # Ablation A
    a_rows = []
    for ma in [10, 30, 50, 70]:
        logger.info(f"[A] max_age={ma}")
        r = run_ablation(
            sequences, ev,
            os.path.join(OUT_ROOT, "ablation_a", f"max_age_{ma}"),
            tracker_factory=lambda ma=ma: DeepSORTTracker(max_age=ma, n_init=3, max_cosine_distance=0.2),
        )
        if r:
            r["max_age"] = ma
            a_rows.append(r)
            logger.info(f"  MOTA={r['MOTA']:.1f} IDF1={r['IDF1']:.1f} IDSW={r['IDSW']}")

    # Ablation B
    b_rows = []
    for use_reid, lbl in [(True, "ON"), (False, "OFF")]:
        logger.info(f"[B] Re-ID={lbl}")
        r = run_ablation(
            sequences, ev,
            os.path.join(OUT_ROOT, "ablation_b", "reid_on" if use_reid else "reid_off"),
            tracker_factory=lambda: DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2),
            use_reid=use_reid,
        )
        if r:
            r["Re-ID"] = lbl
            b_rows.append(r)
            logger.info(f"  MOTA={r['MOTA']:.1f} IDF1={r['IDF1']:.1f} IDSW={r['IDSW']}")

    # Ablation C
    c_rows = []
    for cd in [0.1, 0.2, 0.3, 0.4]:
        logger.info(f"[C] cos_dist={cd}")
        r = run_ablation(
            sequences, ev,
            os.path.join(OUT_ROOT, "ablation_c", f"cos_{cd}"),
            tracker_factory=lambda cd=cd: DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=cd),
        )
        if r:
            r["cos_dist"] = cd
            c_rows.append(r)
            logger.info(f"  MOTA={r['MOTA']:.1f} IDF1={r['IDF1']:.1f} IDSW={r['IDSW']}")

    # Ablation D
    d_rows = []
    for bg in [20, 50, 100, None]:
        tag = "none" if bg is None else str(bg)
        logger.info(f"[D] nn_budget={bg}")
        r = run_ablation(
            sequences, ev,
            os.path.join(OUT_ROOT, "ablation_d", f"nn_budget_{tag}"),
            tracker_factory=lambda bg=bg: DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2, nn_budget=bg),
        )
        if r:
            r["nn_budget"] = "None" if bg is None else str(bg)
            d_rows.append(r)
            logger.info(f"  MOTA={r['MOTA']:.1f} IDF1={r['IDF1']:.1f} IDSW={r['IDSW']}")

    # Summary
    def fmt(rows, head_key):
        display = [{k: v for k, v in r.items() if k in (head_key, "MOTA", "IDF1", "IDSW", "FPS")} for r in rows]
        return tabulate(display, headers="keys", floatfmt=".1f", tablefmt="grid")

    with open(os.path.join(OUT_ROOT, "summary.txt"), "w") as f:
        f.write("Experiment 2: DeepSORT Parameter Ablation\n\n")
        if a_rows:
            f.write("--- Ablation A: max_age ---\n" + fmt(a_rows, "max_age") + "\n\n")
        if b_rows:
            f.write("--- Ablation B: Re-ID on/off ---\n" + fmt(b_rows, "Re-ID") + "\n\n")
        if c_rows:
            f.write("--- Ablation C: max_cosine_distance ---\n" + fmt(c_rows, "cos_dist") + "\n\n")
        if d_rows:
            f.write("--- Ablation D: nn_budget ---\n" + fmt(d_rows, "nn_budget") + "\n\n")

    print("\n===== FINAL =====")
    if a_rows:
        print("\n--- A: max_age ---\n" + fmt(a_rows, "max_age"))
    if b_rows:
        print("\n--- B: Re-ID ---\n" + fmt(b_rows, "Re-ID"))
    if c_rows:
        print("\n--- C: cos_dist ---\n" + fmt(c_rows, "cos_dist"))
    if d_rows:
        print("\n--- D: nn_budget ---\n" + fmt(d_rows, "nn_budget"))

    # Regenerate plots
    try:
        from src.utils.plot_results import generate_all_exp2_charts
        generate_all_exp2_charts(a_rows, b_rows, c_rows, d_rows, OUT_ROOT)
        logger.info("Charts regenerated")
    except Exception as e:
        logger.warning(f"Chart regen failed: {e}")


if __name__ == "__main__":
    main()
