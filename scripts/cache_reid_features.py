"""Precompute Re-ID features per frame for MOT17-DET detections and cache to .npz.

Each .npz contains one array per frame: 'features_<fid>' with shape (N, D),
and 'bboxes_<fid>' for sanity checks. Loading avoids re-running OSNet per run.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector.base import build_detector
from src.reid.feature_extractor import ReIDExtractor
from src.utils.io import load_sequences, get_image_path


def cache_features(data_root, confidences, cache_root, reid_model="osnet_x0_25"):
    reid = ReIDExtractor(model_name=reid_model, weights=None)
    sequences = load_sequences(data_root, split="train")

    for conf_key, conf_val in confidences.items():
        cache_dir = os.path.join(cache_root, conf_key)
        os.makedirs(cache_dir, exist_ok=True)

        for seq in sequences:
            out_path = os.path.join(cache_dir, f"{seq['name']}.npz")
            if os.path.isfile(out_path):
                logger.info(f"[{conf_key}] cache hit for {seq['name']} - skip")
                continue

            det_cfg = {"name": "mot17_det", "confidence_threshold": conf_val, "nms_threshold": 0.4}
            detector = build_detector(det_cfg)
            detector.load(seq["det_file"])

            arrays = {}
            for fid in range(1, seq["seq_length"] + 1):
                img_path = get_image_path(seq["img_dir"], fid, seq.get("im_ext", ".jpg"))
                img = cv2.imread(img_path) if os.path.isfile(img_path) else None
                dets = detector.detect(fid, img)
                if img is not None and len(dets) > 0:
                    bb = np.array([d.tlbr for d in dets])
                    feats = reid.extract(img, bb)
                    arrays[f"feat_{fid}"] = feats.astype(np.float32)
                    arrays[f"bbox_{fid}"] = bb.astype(np.float32)
                else:
                    arrays[f"feat_{fid}"] = np.zeros((0, 512), dtype=np.float32)
                    arrays[f"bbox_{fid}"] = np.zeros((0, 4), dtype=np.float32)

            np.savez_compressed(out_path, **arrays)
            logger.info(f"[{conf_key}] cached {seq['name']} -> {out_path}")


if __name__ == "__main__":
    cache_features(
        data_root="data/MOT17",
        confidences={"conf_0p5": 0.5, "conf_0p95": 0.95},
        cache_root="outputs/_reid_cache",
    )
