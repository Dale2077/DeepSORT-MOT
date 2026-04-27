"""MOT evaluation metrics using motmetrics and/or custom implementation."""

import os
from collections import defaultdict

import numpy as np
from loguru import logger


if not hasattr(np, "asfarray"):
    def _asfarray_compat(array_like, dtype=float):
        return np.asarray(array_like, dtype=dtype)

    np.asfarray = _asfarray_compat


class MOTEvaluator:
    """Evaluate tracking results against ground truth using MOTChallenge metrics.

    Computes: MOTA, MOTP, IDF1, MT, ML, FP, FN, IDSW, HOTA (if available).
    """

    def __init__(self):
        self._accumulator = None

    def evaluate(self, track_file: str, gt_file: str, seq_name: str = "") -> dict:
        """Evaluate a single sequence.

        Parameters
        ----------
        track_file : str
            Path to tracking results file (MOTChallenge format).
        gt_file : str
            Path to ground truth file.
        seq_name : str
            Sequence name for display.

        Returns
        -------
        metrics : dict
            Dictionary of metric name -> value.
        """
        try:
            return self._evaluate_motmetrics(track_file, gt_file, seq_name)
        except ImportError:
            logger.warning("motmetrics not installed, using built-in evaluator")
            return self._evaluate_builtin(track_file, gt_file, seq_name)

    def _evaluate_motmetrics(self, track_file: str, gt_file: str, seq_name: str) -> dict:
        """Evaluate using the motmetrics library."""
        import motmetrics as mm

        gt = self._load_mot_file(gt_file, is_gt=True)
        dt = self._load_mot_file(track_file, is_gt=False)

        acc = mm.MOTAccumulator(auto_id=True)

        all_frames = sorted(set(list(gt.keys()) + list(dt.keys())))
        for frame_id in all_frames:
            gt_ids = []
            gt_bboxes = []
            if frame_id in gt:
                for row in gt[frame_id]:
                    gt_ids.append(int(row[0]))
                    gt_bboxes.append(row[1:5])
            gt_bboxes = np.array(gt_bboxes) if gt_bboxes else np.empty((0, 4))

            dt_ids = []
            dt_bboxes = []
            if frame_id in dt:
                for row in dt[frame_id]:
                    dt_ids.append(int(row[0]))
                    dt_bboxes.append(row[1:5])
            dt_bboxes = np.array(dt_bboxes) if dt_bboxes else np.empty((0, 4))

            # Compute IoU distance
            if len(gt_bboxes) > 0 and len(dt_bboxes) > 0:
                dists = mm.distances.iou_matrix(gt_bboxes, dt_bboxes, max_iou=0.5)
            else:
                dists = np.empty((len(gt_bboxes), len(dt_bboxes)))
                dists[:] = np.nan

            acc.update(gt_ids, dt_ids, dists)

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=[
                "mota", "motp", "idf1",
                "mostly_tracked", "mostly_lost",
                "num_false_positives", "num_misses", "num_switches",
                "num_objects", "num_unique_objects",
            ],
            name=seq_name,
        )

        num_gt_objs = int(summary["num_unique_objects"].iloc[0]) if summary["num_unique_objects"].iloc[0] > 0 else 1

        return {
            "MOTA": float(summary["mota"].iloc[0]) * 100,
            "MOTP": (1 - float(summary["motp"].iloc[0])) * 100 if not np.isnan(summary["motp"].iloc[0]) else 0.0,
            "IDF1": float(summary["idf1"].iloc[0]) * 100,
            "MT": int(summary["mostly_tracked"].iloc[0]),
            "ML": int(summary["mostly_lost"].iloc[0]),
            "FP": int(summary["num_false_positives"].iloc[0]),
            "FN": int(summary["num_misses"].iloc[0]),
            "IDSW": int(summary["num_switches"].iloc[0]),
            "MT%": float(summary["mostly_tracked"].iloc[0]) / num_gt_objs * 100,
            "ML%": float(summary["mostly_lost"].iloc[0]) / num_gt_objs * 100,
        }

    def _evaluate_builtin(self, track_file: str, gt_file: str, seq_name: str) -> dict:
        """Simple built-in evaluation when motmetrics is not available."""
        gt = self._load_mot_file(gt_file, is_gt=True)
        dt = self._load_mot_file(track_file, is_gt=False)

        tp, fp, fn, idsw = 0, 0, 0, 0
        prev_match = {}  # gt_id -> dt_id
        iou_sum, iou_count = 0.0, 0

        all_frames = sorted(set(list(gt.keys()) + list(dt.keys())))

        for frame_id in all_frames:
            gt_items = gt.get(frame_id, [])
            dt_items = dt.get(frame_id, [])

            if len(gt_items) == 0:
                fp += len(dt_items)
                continue
            if len(dt_items) == 0:
                fn += len(gt_items)
                continue

            gt_bboxes = np.array([item[1:5] for item in gt_items])
            dt_bboxes = np.array([item[1:5] for item in dt_items])
            gt_ids = [int(item[0]) for item in gt_items]
            dt_ids = [int(item[0]) for item in dt_items]

            iou_matrix = self._compute_iou(gt_bboxes, dt_bboxes)

            matched_gt = set()
            matched_dt = set()

            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                if max_iou < 0.5:
                    break

                gi, di = max_idx
                matched_gt.add(gi)
                matched_dt.add(di)

                tp += 1
                iou_sum += max_iou
                iou_count += 1

                # Check for ID switch
                gt_id = gt_ids[gi]
                dt_id = dt_ids[di]
                if gt_id in prev_match and prev_match[gt_id] != dt_id:
                    idsw += 1
                prev_match[gt_id] = dt_id

                iou_matrix[gi, :] = 0
                iou_matrix[:, di] = 0

            fp += len(dt_items) - len(matched_dt)
            fn += len(gt_items) - len(matched_gt)

        num_gt_total = sum(len(v) for v in gt.values())
        mota = (1 - (fp + fn + idsw) / max(num_gt_total, 1)) * 100
        motp = (iou_sum / max(iou_count, 1)) * 100

        return {
            "MOTA": mota,
            "MOTP": motp,
            "IDF1": 0.0,  # Requires more complex computation
            "MT": 0,
            "ML": 0,
            "FP": fp,
            "FN": fn,
            "IDSW": idsw,
        }

    @staticmethod
    def _load_mot_file(filepath: str, is_gt: bool = False) -> dict:
        """Load a MOT-format file into a dict of frame_id -> list of entries.

        MOT17 gt.txt columns (0-indexed):
            0 frame, 1 id, 2 x, 3 y, 4 w, 5 h, 6 conf, 7 class, 8 visibility

        For GT we follow the MOTChallenge convention:
          - drop entries with ``conf == 0`` (marked "do not evaluate"),
          - keep only ``class == 1`` (pedestrian),
          - drop zero-visibility entries (fully occluded / outside frame).
        """
        result = defaultdict(list)
        if not os.path.isfile(filepath):
            return result

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame_id = int(float(parts[0]))
                obj_id = int(float(parts[1]))
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                if is_gt and len(parts) > 7:
                    conf_flag = float(parts[6]) if len(parts) > 6 else 1.0
                    cls = int(float(parts[7]))
                    vis = float(parts[8]) if len(parts) > 8 else 1.0
                    # MOT17: conf_flag==0 means the box should be ignored
                    # during evaluation (distractor, reflection, crowd, etc.).
                    if conf_flag <= 0 or cls != 1 or vis <= 0:
                        continue

                result[frame_id].append([obj_id, x, y, w, h])

        return dict(result)

    @staticmethod
    def _compute_iou(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of [x, y, w, h] bboxes."""
        # Convert to [x1, y1, x2, y2]
        a = bboxes_a.copy()
        a[:, 2] += a[:, 0]
        a[:, 3] += a[:, 1]
        b = bboxes_b.copy()
        b[:, 2] += b[:, 0]
        b[:, 3] += b[:, 1]

        x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
        y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter

        return np.where(union > 0, inter / union, 0.0)

    @staticmethod
    def format_results(results: dict, seq_name: str = "") -> str:
        """Format evaluation results as a nicely aligned string."""
        header = f"{'Sequence':<20} {'MOTA':>8} {'MOTP':>8} {'IDF1':>8} {'MT':>6} {'ML':>6} {'FP':>8} {'FN':>8} {'IDSW':>8}"
        line = (
            f"{seq_name:<20} "
            f"{results.get('MOTA', 0):>8.1f} "
            f"{results.get('MOTP', 0):>8.1f} "
            f"{results.get('IDF1', 0):>8.1f} "
            f"{results.get('MT', 0):>6d} "
            f"{results.get('ML', 0):>6d} "
            f"{results.get('FP', 0):>8d} "
            f"{results.get('FN', 0):>8d} "
            f"{results.get('IDSW', 0):>8d}"
        )
        return f"{header}\n{'-' * len(header)}\n{line}"
