"""MOT17 public detection loader."""

import os
import numpy as np
from loguru import logger

from .base import Detection


class MOTDetector:
    """Load pre-computed detections from MOT17 det/det.txt files.

    MOT17 detection format per line:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    """

    def __init__(self, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._detections = {}

    def load(self, det_file: str):
        """Load detections from a MOT-format det.txt file.

        Parameters
        ----------
        det_file : str
            Path to det.txt file.
        """
        if not os.path.isfile(det_file):
            raise FileNotFoundError(f"Detection file not found: {det_file}")

        raw = np.loadtxt(det_file, delimiter=",")
        self._detections = {}
        for row in raw:
            frame_id = int(row[0])
            bbox = row[2:6]  # [x, y, w, h]
            conf = row[6]
            if conf < self.confidence_threshold:
                continue
            if frame_id not in self._detections:
                self._detections[frame_id] = []
            self._detections[frame_id].append(Detection(tlwh=bbox, confidence=conf))

        # Apply NMS per frame
        for frame_id in self._detections:
            self._detections[frame_id] = self._nms(self._detections[frame_id])

        logger.info(
            f"Loaded {sum(len(v) for v in self._detections.values())} detections "
            f"from {len(self._detections)} frames"
        )

    def detect(self, frame_id: int, image=None) -> list:
        """Get detections for a specific frame.

        Parameters
        ----------
        frame_id : int
            1-based frame index.
        image : ndarray or None
            Not used for pre-computed detections.

        Returns
        -------
        detections : list[Detection]
        """
        return self._detections.get(frame_id, [])

    def _nms(self, detections: list) -> list:
        """Non-maximum suppression."""
        if len(detections) <= 1:
            return detections

        bboxes = np.array([d.tlbr for d in detections])
        scores = np.array([d.confidence for d in detections])

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]
