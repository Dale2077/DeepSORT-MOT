"""Detection data structures and detector factory."""

import numpy as np


class Detection:
    """A single bounding box detection.

    Attributes
    ----------
    tlwh : ndarray (4,)
        Bounding box in (top-left x, top-left y, width, height).
    confidence : float
        Detection confidence score.
    feature : ndarray or None
        Re-ID feature vector.
    class_id : int
        Object class (0 for pedestrian by default).
    """

    def __init__(self, tlwh, confidence: float, feature=None, class_id: int = 0):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = feature
        self.class_id = class_id

    @property
    def tlbr(self) -> np.ndarray:
        """Convert to (x1, y1, x2, y2)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert to (center_x, center_y, aspect_ratio, height)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2  # center
        ret[2] /= ret[3]  # aspect ratio
        return ret

    def to_xyxy(self) -> np.ndarray:
        """Alias for tlbr."""
        return self.tlbr

    def __repr__(self):
        return f"Detection(tlwh={self.tlwh}, conf={self.confidence:.2f})"


def build_detector(config: dict):
    """Factory function to build a detector from config.

    Parameters
    ----------
    config : dict
        Detector configuration with 'name' key.

    Returns
    -------
    detector : MOTDetector or YOLOv8Detector
    """
    name = config["name"]
    if name == "mot17_det":
        from .mot_detector import MOTDetector
        return MOTDetector(
            confidence_threshold=config.get("confidence_threshold", 0.5),
            nms_threshold=config.get("nms_threshold", 0.4),
        )
    elif name.startswith("yolov8"):
        from .yolov8_detector import YOLOv8Detector
        return YOLOv8Detector(
            model_name=name,
            weights=config.get("weights"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            nms_threshold=config.get("nms_threshold", 0.4),
        )
    else:
        raise ValueError(f"Unknown detector: {name}")
