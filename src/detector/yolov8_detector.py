"""YOLOv8 detector wrapper using Ultralytics."""

from pathlib import Path

import numpy as np
from loguru import logger

from .base import Detection


class YOLOv8Detector:
    """YOLOv8 object detector wrapper.

    Parameters
    ----------
    model_name : str
        Model variant: "yolov8n", "yolov8s", "yolov8m", etc.
    confidence_threshold : float
    nms_threshold : float
    device : str
        Compute device ("cuda:0", "cpu", etc.).
    """

    def __init__(
        self,
        model_name: str = "yolov8s",
        weights: str = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "",
    ):
        self.model_name = model_name
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.model = None
        self._warned_missing_image = False

    def load(self, det_file: str = None):
        """Initialize the YOLOv8 model.

        Parameters
        ----------
        det_file : str or None
            Not used; included for API compatibility with MOTDetector.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOv8 detection. "
                "Install with: pip install ultralytics"
            )

        # Prefer explicit weights, then a locally fine-tuned MOT17 checkpoint,
        # then the Ultralytics release identified by the model name.
        repo_root = Path(__file__).resolve().parents[2]
        model_path = None
        if self.weights:
            candidates = [Path(self.weights)]
            if not Path(self.weights).is_absolute():
                candidates.append(repo_root / self.weights)
            for candidate in candidates:
                if candidate.is_file():
                    model_path = str(candidate.resolve())
                    break
            if model_path is None:
                logger.warning(f"YOLO weights not found: {self.weights}; falling back to defaults")

        if model_path is None:
            fine_tuned = repo_root / "models" / f"{self.model_name}_mot17.pt"
            if fine_tuned.is_file():
                model_path = str(fine_tuned.resolve())
            else:
                base_pretrained = repo_root / "models" / f"{self.model_name}.pt"
                if base_pretrained.is_file():
                    model_path = str(base_pretrained.resolve())
                else:
                    model_path = f"{self.model_name}.pt"

        self.model = YOLO(model_path)
        if self.device:
            self.model.to(self.device)
        logger.info(f"Loaded YOLOv8 model: {model_path}")

    def detect(self, frame_id: int = None, image: np.ndarray = None) -> list:
        """Run detection on an image.

        Parameters
        ----------
        frame_id : int or None
            Not used for live detection; included for API compatibility.
        image : ndarray
            BGR image (H, W, 3).

        Returns
        -------
        detections : list[Detection]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        if image is None:
            if not self._warned_missing_image:
                logger.warning("Input image is missing; returning no detections for missing YOLO frames")
                self._warned_missing_image = True
            return []

        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=[0],  # person class only
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                # Convert xyxy to tlwh
                x1, y1, x2, y2 = xyxy
                tlwh = np.array([x1, y1, x2 - x1, y2 - y1])
                detections.append(Detection(tlwh=tlwh, confidence=conf))

        return detections
