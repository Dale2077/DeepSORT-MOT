"""
SORT: Simple Online and Realtime Tracking (Bewley et al., ICIP 2016)

Algorithm:
1. Predict existing tracks using Kalman filter.
2. Compute IoU cost matrix between predicted tracks and detections.
3. Solve assignment using Hungarian algorithm with IoU threshold.
4. Update matched tracks, create new tracks for unmatched detections,
   delete old unmatched tracks.
"""

import numpy as np
from loguru import logger

from .track import Track, TrackState
from ..motion.kalman_filter import KalmanFilter
from ..association.matching import linear_assignment
from ..association.iou_matching import iou_distance


class SORTTracker:
    """SORT multi-object tracker.

    Parameters
    ----------
    max_age : int
        Maximum number of missed frames before track deletion.
    min_hits : int
        Minimum number of hits before track is considered valid for output.
    iou_threshold : float
        IoU threshold for matching.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    @classmethod
    def from_config(cls, config: dict) -> "SORTTracker":
        """Create tracker from config dict."""
        tc = config["tracker"]
        return cls(
            max_age=tc.get("max_age", 30),
            min_hits=tc.get("min_hits", 3),
            iou_threshold=tc.get("iou_threshold", 0.3),
        )

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections: list) -> np.ndarray:
        """Run one tracking step.

        Parameters
        ----------
        detections : list[Detection]
            Detections for the current frame.

        Returns
        -------
        outputs : ndarray (K, 5)
            Active track outputs as [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        # Predict existing tracks
        for track in self.tracks:
            track.predict(self.kf)

        # Match predictions with detections using IoU
        cost_matrix = iou_distance(self.tracks, detections)
        matches, unmatched_tracks, unmatched_dets = linear_assignment(
            cost_matrix, thresh=1.0 - self.iou_threshold
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx])

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Collect outputs (only confirmed tracks or those with enough hits)
        outputs = []
        for track in self.tracks:
            if track.time_since_update > 0:
                continue
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = track.to_tlbr()
                outputs.append([*bbox, track.track_id])

        return np.array(outputs) if outputs else np.empty((0, 5))

    def _initiate_track(self, detection):
        """Create a new track from a detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance, self._next_id, n_init=1, max_age=self.max_age)
        )
        self._next_id += 1
