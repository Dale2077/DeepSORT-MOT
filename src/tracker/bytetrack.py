"""
ByteTrack (Zhang et al., ECCV 2022)

Algorithm:
1. Split detections into high-score and low-score groups.
2. First association: match high-score detections with existing tracks using IoU.
3. Second association: match remaining (unmatched) tracks with low-score detections.
4. Create new tracks from unmatched high-score detections.
5. Delete long-lost tracks.
"""

import numpy as np
from loguru import logger

from .track import Track, TrackState
from ..motion.kalman_filter import KalmanFilter
from ..association.matching import linear_assignment
from ..association.iou_matching import iou_distance


class ByteTracker:
    """ByteTrack multi-object tracker.

    Parameters
    ----------
    max_age : int
        Maximum number of missed frames before track deletion.
    min_hits : int
        Minimum consecutive hits for track output.
    high_threshold : float
        Confidence threshold for high-score detections.
    low_threshold : float
        Confidence threshold for low-score detections.
    iou_threshold : float
        IoU threshold for first association.
    second_iou_threshold : float
        IoU threshold for second association with low-score detections.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        second_iou_threshold: float = 0.5,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.second_iou_threshold = second_iou_threshold

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    @classmethod
    def from_config(cls, config: dict) -> "ByteTracker":
        """Create tracker from config dict."""
        tc = config["tracker"]
        return cls(
            max_age=tc.get("max_age", 30),
            min_hits=tc.get("min_hits", 3),
            high_threshold=tc.get("high_threshold", 0.6),
            low_threshold=tc.get("low_threshold", 0.1),
            iou_threshold=tc.get("iou_threshold", 0.3),
            second_iou_threshold=tc.get("second_iou_threshold", 0.5),
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
            All detections for the current frame (including low-score).

        Returns
        -------
        outputs : ndarray (K, 5)
            Active track outputs as [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        # === Step 1: Split detections by confidence ===
        high_dets = []
        low_dets = []
        high_indices = []
        low_indices = []
        for i, det in enumerate(detections):
            if det.confidence >= self.high_threshold:
                high_dets.append(det)
                high_indices.append(i)
            elif det.confidence >= self.low_threshold:
                low_dets.append(det)
                low_indices.append(i)

        # === Step 2: Predict existing tracks ===
        for track in self.tracks:
            track.predict(self.kf)

        # Separate confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        # === Step 3: First association - high score detections ===
        confirmed_track_objs = [self.tracks[i] for i in confirmed_tracks]
        cost_first = iou_distance(confirmed_track_objs, high_dets)
        matches_first, unmatched_tracks_first, unmatched_dets_first = linear_assignment(
            cost_first, thresh=1.0 - self.iou_threshold
        )

        # Map back to global indices
        matches_first = [(confirmed_tracks[r], c) for r, c in matches_first]
        unmatched_track_indices = [confirmed_tracks[i] for i in unmatched_tracks_first]

        # === Step 4: Second association - low score detections with remaining tracks ===
        remaining_track_objs = [self.tracks[i] for i in unmatched_track_indices]
        cost_second = iou_distance(remaining_track_objs, low_dets)
        matches_second, unmatched_tracks_second, _ = linear_assignment(
            cost_second, thresh=1.0 - self.second_iou_threshold
        )

        matches_second = [(unmatched_track_indices[r], c) for r, c in matches_second]
        re_unmatched_tracks = [unmatched_track_indices[i] for i in unmatched_tracks_second]

        # === Step 5: Handle unconfirmed tracks with remaining high dets ===
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets_first]
        unconfirmed_track_objs = [self.tracks[i] for i in unconfirmed_tracks]
        cost_unconf = iou_distance(unconfirmed_track_objs, unmatched_high_dets)
        matches_unconf, unmatched_unconf, unmatched_high_final = linear_assignment(
            cost_unconf, thresh=1.0 - self.iou_threshold
        )
        matches_unconf = [
            (unconfirmed_tracks[r], unmatched_dets_first[c]) for r, c in matches_unconf
        ]

        # === Step 6: Apply updates ===
        all_matches = matches_first + matches_second + matches_unconf

        # Map detection indices for second matches (they index into low_dets)
        for track_idx, det_idx in matches_first:
            self.tracks[track_idx].update(self.kf, high_dets[det_idx])

        for track_idx, det_idx in matches_second:
            self.tracks[track_idx].update(self.kf, low_dets[det_idx])

        for track_idx, det_idx in matches_unconf:
            self.tracks[track_idx].update(self.kf, high_dets[det_idx])

        # Mark unmatched tracks as missed
        for idx in re_unmatched_tracks:
            self.tracks[idx].mark_missed()
        for idx in [unconfirmed_tracks[i] for i in unmatched_unconf]:
            self.tracks[idx].mark_missed()

        # Create new tracks from unmatched high-score detections
        for det_idx in unmatched_high_final:
            global_det_idx = unmatched_dets_first[det_idx]
            self._initiate_track(high_dets[global_det_idx])

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # === Collect outputs ===
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
            Track(mean, covariance, self._next_id, n_init=self.min_hits, max_age=self.max_age)
        )
        self._next_id += 1
