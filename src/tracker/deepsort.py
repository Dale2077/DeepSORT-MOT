"""
DeepSORT: Deep SORT (Wojke et al., ICIP 2017)

Algorithm:
1. Predict existing tracks using Kalman filter.
2. Extract Re-ID features for detections.
3. Cascade matching:
   a. For each cascade level (time_since_update), compute combined
      appearance + motion cost.
   b. Apply Mahalanobis gating.
   c. Solve assignment using Hungarian algorithm.
4. IoU matching for remaining unmatched tracks and detections.
5. Update matched tracks, create new tracks, delete old tracks.
"""

import numpy as np
from loguru import logger

from .track import Track, TrackState
from ..motion.kalman_filter import KalmanFilter
from ..association.matching import linear_assignment, matching_cascade, gate_cost_matrix
from ..association.iou_matching import iou_distance
from ..association.cosine_matching import cosine_distance


class DeepSORTTracker:
    """DeepSORT multi-object tracker.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without an update before deletion.
    n_init : int
        Number of consecutive detections before track is confirmed.
    max_iou_distance : float
        Maximum IoU distance (1 - IoU) for fallback IoU matching.
    max_cosine_distance : float
        Maximum cosine distance for appearance matching.
    nn_budget : int or None
        Maximum size of the appearance feature gallery per track.
    lambda_weight : float
        Retained for configuration compatibility. The original DeepSORT
        reference implementation (Wojke 2017) uses pure appearance distance
        gated by Mahalanobis; motion is a binary gate rather than a weighted
        term, so this value is only applied if explicitly < 1.0.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        lambda_weight: float = 0.98,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.lambda_weight = lambda_weight

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    @classmethod
    def from_config(cls, config: dict) -> "DeepSORTTracker":
        """Create tracker from config dict."""
        tc = config["tracker"]
        return cls(
            max_age=tc.get("max_age", 30),
            n_init=tc.get("n_init", 3),
            max_iou_distance=tc.get("max_iou_distance", 0.7),
            max_cosine_distance=tc.get("max_cosine_distance", 0.2),
            nn_budget=tc.get("nn_budget", 100),
            lambda_weight=tc.get("lambda_weight", 0.98),
        )

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self._next_id = 1
        self.frame_count = 0

    def update(self, detections: list, features: np.ndarray = None) -> np.ndarray:
        """Run one tracking step.

        Parameters
        ----------
        detections : list[Detection]
            Detections for the current frame.
        features : ndarray (N, D) or None
            Re-ID features for each detection. Required for appearance matching.

        Returns
        -------
        outputs : ndarray (K, 5)
            Active track outputs as [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        if features is None:
            features = np.array([])

        # === Step 1: Predict ===
        for track in self.tracks:
            track.predict(self.kf)

        # === Step 2: Cascade matching (confirmed tracks) ===
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        if features.size > 0:
            # Pure appearance cost gated by Mahalanobis (original Wojke 2017
            # DeepSORT). Motion gating sets invalid entries to infinity, so
            # blending IoU into the cost is redundant and dilutes Re-ID's
            # signal on clean detections where IoU alone already matches most
            # tracks correctly.
            def _appearance_metric(tracks, dets, track_indices, det_indices):
                selected_tracks = [tracks[i] for i in track_indices]
                selected_dets = [dets[i] for i in det_indices]
                selected_features = (
                    features[det_indices] if len(det_indices) > 0 else np.array([])
                )
                cost = cosine_distance(selected_tracks, selected_dets, selected_features)
                cost = gate_cost_matrix(
                    self.kf, cost, tracks, dets, track_indices, det_indices
                )
                return cost

            matches_a, unmatched_tracks_a, unmatched_dets = matching_cascade(
                _appearance_metric,
                self.max_cosine_distance,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
            )
        else:
            # Re-ID disabled: fall back to a single-pass IoU assignment on
            # confirmed tracks (equivalent to SORT's primary match). Without
            # this fallback the cascade receives a zero cost matrix and
            # degenerates into arbitrary Hungarian assignments.
            if confirmed_tracks and detections:
                confirmed_track_objs = [self.tracks[i] for i in confirmed_tracks]
                iou_cost = iou_distance(confirmed_track_objs, detections)
                matches_raw, unmatched_rows, unmatched_cols = linear_assignment(
                    iou_cost, thresh=self.max_iou_distance
                )
                matches_a = [(confirmed_tracks[r], c) for r, c in matches_raw]
                unmatched_tracks_a = [confirmed_tracks[i] for i in unmatched_rows]
                unmatched_dets = list(unmatched_cols)
            else:
                matches_a = []
                unmatched_tracks_a = list(confirmed_tracks)
                unmatched_dets = list(range(len(detections)))

        # === Step 3: IoU matching for remaining ===
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]

        if iou_track_candidates and unmatched_dets:
            iou_tracks_for_match = [self.tracks[i] for i in iou_track_candidates]
            iou_dets_for_match = [detections[i] for i in unmatched_dets]
            cost_matrix = iou_distance(iou_tracks_for_match, iou_dets_for_match)
            matches_b, unmatched_tracks_b, unmatched_dets_b = linear_assignment(
                cost_matrix, thresh=self.max_iou_distance
            )
            # Map back to global indices
            matches_b = [
                (iou_track_candidates[r], unmatched_dets[c]) for r, c in matches_b
            ]
            unmatched_tracks_b = [iou_track_candidates[i] for i in unmatched_tracks_b]
            unmatched_dets = [unmatched_dets[i] for i in unmatched_dets_b]
        else:
            matches_b = []
            unmatched_tracks_b = list(iou_track_candidates)

        matches = matches_a + matches_b
        unmatched_tracks = list(unmatched_tracks_a) + unmatched_tracks_b

        # === Step 4: Update tracks ===
        for track_idx, det_idx in matches:
            feat = features[det_idx] if features.size > 0 and det_idx < len(features) else None
            self.tracks[track_idx].update(self.kf, detections[det_idx], feature=feat)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_dets:
            feat = features[det_idx] if features.size > 0 and det_idx < len(features) else None
            self._initiate_track(detections[det_idx], feature=feat)

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Trim feature galleries
        for track in self.tracks:
            if self.nn_budget and len(track.features) > self.nn_budget:
                track.features = track.features[-self.nn_budget :]

        # === Collect outputs ===
        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            bbox = track.to_tlbr()
            outputs.append([*bbox, track.track_id])

        return np.array(outputs) if outputs else np.empty((0, 5))

    def _initiate_track(self, detection, feature=None):
        """Create a new track from a detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance, self._next_id, n_init=self.n_init,
                  max_age=self.max_age, feature=feature)
        )
        self._next_id += 1
