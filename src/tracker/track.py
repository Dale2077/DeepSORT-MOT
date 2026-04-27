"""Track data structure and lifecycle management."""

import numpy as np


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """A single tracked object with Kalman filter state.

    Parameters
    ----------
    mean : ndarray
        Initial state mean vector (8-dimensional).
    covariance : ndarray
        Initial state covariance matrix (8x8).
    track_id : int
        Unique track identifier.
    n_init : int
        Number of consecutive hits before the track is confirmed.
    max_age : int
        Maximum number of missed frames before deletion.
    feature : ndarray or None
        Initial appearance feature vector.
    """

    def __init__(self, mean, covariance, track_id: int, n_init: int, max_age: int, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = []
        self.smooth_feature = None
        if feature is not None:
            self.features.append(feature)
            self.smooth_feature = feature / (np.linalg.norm(feature) + 1e-12)
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self) -> np.ndarray:
        """Get current position in (top-left x, top-left y, w, h) format."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]       # width = aspect_ratio * height
        ret[:2] -= ret[2:] / 2  # top-left = center - size/2
        return ret

    def to_tlbr(self) -> np.ndarray:
        """Get current position in (x1, y1, x2, y2) format."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using KF prediction."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, feature=None, ema_alpha: float = 0.9):
        """Perform Kalman filter measurement update and optionally store feature.

        Feature gallery trimming is delegated to the tracker (``nn_budget`` in
        DeepSORT) so an explicit ``None`` budget really means unlimited.

        An EMA-smoothed feature is also maintained (StrongSORT-style), which
        stabilizes appearance matching against momentary occlusions and
        noisy detections — without it, the min-over-gallery distance is
        overly sensitive to any single bad crop.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        if feature is not None:
            self.features.append(feature)
            feat_norm = feature / (np.linalg.norm(feature) + 1e-12)
            if self.smooth_feature is None:
                self.smooth_feature = feat_norm
            else:
                self.smooth_feature = ema_alpha * self.smooth_feature + (1 - ema_alpha) * feat_norm
                self.smooth_feature /= np.linalg.norm(self.smooth_feature) + 1e-12
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
