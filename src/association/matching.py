"""Linear assignment and matching cascade for data association."""

import numpy as np
import lap

from .iou_matching import iou_distance
from ..motion.kalman_filter import CHI2_95


INFTY_COST = 1e5


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Solve linear assignment problem with threshold.

    Parameters
    ----------
    cost_matrix : ndarray (N, M)
        Cost matrix.
    thresh : float
        Maximum allowable cost for a valid assignment.

    Returns
    -------
    matches : list[(int, int)]
        Matched (row, col) index pairs.
    unmatched_rows : list[int]
    unmatched_cols : list[int]
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    # Use lap (Jonker-Volgenant) for efficient linear assignment
    cost = cost_matrix.copy()
    cost[cost > thresh] = thresh + 1e-4

    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh + 1e-4)

    matches = []
    unmatched_rows = []
    unmatched_cols = list(range(cost_matrix.shape[1]))

    for i in range(cost_matrix.shape[0]):
        if x[i] >= 0 and cost_matrix[i, x[i]] <= thresh:
            matches.append((i, x[i]))
            if x[i] in unmatched_cols:
                unmatched_cols.remove(x[i])
        else:
            unmatched_rows.append(i)

    return matches, unmatched_rows, unmatched_cols


def gate_cost_matrix(
    kf,
    cost_matrix: np.ndarray,
    tracks,
    detections,
    track_indices: list,
    detection_indices: list,
    gated_cost: float = INFTY_COST,
    only_position: bool = False,
) -> np.ndarray:
    """Apply Mahalanobis gating to a cost matrix.

    Invalidate entries where the Mahalanobis distance exceeds the
    chi-squared 95% gate.

    Parameters
    ----------
    kf : KalmanFilter
    cost_matrix : ndarray (N, M)
    tracks : list[Track]
    detections : list[Detection]
    track_indices : list[int]
    detection_indices : list[int]
    gated_cost : float
        Replacement cost for gated entries.
    only_position : bool

    Returns
    -------
    cost_matrix : ndarray (N, M) with gated entries set to `gated_cost`.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = CHI2_95[gating_dim]

    measurements = np.array([detections[i].to_xyah() for i in detection_indices])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix


def matching_cascade(
    distance_metric,
    max_distance: float,
    cascade_depth: int,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Run matching cascade (DeepSORT Algorithm 1).

    Perform matching in a cascaded fashion: for each age level from
    most-recently-seen to oldest, match tracks of that age with
    remaining unmatched detections.

    Parameters
    ----------
    distance_metric : callable
        (tracks, detections, track_indices, detection_indices) -> cost_matrix
    max_distance : float
        Maximum matching distance.
    cascade_depth : int
        Maximum number of cascade levels (typically max_age).
    tracks : list[Track]
    detections : list[Detection]
    track_indices : list[int] or None
    detection_indices : list[int] or None

    Returns
    -------
    matches : list[(int, int)]
        Matched (track_idx, detection_idx) pairs.
    unmatched_tracks : list[int]
    unmatched_detections : list[int]
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        # Select tracks at this cascade level
        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:
            continue

        cost_matrix = distance_metric(
            tracks, detections, track_indices_l, unmatched_detections
        )
        matches_l, _, unmatched_dets_l = linear_assignment(
            cost_matrix, max_distance
        )

        # Map local indices back to global
        for row, col in matches_l:
            matches.append((track_indices_l[row], unmatched_detections[col]))

        # Update unmatched detections: keep only those whose local column
        # index was not matched
        matched_det_local = {col for _, col in matches_l}
        unmatched_detections = [
            k for i, k in enumerate(unmatched_detections) if i not in matched_det_local
        ]

    unmatched_tracks = [k for k in track_indices if k not in {m[0] for m in matches}]
    return matches, unmatched_tracks, unmatched_detections
