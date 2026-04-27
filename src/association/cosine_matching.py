"""Cosine distance matching for Re-ID features."""

import numpy as np


def _pdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distance.

    Parameters
    ----------
    a : ndarray (N, D)
    b : ndarray (M, D)

    Returns
    -------
    distances : ndarray (N, M)
    """
    a2 = np.sum(a * a, axis=1)
    b2 = np.sum(b * b, axis=1)
    r2 = -2.0 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    np.clip(r2, 0.0, None, out=r2)
    return r2


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance.

    Parameters
    ----------
    a : ndarray (N, D) - L2-normalized feature vectors.
    b : ndarray (M, D) - L2-normalized feature vectors.

    Returns
    -------
    distances : ndarray (N, M) in [0, 2].
    """
    a = np.asarray(a) / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = np.asarray(b) / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.dot(a, b.T)


def cosine_distance(tracks, detections, features: np.ndarray) -> np.ndarray:
    """Compute cosine distance between track appearance gallery and detection features.

    Parameters
    ----------
    tracks : list[Track]
        Tracks with `.features` attribute (list of feature vectors).
    detections : list[Detection]
        Detection objects.
    features : ndarray (M, D)
        Feature vectors for each detection.

    Returns
    -------
    cost_matrix : ndarray (N, M)
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))
    if features.size == 0:
        return cost_matrix

    det_features = features

    for i, track in enumerate(tracks):
        # Prefer EMA-smoothed feature (StrongSORT) — more stable than
        # min-over-gallery when the gallery contains bad crops.
        if getattr(track, "smooth_feature", None) is not None:
            track_features = np.asarray(track.smooth_feature).reshape(1, -1)
        elif len(track.features) > 0:
            track_features = np.array(track.features)
        else:
            cost_matrix[i, :] = 1e5
            continue
        distances = _cosine_distance(track_features, det_features)
        cost_matrix[i, :] = distances.min(axis=0)

    return cost_matrix


def nearest_neighbor_distance(track_features: np.ndarray, det_features: np.ndarray) -> np.ndarray:
    """Compute nearest neighbor cosine distance.

    For each detection feature, find the minimum cosine distance to any
    feature in the track gallery.

    Parameters
    ----------
    track_features : ndarray (G, D) - gallery features from a track.
    det_features : ndarray (M, D) - detection features.

    Returns
    -------
    distances : ndarray (M,) - minimum distance for each detection.
    """
    distances = _cosine_distance(track_features, det_features)
    return np.min(distances, axis=0)
