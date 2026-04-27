"""IoU-based matching utilities."""

import numpy as np


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Parameters
    ----------
    bboxes_a : ndarray (N, 4) in [x1, y1, x2, y2]
    bboxes_b : ndarray (M, 4) in [x1, y1, x2, y2]

    Returns
    -------
    iou_matrix : ndarray (N, M)
    """
    bboxes_a = np.asarray(bboxes_a, dtype=np.float64)
    bboxes_b = np.asarray(bboxes_b, dtype=np.float64)

    if bboxes_a.size == 0 or bboxes_b.size == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    # Intersection
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    iou_matrix = np.where(union > 0, intersection / union, 0.0)
    return iou_matrix


def iou_distance(tracks, detections) -> np.ndarray:
    """Compute cost matrix based on IoU distance.

    Parameters
    ----------
    tracks : list[Track]
        List of Track objects (must have `to_tlbr()` method).
    detections : list[Detection]
        List of Detection objects (must have `tlbr` attribute).

    Returns
    -------
    cost_matrix : ndarray (N, M)
        IoU distance matrix = 1 - IoU.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    track_bboxes = np.array([t.to_tlbr() for t in tracks])
    det_bboxes = np.array([d.tlbr for d in detections])
    iou_matrix = iou_batch(track_bboxes, det_bboxes)
    return 1.0 - iou_matrix
