"""Video I/O helpers for running trackers on raw video files.

The MOT17/MOT20 pipelines work on pre-extracted frame directories with a
seqinfo.ini; real-world videos don't have that metadata, so this module
provides thin wrappers that read/write frames via OpenCV and compose
side-by-side comparison grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int


def open_video(path: str) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open a video file and return the capture plus metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    return cap, info


def iter_frames(cap: cv2.VideoCapture):
    """Yield (frame_id, frame_bgr) pairs starting from frame_id=1."""
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        yield frame_id, frame


def make_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Open an mp4 writer; parent directory is created if missing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def list_videos(path: str) -> list[str]:
    """Return video files. ``path`` may be a file or a directory."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)
    raise FileNotFoundError(f"Video path not found: {path}")


def compose_grid(frames: list[np.ndarray], labels: list[str],
                 cell_size: tuple[int, int] | None = None,
                 cols: int | None = None) -> np.ndarray:
    """Tile N annotated frames into a single grid image.

    Parameters
    ----------
    frames : list of BGR ndarrays; all panels are resized to ``cell_size``.
    labels : text drawn on top-left of each panel (e.g. tracker name).
    cell_size : (w, h) for each panel. Defaults to the first frame's size.
    cols : number of columns. Defaults to ``len(frames)`` (single row).
    """
    if not frames:
        raise ValueError("compose_grid requires at least one frame")
    if cell_size is None:
        h, w = frames[0].shape[:2]
        cell_size = (w, h)
    cw, ch = cell_size
    cols = cols or len(frames)
    rows = (len(frames) + cols - 1) // cols

    canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for idx, (frame, label) in enumerate(zip(frames, labels)):
        r, c = divmod(idx, cols)
        tile = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)

        # Label banner
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(tile, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = tile
    return canvas


def overlay_stats(frame: np.ndarray, lines: list[str],
                  origin: tuple[int, int] = (10, 30)) -> None:
    """Draw a stack of text lines onto ``frame`` (in place)."""
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (x, y + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
