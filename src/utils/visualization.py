"""Visualization utilities for tracking results."""

import numpy as np

# Color palette for track IDs (BGR)
_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
]


def get_color(track_id: int) -> tuple:
    """Get consistent color for a track ID."""
    return _PALETTE[track_id % len(_PALETTE)]


class Visualizer:
    """Draw tracking results on images.

    Parameters
    ----------
    im_width : int
    im_height : int
    save_video : bool
        Whether to save a video file.
    output_path : str or None
        Path for output video file.
    fps : int
        Video frame rate.
    """

    def __init__(
        self,
        im_width: int = 1920,
        im_height: int = 1080,
        save_video: bool = False,
        output_path: str = None,
        fps: int = 25,
    ):
        self.im_width = im_width
        self.im_height = im_height
        self.save_video = save_video
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        self._trail = {}  # track_id -> list of center points

    def start(self):
        """Initialize video writer if saving."""
        if self.save_video and self.output_path:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.im_width, self.im_height)
            )

    def draw_frame(self, image: np.ndarray, tracks: np.ndarray, frame_id: int = 0,
                   detections: list = None, show_trail: bool = True) -> np.ndarray:
        """Draw tracking results on an image.

        Parameters
        ----------
        image : ndarray (H, W, 3) BGR
        tracks : ndarray (K, 5) [x1, y1, x2, y2, track_id]
        frame_id : int
        detections : list[Detection] or None
        show_trail : bool

        Returns
        -------
        annotated : ndarray (H, W, 3) BGR
        """
        import cv2

        img = image.copy()

        # Draw detections (thin gray boxes)
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2 = det.tlbr.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            track_id = int(track[4])
            color = get_color(track_id)

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Trail
            if show_trail:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id not in self._trail:
                    self._trail[track_id] = []
                self._trail[track_id].append((cx, cy))
                # Keep last 30 points
                if len(self._trail[track_id]) > 30:
                    self._trail[track_id] = self._trail[track_id][-30:]

                points = self._trail[track_id]
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(float(i) / len(points)) * 2) + 1
                    cv2.line(img, points[i - 1], points[i], color, thickness)

        # Frame info
        cv2.putText(img, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Tracks: {len(tracks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Write to video
        if self.writer is not None:
            self.writer.write(img)

        return img

    def finish(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self._trail.clear()
