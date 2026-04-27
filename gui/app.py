"""
GUI application for multi-object tracking visualization.

Built with PySide6, supports:
  - Sequence selection from MOT17 / MOT20 datasets (quick switch)
  - Raw video-file mode (data/videos/*.mp4) via YOLO detector
  - Tracker selection (SORT / DeepSORT / ByteTrack)
  - Detector selection (MOT public det / YOLOv8 variants)
  - Real-time playback with tracking overlay
  - Play / Pause / Step / Speed control
  - Export tracked video
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSlider, QFileDialog, QGroupBox,
    QStatusBar, QSpinBox, QCheckBox, QSplitter, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

from src.detector.base import build_detector
from src.tracker.sort import SORTTracker
from src.tracker.deepsort import DeepSORTTracker
from src.tracker.bytetrack import ByteTracker
from src.reid.feature_extractor import ReIDExtractor
from src.utils.io import load_sequences, get_image_path
from src.utils.visualization import Visualizer, get_color


class TrackingWorker:
    """Manages tracking state for the GUI.

    Supports two source types via ``seq_info["source"]``:
      * "mot" (default): frame directory + public detections (MOT17/MOT20 style)
      * "video": raw video file processed on the fly with a YOLO detector
    """

    def __init__(self):
        self.tracker = None
        self.detector = None
        self.reid = None
        self.seq_info = None
        self.visualizer = None
        self.current_frame = 0
        self.tracks_cache = {}
        self._cap = None
        self._cap_pos = 0  # next frame_id the capture will return

    def setup(self, tracker_name, detector_name, seq_info, use_reid=True):
        """Initialize tracker, detector, and sequence."""
        self._release_cap()
        self.seq_info = seq_info
        self.current_frame = 0
        self.tracks_cache = {}

        source = seq_info.get("source", "mot")

        # Build detector (video mode forces YOLO since no public det exists).
        if source == "video":
            # Use a permissive conf so ByteTrack-like low-score recovery still works.
            det_config = {
                "name": detector_name if detector_name.startswith("yolov8") else "yolov8m",
                "confidence_threshold": 0.1 if tracker_name == "ByteTrack" else 0.25,
                "nms_threshold": 0.45,
            }
            self.detector = build_detector(det_config)
            self.detector.load()  # YOLO ignores the argument
            self._cap = cv2.VideoCapture(seq_info["video_path"])
            self._cap_pos = 1
        else:
            det_config = {
                "name": detector_name,
                "confidence_threshold": 0.25 if detector_name.startswith("yolov8") else 0.5,
                "nms_threshold": 0.45 if detector_name.startswith("yolov8") else 0.4,
            }
            self.detector = build_detector(det_config)
            self.detector.load(seq_info["det_file"])

        # Build tracker
        if tracker_name == "SORT":
            self.tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        elif tracker_name == "DeepSORT":
            self.tracker = DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2)
            if use_reid:
                self.reid = ReIDExtractor(model_name="osnet_x0_25")
        elif tracker_name == "ByteTrack":
            self.tracker = ByteTracker(max_age=30, min_hits=3, high_threshold=0.6)

        self.visualizer = Visualizer(
            im_width=seq_info["im_width"],
            im_height=seq_info["im_height"],
        )

    def _read_frame(self, frame_id):
        """Fetch the image for ``frame_id`` from the active source."""
        source = self.seq_info.get("source", "mot")
        if source == "video":
            if self._cap is None:
                return None
            # Seek only when the requested frame isn't the next one — seeking
            # is expensive on many codecs, so sequential playback stays fast.
            if frame_id != self._cap_pos:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                self._cap_pos = frame_id
            ok, image = self._cap.read()
            self._cap_pos += 1 if ok else 0
            return image if ok else None

        img_path = get_image_path(
            self.seq_info["img_dir"], frame_id, self.seq_info.get("im_ext", ".jpg")
        )
        if not os.path.isfile(img_path):
            return None
        return cv2.imread(img_path)

    def process_frame(self, frame_id):
        """Process a single frame and return annotated image."""
        if self.seq_info is None or self.tracker is None:
            return None, np.empty((0, 5))

        image = self._read_frame(frame_id)
        if image is None:
            return None, np.empty((0, 5))

        detections = self.detector.detect(frame_id, image)

        features = None
        if (
            self.reid is not None
            and isinstance(self.tracker, DeepSORTTracker)
            and len(detections) > 0
        ):
            bboxes = np.array([d.tlbr for d in detections])
            features = self.reid.extract(image, bboxes)

        if isinstance(self.tracker, DeepSORTTracker):
            tracks = self.tracker.update(detections, features=features)
        else:
            tracks = self.tracker.update(detections)

        annotated = self.visualizer.draw_frame(image, tracks, frame_id, detections)
        return annotated, tracks

    def reset(self):
        if self.tracker:
            self.tracker.reset()
        self.current_frame = 0
        self.tracks_cache = {}
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._cap_pos = 1

    def _release_cap(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._cap_pos = 0


class MainWindow(QMainWindow):
    """Main GUI window for MOT visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Object Tracking System")
        self.setMinimumSize(1280, 800)

        self.worker = TrackingWorker()
        self.sequences = []
        self.playing = False
        self.current_frame = 1
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)

        self._setup_ui()
        self._load_sequences()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: controls
        left_panel = QVBoxLayout()
        left_panel.setMaximumWidth(320)

        # Dataset group
        dataset_group = QGroupBox("数据集")
        dataset_layout = QVBoxLayout(dataset_group)

        # Quick-switch row: MOT17 / MOT20 / 视频文件
        quick_row = QHBoxLayout()
        self.btn_mot17 = QPushButton("MOT17")
        self.btn_mot17.clicked.connect(lambda: self._use_dataset("data/MOT17"))
        self.btn_mot20 = QPushButton("MOT20")
        self.btn_mot20.clicked.connect(lambda: self._use_dataset("data/MOT20"))
        self.btn_load_video = QPushButton("视频文件...")
        self.btn_load_video.clicked.connect(self._load_video)
        quick_row.addWidget(self.btn_mot17)
        quick_row.addWidget(self.btn_mot20)
        quick_row.addWidget(self.btn_load_video)
        dataset_layout.addLayout(quick_row)

        self.data_root_label = QLabel("data/MOT17")
        self.data_root_label.setWordWrap(True)
        self.btn_browse = QPushButton("浏览 MOT 根目录...")
        self.btn_browse.clicked.connect(self._browse_data)
        self.combo_sequence = QComboBox()
        self.combo_sequence.currentIndexChanged.connect(self._on_sequence_changed)
        dataset_layout.addWidget(QLabel("当前数据路径:"))
        dataset_layout.addWidget(self.data_root_label)
        dataset_layout.addWidget(self.btn_browse)
        dataset_layout.addWidget(QLabel("序列 / 视频:"))
        dataset_layout.addWidget(self.combo_sequence)
        left_panel.addWidget(dataset_group)

        # Tracker group
        tracker_group = QGroupBox("跟踪器设置")
        tracker_layout = QVBoxLayout(tracker_group)
        self.combo_tracker = QComboBox()
        self.combo_tracker.addItems(["SORT", "DeepSORT", "ByteTrack"])
        self.combo_tracker.setCurrentIndex(1)
        self.combo_detector = QComboBox()
        self.combo_detector.addItems(["mot17_det", "yolov8n", "yolov8s", "yolov8m"])
        self.check_reid = QCheckBox("启用 Re-ID 特征")
        self.check_reid.setChecked(True)
        tracker_layout.addWidget(QLabel("跟踪算法:"))
        tracker_layout.addWidget(self.combo_tracker)
        tracker_layout.addWidget(QLabel("检测器:"))
        tracker_layout.addWidget(self.combo_detector)
        tracker_layout.addWidget(self.check_reid)
        left_panel.addWidget(tracker_group)

        # Control group
        control_group = QGroupBox("播放控制")
        control_layout = QVBoxLayout(control_group)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始跟踪")
        self.btn_start.clicked.connect(self._start_tracking)
        self.btn_play = QPushButton("⏸ 暂停")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setEnabled(False)
        self.btn_step = QPushButton("⏭ 单步")
        self.btn_step.clicked.connect(self._step_frame)
        self.btn_step.setEnabled(False)
        self.btn_reset = QPushButton("⟲ 重置")
        self.btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_step)
        btn_row.addWidget(self.btn_reset)
        control_layout.addLayout(btn_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("速度:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(25)
        self.speed_spin.setSuffix(" FPS")
        self.speed_spin.valueChanged.connect(self._update_speed)
        speed_row.addWidget(self.speed_spin)
        control_layout.addLayout(speed_row)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.valueChanged.connect(self._on_slider)
        control_layout.addWidget(self.slider)
        self.frame_label = QLabel("帧: 1 / -")
        control_layout.addWidget(self.frame_label)
        left_panel.addWidget(control_group)

        # Export
        self.btn_export = QPushButton("📁 导出视频")
        self.btn_export.clicked.connect(self._export_video)
        self.btn_export.setEnabled(False)
        left_panel.addWidget(self.btn_export)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_panel.addWidget(self.progress)

        left_panel.addStretch()

        # Right panel: image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #2d2d2d; border: 1px solid #555;")

        # Info panel
        self.info_label = QLabel("就绪")
        self.info_label.setFont(QFont("Courier", 10))

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label, stretch=1)
        right_layout.addWidget(self.info_label)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(320)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, stretch=1)

        # Status bar
        self.statusBar().showMessage("就绪 — 请选择数据集和跟踪器")

    def _load_sequences(self, data_root="data/MOT17"):
        """Load available MOT sequences from train + test splits."""
        self.combo_sequence.clear()
        sequences = []
        for split in ("train", "test"):
            try:
                for info in load_sequences(data_root, split=split):
                    info["source"] = "mot"
                    sequences.append(info)
            except FileNotFoundError:
                continue
        self.sequences = sequences

        for seq in self.sequences:
            self.combo_sequence.addItem(seq["name"])

        if not self.sequences:
            self.statusBar().showMessage(f"在 {data_root} 下未找到序列")

    def _use_dataset(self, data_root):
        """One-click switch to MOT17 / MOT20 (or any preset path)."""
        self.data_root_label.setText(data_root)
        self._load_sequences(data_root)

    def _browse_data(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择 MOT 数据根目录 (含 train/test)")
        if dir_path:
            self.data_root_label.setText(dir_path)
            self._load_sequences(dir_path)

    def _load_video(self):
        """Load a raw video file as a pseudo-sequence (YOLO detector required)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择测试视频",
            "data/videos",
            "Video (*.mp4 *.mov *.avi *.mkv *.m4v)"
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.statusBar().showMessage(f"无法打开视频: {path}")
            return
        info = {
            "source": "video",
            "name": f"[video] {Path(path).name}",
            "video_path": path,
            "im_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "im_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_rate": int(cap.get(cv2.CAP_PROP_FPS) or 25),
            "seq_length": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "im_ext": None,
            "img_dir": None,
            "det_file": None,
            "gt_file": None,
        }
        cap.release()

        self.sequences.append(info)
        self.combo_sequence.addItem(info["name"])
        self.combo_sequence.setCurrentIndex(len(self.sequences) - 1)
        # Video mode cannot use MOT public detections — default to YOLOv8m.
        idx = self.combo_detector.findText("yolov8m")
        if idx >= 0:
            self.combo_detector.setCurrentIndex(idx)
        self.data_root_label.setText(f"视频: {path}")
        self.statusBar().showMessage(
            f"已加载视频: {info['name']} | {info['seq_length']}帧 "
            f"{info['im_width']}x{info['im_height']} @ {info['frame_rate']}fps"
        )

    def _on_sequence_changed(self, index):
        if 0 <= index < len(self.sequences):
            seq = self.sequences[index]
            self.slider.setMaximum(seq["seq_length"])
            self.frame_label.setText(f"帧: 1 / {seq['seq_length']}")
            self.statusBar().showMessage(
                f"序列: {seq['name']} | {seq['seq_length']}帧 | {seq['im_width']}x{seq['im_height']} @ {seq['frame_rate']}fps"
            )

    def _start_tracking(self):
        idx = self.combo_sequence.currentIndex()
        if idx < 0 or idx >= len(self.sequences):
            self.statusBar().showMessage("请先选择一个序列")
            return

        seq = self.sequences[idx]
        tracker_name = self.combo_tracker.currentText()
        detector_name = self.combo_detector.currentText()

        self.statusBar().showMessage(f"初始化 {tracker_name} + {detector_name}...")
        QApplication.processEvents()

        try:
            self.worker.setup(tracker_name, detector_name, seq, self.check_reid.isChecked())
        except Exception as e:
            self.statusBar().showMessage(f"初始化失败: {e}")
            return

        self.current_frame = 1
        self.worker.reset()
        self.playing = True
        self.btn_play.setEnabled(True)
        self.btn_step.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_play.setText("⏸ 暂停")
        self._update_speed()
        self.timer.start()
        self.statusBar().showMessage(f"跟踪中: {tracker_name} on {seq['name']}")

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ 暂停")
            self.timer.start()
        else:
            self.btn_play.setText("▶ 播放")
            self.timer.stop()

    def _step_frame(self):
        self.playing = False
        self.timer.stop()
        self.btn_play.setText("▶ 播放")
        self._process_next_frame()

    def _reset(self):
        self.timer.stop()
        self.playing = False
        self.current_frame = 1
        self.worker.reset()
        self.btn_play.setText("▶ 播放")
        self.slider.setValue(1)
        self.image_label.clear()
        self.info_label.setText("已重置")
        self.statusBar().showMessage("已重置")

    def _update_speed(self):
        fps = self.speed_spin.value()
        self.timer.setInterval(int(1000 / fps))

    def _on_slider(self, value):
        if not self.playing:
            self.current_frame = value
            self.frame_label.setText(
                f"帧: {value} / {self.slider.maximum()}"
            )

    def _on_timer(self):
        self._process_next_frame()

    def _process_next_frame(self):
        if self.worker.seq_info is None:
            return

        seq_len = self.worker.seq_info["seq_length"]
        if self.current_frame > seq_len:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("▶ 播放")
            self.statusBar().showMessage("跟踪完成")
            return

        annotated, tracks = self.worker.process_frame(self.current_frame)

        if annotated is not None:
            self._display_image(annotated)
            n_tracks = len(tracks) if len(tracks) > 0 else 0
            self.info_label.setText(
                f"帧: {self.current_frame}/{seq_len} | 跟踪目标: {n_tracks}"
            )

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.frame_label.setText(f"帧: {self.current_frame} / {seq_len}")
        self.current_frame += 1

    def _display_image(self, image):
        """Convert OpenCV image to QPixmap and display."""
        h, w, c = image.shape
        bytes_per_line = c * w
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label
        label_size = self.image_label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _export_video(self):
        if self.worker.seq_info is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存视频", "output.mp4", "Video (*.mp4)"
        )
        if not path:
            return

        seq = self.worker.seq_info
        self.progress.setVisible(True)
        self.progress.setMaximum(seq["seq_length"])
        self.statusBar().showMessage("正在导出视频...")

        # Reset tracker for full run
        self.worker.reset()
        vis = Visualizer(
            im_width=seq["im_width"],
            im_height=seq["im_height"],
            save_video=True,
            output_path=path,
            fps=seq["frame_rate"],
        )
        vis.start()

        for frame_id in range(1, seq["seq_length"] + 1):
            annotated, tracks = self.worker.process_frame(frame_id)
            if annotated is not None:
                vis.writer.write(annotated)
            self.progress.setValue(frame_id)
            QApplication.processEvents()

        vis.finish()
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"视频已保存至: {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
