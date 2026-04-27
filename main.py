"""
Multi-Object Tracking System — Main Entry Point

Usage:
    # Run tracking on a sequence
    python main.py track --config configs/deepsort.yaml --sequence MOT17-02-SDP

    # Run tracking on MOT20 (override data root from CLI)
    python main.py track --config configs/deepsort.yaml \
        --data-root data/MOT20 --sequence MOT20-01

    # Run all experiments (defaults to MOT17; pass --data-root data/MOT20 for MOT20)
    python main.py experiment --exp all
    python main.py experiment --exp 1 --data-root data/MOT20

    # Run tracking on raw video files (saves annotated mp4 to outputs/videos/)
    python main.py video --input data/videos/test_video_1.mp4
    python main.py video --input data/videos --compare    # all 3 trackers + side-by-side

    # Launch GUI
    python main.py gui
"""

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.detector.base import build_detector
from src.tracker.sort import SORTTracker
from src.tracker.deepsort import DeepSORTTracker
from src.tracker.bytetrack import ByteTracker
from src.reid.feature_extractor import ReIDExtractor
from src.utils.io import load_config, load_sequences, save_tracks, get_image_path
from src.utils.visualization import Visualizer
from src.utils.metrics import MOTEvaluator


def build_tracker(config: dict):
    """Build tracker from config."""
    name = config["tracker"]["name"]
    if name == "SORT":
        return SORTTracker.from_config(config)
    elif name == "DeepSORT":
        return DeepSORTTracker.from_config(config)
    elif name == "ByteTrack":
        return ByteTracker.from_config(config)
    else:
        raise ValueError(f"Unknown tracker: {name}")


def cmd_track(args):
    """Run tracking on specified sequences."""
    config = load_config(args.config)

    # Override from CLI
    if args.sequence:
        config["dataset"]["sequences"] = [args.sequence]
    if args.detector:
        config["detector"]["name"] = args.detector
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    if args.split:
        config["dataset"]["split"] = args.split

    data_root = config["dataset"]["root"]
    split = config["dataset"]["split"]
    output_dir = config["output"]["dir"]
    sequences = load_sequences(data_root, split, config["dataset"].get("sequences"))

    if not sequences:
        logger.error("No sequences found.")
        return

    # Build components
    detector = build_detector(config["detector"])
    tracker = build_tracker(config)

    need_reid = config["tracker"]["name"] == "DeepSORT"
    reid = None
    if need_reid and "reid" in config:
        reid = ReIDExtractor(
            model_name=config["reid"].get("model", "osnet_x0_25"),
            weights=config["reid"].get("weights"),
            input_size=tuple(config["reid"].get("input_size", [256, 128])),
            batch_size=config["reid"].get("batch_size", 32),
        )

    evaluator = MOTEvaluator()

    for seq_info in sequences:
        logger.info(f"\nProcessing: {seq_info['name']} ({seq_info['seq_length']} frames)")
        tracker.reset()
        detector.load(seq_info["det_file"])

        # Setup visualization
        vis = None
        if config["output"].get("save_video") or config["output"].get("visualize"):
            video_path = os.path.join(output_dir, "videos", f"{seq_info['name']}.mp4") if config["output"].get("save_video") else None
            vis = Visualizer(
                im_width=seq_info["im_width"],
                im_height=seq_info["im_height"],
                save_video=config["output"].get("save_video", False),
                output_path=video_path,
                fps=seq_info["frame_rate"],
            )
            vis.start()

        tracks_per_frame = {}
        total_time = 0.0

        for frame_id in range(1, seq_info["seq_length"] + 1):
            img_path = get_image_path(seq_info["img_dir"], frame_id, seq_info.get("im_ext", ".jpg"))
            image = cv2.imread(img_path) if os.path.isfile(img_path) else None
            detections = detector.detect(frame_id, image)

            t0 = time.perf_counter()

            features = None
            if need_reid and reid is not None and image is not None and len(detections) > 0:
                bboxes = np.array([d.tlbr for d in detections])
                features = reid.extract(image, bboxes)

            if isinstance(tracker, DeepSORTTracker):
                outputs = tracker.update(detections, features=features)
            else:
                outputs = tracker.update(detections)

            total_time += time.perf_counter() - t0
            tracks_per_frame[frame_id] = outputs

            # Visualization
            if vis is not None and image is not None:
                annotated = vis.draw_frame(image, outputs, frame_id, detections)
                if config["output"].get("visualize"):
                    cv2.imshow(f"Tracking: {seq_info['name']}", annotated)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break

            if frame_id % 100 == 0:
                fps = frame_id / max(total_time, 1e-6)
                logger.info(f"  Frame {frame_id}/{seq_info['seq_length']} | FPS: {fps:.1f}")

        if vis is not None:
            vis.finish()
            if config["output"].get("visualize"):
                cv2.destroyAllWindows()

        # Save tracking results
        if config["output"].get("save_tracks", True):
            tracker_name = config["tracker"]["name"]
            track_file = os.path.join(output_dir, "tracks", tracker_name, f"{seq_info['name']}.txt")
            save_tracks(tracks_per_frame, track_file)

        # Evaluate
        fps = seq_info["seq_length"] / max(total_time, 1e-6)
        logger.info(f"  Total time: {total_time:.2f}s | FPS: {fps:.1f}")

        gt_file = seq_info["gt_file"]
        if os.path.isfile(gt_file) and config["output"].get("save_tracks", True):
            track_file = os.path.join(output_dir, "tracks", tracker_name, f"{seq_info['name']}.txt")
            metrics = evaluator.evaluate(track_file, gt_file, seq_info["name"])
            logger.info(evaluator.format_results(metrics, seq_info["name"]))


def cmd_experiment(args):
    """Run experiment scripts."""
    exp_map = {
        "1": "experiments/exp1_algorithm_compare.py",
        "2": "experiments/exp2_deepsort_ablation.py",
        "3": "experiments/exp3_detector_ablation.py",
    }

    if args.exp == "all":
        exps = ["1", "2", "3"]
    else:
        exps = [args.exp]

    for exp_id in exps:
        script = exp_map.get(exp_id)
        if script is None:
            logger.error(f"Unknown experiment: {exp_id}")
            continue
        logger.info(f"\nRunning Experiment {exp_id}: {script}")
        os.system(f"{sys.executable} {script} --data-root {args.data_root}")


_TRACKER_SPECS = {
    "SORT": dict(
        build=lambda: SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3),
        # SORT ignores low-score detections by design — filter at the tracker input.
        conf_filter=0.25,
        needs_reid=False,
    ),
    "DeepSORT": dict(
        build=lambda: DeepSORTTracker(max_age=30, n_init=3, max_cosine_distance=0.2),
        conf_filter=0.25,
        needs_reid=True,
    ),
    "ByteTrack": dict(
        # ByteTrack consumes low-score detections internally; no pre-filter.
        build=lambda: ByteTracker(max_age=30, min_hits=3,
                                  high_threshold=0.6, low_threshold=0.1),
        conf_filter=0.0,
        needs_reid=False,
    ),
}


def _run_trackers_on_video(video_path, tracker_names, yolo_name, yolo_weights,
                           reid_model, reid_weights, output_dir, show=False):
    """Run one or more trackers on a single video, writing annotated outputs.

    A single YOLO detector pass is reused across trackers (detections are
    filtered per-tracker by confidence). Panels for every tracker are composed
    into a ``comparison.mp4`` grid video when more than one tracker is active.
    """
    from src.utils.video import (
        open_video, iter_frames, make_video_writer,
        compose_grid, overlay_stats,
    )
    from src.utils.visualization import Visualizer

    video_name = Path(video_path).stem
    out_root = Path(output_dir) / "videos" / video_name
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Detector ---------------------------------------------------------
    # Use the lowest threshold any active tracker needs so a single pass
    # serves them all; per-tracker filtering happens below.
    min_conf = min(_TRACKER_SPECS[n]["conf_filter"] for n in tracker_names)
    detector = build_detector({
        "name": yolo_name,
        "weights": yolo_weights,
        "confidence_threshold": max(min_conf, 0.05),
        "nms_threshold": 0.45,
    })
    detector.load()

    # --- Re-ID (lazy; only if DeepSORT is included) -----------------------
    reid = None
    if any(_TRACKER_SPECS[n]["needs_reid"] for n in tracker_names):
        reid = ReIDExtractor(model_name=reid_model, weights=reid_weights)

    # --- Trackers + per-tracker visualisers + writers ---------------------
    cap, info = open_video(video_path)
    logger.info(f"Video: {video_path} | {info.width}x{info.height} @ {info.fps:.1f} FPS, "
                f"{info.total_frames} frames")

    trackers = {name: _TRACKER_SPECS[name]["build"]() for name in tracker_names}
    visualisers = {
        name: Visualizer(im_width=info.width, im_height=info.height, save_video=False)
        for name in tracker_names
    }
    writers = {
        name: make_video_writer(
            str(out_root / f"{name}.mp4"), info.width, info.height, info.fps
        )
        for name in tracker_names
    }
    grid_writer = None
    grid_cell = None  # (cell_w, cell_h)
    if len(tracker_names) > 1:
        # Single-row layout; shrink panels so a 3-tracker grid fits 1920px wide.
        target_total_w = 1920
        cell_w = max(320, min(info.width, target_total_w // len(tracker_names)))
        cell_h = int(round(info.height * (cell_w / info.width)))
        grid_cell = (cell_w, cell_h)
        grid_writer = make_video_writer(
            str(out_root / "comparison.mp4"),
            cell_w * len(tracker_names), cell_h, info.fps,
        )

    metrics_path = out_root / "metrics.csv"
    metrics_fh = metrics_path.open("w")
    metrics_fh.write("frame," + ",".join(
        f"{n}_tracks,{n}_fps" for n in tracker_names) + "\n")

    try:
        import time as _time
        total_start = _time.perf_counter()
        per_tracker_elapsed = {n: 0.0 for n in tracker_names}

        for frame_id, frame in iter_frames(cap):
            det_all = detector.detect(frame_id, frame)

            row = [str(frame_id)]
            panels = []
            panel_labels = []
            for name in tracker_names:
                spec = _TRACKER_SPECS[name]
                dets = [d for d in det_all if d.confidence >= spec["conf_filter"]]

                t0 = _time.perf_counter()
                if spec["needs_reid"]:
                    features = None
                    if reid is not None and dets:
                        bboxes = np.array([d.tlbr for d in dets])
                        features = reid.extract(frame, bboxes)
                    outputs = trackers[name].update(dets, features=features)
                else:
                    outputs = trackers[name].update(dets)
                elapsed = _time.perf_counter() - t0
                per_tracker_elapsed[name] += elapsed
                fps = frame_id / max(per_tracker_elapsed[name], 1e-6)

                annotated = visualisers[name].draw_frame(
                    frame, outputs, frame_id, detections=None
                )
                overlay_stats(annotated, [
                    f"{name}",
                    f"Tracks: {len(outputs)}",
                    f"FPS: {fps:.1f}",
                ], origin=(10, info.height - 90))

                writers[name].write(annotated)
                if grid_writer is not None:
                    panels.append(annotated)
                    panel_labels.append(name)

                row.extend([str(len(outputs)), f"{fps:.2f}"])

            if grid_writer is not None and panels:
                grid = compose_grid(
                    panels, panel_labels,
                    cell_size=grid_cell,
                    cols=len(panels),
                )
                grid_writer.write(grid)

            metrics_fh.write(",".join(row) + "\n")

            if show:
                display = panels[0] if panels else annotated
                cv2.imshow(f"Tracking: {video_name}", display)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            if frame_id % 50 == 0:
                total_elapsed = _time.perf_counter() - total_start
                wall_fps = frame_id / max(total_elapsed, 1e-6)
                logger.info(f"  Frame {frame_id}/{info.total_frames or '?'} "
                            f"| wall {wall_fps:.1f} FPS")
    finally:
        cap.release()
        for w in writers.values():
            w.release()
        if grid_writer is not None:
            grid_writer.release()
        metrics_fh.close()
        if show:
            cv2.destroyAllWindows()

    logger.info(f"Saved annotated videos to {out_root}")
    logger.info(f"  per-tracker:  {', '.join(f'{n}.mp4' for n in tracker_names)}")
    if len(tracker_names) > 1:
        logger.info(f"  side-by-side: comparison.mp4")
    logger.info(f"  metrics csv:  {metrics_path.name}")


def cmd_video(args):
    """Run tracking on raw video files (data/videos/*.mp4 by default)."""
    from src.utils.video import list_videos

    if args.compare:
        tracker_names = ["SORT", "DeepSORT", "ByteTrack"]
    else:
        tracker_map = {"sort": "SORT", "deepsort": "DeepSORT", "bytetrack": "ByteTrack"}
        name = tracker_map.get(args.tracker.lower())
        if name is None:
            logger.error(f"Unknown tracker: {args.tracker}. "
                         "Use sort|deepsort|bytetrack, or pass --compare.")
            return
        tracker_names = [name]

    videos = list_videos(args.input)
    if not videos:
        logger.error(f"No videos found at {args.input}")
        return
    logger.info(f"Found {len(videos)} video(s). Trackers: {tracker_names}")

    for video in videos:
        logger.info(f"\n{'='*60}\nProcessing: {video}\n{'='*60}")
        _run_trackers_on_video(
            video_path=video,
            tracker_names=tracker_names,
            yolo_name=args.detector,
            yolo_weights=args.weights,
            reid_model=args.reid_model,
            reid_weights=args.reid_weights,
            output_dir=args.output_dir,
            show=args.show,
        )


def cmd_gui(args):
    """Launch GUI."""
    from gui.app import main as gui_main
    gui_main()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Object Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Track command
    p_track = subparsers.add_parser("track", help="Run tracking on sequences")
    p_track.add_argument("--config", required=True, help="Path to config YAML")
    p_track.add_argument("--sequence", default=None, help="Specific sequence name")
    p_track.add_argument("--detector", default=None, help="Override detector")
    p_track.add_argument("--data-root", default=None,
                         help="Override dataset.root in config (e.g. data/MOT20)")
    p_track.add_argument("--split", default=None,
                         help="Override dataset.split in config (train|test)")
    p_track.set_defaults(func=cmd_track)

    # Experiment command
    p_exp = subparsers.add_parser("experiment", help="Run experiments")
    p_exp.add_argument("--exp", default="all", choices=["1", "2", "3", "all"], help="Experiment number")
    p_exp.add_argument("--data-root", default="data/MOT17",
                       help="Dataset root (e.g. data/MOT17 or data/MOT20)")
    p_exp.set_defaults(func=cmd_experiment)

    # Video command — run trackers on raw mp4 files (not MOT sequences)
    p_video = subparsers.add_parser(
        "video", help="Run tracker(s) on raw video file(s) and save annotated output"
    )
    p_video.add_argument("--input", default="data/videos",
                         help="Video file or directory (default: data/videos)")
    p_video.add_argument("--tracker", default="deepsort",
                         choices=["sort", "deepsort", "bytetrack"],
                         help="Tracker to run when --compare is not set")
    p_video.add_argument("--compare", action="store_true",
                         help="Run all three trackers and save a side-by-side comparison video")
    p_video.add_argument("--detector", default="yolov8m",
                         help="YOLO variant (yolov8n|yolov8s|yolov8m)")
    p_video.add_argument("--weights", default="models/yolov8m_mot17.pt",
                         help="YOLO weights path")
    p_video.add_argument("--reid-model", default="osnet_x0_25",
                         help="Re-ID model name (DeepSORT only)")
    p_video.add_argument("--reid-weights", default="models/osnet_x0_25_msmt17.pth",
                         help="Re-ID weights path (DeepSORT only)")
    p_video.add_argument("--output-dir", default="outputs",
                         help="Output root (videos go to {output}/videos/{stem}/)")
    p_video.add_argument("--show", action="store_true",
                         help="Display frames in a window while processing (ESC to stop)")
    p_video.set_defaults(func=cmd_video)

    # GUI command
    p_gui = subparsers.add_parser("gui", help="Launch GUI")
    p_gui.set_defaults(func=cmd_gui)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add("outputs/logs/{time:YYYY-MM-DD}.log", rotation="10 MB", level="DEBUG")

    args.func(args)


if __name__ == "__main__":
    main()
