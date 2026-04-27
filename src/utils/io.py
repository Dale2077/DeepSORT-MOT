"""I/O utilities for config loading, dataset management, and result output."""

import os
import configparser

import yaml
import numpy as np
from loguru import logger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file.

    Returns
    -------
    config : dict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config_path}")
    return config


def load_sequences(data_root: str, split: str = "train", sequences: list = None) -> list:
    """Discover and load MOT17 sequence metadata.

    Parameters
    ----------
    data_root : str
        Root directory of the MOT17 dataset (e.g., "data/MOT17").
    split : str
        "train" or "test".
    sequences : list or None
        Specific sequence names. None = all sequences.

    Returns
    -------
    seq_infos : list[dict]
        List of sequence info dicts with keys:
        name, dir, det_file, gt_file, img_dir, frame_rate, seq_length,
        im_width, im_height.
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    seq_dirs = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith(".")
    ])

    if sequences:
        seq_dirs = [d for d in seq_dirs if d in sequences]

    seq_infos = []
    for seq_name in seq_dirs:
        seq_dir = os.path.join(split_dir, seq_name)
        ini_file = os.path.join(seq_dir, "seqinfo.ini")

        if not os.path.isfile(ini_file):
            logger.warning(f"Skipping {seq_name}: no seqinfo.ini")
            continue

        config = configparser.ConfigParser()
        config.read(ini_file)
        seq_section = config["Sequence"]

        info = {
            "name": seq_section.get("name", seq_name),
            "dir": seq_dir,
            "det_file": os.path.join(seq_dir, "det", "det.txt"),
            "gt_file": os.path.join(seq_dir, "gt", "gt.txt"),
            "img_dir": os.path.join(seq_dir, seq_section.get("imDir", "img1")),
            "frame_rate": int(seq_section.get("frameRate", 30)),
            "seq_length": int(seq_section.get("seqLength", 0)),
            "im_width": int(seq_section.get("imWidth", 1920)),
            "im_height": int(seq_section.get("imHeight", 1080)),
            "im_ext": seq_section.get("imExt", ".jpg"),
        }
        seq_infos.append(info)
        logger.debug(f"Found sequence: {info['name']} ({info['seq_length']} frames)")

    logger.info(f"Loaded {len(seq_infos)} sequences from {split_dir}")
    return seq_infos


def load_groundtruth(gt_file: str) -> dict:
    """Load MOT ground truth annotations.

    Parameters
    ----------
    gt_file : str
        Path to gt.txt file.

    Returns
    -------
    gt_dict : dict[int, ndarray]
        Mapping from frame_id to (N, 6) array [id, x1, y1, w, h, conf].
    """
    if not os.path.isfile(gt_file):
        return {}

    raw = np.loadtxt(gt_file, delimiter=",")
    gt_dict = {}

    for row in raw:
        frame_id = int(row[0])
        obj_id = int(row[1])
        bbox = row[2:6]  # [x, y, w, h]
        conf = row[6]
        cls = int(row[7]) if len(row) > 7 else 1
        vis = row[8] if len(row) > 8 else 1.0

        # MOT17: only consider pedestrians (class 1) with visibility > 0
        if cls != 1 or vis <= 0:
            continue

        if frame_id not in gt_dict:
            gt_dict[frame_id] = []
        gt_dict[frame_id].append([obj_id, *bbox, conf])

    return {k: np.array(v) for k, v in gt_dict.items()}


def save_tracks(tracks_per_frame: dict, output_file: str):
    """Save tracking results in MOTChallenge format.

    Parameters
    ----------
    tracks_per_frame : dict[int, ndarray]
        Mapping from frame_id to (K, 5) array [x1, y1, x2, y2, track_id].
    output_file : str
        Output file path.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = []
    for frame_id in sorted(tracks_per_frame.keys()):
        tracks = tracks_per_frame[frame_id]
        if len(tracks) == 0:
            continue
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                w, h = x2 - x1, y2 - y1
                # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
                lines.append(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    with open(output_file, "w") as f:
        f.writelines(lines)
    logger.info(f"Saved {len(lines)} track entries to {output_file}")


def get_image_path(img_dir: str, frame_id: int, ext: str = ".jpg") -> str:
    """Get image file path for a given frame.

    Parameters
    ----------
    img_dir : str
        Image directory.
    frame_id : int
        1-based frame index.
    ext : str
        Image file extension.

    Returns
    -------
    path : str
    """
    return os.path.join(img_dir, f"{frame_id:06d}{ext}")
