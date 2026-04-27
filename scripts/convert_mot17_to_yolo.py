"""
Convert the MOT17 training set into Ultralytics YOLO format.

The MOT17 ``gt/gt.txt`` schema is::

    frame, id, x, y, w, h, conf, class, visibility

For single-class pedestrian detection we keep rows where ``conf == 1`` and
``class`` is in ``{1}`` (pedestrian). Rows for class 7 (static person) /
class 2 (person on vehicle) can optionally be included via ``--include-classes``.
Boxes with very low visibility (partially occluded / out-of-frame) are filtered
by ``--min-visibility`` to avoid polluting the training signal.

Each train sequence is split temporally (80/20 by default) so that the
validation set mirrors the distribution of the train set without leaking
identities across frames of the same shot.

Output layout (Ultralytics compatible)::

    data/MOT17_yolo/
    ├── dataset.yaml
    ├── images/
    │   ├── train/<seq>_<frame>.jpg   (symlink to the original frame)
    │   └── val/...
    └── labels/
        ├── train/<seq>_<frame>.txt
        └── val/...

Usage::

    python scripts/convert_mot17_to_yolo.py \\
        --mot-root data/MOT17 \\
        --out-root data/MOT17_yolo \\
        --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from loguru import logger


DEFAULT_TRAIN_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


def _read_seqinfo(seq_dir: Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    section = parser["Sequence"]
    return {
        "name": section["name"],
        "img_dir": seq_dir / section.get("imDir", "img1"),
        "frame_rate": int(section.get("frameRate", 30)),
        "seq_length": int(section["seqLength"]),
        "im_width": int(section["imWidth"]),
        "im_height": int(section["imHeight"]),
        "im_ext": section.get("imExt", ".jpg"),
    }


def _load_gt(gt_path: Path) -> np.ndarray:
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    return np.loadtxt(gt_path, delimiter=",", dtype=np.float64)


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def convert_sequence(
    seq_dir: Path,
    out_root: Path,
    include_classes: set[int],
    min_visibility: float,
    val_ratio: float,
    copy_images: bool,
) -> tuple[int, int]:
    info = _read_seqinfo(seq_dir)
    name = info["name"]
    W, H = info["im_width"], info["im_height"]

    gt = _load_gt(seq_dir / "gt" / "gt.txt")
    # gt columns: frame, id, x, y, w, h, conf, class, visibility
    mask = (
        (gt[:, 6] == 1)
        & np.isin(gt[:, 7].astype(int), list(include_classes))
        & (gt[:, 8] >= min_visibility)
    )
    gt = gt[mask]

    # Temporal split: first (1 - val_ratio) frames -> train, rest -> val.
    seq_len = info["seq_length"]
    cutoff = int(round(seq_len * (1.0 - val_ratio)))
    cutoff = max(1, min(seq_len - 1, cutoff))

    n_train = n_val = 0
    for frame_id in range(1, seq_len + 1):
        split = "train" if frame_id <= cutoff else "val"
        fname = f"{name}_{frame_id:06d}"

        # Source frame path, with 6-digit zero-padded frame id (MOT17 default).
        src_img = info["img_dir"] / f"{frame_id:06d}{info['im_ext']}"
        if not src_img.is_file():
            # Fall back to whatever extension is present.
            candidates = list(info["img_dir"].glob(f"{frame_id:06d}.*"))
            if not candidates:
                continue
            src_img = candidates[0]

        dst_img = out_root / "images" / split / f"{fname}{info['im_ext']}"
        dst_lbl = out_root / "labels" / split / f"{fname}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        _link_or_copy(src_img, dst_img, copy_images)

        rows = gt[gt[:, 0] == frame_id]
        lines = []
        for row in rows:
            x, y, w, h = row[2], row[3], row[4], row[5]
            # Clip to image bounds, convert tlwh -> normalized cxcywh.
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            nw = bw / W
            nh = bh / H
            # Single class "person" -> id 0.
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_lbl.write_text("\n".join(lines))
        if split == "train":
            n_train += 1
        else:
            n_val += 1

    logger.info(f"{name}: train={n_train} val={n_val} (cutoff frame {cutoff}/{seq_len})")
    return n_train, n_val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mot-root", default="data/MOT17", help="Root of the MOT17 dataset.")
    ap.add_argument("--out-root", default="data/MOT17_yolo", help="Output YOLO dataset root.")
    ap.add_argument(
        "--sequences",
        nargs="*",
        default=DEFAULT_TRAIN_SEQS,
        help="Sequences to convert. Defaults to all 7 SDP train sequences.",
    )
    ap.add_argument(
        "--include-classes",
        nargs="*",
        type=int,
        default=[1],
        help="MOT17 GT class ids to keep (1=pedestrian, 2=person-on-vehicle, 7=static-person).",
    )
    ap.add_argument(
        "--min-visibility",
        type=float,
        default=0.25,
        help="Drop GT boxes with visibility below this threshold.",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of each sequence (temporally, from the tail) used as validation.",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating symlinks (use on filesystems without symlink support).",
    )
    args = ap.parse_args()

    mot_root = Path(args.mot_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not (mot_root / "train").is_dir():
        raise SystemExit(f"MOT17 train dir not found under {mot_root}")

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0
    for seq_name in args.sequences:
        seq_dir = mot_root / "train" / seq_name
        if not seq_dir.is_dir():
            logger.warning(f"Skipping missing sequence: {seq_dir}")
            continue
        n_tr, n_val = convert_sequence(
            seq_dir=seq_dir,
            out_root=out_root,
            include_classes=set(args.include_classes),
            min_visibility=args.min_visibility,
            val_ratio=args.val_ratio,
            copy_images=args.copy,
        )
        total_train += n_tr
        total_val += n_val

    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
                "nc": 1,
            },
            sort_keys=False,
        )
    )
    logger.info(f"Wrote {yaml_path}")
    logger.info(f"Totals: train={total_train} frames, val={total_val} frames")


if __name__ == "__main__":
    main()
