"""
Fine-tune YOLOv8m on MOT17 for single-class pedestrian detection.

Hardware target
---------------
NVIDIA RTX 5090 (32 GB) — the defaults (imgsz=1280, batch=16, AMP, RAM cache)
are sized for this card. On smaller GPUs drop ``--batch`` or ``--imgsz``.

Training recipe
---------------
- Backbone: ``yolov8m.pt`` pretrained on COCO; the "person" class transfers
  well so we start from COCO rather than training from scratch.
- Image size 1280 (MOT17 is 1920x1080, using 640 drops a lot of small
  pedestrians). Rectangular batches keep aspect ratio close to native.
- SGD + cosine LR + 3-epoch warmup — the recipe the YOLOv8 authors use for
  COCO and the one that consistently produces strong mAP on MOT17.
- Mosaic + MixUp + copy-paste during the bulk of training; ``close_mosaic=10``
  turns mosaic off for the final 10 epochs so the model aligns to the native
  distribution before validation is measured.
- 50 epochs with ``patience=30`` early stopping: an earlier 80-epoch run
  peaked at epoch 29 and plateaued, so 50 is enough headroom without
  wasting compute on the flat tail.

Target metrics
--------------
With these settings on MOT17 val (20% tail split per sequence) expect
roughly AP50 ≥ 0.90 / AP50-95 ≥ 0.60 for the pedestrian class — in line
with or above the YOLOX-m / YOLOv7 baselines commonly reported in the
MOT tracking literature.

Usage
-----
::

    # 1. Convert MOT17 GT into YOLO format.
    python scripts/convert_mot17_to_yolo.py \\
        --mot-root data/MOT17 --out-root data/MOT17_yolo

    # 2. Train.
    python scripts/train_yolov8_mot17.py \\
        --data data/MOT17_yolo/dataset.yaml \\
        --weights models/yolov8m.pt \\
        --epochs 50 --batch 16 --imgsz 1280 --device 0

The best weights are written to
``runs/detect/yolov8m_mot17/weights/best.pt``; the script also copies them
to ``models/yolov8m_mot17.pt`` for use by the tracker configs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data/MOT17_yolo/dataset.yaml",
                    help="Path to the Ultralytics dataset YAML produced by convert_mot17_to_yolo.py.")
    ap.add_argument("--weights", default="models/yolov8m.pt",
                    help="Initial weights. COCO-pretrained yolov8m.pt is recommended.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size. Use -1 for Ultralytics auto-batch (~60%% VRAM).")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Training image size. 1280 preserves small pedestrians; 960 is a faster fallback.")
    ap.add_argument("--device", default="0", help="CUDA device id, e.g. '0' or '0,1'.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="yolov8m_mot17")
    ap.add_argument("--patience", type=int, default=30,
                    help="Early-stopping patience in epochs.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the last checkpoint of {project}/{name}.")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                    help="Image cache location. 'ram' is fastest; 'false' disables caching.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-to", default="models/yolov8m_mot17.pt",
                    help="Copy the best checkpoint here after training (set to '' to skip).")
    ap.add_argument("--no-val", action="store_true",
                    help="Skip the final val pass (useful for throughput-only runs).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - import-time guard
        raise SystemExit(
            "Ultralytics is required. Install with `pip install ultralytics>=8.1.0`."
        ) from e

    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)} — {torch.cuda.device_count()} device(s)")
    else:
        logger.warning("CUDA not available. YOLOv8m training on CPU is not practical.")

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise SystemExit(
            f"Dataset YAML not found: {data_yaml}. Run scripts/convert_mot17_to_yolo.py first."
        )

    cache = False if args.cache == "false" else args.cache

    model = YOLO(args.weights)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        seed=args.seed,
        patience=args.patience,
        cache=cache,
        # --- optimizer / schedule -----------------------------------------
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        # --- loss weights (Ultralytics defaults tuned for single-class) ---
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # --- regularisation / augmentation --------------------------------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        # --- runtime ------------------------------------------------------
        amp=True,
        rect=False,
        multi_scale=False,
        plots=True,
        val=not args.no_val,
        save=True,
        save_period=-1,
        verbose=True,
    )

    logger.info("Launching training with the following key arguments:")
    for k in ("data", "epochs", "batch", "imgsz", "device", "optimizer", "lr0", "cos_lr",
              "mosaic", "mixup", "close_mosaic", "amp", "cache"):
        logger.info(f"  {k}: {train_kwargs[k]}")

    # Guardrails for settings that historically produced misleading checkpoints.
    if train_kwargs["imgsz"] < 1280:
        logger.warning(
            f"imgsz={train_kwargs['imgsz']} < 1280: small pedestrians in MOT17 "
            "(1920x1080 native) will be downsampled heavily; recall will drop "
            "~5-10 points. Use --imgsz 1280 unless you are only probing throughput."
        )
    if not train_kwargs["val"]:
        logger.warning(
            "val=False: per-epoch validation disabled, so results.csv metrics "
            "will be 0 for all non-final epochs and early stopping cannot engage."
        )
    if args.epochs < 30:
        logger.warning(
            f"epochs={args.epochs} < 30: previous runs peaked around epoch 29, "
            "so going below 30 risks stopping before the model has converged."
        )

    results = model.train(**train_kwargs)

    if results is not None and hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)
    else:
        save_dir = Path(args.project) / args.name

    best = save_dir / "weights" / "best.pt"
    if best.is_file():
        logger.info(f"Training finished. Best weights: {best}")
        if args.export_to:
            export = Path(args.export_to)
            export.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, export)
            logger.info(f"Exported best weights to {export}")
    else:
        logger.warning(f"Could not find best.pt under {best.parent}")

    # Ultralytics stores final val metrics on results.results_dict.
    if results is not None and hasattr(results, "results_dict"):
        logger.info("Final metrics:")
        for k, v in results.results_dict.items():
            logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
