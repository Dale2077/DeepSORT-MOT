"""Generate a single Markdown report from training and experiment outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-results",
        default="runs/detect/runs/feasibility/yolov8m_mot17_probe/results.csv",
        help="Path to Ultralytics results.csv",
    )
    parser.add_argument(
        "--training-weights",
        default="models/yolov8m_mot17.pt",
        help="Path to the exported detector checkpoint",
    )
    parser.add_argument(
        "--exp1-summary",
        default="outputs/exp1_algorithm_compare/summary.txt",
        help="Experiment 1 summary.txt path",
    )
    parser.add_argument(
        "--exp2-summary",
        default="outputs/exp2_deepsort_ablation/summary.txt",
        help="Experiment 2 summary.txt path",
    )
    parser.add_argument(
        "--exp3-summary",
        default="outputs/exp3_detector_ablation/summary.txt",
        help="Experiment 3 summary.txt path",
    )
    parser.add_argument(
        "--output",
        default="outputs/training_experiment_report.md",
        help="Output Markdown path",
    )
    return parser.parse_args()


def read_text(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_file():
        return f"[Missing] {path}"
    return path.read_text(encoding="utf-8")


def load_training_metrics(csv_path: str) -> dict:
    path = Path(csv_path)
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}

    return rows[-1]


def render_training_section(metrics: dict, weights_path: str) -> str:
    if not metrics:
        return "## Training\n\nTraining metrics file was not found.\n"

    rows = [
        ("Epoch", metrics.get("epoch", "")),
        ("Time (s)", metrics.get("time", "")),
        ("Train Box Loss", metrics.get("train/box_loss", "")),
        ("Train Cls Loss", metrics.get("train/cls_loss", "")),
        ("Train DFL Loss", metrics.get("train/dfl_loss", "")),
        ("Precision", metrics.get("metrics/precision(B)", "")),
        ("Recall", metrics.get("metrics/recall(B)", "")),
        ("mAP50", metrics.get("metrics/mAP50(B)", "")),
        ("mAP50-95", metrics.get("metrics/mAP50-95(B)", "")),
        ("Val Box Loss", metrics.get("val/box_loss", "")),
        ("Val Cls Loss", metrics.get("val/cls_loss", "")),
        ("Val DFL Loss", metrics.get("val/dfl_loss", "")),
        ("Weights", weights_path),
    ]

    table = ["| Metric | Value |", "| --- | --- |"]
    table.extend(f"| {key} | {value} |" for key, value in rows)
    return "## Training\n\n" + "\n".join(table) + "\n"


def render_summary_section(title: str, content: str) -> str:
    return f"## {title}\n\n```text\n{content.rstrip()}\n```\n"


def main() -> None:
    args = parse_args()

    training_metrics = load_training_metrics(args.training_results)
    exp1 = read_text(args.exp1_summary)
    exp2 = read_text(args.exp2_summary)
    exp3 = read_text(args.exp3_summary)

    report = [
        "# Training And Experiment Report",
        "",
        render_training_section(training_metrics, args.training_weights),
        render_summary_section("Experiment 1", exp1),
        render_summary_section("Experiment 2", exp2),
        render_summary_section("Experiment 3", exp3),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
