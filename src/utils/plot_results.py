"""
Data chart visualization for MOT experiment results.

Generates publication-quality figures:
  - Algorithm comparison bar charts & radar plots
  - Ablation line/bar charts
  - Per-sequence heatmaps
  - Metric correlation scatter plots
  - Track count over time curves

All plots use a consistent academic style with English labels.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

from loguru import logger

# ── Style configuration ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Tracker color scheme
TRACKER_COLORS = {
    "SORT": "#4C72B0",
    "DeepSORT": "#DD8452",
    "ByteTrack": "#55A868",
}

METRIC_LABELS = {
    "MOTA": "MOTA (%)",
    "MOTP": "MOTP (%)",
    "IDF1": "IDF1 (%)",
    "FP": "False Positives",
    "FN": "False Negatives",
    "IDSW": "ID Switches",
    "FPS": "FPS",
    "MT": "Mostly Tracked",
    "ML": "Mostly Lost",
}


def _save_fig(fig, output_path: str):
    """Save figure and close."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved chart: {output_path}")


# =====================================================================
#  Experiment 1: Algorithm Comparison Charts
# =====================================================================

def plot_algorithm_comparison_bars(
    results: Dict[str, dict],
    output_dir: str,
    metrics: List[str] = None,
):
    """Bar chart comparing trackers across key metrics.

    Parameters
    ----------
    results : dict
        {tracker_name: {"MOTA": float, "IDF1": float, ...}}
    output_dir : str
    metrics : list[str] or None
        Metrics to plot. Default: MOTA, IDF1, IDSW, FPS.
    """
    if metrics is None:
        metrics = ["MOTA", "IDF1", "IDSW", "FPS"]

    trackers = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [results[t].get(metric, 0) for t in trackers]
        colors = [TRACKER_COLORS.get(t, "#999999") for t in trackers]

        bars = ax.bar(trackers, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}" if isinstance(val, float) else str(int(val))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontweight="bold", fontsize=10)

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))

        # Highlight best
        if metric in ("MOTA", "IDF1", "FPS", "MT"):
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor("#333333")
        bars[best_idx].set_linewidth(2)

    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "algorithm_comparison_bars.png"))


def plot_algorithm_radar(
    results: Dict[str, dict],
    output_dir: str,
    metrics: List[str] = None,
):
    """Radar (spider) chart comparing trackers.

    Parameters
    ----------
    results : dict
        {tracker_name: {"MOTA": float, ...}}
    output_dir : str
    metrics : list[str]
    """
    if metrics is None:
        metrics = ["MOTA", "IDF1", "FPS"]

    trackers = list(results.keys())
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for tracker in trackers:
        values = [results[tracker].get(m, 0) for m in metrics]
        # Normalize to [0, 1] for display
        max_vals = [max(results[t].get(m, 0) for t in trackers) for m in metrics]
        norm_values = [v / (mv if mv > 0 else 1) for v, mv in zip(values, max_vals)]
        norm_values += norm_values[:1]

        color = TRACKER_COLORS.get(tracker, "#999999")
        ax.plot(angles, norm_values, "o-", linewidth=2, label=tracker, color=color)
        ax.fill(angles, norm_values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Algorithm Radar Chart", fontsize=14, fontweight="bold", pad=20)

    _save_fig(fig, os.path.join(output_dir, "algorithm_radar.png"))


def plot_per_sequence_comparison(
    all_results: Dict[str, List[dict]],
    output_dir: str,
    metric: str = "MOTA",
):
    """Grouped bar chart showing per-sequence metric for each tracker.

    Parameters
    ----------
    all_results : dict
        {tracker_name: [{"sequence": str, "MOTA": float, ...}, ...]}
    output_dir : str
    metric : str
    """
    trackers = list(all_results.keys())
    # Get unique sequences preserving order
    sequences = []
    for r_list in all_results.values():
        for r in r_list:
            s = r.get("sequence", "")
            if s and s not in sequences:
                sequences.append(s)

    if not sequences:
        return

    x = np.arange(len(sequences))
    width = 0.8 / len(trackers)

    fig, ax = plt.subplots(figsize=(max(10, len(sequences) * 1.5), 6))

    for i, tracker in enumerate(trackers):
        seq_map = {r["sequence"]: r.get(metric, 0) for r in all_results[tracker]}
        values = [seq_map.get(s, 0) for s in sequences]
        color = TRACKER_COLORS.get(tracker, "#999999")
        ax.bar(x + i * width - 0.4 + width / 2, values,
               width=width, label=tracker, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("MOT17-", "") for s in sequences], rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Per-Sequence {metric}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"per_sequence_{metric.lower()}.png"))


# =====================================================================
#  Experiment 2: Ablation Charts
# =====================================================================

def plot_ablation_line(
    results: List[dict],
    x_key: str,
    y_keys: List[str],
    output_dir: str,
    title: str = "Ablation Study",
    filename: str = "ablation.png",
):
    """Line chart for parameter ablation.

    Parameters
    ----------
    results : list[dict]
        [{x_key: val, y_key1: val, y_key2: val, ...}, ...]
    x_key : str
        Key for x-axis variable.
    y_keys : list[str]
        Keys for y-axis metrics (one line per metric).
    output_dir : str
    title : str
    filename : str
    """
    if not results:
        return

    x_values = [r[x_key] for r in results]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    n_metrics = len(y_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, y_key, color in zip(axes, y_keys, colors):
        y_values = [r.get(y_key, 0) for r in results]
        ax.plot(x_values, y_values, "o-", color=color, linewidth=2, markersize=8, markeredgecolor="white")

        # Annotate points
        for xv, yv in zip(x_values, y_values):
            fmt = f"{yv:.1f}" if isinstance(yv, float) else str(yv)
            ax.annotate(fmt, (xv, yv), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

        ax.set_xlabel(x_key)
        ax.set_ylabel(METRIC_LABELS.get(y_key, y_key))
        ax.set_title(METRIC_LABELS.get(y_key, y_key))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, filename))


def plot_ablation_grouped_bar(
    results: List[dict],
    x_key: str,
    y_keys: List[str],
    output_dir: str,
    title: str = "Ablation Study",
    filename: str = "ablation_bar.png",
):
    """Grouped bar chart for ablation comparison (e.g., Re-ID on/off).

    Parameters
    ----------
    results : list[dict]
    x_key : str
    y_keys : list[str]
    output_dir : str
    title : str
    filename : str
    """
    if not results:
        return

    labels = [str(r[x_key]) for r in results]
    x = np.arange(len(labels))
    width = 0.8 / len(y_keys)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))

    for i, (y_key, color) in enumerate(zip(y_keys, colors)):
        values = [r.get(y_key, 0) for r in results]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values,
                      width=width, label=METRIC_LABELS.get(y_key, y_key),
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            fmt = f"{val:.1f}" if isinstance(val, float) else str(int(val))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_key)
    ax.legend()
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, filename))


# =====================================================================
#  Experiment 3: Detector Ablation Charts
# =====================================================================

def plot_detector_comparison(
    results: Dict[str, dict],
    output_dir: str,
):
    """Combined chart for detector ablation: bars + table.

    Parameters
    ----------
    results : dict
        {detector_label: {"MOTA": float, "IDF1": float, "IDSW": int, "FPS": float, ...}}
    output_dir : str
    """
    detectors = list(results.keys())
    metrics_top = ["MOTA", "IDF1"]
    metrics_bottom = ["IDSW", "FP", "FN"]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(detectors) * 2.5), 10))

    # Top: accuracy metrics
    x = np.arange(len(detectors))
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]
    for i, m in enumerate(metrics_top):
        vals = [results[d].get(m, 0) for d in detectors]
        bars = axes[0].bar(x + i * width - width / 2, vals, width=width,
                           label=METRIC_LABELS.get(m, m), color=colors[i],
                           edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(detectors)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Detector Ablation — Accuracy Metrics", fontsize=13, fontweight="bold")
    axes[0].legend()

    # Bottom: error metrics (stacked)
    error_colors = ["#C44E52", "#937860", "#8C8C8C"]
    bottom = np.zeros(len(detectors))
    for m, color in zip(metrics_bottom, error_colors):
        vals = np.array([results[d].get(m, 0) for d in detectors], dtype=float)
        axes[1].bar(detectors, vals, bottom=bottom, label=METRIC_LABELS.get(m, m),
                    color=color, edgecolor="white", linewidth=0.5)
        bottom += vals

    axes[1].set_ylabel("Count")
    axes[1].set_title("Detector Ablation — Error Metrics", fontsize=13, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, "detector_ablation.png"))


def plot_fps_comparison(
    results: Dict[str, dict],
    output_dir: str,
    title: str = "FPS Comparison",
    filename: str = "fps_comparison.png",
):
    """Horizontal bar chart comparing FPS.

    Parameters
    ----------
    results : dict
        {label: {"FPS": float, ...}}
    output_dir : str
    """
    labels = list(results.keys())
    fps_vals = [results[l].get("FPS", 0) for l in labels]
    colors = [TRACKER_COLORS.get(l, "#4C72B0") for l in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.8)))
    bars = ax.barh(labels, fps_vals, color=colors, edgecolor="white", linewidth=0.8, height=0.5)

    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_width() + max(fps_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontweight="bold", fontsize=10)

    ax.set_xlabel("Frames Per Second (FPS)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, filename))


# =====================================================================
#  General-purpose Charts
# =====================================================================

def plot_track_count_over_time(
    tracks_per_frame: dict,
    output_dir: str,
    seq_name: str = "",
    gt_per_frame: dict = None,
):
    """Line chart showing number of active tracks per frame.

    Parameters
    ----------
    tracks_per_frame : dict[int, ndarray]
        {frame_id: (K, 5) track array}
    output_dir : str
    seq_name : str
    gt_per_frame : dict or None
        Ground truth counts per frame.
    """
    frames = sorted(tracks_per_frame.keys())
    counts = [len(tracks_per_frame[f]) for f in frames]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frames, counts, color="#4C72B0", linewidth=1.5, alpha=0.8, label="Tracked")
    ax.fill_between(frames, counts, alpha=0.1, color="#4C72B0")

    if gt_per_frame:
        gt_counts = [len(gt_per_frame.get(f, [])) for f in frames]
        ax.plot(frames, gt_counts, color="#DD8452", linewidth=1.5, alpha=0.8, linestyle="--", label="Ground Truth")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of Targets")
    ax.set_title(f"Track Count Over Time {f'({seq_name})' if seq_name else ''}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    fname = f"track_count_{seq_name}.png" if seq_name else "track_count.png"
    _save_fig(fig, os.path.join(output_dir, fname))


def plot_metric_heatmap(
    all_results: Dict[str, List[dict]],
    output_dir: str,
    metrics: List[str] = None,
    filename: str = "metric_heatmap.png",
):
    """Heatmap of metrics across trackers and sequences.

    Parameters
    ----------
    all_results : dict
        {tracker_name: [{sequence: str, MOTA: float, ...}, ...]}
    output_dir : str
    metrics : list[str]
    filename : str
    """
    if metrics is None:
        metrics = ["MOTA", "IDF1", "IDSW"]

    trackers = list(all_results.keys())
    sequences = []
    for r_list in all_results.values():
        for r in r_list:
            s = r.get("sequence", "")
            if s and s not in sequences:
                sequences.append(s)

    if not sequences or not trackers:
        return

    for metric in metrics:
        data = np.zeros((len(trackers), len(sequences)))
        for i, t in enumerate(trackers):
            seq_map = {r["sequence"]: r.get(metric, 0) for r in all_results[t]}
            for j, s in enumerate(sequences):
                data[i, j] = seq_map.get(s, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(sequences) * 1.2), max(3, len(trackers) * 1.2)))

        # Choose colormap based on metric direction
        if metric in ("MOTA", "IDF1", "FPS", "MT", "MOTP"):
            cmap = "YlGn"
        else:
            cmap = "YlOrRd"

        im = ax.imshow(data, cmap=cmap, aspect="auto")

        # Annotate cells
        for i in range(len(trackers)):
            for j in range(len(sequences)):
                val = data[i, j]
                fmt = f"{val:.1f}" if isinstance(val, float) and val != int(val) else f"{int(val)}"
                text_color = "white" if val > (data.max() + data.min()) / 2 else "black"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=9, color=text_color)

        ax.set_xticks(range(len(sequences)))
        ax.set_xticklabels([s.replace("MOT17-", "") for s in sequences], rotation=45, ha="right")
        ax.set_yticks(range(len(trackers)))
        ax.set_yticklabels(trackers)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} Heatmap",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, os.path.join(output_dir, f"heatmap_{metric.lower()}.png"))


def plot_error_pie(
    results: dict,
    output_dir: str,
    tracker_name: str = "",
    filename: str = "error_breakdown.png",
):
    """Pie chart showing error breakdown (FP / FN / IDSW).

    Parameters
    ----------
    results : dict
        {"FP": int, "FN": int, "IDSW": int}
    output_dir : str
    tracker_name : str
    filename : str
    """
    labels = ["FP (False Positives)", "FN (False Negatives)", "IDSW (ID Switches)"]
    values = [results.get("FP", 0), results.get("FN", 0), results.get("IDSW", 0)]
    colors = ["#C44E52", "#937860", "#4C72B0"]

    if sum(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75, textprops={"fontsize": 11}
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    title = f"Error Breakdown"
    if tracker_name:
        title += f" ({tracker_name})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, os.path.join(output_dir, filename))


def generate_all_exp1_charts(
    all_results: Dict[str, List[dict]],
    summary: Dict[str, dict],
    output_dir: str,
):
    """Generate all charts for Experiment 1.

    Parameters
    ----------
    all_results : dict
        {tracker: [per-sequence metrics]}
    summary : dict
        {tracker: averaged metrics}
    output_dir : str
    """
    plot_dir = os.path.join(output_dir, "plots")

    plot_algorithm_comparison_bars(summary, plot_dir)
    plot_algorithm_radar(summary, plot_dir, metrics=["MOTA", "IDF1", "FPS"])
    plot_fps_comparison(summary, plot_dir)
    plot_per_sequence_comparison(all_results, plot_dir, metric="MOTA")
    plot_per_sequence_comparison(all_results, plot_dir, metric="IDSW")
    plot_metric_heatmap(all_results, plot_dir, metrics=["MOTA", "IDSW"])

    for tracker_name, metrics in summary.items():
        plot_error_pie(metrics, plot_dir, tracker_name,
                       filename=f"error_pie_{tracker_name.lower()}.png")

    logger.info(f"Generated {7 + len(summary)} charts for Experiment 1 in {plot_dir}")


def generate_all_exp2_charts(
    ablation_a: List[dict],
    ablation_b: List[dict],
    ablation_c: List[dict],
    ablation_d: List[dict],
    output_dir: str,
):
    """Generate all charts for Experiment 2.

    Parameters
    ----------
    ablation_a : list - max_age results
    ablation_b : list - Re-ID on/off results
    ablation_c : list - max_cosine_distance results
    ablation_d : list - nn_budget results
    output_dir : str
    """
    plot_dir = os.path.join(output_dir, "plots")

    if ablation_a:
        plot_ablation_line(ablation_a, "max_age", ["MOTA", "IDF1", "IDSW"],
                           plot_dir, title="Ablation A: Effect of max_age",
                           filename="ablation_a_max_age.png")

    if ablation_b:
        plot_ablation_grouped_bar(ablation_b, "Re-ID", ["MOTA", "IDF1"],
                                  plot_dir, title="Ablation B: Re-ID Feature On/Off",
                                  filename="ablation_b_reid.png")

    if ablation_c:
        plot_ablation_line(ablation_c, "cos_dist", ["MOTA", "IDF1", "IDSW"],
                           plot_dir, title="Ablation C: Effect of max_cosine_distance",
                           filename="ablation_c_cos_dist.png")

    if ablation_d:
        plot_ablation_line(ablation_d, "nn_budget", ["MOTA", "IDF1", "IDSW"],
                           plot_dir, title="Ablation D: Effect of nn_budget",
                           filename="ablation_d_nn_budget.png")

    count = sum(1 for x in [ablation_a, ablation_b, ablation_c, ablation_d] if x)
    logger.info(f"Generated {count} chart(s) for Experiment 2 in {plot_dir}")


def generate_all_exp3_charts(
    results: Dict[str, dict],
    output_dir: str,
):
    """Generate all charts for Experiment 3.

    Parameters
    ----------
    results : dict
        {detector_label: averaged metrics}
    output_dir : str
    """
    plot_dir = os.path.join(output_dir, "plots")

    plot_detector_comparison(results, plot_dir)
    plot_fps_comparison(results, plot_dir,
                        title="Detector FPS Comparison",
                        filename="detector_fps.png")

    logger.info(f"Generated 2 charts for Experiment 3 in {plot_dir}")
