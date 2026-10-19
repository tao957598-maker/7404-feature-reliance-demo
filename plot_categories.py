#!/usr/bin/env python3
"""Plot Experiment I accuracy by image category from saved predictions."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


MODELS = (
    ("standard_proxy", "ResNet50-standard (proxy)", "#3049a5", "*", -0.19, 76),
    ("resnet50_sota", "ResNet50 A1", "#3f874a", "s", 0.0, 37),
    ("convnextv2", "ConvNeXtV2 Tiny", "#e98d3b", "D", 0.19, 38),
)
CONDITIONS = (
    ("global_shape", "(a) Global shape · 3×3 shuffle"),
    ("local_shape", "(b) Local shape · 6×6 shuffle"),
    ("texture", "(c) Texture · bilateral filter"),
    ("color", "(d) Color · grayscale"),
)


def make_category_plot(results_dir: Path) -> None:
    """Write category_metrics.csv and category_comparison.png using existing CSVs."""
    with (results_dir / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    with (results_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = {(row["model"], row["condition"]): row for row in csv.DictReader(stream)}
    if not predictions or not metrics:
        raise ValueError("Missing predictions or aggregate metrics")

    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in predictions:
        key = (row["model"], row["condition"], row["category"])
        counts[key][0] += int(row["correct"])
        counts[key][1] += 1
    categories = sorted({key[2] for key in counts})
    if len(categories) != 16:
        raise ValueError(f"Expected 16 categories, found {len(categories)}")
    for model, *_ in MODELS:
        for condition in ("original", *(item[0] for item in CONDITIONS)):
            expected = metrics.get((model, condition))
            groups = [counts.get((model, condition, category)) for category in categories]
            if expected is None or any(group is None for group in groups):
                raise ValueError(f"Missing category data for {model}, {condition}")
            if sum(group[0] for group in groups) != int(expected["correct"]) or sum(group[1] for group in groups) != int(expected["count"]):
                raise ValueError(f"Category results disagree with metrics.csv: {model}, {condition}")

    rows = [
        {"model": model, "condition": condition, "category": category,
         "correct": counts[(model, condition, category)][0],
         "count": counts[(model, condition, category)][1],
         "accuracy": counts[(model, condition, category)][0] / counts[(model, condition, category)][1]}
        for model, *_ in MODELS
        for condition in ("original", *(item[0] for item in CONDITIONS))
        for category in categories
    ]
    with (results_dir / "category_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 4, figsize=(19, 8.7), sharey=True)
    y = list(range(len(categories)))
    for axis, (condition, title) in zip(axes, CONDITIONS):
        for model, display, color, marker, offset, size in MODELS:
            values = [counts[(model, condition, category)][0] / counts[(model, condition, category)][1] for category in categories]
            axis.scatter(values, [position + offset for position in y], color=color, marker=marker, s=size, zorder=3, label=display)
            axis.axvline(float(metrics[(model, condition)]["accuracy"]), color=color, linewidth=1.5, alpha=0.75, zorder=2)
        axis.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
        axis.set_xlim(-0.025, 1.025)
        axis.set_xticks([i / 5 for i in range(6)])
        axis.set_xlabel("Fraction of correct decisions", fontsize=10)
        axis.grid(axis="x", alpha=0.22)
        axis.grid(axis="y", alpha=0.12)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=9)
    axes[0].set_yticks(y, [category.capitalize() for category in categories])
    axes[0].set_ylabel("Image category", fontsize=11)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=10)
    for axis in axes[1:]:
        axis.tick_params(axis="y", left=False, labelleft=False)

    handles = [Line2D([], [], color=color, marker=marker, linestyle="-", markersize=8, linewidth=1.5, label=display)
               for _, display, color, marker, _, _ in MODELS]
    fig.suptitle("Accuracy by image category — Experiment I method demo", fontsize=15, y=0.975)
    fig.text(0.5, 0.93, "Dots: accuracy within each category  ·  Vertical lines: overall accuracy  ·  CNNs only; no human data",
             ha="center", fontsize=10)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.91), frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.82, wspace=0.13)
    fig.savefig(results_dir / "category_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?", default=ROOT / "results" / "demo_400")
    args = parser.parse_args()
    make_category_plot(args.results_dir)
    print(f"Saved {args.results_dir / 'category_comparison.png'}")
