#!/usr/bin/env python3
"""Check that a completed demo produced plausible results for the fixed sample list."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


MODELS = {"standard_proxy", "resnet50_sota", "convnextv2"}
CONDITIONS = {"original", "global_shape", "local_shape", "texture", "color"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(results_dir: Path) -> None:
    report = json.loads((results_dir / "report.json").read_text(encoding="utf-8"))
    sample_count = report["data"]["sample_count"]
    if sample_count not in (16, 800):
        raise ValueError(f"Unexpected sample count: {sample_count}")

    samples = read_csv(results_dir / "sample_manifest.csv")
    per_class = sample_count // 16
    if len(samples) != sample_count or set(Counter(row["category"] for row in samples).values()) != {per_class}:
        raise ValueError("Sample manifest has the wrong image or category counts")

    metrics = read_csv(results_dir / "metrics.csv")
    expected_pairs = {(model, condition) for model in MODELS for condition in CONDITIONS}
    actual_pairs = [(row["model"], row["condition"]) for row in metrics]
    if len(metrics) != 15 or set(actual_pairs) != expected_pairs or len(set(actual_pairs)) != 15:
        raise ValueError("Expected exactly 15 distinct model/condition metrics")
    for row in metrics:
        correct, count, accuracy = int(row["correct"]), int(row["count"]), float(row["accuracy"])
        if count != sample_count or not 0 <= correct <= count or not math.isclose(accuracy, correct / count, abs_tol=1e-10):
            raise ValueError(f"Invalid metric row: {row}")

    # These broad bounds only catch broken inference, not differences from the paper.
    min_original_accuracy = 0.75 if sample_count == 800 else 0.5
    baselines = {row["model"]: float(row["accuracy"]) for row in metrics if row["condition"] == "original"}
    for model, accuracy in baselines.items():
        if accuracy < min_original_accuracy:
            raise ValueError(f"Implausible original accuracy for {model}: {accuracy:.4f}")
    for row in metrics:
        expected = float(row["accuracy"]) / baselines[row["model"]]
        if not math.isclose(float(row["relative_accuracy"]), expected, abs_tol=1e-10):
            raise ValueError(f"Invalid relative accuracy: {row}")

    predictions = read_csv(results_dir / "predictions.csv")
    if Counter((row["model"], row["condition"]) for row in predictions) != Counter({pair: sample_count for pair in expected_pairs}):
        raise ValueError("Prediction count differs from the metrics")
    for row in predictions:
        probability = float(row["category_probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid category probability for {row['path']}: {probability}")

    if len(read_csv(results_dir / "category_metrics.csv")) != 15 * 16:
        raise ValueError("Expected 240 category-level metrics")
    for name in ("comparison.png", "category_comparison.png"):
        if (results_dir / name).stat().st_size <= 1000:
            raise ValueError(f"Missing or empty chart: {name}")
    if report["runtime"]["platform"].startswith("Windows") and report["runtime"].get("torch_cpu_threads") != 1:
        raise ValueError("Windows run did not use the validated single CPU thread")
    print(f"Validated {sample_count} images, 15 metrics, and both charts in {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    validate(parser.parse_args().results_dir)
