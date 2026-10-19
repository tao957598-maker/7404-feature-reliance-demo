#!/usr/bin/env python3
"""Runnable, explicitly approximate demonstration of Experiment I in Burgert et al.

The curated study images and the author's ResNet50-standard checkpoint are not public.
This uses a separate 16-class ImageNet subset and a clearly labelled proxy checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import platform
import random
import re
import resource
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import timm
import torch
from PIL import Image
from timm.data import ImageNetInfo

from plot_categories import make_category_plot


ROOT = Path(__file__).resolve().parent
DATA_URL = "https://zenodo.org/api/records/8027520/files/imagenet16.zip/content"
DATA_SIZE = 885_304_386
DATA_MD5 = "9f3101db921b47210d71c30f0495fa39"
TAIL_BYTES = 2 * 1024 * 1024
IMAGE_SIZE = 224
SEED = 1
CONDITIONS = ("original", "global_shape", "local_shape", "texture", "color")
MODELS = (
    ("standard_proxy", "resnet50.tv_in1k", "timm/resnet50.tv_in1k"),
    ("resnet50_sota", "resnet50.a1_in1k", "timm/resnet50.a1_in1k"),
    ("convnextv2", "convnextv2_tiny.fcmae_ft_in22k_in1k", "timm/convnextv2_tiny.fcmae_ft_in22k_in1k"),
)
USER_AGENT = "feature-reliance-experiment1-demo/1.0"


def request_bytes(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    last_error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
                if start is not None:
                    expected_range = f"bytes {start}-{end}/"
                    actual_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not actual_range.startswith(expected_range):
                        raise RuntimeError(f"Server ignored requested byte range: {actual_range}")
                data = response.read()
            if start is not None and len(data) != end - start + 1:
                raise RuntimeError(f"Short range response: {len(data)} bytes for {start}-{end}")
            return data
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocated_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_blocks * 512 for item in path.rglob("*") if item.is_file())


def load_category_map(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    mappings: dict[str, list[str]] = {}
    for match in re.finditer(r"^\s*([a-z]+)\s*=\s*\[([^]]+)\]", text, flags=re.MULTILINE):
        mappings[match.group(1)] = re.findall(r"n\d{8}", match.group(2))
    if len(mappings) != 16:
        raise ValueError(f"Expected 16 categories in {path}; found {len(mappings)}")
    return mappings


def category_indices(category_map: dict[str, list[str]]) -> dict[str, list[int]]:
    synsets = ImageNetInfo("imagenet-1k").label_names()
    synset_to_index = {wnid: index for index, wnid in enumerate(synsets)}
    if len(synset_to_index) != 1000:
        raise ValueError("timm did not provide the ImageNet-1k class order")
    indices = {
        category: [synset_to_index[wnid] for wnid in wnids if wnid in synset_to_index]
        for category, wnids in category_map.items()
    }
    if any(not values for values in indices.values()):
        raise ValueError("A category has no ImageNet-1k subclasses")
    return indices


def select_samples(infos: list[zipfile.ZipInfo], per_class: int, category_map: dict[str, list[str]]) -> list[dict]:
    category_for_wnid = {wnid: category for category, wnids in category_map.items() for wnid in wnids}
    candidates: dict[str, list[tuple[int, zipfile.ZipInfo]]] = defaultdict(list)
    for index, info in enumerate(infos):
        parts = info.filename.split("/")
        if len(parts) == 4 and parts[1] == "val" and parts[3].lower().endswith((".jpeg", ".jpg", ".png")):
            candidates[parts[2]].append((index, info))
    if len(candidates) != 16:
        raise ValueError(f"Expected 16 validation folders; found {len(candidates)}")
    samples = []
    for wnid, entries in candidates.items():
        if wnid not in category_for_wnid:
            raise ValueError(f"Unmapped validation class: {wnid}")
        if len(entries) < per_class:
            raise ValueError(f"Only {len(entries)} validation images for {wnid}")
        for archive_index, info in sorted(entries, key=lambda pair: pair[1].filename)[:per_class]:
            samples.append({"path": info.filename, "wnid": wnid, "category": category_for_wnid[wnid], "archive_index": archive_index})
    samples.sort(key=lambda item: (item["category"], item["path"]))
    if len(samples) != 16 * per_class or len({s["category"] for s in samples}) != 16:
        raise ValueError("Validation sample count or category coverage is incorrect")
    return samples


def prepare_sparse_zip(per_class: int, category_map: dict[str, list[str]], data_dir: Path) -> tuple[Path, list[dict]]:
    """Retrieve the ZIP directory and only the selected image byte ranges."""
    data_dir.mkdir(parents=True, exist_ok=True)
    sparse_path = data_dir / f"imagenet16_n{per_class}_sparse.zip"
    marker = data_dir / f"imagenet16_n{per_class}_sparse.complete.json"
    if sparse_path.exists() and marker.exists():
        try:
            with zipfile.ZipFile(sparse_path) as archive:
                samples = select_samples(archive.infolist(), per_class, category_map)
                for sample in samples:
                    archive.read(sample["path"])  # checks CRC for every selected entry
            print(f"Using verified sparse ZIP: {sparse_path}", flush=True)
            return sparse_path, samples
        except (OSError, ValueError, zipfile.BadZipFile):
            sparse_path.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)

    print("Downloading ZIP index and selected image ranges...", flush=True)
    tail_start = DATA_SIZE - TAIL_BYTES
    tail = request_bytes(DATA_URL, tail_start, DATA_SIZE - 1)
    with zipfile.ZipFile(io.BytesIO(tail)) as archive:
        infos = archive.infolist()
    samples = select_samples(infos, per_class, category_map)
    correction = DATA_SIZE - len(tail)
    ranges = []
    for wnid in sorted({sample["wnid"] for sample in samples}):
        chosen = [sample["archive_index"] for sample in samples if sample["wnid"] == wnid]
        first = min(infos[index].header_offset for index in chosen) + correction
        last = max(infos[index + 1].header_offset for index in chosen) + correction - 1
        ranges.append((wnid, first, last))

    temporary = sparse_path.with_suffix(".part")
    try:
        with temporary.open("wb") as stream:
            stream.truncate(DATA_SIZE)  # sparse file: holes do not occupy physical disk space
            stream.seek(tail_start)
            stream.write(tail)
            for number, (wnid, start, end) in enumerate(ranges, 1):
                print(f"  class {number:02d}/16 {wnid}: {(end - start + 1) / 1e6:.1f} MB", flush=True)
                stream.seek(start)
                stream.write(request_bytes(DATA_URL, start, end))
        with zipfile.ZipFile(temporary) as archive:
            for sample in samples:
                archive.read(sample["path"])
        os.replace(temporary, sparse_path)
        marker.write_text(json.dumps({"url": DATA_URL, "size": DATA_SIZE, "mode": "selected-byte-ranges", "per_class": per_class, "selected_entries_crc_checked": len(samples)}, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    return sparse_path, samples


def prepare_supplied_zip(path: Path, per_class: int, category_map: dict[str, list[str]]) -> tuple[Path, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != DATA_SIZE:
        raise ValueError(f"Expected Zenodo ZIP size {DATA_SIZE}, got {path.stat().st_size}")
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DATA_MD5:
        raise ValueError("Supplied ZIP failed the Zenodo MD5 check")
    with zipfile.ZipFile(path) as archive:
        samples = select_samples(archive.infolist(), per_class, category_map)
        for sample in samples:
            archive.read(sample["path"])
    return path, samples


def transform_image(image: np.ndarray, condition: str, sample_index: int) -> np.ndarray:
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    if condition == "original":
        result = resized
    elif condition in ("global_shape", "local_shape"):
        grid = 3 if condition == "global_shape" else 6
        random.seed(SEED + sample_index * 31 + grid)
        np.random.seed(SEED + sample_index * 31 + grid)
        result = A.RandomGridShuffle(grid=(grid, grid), p=1.0)(image=resized)["image"]
    elif condition == "texture":
        result = cv2.bilateralFilter(resized, d=12, sigmaColor=170, sigmaSpace=75)
    elif condition == "color":
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    if result.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or result.dtype != np.uint8:
        raise ValueError(f"Invalid transformed image: {result.shape}, {result.dtype}")
    return result


def prepare_images(archive_path: Path, samples: list[dict], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_key = hashlib.sha256("\n".join(sample["path"] for sample in samples).encode()).hexdigest()[:12]
    array_path = cache_dir / f"images_{len(samples)}_{manifest_key}.npy"
    marker = array_path.with_suffix(".complete")
    if array_path.exists() and marker.exists():
        array = np.load(array_path, mmap_mode="r")
        if array.shape == (len(CONDITIONS), len(samples), IMAGE_SIZE, IMAGE_SIZE, 3):
            print(f"Using processed image cache: {array_path}", flush=True)
            return array_path
    temporary = array_path.with_suffix(".part.npy")
    transformed = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8,
        shape=(len(CONDITIONS), len(samples), IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    started = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        for index, sample in enumerate(samples):
            with archive.open(sample["path"]) as stream:
                with Image.open(stream) as pil_image:
                    image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
            for condition_index, condition in enumerate(CONDITIONS):
                transformed[condition_index, index] = transform_image(image, condition, index)
            if (index + 1) % 25 == 0 or index + 1 == len(samples):
                print(f"  transformed {index + 1}/{len(samples)} images", flush=True)
    transformed.flush()
    del transformed
    os.replace(temporary, array_path)
    marker.write_text(f"seed={SEED}; seconds={time.monotonic() - started:.2f}\n")
    return array_path


def model_weights(label: str, repository: str, weights_dir: Path) -> tuple[Path, dict]:
    weights_dir.mkdir(parents=True, exist_ok=True)
    locked = json.loads((ROOT / "config" / "model_weights.json").read_text(encoding="utf-8"))[label]
    revision = locked["revision"]
    expected_sha256 = locked["sha256"]
    if locked["repository"] != repository or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError(f"Invalid locked model identity for {label}")
    path = weights_dir / f"{label}_{revision}.safetensors"
    if path.exists() and sha256_file(path) != expected_sha256:
        print(f"Cached checkpoint checksum mismatch; downloading {label} again", flush=True)
        path.unlink()
    if not path.exists():
        url = f"https://huggingface.co/{repository}/resolve/{revision}/model.safetensors"
        print(f"Downloading model {repository} @ {revision[:12]}...", flush=True)
        temporary = path.with_suffix(".part")
        try:
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=180) as response, temporary.open("wb") as stream:
                        expected_size = int(response.headers.get("Content-Length", "0"))
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                    if expected_size and temporary.stat().st_size != expected_size:
                        raise RuntimeError(f"Short checkpoint download: {temporary.stat().st_size} != {expected_size}")
                    if sha256_file(temporary) != expected_sha256:
                        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {repository} @ {revision}")
                    os.replace(temporary, path)
                    break
                except (OSError, urllib.error.URLError, RuntimeError):
                    temporary.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        finally:
            temporary.unlink(missing_ok=True)
    return path, {"repository": repository, "revision": revision, "url": f"https://huggingface.co/{repository}", "sha256": expected_sha256}


def decide_device(requested: str) -> torch.device:
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but is unavailable; use --device auto or cpu")
    return torch.device("mps" if (requested == "mps" or (requested == "auto" and available)) else "cpu")


def score_probs(probabilities: np.ndarray, category: str, indices: dict[str, list[int]]) -> np.ndarray:
    # Paper's rule: sum ImageNet subclass softmax probabilities, then threshold >0.5.
    return probabilities[:, indices[category]].sum(axis=1)


def run_model(model: torch.nn.Module, images_path: Path, samples: list[dict], indices: dict[str, list[int]], device: torch.device, batch_size: int) -> tuple[list[dict], list[dict]]:
    model = model.to(device).eval()
    images = np.load(images_path, mmap_mode="r")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rows: list[dict] = []
    predictions: list[dict] = []
    with torch.inference_mode():
        for condition_index, condition in enumerate(CONDITIONS):
            correct = 0
            for start in range(0, len(samples), batch_size):
                end = min(start + batch_size, len(samples))
                batch = np.array(images[condition_index, start:end], copy=True)
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
                tensor = (tensor / 255.0 - mean) / std
                probabilities = model(tensor).softmax(dim=1).cpu().numpy()
                if probabilities.shape != (end - start, 1000):
                    raise ValueError(f"Expected 1000 ImageNet logits, got {probabilities.shape}")
                for offset, sample in enumerate(samples[start:end]):
                    probability = float(score_probs(probabilities[offset:offset + 1], sample["category"], indices)[0])
                    is_correct = probability > 0.5
                    correct += is_correct
                    predictions.append({"condition": condition, "path": sample["path"], "category": sample["category"], "category_probability": probability, "correct": int(is_correct)})
            rows.append({"condition": condition, "correct": correct, "count": len(samples), "accuracy": correct / len(samples)})
            print(f"    {condition}: {correct}/{len(samples)} = {correct / len(samples):.4f}", flush=True)
    return rows, predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict], path: Path) -> None:
    names = ["Original", "Global shape", "Local shape", "Texture", "Color"]
    display_names = {
        "standard_proxy": "ResNet50-standard (proxy)",
        "resnet50_sota": "ResNet50 A1",
        "convnextv2": "ConvNeXtV2 Tiny",
    }
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for label, _, _ in MODELS:
        own = [row for row in rows if row["model"] == label]
        if len(own) != len(CONDITIONS):
            raise ValueError(f"Missing results for {label}")
        axes[0].plot(x, [row["accuracy"] for row in own], marker="o", label=display_names[label])
        axes[1].plot(x, [row["relative_accuracy"] for row in own], marker="o", label=display_names[label])
    for axis, title, ylabel in ((axes[0], "Absolute accuracy", "Accuracy"), (axes[1], "Relative accuracy", "Suppressed / original")):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Experiment I method demo — public images; standard model is a proxy", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate, runnable Experiment I demo (not paper-number reproduction).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="1 image per class (16 total)")
    group.add_argument("--half", action="store_true", help="25 images per class (400 total)")
    group.add_argument("--full", action="store_true", help="50 images per class (800 total; also the default)")
    parser.add_argument("--data-zip", type=Path, help="Locally downloaded, MD5-verified Zenodo imagenet16.zip")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Download / image cache directory")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "data" / "weights", help="Model checkpoint cache")
    parser.add_argument("--output", type=Path, help="Output directory (default: results/demo_N)")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 8:
        parser.error("--batch-size must be between 1 and 8 on this small-memory demo")
    per_class = 1 if args.smoke else 25 if args.half else 50
    output = args.output or ROOT / "results" / f"demo_{16 * per_class}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(1)
    category_map = load_category_map(ROOT / "config" / "imagenet16_category_map.txt")
    indices = category_indices(category_map)
    if args.data_zip:
        archive_path, samples = prepare_supplied_zip(args.data_zip, per_class, category_map)
        data_mode = "full-zip-md5-verified"
    else:
        archive_path, samples = prepare_sparse_zip(per_class, category_map, args.data_dir)
        data_mode = "sparse-zip-selected-entries-crc-verified"
    images_path = prepare_images(archive_path, samples, args.data_dir / "processed")
    write_csv(output / "sample_manifest.csv", [{key: sample[key] for key in ("path", "wnid", "category")} for sample in samples])
    device = decide_device(args.device)
    print(f"Device: {device}; images: {len(samples)}; batch size: {args.batch_size}", flush=True)

    all_results: list[dict] = []
    all_predictions: list[dict] = []
    model_meta = {}
    for label, model_name, repository in MODELS:
        checkpoint, provenance = model_weights(label, repository, args.weights_dir)
        print(f"Evaluating {label} ({model_name})...", flush=True)
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        state = safetensors.torch.load_file(str(checkpoint), device="cpu")
        model.load_state_dict(state, strict=True)
        del state
        model_started = time.monotonic()
        actual_device = device
        try:
            result_rows, predictions = run_model(model, images_path, samples, indices, device, args.batch_size)
        except RuntimeError as exc:
            if args.device != "auto" or device.type != "mps":
                raise
            print(f"MPS failed ({exc}); retrying {label} on CPU", file=sys.stderr, flush=True)
            model = model.to("cpu")
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
            actual_device = torch.device("cpu")
            result_rows, predictions = run_model(model, images_path, samples, indices, actual_device, args.batch_size)
        baseline = result_rows[0]["accuracy"]
        for row in result_rows:
            row["model"] = label
            row["relative_accuracy"] = row["accuracy"] / baseline if baseline else None
            all_results.append(row)
        for row in predictions:
            row["model"] = label
            all_predictions.append(row)
        provenance.update({"timm_model": model_name, "device": str(actual_device), "seconds": round(time.monotonic() - model_started, 2), "standard_proxy": label == "standard_proxy"})
        model_meta[label] = provenance
        del model
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    ordered_results = [{key: row[key] for key in ("model", "condition", "correct", "count", "accuracy", "relative_accuracy")} for row in all_results]
    ordered_predictions = [{key: row[key] for key in ("model", "condition", "path", "category", "category_probability", "correct")} for row in all_predictions]
    write_csv(output / "metrics.csv", ordered_results)
    write_csv(output / "predictions.csv", ordered_predictions)
    plot_results(ordered_results, output / "comparison.png")
    make_category_plot(output)
    report = {
        "disclaimer": "Method demo: public ImageNet16 subset; ResNet50-standard is a proxy; not a reproduction of the paper's Figure 3 values.",
        "paper": "Burgert et al., NeurIPS 2025, Experiment I",
        "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf",
        "source_code": "https://github.com/tomburgert/feature-reliance",
        "data": {"url": DATA_URL, "zenodo_md5": DATA_MD5, "mode": data_mode, "sample_count": len(samples), "per_class": per_class, "seed": SEED, "selection": "first N filenames in each validation WNID folder, lexicographically sorted"},
        "method": {"conditions": list(CONDITIONS), "image_size": IMAGE_SIZE, "patch_shuffle_grids": [3, 6], "bilateral": {"d": 12, "sigma_color": 170, "sigma_space": 75}, "decision": "sum ImageNet-1k subclass softmax probabilities > 0.5"},
        "models": model_meta,
        "runtime": {
            "platform": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "timm": timm.__version__,
            "wall_seconds": round(time.monotonic() - started, 2),
            "peak_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024), 1),
            "venv_disk_mib": round(allocated_bytes(ROOT / ".venv") / (1024**2), 1),
            "data_disk_mib": round(allocated_bytes(args.data_dir) / (1024**2), 1),
        },
        "results": ordered_results,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Done in {report['runtime']['wall_seconds']:.1f}s. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
