#!/usr/bin/env python3
"""Optimized second-round training for the elevator-button YOLO26 model.

This is a new experiment and does not overwrite the original training run.
Compared with train_detector.py, it uses gentler geometry augmentation, disables
horizontal flips for direction-sensitive door icons, and applies a fresh AdamW
learning-rate schedule.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "dataset.yaml"
DEFAULT_MODEL = "yolo26m.pt"
DEFAULT_PROJECT = ROOT / "runs" / "detect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch optimized YOLO26 elevator-button training")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset YAML path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics base-model name or local initial weights")
    parser.add_argument("--epochs", type=int, default=200, help="maximum training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=4, help="batch size for a 10 GB RTX 3080")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="elevator_yolo26m_optimized")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="cache images in RAM")
    parser.add_argument("--resume", action="store_true", help="resume this experiment from last.pt")
    parser.add_argument(
        "--from-best",
        action="store_true",
        help="fine-tune from the original run's best.pt instead of yolo26m.pt",
    )
    return parser.parse_args()


def audit_dataset(data_yaml: Path) -> None:
    """Fail early on invalid paths and print the long-tail class distribution."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names", [])
    num_classes = int(config.get("nc", len(names)))
    if len(names) != num_classes:
        raise ValueError(f"data.yaml declares nc={num_classes}, but contains {len(names)} names")

    dataset_root = (data_yaml.parent / config.get("path", ".")).resolve()
    print("Dataset audit:")
    for split_key in ("train", "val"):
        relative_images = config.get(split_key)
        if not relative_images:
            raise ValueError(f"Missing '{split_key}' path in {data_yaml}")
        image_dir = (dataset_root / relative_images).resolve()
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split_key} images or labels: {image_dir}, {label_dir}")

        image_count = sum(path.is_file() for path in image_dir.iterdir())
        counts: Counter[int] = Counter()
        for label_path in label_dir.glob("*.txt"):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid label row: {label_path}:{line_number}")
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if not 0 <= class_id < num_classes or any(not 0 <= value <= 1 for value in coordinates):
                    raise ValueError(f"Out-of-range label: {label_path}:{line_number}")
                counts[class_id] += 1

        rare_classes = sum(0 < counts[class_id] < 10 for class_id in range(num_classes))
        absent_classes = sum(counts[class_id] == 0 for class_id in range(num_classes))
        print(
            f"  {split_key}: images={image_count}, instances={sum(counts.values())}, "
            f"present_classes={len(counts)}/{num_classes}, classes_lt_10={rare_classes}, "
            f"absent_classes={absent_classes}"
        )


def main() -> None:
    args = parse_args()
    data = args.data.expanduser().resolve()
    initial_weights: str | Path = args.model
    run_dir = DEFAULT_PROJECT / args.name

    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu explicitly for CPU training.")

    if args.resume:
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        print(f"Resuming interrupted run: {checkpoint}")
        YOLO(checkpoint).train(resume=True)
        return

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Choose another --name, or use --resume for an interrupted run."
        )

    if args.from_best:
        initial_weights = ROOT / "models" / "best.pt"
        if not initial_weights.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {initial_weights}")

    audit_dataset(data)
    print(f"Initial weights: {initial_weights}")
    print(f"Output directory: {run_dir}")
    print("Starting optimized training...")

    model = YOLO(initial_weights)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=args.cache,
        cos_lr=True,
        # Direction-sensitive open/close icons must not be mirrored.
        fliplr=0.0,
        flipud=0.0,
        # The Roboflow export is already augmented 3x, so use gentler online augmentation.
        mosaic=0.5,
        close_mosaic=20,
        translate=0.05,
        scale=0.3,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
