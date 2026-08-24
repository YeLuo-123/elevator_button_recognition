#!/usr/bin/env python3
"""Train YOLO26 on the elevator-button dataset.

Running this file starts training. Paths are resolved relative to this script, so
the command works from any current working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "dataset.yaml"
DEFAULT_MODEL = "yolo26m.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an elevator-button YOLO26 detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset YAML path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics base-model name or local initial weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="batch size; -1 lets Ultralytics target about 60%% GPU-memory use",
    )
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="elevator_yolo26m")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="cache images in RAM")
    parser.add_argument("--resume", action="store_true", help="resume the most recent run for --name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.expanduser().resolve()
    run_dir = ROOT / "runs" / "detect" / args.name

    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu explicitly for CPU training.")

    if args.resume:
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(checkpoint)
        model.train(resume=True)
        return

    # A bare model name is downloaded and cached automatically by Ultralytics.
    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=args.cache,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
