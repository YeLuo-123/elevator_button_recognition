#!/usr/bin/env python3
"""Run YOLO26 on a video and write an annotated MP4."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "detect" / "elevator_yolo26m" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "fa476d4327584b23108fbfccce7f07e3.mp4"
DEFAULT_OUTPUT = ROOT / "runs" / "predict" / "video_results" / "fa476d4327584b23108fbfccce7f07e3_predicted.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO26 predictions on a video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video metadata: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    model = YOLO(model_path)
    class_counts: Counter[str] = Counter()
    processed = 0
    try:
        results = model.predict(
            source=str(source),
            stream=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        for processed, result in enumerate(results, start=1):
            writer.write(result.plot())
            if result.boxes is not None:
                class_counts.update(model.names[int(class_id)] for class_id in result.boxes.cls.cpu().tolist())
            if processed == 1 or processed % 50 == 0 or processed == total_frames:
                print(f"Processed {processed}/{total_frames} frames")
    finally:
        writer.release()

    print(f"\nOutput video: {output}")
    print(f"Frames: {processed}, FPS: {fps:.3f}, size: {width}x{height}")
    print("Detections by class:")
    for name, count in class_counts.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
