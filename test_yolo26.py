#!/usr/bin/env python3
"""Run the trained elevator-button detector and save visualized results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "detect" / "elevator_yolo26m" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "dianti(1)" / "dianti"
DEFAULT_OUTPUT = ROOT / "runs" / "predict" / "dianti_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test and visualize the trained YOLO26 model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="trained weights path")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="image directory or image path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="result directory")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    return parser.parse_args()


def make_contact_sheet(images: list[Path], output: Path, columns: int = 4) -> None:
    """Build a compact overview from the annotated images."""
    if not images:
        return
    thumb_size = (400, 300)
    caption_height = 28
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(images):
        with Image.open(path) as image:
            tile = ImageOps.contain(image.convert("RGB"), thumb_size)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + caption_height)
        x_centered = x + (thumb_size[0] - tile.width) // 2
        y_centered = y + (thumb_size[1] - tile.height) // 2
        sheet.paste(tile, (x_centered, y_centered))
        draw.text((x + 8, y + thumb_size[1] + 6), path.name, fill="black")

    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Test source not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    results = model.predict(
        source=str(source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=True,
    )

    annotated_paths: list[Path] = []
    csv_rows: list[list[object]] = []
    total_detections = 0

    for result in results:
        source_path = Path(result.path)
        annotated_path = output / source_path.name
        result.save(filename=str(annotated_path))
        annotated_paths.append(annotated_path)

        boxes = result.boxes
        if boxes is None:
            continue
        for xyxy, confidence, class_id in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
            class_index = int(class_id.item())
            x1, y1, x2, y2 = (round(float(value), 2) for value in xyxy.tolist())
            csv_rows.append(
                [source_path.name, model.names[class_index], class_index, round(float(confidence), 4), x1, y1, x2, y2]
            )
            total_detections += 1

    csv_path = output / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "class_name", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(csv_rows)

    contact_sheet = output / "contact_sheet.jpg"
    make_contact_sheet(annotated_paths, contact_sheet)

    print(f"\nProcessed images: {len(annotated_paths)}")
    print(f"Detections: {total_detections}")
    print(f"Annotated images: {output}")
    print(f"Overview: {contact_sheet}")
    print(f"Detection table: {csv_path}")


if __name__ == "__main__":
    main()
