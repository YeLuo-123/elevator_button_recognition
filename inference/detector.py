"""Reusable YOLO detector independent of camera and robot transports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center_pixel(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


class ElevatorButtonDetector:
    def __init__(self, model_path: str | Path, **predict_options: Any) -> None:
        from ultralytics import YOLO

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model = YOLO(path)
        self.predict_options = predict_options

    def detect(self, image: Any) -> list[Detection]:
        result = self.model.predict(source=image, verbose=False, **self.predict_options)[0]
        if result.boxes is None:
            return []
        detections: list[Detection] = []
        for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu()):
            index = int(class_id.item())
            detections.append(Detection(index, str(self.model.names[index]), float(confidence), tuple(map(float, xyxy.tolist()))))
        return detections
