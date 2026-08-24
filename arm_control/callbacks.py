"""Callbacks joining detections to robot actions."""

from __future__ import annotations

from collections.abc import Callable

from inference.coordinate_transform import HandEyeCalibration
from inference.detector import Detection

from .base import ArmController


def build_detection_callback(
    arm: ArmController,
    calibration: HandEyeCalibration,
    target_label: str | None = None,
    minimum_confidence: float = 0.5,
) -> Callable[[Detection], None]:
    def on_detection(detection: Detection) -> None:
        if detection.confidence < minimum_confidence:
            return
        if target_label is not None and detection.label != target_label:
            return
        x, y, z = calibration.pixel_to_world(*detection.center_pixel)
        arm.press_at(x, y, z, detection.label)

    return on_detection
