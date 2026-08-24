#!/usr/bin/env python3
"""Camera inference entry point with an optional robot-action callback."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from arm_control import MockArmController, build_detection_callback
from inference import ElevatorButtonDetector, HandEyeCalibration

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--calibration", type=Path, default=ROOT / "configs/hand_eye_calibration.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--target", help="only trigger this class label")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--trigger-arm", action="store_true", help="enable callback (mock adapter by default)")
    args = parser.parse_args()

    detector = ElevatorButtonDetector(args.model, conf=args.conf)
    callback = build_detection_callback(MockArmController(), HandEyeCalibration.from_yaml(args.calibration), args.target, args.conf)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            detections = detector.detect(frame)
            if args.trigger_arm:
                for detection in detections:
                    callback(detection)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
