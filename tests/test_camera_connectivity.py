"""Hardware smoke test; opt in with RUN_HARDWARE_TESTS=1."""

import os

import cv2
import pytest


@pytest.mark.skipif(os.getenv("RUN_HARDWARE_TESTS") != "1", reason="hardware test disabled")
def test_default_camera_reads_frame() -> None:
    camera = cv2.VideoCapture(0)
    try:
        assert camera.isOpened()
        ok, frame = camera.read()
        assert ok and frame is not None
    finally:
        camera.release()
