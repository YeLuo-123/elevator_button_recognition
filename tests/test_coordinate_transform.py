from pathlib import Path

import pytest

from inference.coordinate_transform import HandEyeCalibration


def test_identity_example_calibration() -> None:
    config = Path(__file__).parents[1] / "configs/hand_eye_calibration.yaml"
    calibration = HandEyeCalibration.from_yaml(config)
    assert calibration.pixel_to_world(12.5, 30.0) == pytest.approx((12.5, 30.0, 0.0))
