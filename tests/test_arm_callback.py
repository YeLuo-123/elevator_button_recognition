from inference.coordinate_transform import HandEyeCalibration
from inference.detector import Detection
from arm_control.callbacks import build_detection_callback


class RecordingArm:
    def __init__(self) -> None:
        self.calls = []

    def press_at(self, x, y, z, label) -> None:
        self.calls.append((x, y, z, label))


def test_callback_converts_center_and_triggers_arm() -> None:
    import numpy as np

    arm = RecordingArm()
    callback = build_detection_callback(arm, HandEyeCalibration(np.eye(3), 5.0), "3", 0.5)
    callback(Detection(0, "3", 0.9, (10, 20, 30, 40)))
    assert arm.calls == [(20.0, 30.0, 5.0, "3")]
