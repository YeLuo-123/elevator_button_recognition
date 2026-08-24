"""Robot-arm adapters and detection callbacks."""

from .base import ArmController, MockArmController
from .callbacks import build_detection_callback

__all__ = ["ArmController", "MockArmController", "build_detection_callback"]
