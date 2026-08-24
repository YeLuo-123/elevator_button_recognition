"""Hardware-neutral arm interface. Implement this protocol for the real SDK."""

from __future__ import annotations

from typing import Protocol


class ArmController(Protocol):
    def press_at(self, x: float, y: float, z: float, label: str) -> None: ...


class MockArmController:
    """Safe development adapter: logs commands without moving hardware."""

    def press_at(self, x: float, y: float, z: float, label: str) -> None:
        print(f"[DRY RUN] press {label!r} at ({x:.2f}, {y:.2f}, {z:.2f})")
