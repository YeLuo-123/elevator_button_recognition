"""Convert image pixels to robot world coordinates using planar calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class HandEyeCalibration:
    homography: np.ndarray
    fixed_z: float
    units: str = "mm"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HandEyeCalibration":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data.get("method") != "homography":
            raise ValueError("Only planar homography calibration is currently supported")
        matrix = np.asarray(data["homography"], dtype=float)
        if matrix.shape != (3, 3) or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("homography must be an invertible 3x3 matrix")
        return cls(matrix, float(data["fixed_z"]), str(data.get("units", "mm")))

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float, float]:
        projected = self.homography @ np.array([u, v, 1.0], dtype=float)
        if abs(projected[2]) < 1e-12:
            raise ValueError("Pixel maps to a point at infinity")
        x, y = projected[:2] / projected[2]
        return float(x), float(y), self.fixed_z
