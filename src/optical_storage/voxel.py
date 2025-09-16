"""Data structures describing volumetric voxels used for 5D storage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voxel:
    """Single storage voxel in nanostructured glass."""

    x: int
    y: int
    z: int
    intensity: float
    polarization: float

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.z < 0:
            raise ValueError("voxel coordinates must be non-negative")
        if self.intensity < 0:
            raise ValueError("intensity must be non-negative")
        if not (0 <= self.polarization <= 3.14159265358979323846 * 2):
            # Polarization angle is typically in radians. We allow up to 2pi.
            raise ValueError("polarization angle should be within [0, 2*pi]")
