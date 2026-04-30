"""Noise injection utilities for optical data storage simulations."""

from __future__ import annotations

import random
from typing import List

from .storage_pattern import StoragePattern
from .voxel import Voxel


def apply_gaussian_noise(
    pattern: StoragePattern,
    intensity_std: float,
    polarization_std: float,
    seed: int | None = None,
) -> List[Voxel]:
    """Return a new list of voxels with Gaussian measurement noise applied."""

    rng = random.Random(seed)
    noisy_voxels: List[Voxel] = []
    min_i, max_i = pattern.intensity_range
    min_p, max_p = pattern.polarization_range

    for voxel in pattern.voxels:
        noisy_intensity = rng.gauss(voxel.intensity, intensity_std)
        noisy_polarization = rng.gauss(voxel.polarization, polarization_std)
        noisy_intensity = max(min(noisy_intensity, max_i), min_i)
        noisy_polarization = max(min(noisy_polarization, max_p), min_p)
        noisy_voxels.append(
            Voxel(
                x=voxel.x,
                y=voxel.y,
                z=voxel.z,
                intensity=noisy_intensity,
                polarization=noisy_polarization,
            )
        )
    return noisy_voxels
