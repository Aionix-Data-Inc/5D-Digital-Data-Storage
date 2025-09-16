"""Laser writing pipeline for 5D optical storage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from .bit_utils import bits_to_int, chunk_bits, bytes_to_bits
from .error_correction import ErrorCorrectionScheme, Hamming74
from .storage_pattern import StoragePattern
from .voxel import Voxel


@dataclass
class LaserWriter:
    """Translate bytes into five-dimensional voxel patterns."""

    grid_size: Tuple[int, int, int] = (64, 64, 32)
    voxel_pitch: Tuple[float, float, float] = (5.0, 5.0, 20.0)  # micrometres
    intensity_levels: int = 16
    polarization_states: int = 8
    intensity_range: Tuple[float, float] = (0.15, 1.0)
    polarization_range: Tuple[float, float] = (0.0, math.pi)
    error_correction: ErrorCorrectionScheme = field(default_factory=Hamming74)

    def __post_init__(self) -> None:
        self._validate_parameters()
        self.bits_per_intensity = self._bits_for_levels(self.intensity_levels)
        self.bits_per_polarization = self._bits_for_levels(self.polarization_states)
        self.bits_per_voxel = self.bits_per_intensity + self.bits_per_polarization
        if self.bits_per_voxel == 0:
            raise ValueError("At least one dimension must encode information.")

    def write(self, data: bytes) -> StoragePattern:
        """Encode *data* into a :class:`StoragePattern`."""
        payload_bits = bytes_to_bits(data)
        encoded_bits = self.error_correction.encode(payload_bits)
        encoded_bit_length = len(encoded_bits)

        max_voxels = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        required_voxels = math.ceil(encoded_bit_length / self.bits_per_voxel) if encoded_bit_length else 0
        if required_voxels > max_voxels:
            raise ValueError(
                "Data does not fit inside the configured lattice: "
                f"requires {required_voxels} voxels, only {max_voxels} available"
            )

        padding_bits = required_voxels * self.bits_per_voxel - encoded_bit_length if encoded_bit_length else 0
        padded_bits = list(encoded_bits)
        padded_bits.extend([0] * padding_bits)

        voxels: List[Voxel] = []
        for index, chunk in enumerate(chunk_bits(padded_bits, self.bits_per_voxel, pad=False)):
            if not chunk:
                continue
            x, y, z = self._index_to_coordinates(index)
            intensity_bits = chunk[: self.bits_per_intensity]
            polarization_bits = chunk[self.bits_per_intensity :]
            intensity_level = bits_to_int(intensity_bits) if self.bits_per_intensity else 0
            polarization_level = bits_to_int(polarization_bits) if self.bits_per_polarization else 0
            intensity_value = self._level_to_physical(intensity_level, self.intensity_levels, self.intensity_range)
            polarization_value = self._level_to_physical(
                polarization_level, self.polarization_states, self.polarization_range
            )
            voxels.append(Voxel(x=x, y=y, z=z, intensity=intensity_value, polarization=polarization_value))

        pattern = StoragePattern(
            voxels=voxels,
            grid_size=self.grid_size,
            voxel_pitch=self.voxel_pitch,
            intensity_levels=self.intensity_levels,
            intensity_range=self.intensity_range,
            polarization_states=self.polarization_states,
            polarization_range=self.polarization_range,
            bits_per_voxel=self.bits_per_voxel,
            encoded_bit_length=encoded_bit_length,
            data_bit_length=len(payload_bits),
            padding_bits=padding_bits,
            error_correction=self.error_correction,
            error_correction_metadata=self.error_correction.metadata(),
            data_length_bytes=len(data),
        )
        return pattern

    # ------------------------------------------------------------------
    # Helper methods

    def _validate_parameters(self) -> None:
        gx, gy, gz = self.grid_size
        if gx <= 0 or gy <= 0 or gz <= 0:
            raise ValueError("grid_size components must be positive")
        if any(p <= 0 for p in self.voxel_pitch):
            raise ValueError("voxel_pitch values must be positive")
        if self.intensity_levels <= 0 or self.polarization_states <= 0:
            raise ValueError("quantisation levels must be positive")
        if self.intensity_range[0] >= self.intensity_range[1]:
            raise ValueError("invalid intensity_range")
        if self.polarization_range[0] >= self.polarization_range[1]:
            raise ValueError("invalid polarization_range")

    @staticmethod
    def _bits_for_levels(levels: int) -> int:
        if levels <= 0:
            raise ValueError("levels must be positive")
        if levels & (levels - 1):
            raise ValueError("levels must be a power of two for binary encoding")
        return int(math.log2(levels))

    def _index_to_coordinates(self, index: int) -> Tuple[int, int, int]:
        gx, gy, gz = self.grid_size
        plane_size = gx * gy
        z = index // plane_size
        if z >= gz:
            raise ValueError("Index exceeds lattice depth")
        remainder = index % plane_size
        y = remainder // gx
        x = remainder % gx
        return x, y, z

    @staticmethod
    def _level_to_physical(level: int, levels: int, value_range: Tuple[float, float]) -> float:
        min_val, max_val = value_range
        if levels == 1:
            return (min_val + max_val) / 2.0
        step = (max_val - min_val) / float(levels - 1)
        level = max(0, min(level, levels - 1))
        return min_val + level * step
