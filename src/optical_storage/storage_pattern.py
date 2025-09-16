"""Storage pattern definition for 5D optical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .error_correction import ErrorCorrectionScheme
from .voxel import Voxel


@dataclass
class StoragePattern:
    """Represents the 5D distribution of voxels encoding some data."""

    voxels: List[Voxel]
    grid_size: Tuple[int, int, int]
    voxel_pitch: Tuple[float, float, float]
    intensity_levels: int
    intensity_range: Tuple[float, float]
    polarization_states: int
    polarization_range: Tuple[float, float]
    bits_per_voxel: int
    encoded_bit_length: int
    data_bit_length: int
    padding_bits: int
    error_correction: ErrorCorrectionScheme = field(repr=False)
    error_correction_metadata: Dict[str, Any] = field(default_factory=dict)
    data_length_bytes: int = 0

    def capacity_bits(self) -> int:
        """Return the total raw capacity (without ECC) in bits."""
        x, y, z = self.grid_size
        return x * y * z * self.bits_per_voxel

    @property
    def voxel_count(self) -> int:
        return len(self.voxels)

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary summarising key parameters."""
        return {
            "grid_size": self.grid_size,
            "voxel_pitch": self.voxel_pitch,
            "intensity_levels": self.intensity_levels,
            "polarization_states": self.polarization_states,
            "bits_per_voxel": self.bits_per_voxel,
            "encoded_bit_length": self.encoded_bit_length,
            "data_bit_length": self.data_bit_length,
            "padding_bits": self.padding_bits,
            "error_correction": getattr(self.error_correction, "name", "unknown"),
            "error_correction_metadata": self.error_correction_metadata,
            "data_length_bytes": self.data_length_bytes,
            "voxel_count": self.voxel_count,
        }
