"""Storage pattern definition for 5D optical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .error_correction import ErrorCorrectionScheme
from .voxel import Voxel
from .error_correction import NoErrorCorrection, Hamming74, Parity8


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

    # ------------------------------------------------------------------
    # Serialization helpers
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representing this pattern.

        Note: `error_correction` is represented by its `name`. Consumers
        should map that back to an instance when calling `from_dict`.
        """
        return {
            "voxels": [
                {
                    "x": v.x,
                    "y": v.y,
                    "z": v.z,
                    "intensity": v.intensity,
                    "polarization": v.polarization,
                }
                for v in self.voxels
            ],
            "grid_size": list(self.grid_size),
            "voxel_pitch": list(self.voxel_pitch),
            "intensity_levels": int(self.intensity_levels),
            "intensity_range": list(self.intensity_range),
            "polarization_states": int(self.polarization_states),
            "polarization_range": list(self.polarization_range),
            "bits_per_voxel": int(self.bits_per_voxel),
            "encoded_bit_length": int(self.encoded_bit_length),
            "data_bit_length": int(self.data_bit_length),
            "padding_bits": int(self.padding_bits),
            "error_correction": getattr(self.error_correction, "name", "none"),
            "error_correction_metadata": dict(self.error_correction_metadata),
            "data_length_bytes": int(self.data_length_bytes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoragePattern":
        """Construct a StoragePattern from a dictionary produced by `to_dict`.

        The function maps known `error_correction` names to instances. Unknown
        names default to `NoErrorCorrection()`.
        """
        name = data.get("error_correction", "none")
        mapping = {
            "none": NoErrorCorrection(),
            "hamming74": Hamming74(),
            "parity8": Parity8(),
        }
        ecc = mapping.get(name, NoErrorCorrection())

        voxels = [
            Voxel(
                x=v["x"],
                y=v["y"],
                z=v["z"],
                intensity=v["intensity"],
                polarization=v["polarization"],
            )
            for v in data.get("voxels", [])
        ]

        return cls(
            voxels=voxels,
            grid_size=tuple(data.get("grid_size", (0, 0, 0))),
            voxel_pitch=tuple(data.get("voxel_pitch", (0.0, 0.0, 0.0))),
            intensity_levels=int(data.get("intensity_levels", 1)),
            intensity_range=tuple(data.get("intensity_range", (0.0, 1.0))),
            polarization_states=int(data.get("polarization_states", 1)),
            polarization_range=tuple(data.get("polarization_range", (0.0, 3.14159))),
            bits_per_voxel=int(data.get("bits_per_voxel", 0)),
            encoded_bit_length=int(data.get("encoded_bit_length", 0)),
            data_bit_length=int(data.get("data_bit_length", 0)),
            padding_bits=int(data.get("padding_bits", 0)),
            error_correction=ecc,
            error_correction_metadata=dict(data.get("error_correction_metadata", {})),
            data_length_bytes=int(data.get("data_length_bytes", 0)),
        )
