"""Optical reader that reconstructs data from voxel measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .bit_utils import bits_to_bytes, int_to_bits
from .error_correction import DecodingResult
from .storage_pattern import StoragePattern
from .voxel import Voxel


@dataclass
class ReadResult:
    """Result returned after reading a :class:`StoragePattern`."""

    data: bytes
    corrected_errors: int
    detected_uncorrectable: int
    voxels_used: int
    raw_bitstream: List[int] = field(repr=False)
    decoded_payload_bits: List[int] = field(repr=False)


class LaserReader:
    """Recover the original bytes from a 5D voxel lattice."""

    def __init__(self, pattern: StoragePattern) -> None:
        self.pattern = pattern
        self._validate_pattern()
        self.bits_per_intensity = self._bits_for_levels(pattern.intensity_levels)
        self.bits_per_polarization = self._bits_for_levels(pattern.polarization_states)
        self.bits_per_voxel = self.bits_per_intensity + self.bits_per_polarization

    def read(self, voxels: Sequence[Voxel] | None = None) -> ReadResult:
        voxels = list(voxels) if voxels is not None else list(self.pattern.voxels)
        if not voxels and self.pattern.encoded_bit_length:
            raise ValueError("No voxels provided for decoding")

        required_bits = self.pattern.encoded_bit_length + self.pattern.padding_bits
        collected_bits: List[int] = []
        voxels_used = 0
        for voxel in voxels:
            if required_bits and len(collected_bits) >= required_bits:
                break
            intensity_level = self._physical_to_level(
                voxel.intensity, self.pattern.intensity_levels, self.pattern.intensity_range
            )
            polarization_level = self._physical_to_level(
                voxel.polarization, self.pattern.polarization_states, self.pattern.polarization_range
            )
            if self.bits_per_intensity:
                collected_bits.extend(int_to_bits(intensity_level, self.bits_per_intensity))
            if self.bits_per_polarization:
                collected_bits.extend(int_to_bits(polarization_level, self.bits_per_polarization))
            voxels_used += 1

        if required_bits and len(collected_bits) < required_bits:
            raise ValueError("Insufficient voxel data to reconstruct payload")

        collected_bits = collected_bits[:required_bits] if required_bits else collected_bits
        if self.pattern.padding_bits:
            collected_bits = collected_bits[: -self.pattern.padding_bits]

        ecc_result: DecodingResult = self.pattern.error_correction.decode(collected_bits)
        payload_bits = ecc_result.bits[: self.pattern.data_bit_length]
        data = bits_to_bytes(payload_bits)
        data = data[: self.pattern.data_length_bytes]

        return ReadResult(
            data=data,
            corrected_errors=ecc_result.corrected_errors,
            detected_uncorrectable=ecc_result.detected_uncorrectable,
            voxels_used=voxels_used,
            raw_bitstream=collected_bits,
            decoded_payload_bits=payload_bits,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _validate_pattern(self) -> None:
        if self.pattern.bits_per_voxel <= 0 and self.pattern.encoded_bit_length:
            raise ValueError("Pattern does not contain encodable information")

    @staticmethod
    def _bits_for_levels(levels: int) -> int:
        if levels <= 0:
            raise ValueError("levels must be positive")
        if levels & (levels - 1):
            raise ValueError("levels must be a power of two for binary encoding")
        return 0 if levels == 1 else levels.bit_length() - 1

    @staticmethod
    def _physical_to_level(value: float, levels: int, value_range: tuple[float, float]) -> int:
        min_val, max_val = value_range
        if levels == 1:
            return 0
        value_clamped = max(min(value, max_val), min_val)
        step = (max_val - min_val) / float(levels - 1)
        if step == 0:
            return 0
        normalized = (value_clamped - min_val) / step
        level = int(round(normalized))
        return max(0, min(level, levels - 1))
