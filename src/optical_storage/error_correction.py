"""Error correction primitives for optical 5D data storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .bit_utils import chunk_bits


@dataclass
class DecodingResult:
    """Result returned by :class:`ErrorCorrectionScheme.decode`."""

    bits: List[int]
    corrected_errors: int = 0
    detected_uncorrectable: int = 0


class ErrorCorrectionScheme:
    """Interface for forward-error-correction schemes."""

    name: str = "abstract"

    def encode(self, bits: Sequence[int]) -> List[int]:
        """Encode payload bits and return a new bit sequence."""
        raise NotImplementedError

    def decode(self, bits: Sequence[int]) -> DecodingResult:
        """Decode an encoded bit stream."""
        raise NotImplementedError

    def metadata(self) -> Dict[str, int]:
        """Return human-readable metadata about the scheme."""
        return {}


class NoErrorCorrection(ErrorCorrectionScheme):
    """Passthrough encoder/decoder."""

    name = "none"

    def encode(self, bits: Sequence[int]) -> List[int]:
        return [bit & 0x1 for bit in bits]

    def decode(self, bits: Sequence[int]) -> DecodingResult:
        return DecodingResult(bits=[bit & 0x1 for bit in bits])


class Hamming74(ErrorCorrectionScheme):
    """Classic (7,4) Hamming code able to correct single-bit errors."""

    name = "hamming74"

    def encode(self, bits: Sequence[int]) -> List[int]:
        encoded: List[int] = []
        for chunk in chunk_bits(bits, 4, pad=True):
            d1, d2, d3, d4 = chunk
            p1 = (d1 ^ d2 ^ d4) & 0x1
            p2 = (d1 ^ d3 ^ d4) & 0x1
            p3 = (d2 ^ d3 ^ d4) & 0x1
            encoded.extend([p1, p2, d1, p3, d2, d3, d4])
        return encoded

    def decode(self, bits: Sequence[int]) -> DecodingResult:
        corrected = 0
        decoded: List[int] = []
        for block in chunk_bits(bits, 7, pad=True):
            if len(block) < 7:
                block.extend([0] * (7 - len(block)))
            b1, b2, b3, b4, b5, b6, b7 = block
            s1 = (b1 ^ b3 ^ b5 ^ b7) & 0x1
            s2 = (b2 ^ b3 ^ b6 ^ b7) & 0x1
            s3 = (b4 ^ b5 ^ b6 ^ b7) & 0x1
            error_position = (s3 << 2) | (s2 << 1) | s1
            if error_position:
                index = error_position - 1
                block[index] ^= 0x1
                corrected += 1
            decoded.extend([block[2], block[4], block[5], block[6]])
        return DecodingResult(bits=decoded, corrected_errors=corrected)

    def metadata(self) -> Dict[str, int]:
        return {"data_bits_per_block": 4, "encoded_bits_per_block": 7}
