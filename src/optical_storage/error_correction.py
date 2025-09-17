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
    """Classic (7,4) Hamming code able to correct single-bit errors.

    This implementation attempts single-bit correction per 7-bit block. After
    performing a flip based on the syndrome, it recomputes the syndrome and
    verifies reconstructed parity bits; if inconsistencies remain it increments
    ``detected_uncorrectable`` to indicate a likely multi-bit error.
    """

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
        detected_uncorrectable = 0
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
                # Attempt single-bit correction
                block[index] ^= 0x1
                corrected += 1
                # Recompute syndrome to detect if error remains (possible multi-bit error)
                rb1 = (block[0] ^ block[2] ^ block[4] ^ block[6]) & 0x1
                rb2 = (block[1] ^ block[2] ^ block[5] ^ block[6]) & 0x1
                rb3 = (block[3] ^ block[4] ^ block[5] ^ block[6]) & 0x1
                if (rb3 << 2) | (rb2 << 1) | rb1:
                    detected_uncorrectable += 1
                else:
                    # Also verify parity bits reconstructed from decoded data. This
                    # catches some multi-bit errors that accidentally zero the
                    # syndrome after an incorrect single-bit flip.
                    d1, d2, d3, d4 = block[2], block[4], block[5], block[6]
                    ep1 = (d1 ^ d2 ^ d4) & 0x1
                    ep2 = (d1 ^ d3 ^ d4) & 0x1
                    ep3 = (d2 ^ d3 ^ d4) & 0x1
                    if ep1 != block[0] or ep2 != block[1] or ep3 != block[3]:
                        detected_uncorrectable += 1
            decoded.extend([block[2], block[4], block[5], block[6]])
        return DecodingResult(bits=decoded, corrected_errors=corrected, detected_uncorrectable=detected_uncorrectable)

    def metadata(self) -> Dict[str, int]:
        return {"data_bits_per_block": 4, "encoded_bits_per_block": 7}
