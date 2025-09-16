"""Utilities for converting between bit-level representations."""

from __future__ import annotations

from typing import Iterator, List, Sequence


def bytes_to_bits(data: bytes) -> List[int]:
    """Convert a byte string to a list of bits (MSB first)."""
    bits: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 0x1)
    return bits


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    """Convert a sequence of bits (MSB first) into bytes."""
    out = bytearray()
    for chunk in chunk_bits(bits, 8, pad=True):
        value = 0
        for bit in chunk:
            value = (value << 1) | (bit & 0x1)
        out.append(value)
    return bytes(out)


def chunk_bits(bits: Sequence[int], size: int, pad: bool = False, pad_value: int = 0) -> Iterator[List[int]]:
    """Yield fixed-size chunks from *bits*.

    Args:
        bits: The source sequence of bits.
        size: The chunk size to yield.
        pad: When ``True`` the final chunk is padded to ``size`` with ``pad_value``.
        pad_value: The value to use for padding bits.

    Yields:
        A list of bits with length ``size`` (last chunk may be shorter when ``pad``
        is ``False``).
    """

    if size <= 0:
        raise ValueError("size must be greater than zero")

    accumulator: List[int] = []
    for bit in bits:
        accumulator.append(bit & 0x1)
        if len(accumulator) == size:
            yield accumulator
            accumulator = []

    if accumulator:
        if pad:
            accumulator.extend([pad_value & 0x1] * (size - len(accumulator)))
            yield accumulator
        elif not pad:
            yield accumulator


def bits_to_int(bits: Sequence[int]) -> int:
    """Interpret *bits* (MSB first) as an integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | (bit & 0x1)
    return value


def int_to_bits(value: int, width: int) -> List[int]:
    """Return the ``width``-bit big-endian representation of *value*."""
    if width < 0:
        raise ValueError("width must be non-negative")
    if value < 0:
        raise ValueError("value must be non-negative")

    bits = [(value >> shift) & 0x1 for shift in range(width - 1, -1, -1)] if width else []
    if value >= (1 << width) and width > 0:
        raise ValueError("value does not fit into specified width")
    return bits
