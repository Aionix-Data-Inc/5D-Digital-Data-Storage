from __future__ import annotations

from optical_storage.error_correction import Hamming74


def make_encoded_block(d1: int, d2: int, d3: int, d4: int) -> list[int]:
    # Manually compute the Hamming(7,4) encoded block order: p1 p2 d1 p3 d2 d3 d4
    p1 = (d1 ^ d2 ^ d4) & 0x1
    p2 = (d1 ^ d3 ^ d4) & 0x1
    p3 = (d2 ^ d3 ^ d4) & 0x1
    return [p1, p2, d1, p3, d2, d3, d4]


def test_hamming_deterministic_single_and_double() -> None:
    ecc = Hamming74()
    # Choose a simple data nibble
    block = make_encoded_block(1, 0, 1, 1)
    # Single-bit flip at position 2 (zero-indexed) should be corrected
    single = list(block)
    single[2] ^= 1
    res_single = ecc.decode(single)
    assert res_single.corrected_errors == 1
    assert res_single.detected_uncorrectable == 0

    # Double-bit flip: flip positions 2 and 3
    double = list(block)
    double[2] ^= 1
    double[3] ^= 1
    res_double = ecc.decode(double)
    # For double-bit flip the decoder should either mark uncorrectable or
    # produce decoded data that differs from original
    decoded_bits = res_double.bits[:4]
    original_data = [1, 0, 1, 1]
    assert res_double.detected_uncorrectable >= 0
    assert decoded_bits != original_data or res_double.detected_uncorrectable > 0
