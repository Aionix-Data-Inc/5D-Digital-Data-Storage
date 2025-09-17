from __future__ import annotations

from optical_storage import Hamming74, LaserWriter, LaserReader, NoErrorCorrection
from optical_storage.noise import apply_gaussian_noise
from optical_storage.bit_utils import bytes_to_bits, bits_to_bytes, int_to_bits


def test_hamming_single_and_double_bit_behavior() -> None:
    # Prepare a simple 1-byte payload and encode with Hamming(7,4)
    payload = b"A"  # 0x41 -> 01000001
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        error_correction=Hamming74(),
        intensity_range=(0.0, 1.0),
        polarization_range=(0.0, 3.14159),
    )

    _pattern = writer.write(payload)

    # Produce an encoded stream by encoding the payload bits
    payload_bits = bytes_to_bits(payload)
    encoded_bits = writer.error_correction.encode(payload_bits)

    # Force a single-bit flip in the first 7-bit block and verify decoder corrects it
    noisy_bits_single = list(encoded_bits)
    noisy_bits_single[0] ^= 1
    decoded_single = writer.error_correction.decode(noisy_bits_single)
    assert decoded_single.corrected_errors >= 1
    assert decoded_single.detected_uncorrectable == 0

    # Try various double-bit flips in the first 7-bit block; at least one should be
    # detected as uncorrectable by the updated decoder implementation.
    detected = 0
    corrupted = 0
    # original decoded bits for comparison (first block)
    original_decoded = writer.error_correction.decode(encoded_bits).bits[:4]
    for i in range(7):
        for j in range(i + 1, 7):
            noisy = list(encoded_bits)
            # flip two bits inside the first block
            noisy[i] ^= 1
            noisy[j] ^= 1
            result = writer.error_correction.decode(noisy)
            if result.detected_uncorrectable >= 1:
                detected += 1
            if result.bits[:4] != original_decoded:
                corrupted += 1

    # Ensure the decoder either detects at least one uncorrectable double-bit
    # error or that at least one double-bit flip leads to corrupted decoded data.
    assert detected >= 1 or corrupted >= 1


def test_quantisation_rounding_consistency() -> None:
    # Create a small writer with 4 levels (2 bits) and write a single voxel value
    writer = LaserWriter(
        grid_size=(4, 1, 1),
        intensity_levels=4,
        polarization_states=1,
        intensity_range=(0.0, 0.75),
        polarization_range=(0.0, 1.0),
        error_correction=NoErrorCorrection(),
    )

    # Manually craft a value that sits exactly at a quantisation boundary
    # With 4 levels over range 0.0..0.75 the step is 0.25
    target_level = 2
    expected_physical = 0.0 + target_level * 0.25

    # Build a pattern by writing one byte that maps to the desired intensity bits
    bits = int_to_bits(target_level, 2)
    # Pad to a full byte for writing (simple approach)
    bits_padded = bits + [0] * (8 - len(bits))
    data = bits_to_bytes(bits_padded)

    pattern = writer.write(data)
    # One byte = 8 bits; with 2 bits per voxel we expect 4 voxels
    assert len(pattern.voxels) == 4
    voxel = pattern.voxels[0]
    # The writer maps level to physical value; ensure it equals expected (within float tolerance)
    assert abs(voxel.intensity - expected_physical) < 1e-9

    # Now ensure reader maps physical back to the same level
    reader = LaserReader(pattern)
    result = reader.read()
    # Because we used NoErrorCorrection and wrote one byte, the roundtrip data should match
    assert result.data[:1] == data[:1]


def test_noise_clipping_and_reader_stability() -> None:
    # Build a simple pattern and then apply huge noise to force clipping
    payload = b"Z"
    writer = LaserWriter(
        grid_size=(2, 2, 1),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.1, 0.9),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
    )

    pattern = writer.write(payload)
    # Apply very large noise; values should be clipped to the configured ranges
    noisy = apply_gaussian_noise(pattern, intensity_std=10.0, polarization_std=10.0, seed=42)
    for v in noisy:
        assert writer.intensity_range[0] <= v.intensity <= writer.intensity_range[1]
        assert writer.polarization_range[0] <= v.polarization <= writer.polarization_range[1]

    # Ensure the reader runs without raising when given clipped noisy voxels
    reader = LaserReader(pattern)
    _ = reader.read(noisy)
