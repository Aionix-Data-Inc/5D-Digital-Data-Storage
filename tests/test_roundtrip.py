from __future__ import annotations

import pytest

from optical_storage import Hamming74, LaserReader, LaserWriter, NoErrorCorrection
from optical_storage.noise import apply_gaussian_noise


def test_roundtrip_no_noise() -> None:
    payload = b"DataCube" * 4
    writer = LaserWriter(
        grid_size=(16, 16, 4),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.1, 0.9),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
    )

    pattern = writer.write(payload)
    reader = LaserReader(pattern)
    result = reader.read()
    assert result.data == payload
    assert result.corrected_errors == 0
    assert result.detected_uncorrectable == 0


def test_roundtrip_with_noise() -> None:
    payload = b"Femtosecond lasers rock!"
    writer = LaserWriter(
        grid_size=(32, 32, 8),
        intensity_levels=16,
        polarization_states=8,
        intensity_range=(0.2, 1.0),
        polarization_range=(0.0, 3.14159),
        error_correction=Hamming74(),
    )

    pattern = writer.write(payload)
    noisy = apply_gaussian_noise(pattern, intensity_std=0.01, polarization_std=0.01, seed=99)
    reader = LaserReader(pattern)
    result = reader.read(noisy)
    assert result.data == payload
    assert result.corrected_errors >= 0
    assert result.detected_uncorrectable == 0


def test_capacity_guard() -> None:
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
    )

    # With 4x4x1 voxels and 4 intensity/polarization levels we have 4 bits per voxel
    # and 16 voxels -> 64 bits total. Attempt to write 9 bytes should fail.
    payload = b"123456789"
    with pytest.raises(ValueError):
        writer.write(payload)
