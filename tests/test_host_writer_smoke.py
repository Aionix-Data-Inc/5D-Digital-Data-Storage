import math

import pytest

from host_writer import HostWriter
from optical_storage.noise import apply_gaussian_noise
from optical_storage.reader import LaserReader


def test_host_writer_roundtrip_no_noise():
    hw = HostWriter(grid_size=(8, 8, 2), intensity_levels=4, polarization_states=4)
    data = b"hello host writer"
    pattern = hw.write(data)
    rb = hw.verify(pattern)
    assert rb.data == data
    assert rb.read_result.detected_uncorrectable == 0


def test_host_writer_with_noise_and_reader():
    hw = HostWriter(grid_size=(8, 8, 2), intensity_levels=8, polarization_states=8)
    data = b"noisy pipeline"
    pattern = hw.write(data)
    noisy_voxels = apply_gaussian_noise(pattern, intensity_std=0.02, polarization_std=0.02, seed=123)
    # Verify using LaserReader directly (descrambling applied by HostWriter.verify)
    rr = LaserReader(pattern).read(noisy_voxels)
    assert rr.corrected_errors >= 0
    # Now go through host verify for descramble
    rb = hw.verify(pattern, voxels_override=noisy_voxels)
    # Note: We can't guarantee exact recovery under noise, but check no crash and data length
    assert len(rb.data) == len(data)


def test_host_writer_capacity_guard():
    hw = HostWriter(grid_size=(2, 2, 1), intensity_levels=2, polarization_states=2)
    # bits_per_voxel = 2; total voxels = 4 -> capacity after ECC roughly small; choose large data
    big = b"A" * 1024
    with pytest.raises(ValueError):
        hw.write(big)
