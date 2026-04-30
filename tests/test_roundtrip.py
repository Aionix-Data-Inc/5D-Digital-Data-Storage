from __future__ import annotations

import pytest

from optical_storage import Hamming74, LaserInstruction, LaserReader, LaserWriter, NoErrorCorrection, OpticalStoragePipeline, Voxel
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


def test_compare_voxel_states_detects_mismatch() -> None:
    payload = b"CompareState"
    writer = LaserWriter(
        grid_size=(8, 8, 2),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.1, 0.8),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
    )

    pattern = writer.write(payload)
    observed = [
        Voxel(
            x=v.x,
            y=v.y,
            z=v.z,
            intensity=v.intensity + (0.06 if i == 0 else 0.0),
            polarization=v.polarization + (0.03 if i == 0 else 0.0),
        )
        for i, v in enumerate(pattern.voxels)
    ]
    summary = pattern.compare_voxel_states(observed, intensity_tolerance=0.05, polarization_tolerance=0.02)
    assert summary["total_expected"] == len(pattern.voxels)
    assert summary["mismatched_voxels"] >= 1
    assert summary["matched_voxels"] == summary["common_voxels"] - summary["mismatched_voxels"]


def test_read_reports_block_confidence_scores() -> None:
    payload = b"Confidence"
    writer = LaserWriter(
        grid_size=(16, 16, 2),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.2, 0.9),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
    )

    pattern = writer.write(payload)
    reader = LaserReader(pattern)
    result = reader.read()

    assert len(result.block_confidence_scores) == result.voxels_used
    assert all(0.0 <= score <= 1.0 for score in result.block_confidence_scores)
    assert result.confidence_summary["min"] >= 0.0
    assert result.confidence_summary["max"] <= 1.0


def test_adjust_encoding_and_decoding_parameters_dynamically() -> None:
    payload = b"Dynamic"
    writer = LaserWriter(
        grid_size=(16, 16, 2),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.15, 1.0),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
    )
    writer.adjust_encoding_parameters(intensity_levels=16, polarization_states=8, intensity_range=(0.2, 0.9))

    pattern = writer.write(payload, verify=False)
    reader = LaserReader(pattern)
    reader.adjust_read_parameters(intensity_range=(0.2, 0.9), polarization_range=(0.0, 3.14159), uncertainty_threshold=0.2)
    result = reader.read()

    assert result.data == payload
    assert reader._override_intensity_range == (0.2, 0.9)
    assert reader._uncertainty_threshold == 0.2


def test_voxel_retardance_orientation_aliases() -> None:
    voxel = Voxel(x=0, y=0, z=0, intensity=0.5, polarization=1.2)
    assert voxel.retardance == 0.5
    assert voxel.orientation == 1.2

    instruction = LaserInstruction(x=0.0, y=0.0, z=0.0, retardance=0.5, orientation=1.2)
    assert instruction.pulse_energy == 0.5
    assert instruction.polarization_state == 1.2


def test_optical_storage_pipeline_run_recovers_data() -> None:
    payload = b"PipelineTest"
    writer = LaserWriter(
        grid_size=(16, 16, 2),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.2, 0.9),
        polarization_range=(0.0, 3.14159),
        error_correction=Hamming74(),
    )
    pipeline = OpticalStoragePipeline(writer)
    result = pipeline.run(
        payload,
        verify=False,
        intensity_std=0.005,
        polarization_std=0.005,
        seed=42,
        read_kwargs={"parallel": False, "optimize_scan": True, "use_ml": False},
    )

    assert result.read_result.data == payload
    assert result.read_result.detected_uncorrectable == 0
    assert result.read_result.corrected_errors >= 0
    assert result.voxel_state_comparison["common_voxels"] == len(result.pattern.voxels)
    assert len(result.instructions) == len(result.pattern.voxels)


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
