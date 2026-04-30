"""Tests for anomaly detection and retry strategies."""

from __future__ import annotations

import math
from pathlib import Path

from optical_storage import (
    AnomalyDetector,
    LaserWriter,
    NoErrorCorrection,
)
from optical_storage.voxel import Voxel


def test_detect_miswritten_voxels_and_noise_spikes(tmp_path: Path) -> None:
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, math.pi),
        error_correction=NoErrorCorrection(),
    )
    pattern = writer.write(b"OK")

    measured_voxels = [v for v in pattern.voxels if not (v.x == 0 and v.y == 0 and v.z == 0)]
    assert len(measured_voxels) < len(pattern.voxels)

    # Introduce an intensity mismatch in a voxel that can be altered.
    mismatch_voxel = next((v for v in measured_voxels if v.intensity < writer.intensity_range[1]), None)
    assert mismatch_voxel is not None
    mismatch_index = measured_voxels.index(mismatch_voxel)
    measured_voxels[mismatch_index] = Voxel(
        x=mismatch_voxel.x,
        y=mismatch_voxel.y,
        z=mismatch_voxel.z,
        intensity=writer.intensity_range[1],
        polarization=mismatch_voxel.polarization,
    )

    detector = AnomalyDetector(log_dir=str(tmp_path), device_id="dev1", media_batch_id="batch1")
    anomalies = detector.detect_miswritten_voxels(pattern, measured_voxels, tolerance=0.1)

    assert any(a.anomaly_type == "missing_voxel" for a in anomalies)
    assert any(a.anomaly_type == "intensity_mismatch" for a in anomalies)

    # Verify noise spike detection reports outliers.
    spike_voxel = pattern.voxels[0]
    noisy_voxels = pattern.voxels + [
        Voxel(
            x=spike_voxel.x,
            y=spike_voxel.y,
            z=spike_voxel.z,
            intensity=writer.intensity_range[1],
            polarization=writer.polarization_range[1],
        )
    ]
    spikes = detector.detect_optical_noise_spikes(noisy_voxels, intensity_threshold=1.0, polarization_threshold=1.0)
    assert any(a.anomaly_type == "intensity_noise_spike" for a in spikes)

    detector.log_anomalies(anomalies + spikes)
    log_file = tmp_path / "dev1_batch1_anomalies.jsonl"
    assert log_file.exists()
    assert len(log_file.read_text().splitlines()) >= 1


def test_detect_hardware_drift() -> None:
    detector = AnomalyDetector(device_id="dev2", media_batch_id="batch2")

    stable_measurements = [
        Voxel(x=0, y=0, z=0, intensity=0.20, polarization=0.5),
        Voxel(x=1, y=0, z=0, intensity=0.21, polarization=0.52),
        Voxel(x=2, y=0, z=0, intensity=0.19, polarization=0.48),
    ]

    for _ in range(10):
        drift_anomalies = detector.detect_hardware_drift(stable_measurements)
        assert drift_anomalies == []

    drifted_measurements = [
        Voxel(x=0, y=0, z=0, intensity=0.80, polarization=0.5),
        Voxel(x=1, y=0, z=0, intensity=0.79, polarization=0.52),
        Voxel(x=2, y=0, z=0, intensity=0.81, polarization=0.48),
    ]
    drift_anomalies = detector.detect_hardware_drift(drifted_measurements)
    assert any(a.anomaly_type == "intensity_drift" for a in drift_anomalies)
    assert drift_anomalies[0].deviation > 0


def test_retry_failed_regions_returns_retry_results(tmp_path: Path) -> None:
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, math.pi),
        error_correction=NoErrorCorrection(),
    )
    pattern = writer.write(b"RE")
    measured_voxels = [v for v in pattern.voxels if not (v.x == 0 and v.y == 0 and v.z == 0)]

    detector = AnomalyDetector(log_dir=str(tmp_path), device_id="dev3", media_batch_id="batch3")
    anomalies = detector.detect_miswritten_voxels(pattern, measured_voxels, tolerance=0.1)
    assert anomalies

    updated_pattern, remaining = writer.retry_failed_regions(pattern, anomalies)
    assert updated_pattern.grid_size == pattern.grid_size
    assert remaining == []
