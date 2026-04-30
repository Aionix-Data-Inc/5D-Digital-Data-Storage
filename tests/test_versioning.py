"""Tests for versioning, firmware metadata, and immutable write logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from optical_storage import (
    EncodingVersion,
    ImmutableWriteLog,
    LaserReader,
    LaserWriter,
    VersionManager,
)
from optical_storage.anomaly_detection import AnomalyDetector
from optical_storage.error_correction import NoErrorCorrection
from optical_storage.versioning import VersionInfo, WriteLogEntry


def test_version_info_compatibility_with_older_patterns(tmp_path: Path) -> None:
    log_dir = tmp_path / "write_logs"
    anomaly_detector = AnomalyDetector(log_dir=str(log_dir), device_id="test_reader", media_batch_id="batch1")
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
        anomaly_detector=anomaly_detector,
        write_log=ImmutableWriteLog(log_dir=str(log_dir)),
    )
    pattern = writer.write(b"v")

    pattern.version_info = VersionInfo(
        encoding_version=EncodingVersion.V1_0,
        firmware_version="1.0.0",
        compatible_versions=["1.0", "1.1", "2.0"],
    )

    reader = LaserReader(pattern)
    assert reader.validate_version_compatibility()


def test_version_manager_encodes_and_decodes_supported_versions() -> None:
    manager = VersionManager()
    for version in [EncodingVersion.V1_0, EncodingVersion.V1_1, EncodingVersion.V1_2, EncodingVersion.V2_0]:
        encoded = manager.encode_with_version(b"abc123", version=version)
        decoded = manager.decode_with_version_compatibility(encoded)

        assert decoded == b"abc123"
        assert encoded["version_info"]["encoding_version"] == version.value
        assert encoded["version_info"]["firmware_version"] == manager.firmware_version


def test_immutable_write_log_chain_integrity(tmp_path: Path) -> None:
    log_dir = tmp_path / "write_logs"
    log = ImmutableWriteLog(log_dir=str(log_dir))

    entry_a = WriteLogEntry(
        timestamp=datetime.now(),
        device_id="dev1",
        media_batch_id="batch1",
        operation_type="write",
        payload_hash="hash-a",
        pattern_hash="pattern-a",
        metadata={"encoding_version": EncodingVersion.V2_0.value},
    )
    hash_a = log.append_entry(entry_a)
    assert hash_a == entry_a.log_hash

    entry_b = WriteLogEntry(
        timestamp=datetime.now(),
        device_id="dev1",
        media_batch_id="batch1",
        operation_type="retry",
        payload_hash="hash-b",
        pattern_hash="pattern-b",
        metadata={"encoding_version": EncodingVersion.V2_0.value},
    )
    hash_b = log.append_entry(entry_b)
    assert hash_b == entry_b.log_hash
    assert log.verify_chain_integrity()
    assert log.get_latest_entry().log_hash == hash_b


def test_writer_rejects_duplicate_write_without_overwrite(tmp_path: Path) -> None:
    write_log_dir = tmp_path / "write_logs"
    anomaly_detector = AnomalyDetector(log_dir=str(write_log_dir), device_id="duplicate_test", media_batch_id="batch1")
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
        anomaly_detector=anomaly_detector,
        write_log=ImmutableWriteLog(log_dir=str(write_log_dir)),
        enable_overwrite_protection=True,
    )

    writer.write(b"x")
    with pytest.raises(ValueError, match="overwrite=True"):
        writer.write(b"x")

    overwritten_pattern = writer.write(b"x", overwrite=True)
    assert overwritten_pattern.write_log_hash is not None


def test_verify_write_detects_corrupted_pattern(tmp_path: Path) -> None:
    log_dir = tmp_path / "write_logs"
    anomaly_detector = AnomalyDetector(log_dir=str(log_dir), device_id="verify_corrupt", media_batch_id="batch1")
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
        anomaly_detector=anomaly_detector,
        write_log=ImmutableWriteLog(log_dir=str(log_dir)),
    )
    pattern = writer.write(b"v")
    pattern.data_hash = "badhash"
    assert writer.verify_write(pattern, b"v") is False


def test_reader_returns_empty_data_when_hash_invalid(tmp_path: Path) -> None:
    log_dir = tmp_path / "write_logs"
    anomaly_detector = AnomalyDetector(log_dir=str(log_dir), device_id="reader_corrupt", media_batch_id="batch1")
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.2, 0.8),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
        anomaly_detector=anomaly_detector,
        write_log=ImmutableWriteLog(log_dir=str(log_dir)),
    )
    pattern = writer.write(b"v")
    pattern.data_hash = "deadbeef"

    reader = LaserReader(pattern)
    result = reader.read()

    assert result.data == b""
    assert result.hash_valid is False
    assert result.detected_uncorrectable == 0
