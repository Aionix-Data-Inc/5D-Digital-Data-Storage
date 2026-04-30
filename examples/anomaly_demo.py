"""Comprehensive demonstration of anomaly detection and retry strategies."""

import sys
import pathlib
import math
import os

# Set up paths
REPO_ROOT = pathlib.Path('/workspaces/5D-Digital-Data-Storage')
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from optical_storage import (
    LaserWriter, LaserReader, NoErrorCorrection,
    AdaptiveCalibration, AnomalyDetector, RetryManager
)
from optical_storage.noise import apply_gaussian_noise


def demonstrate_anomaly_detection():
    """Demonstrate comprehensive anomaly detection and retry functionality."""

    print("🔍 5D Optical Storage: Anomaly Detection & Retry Strategies Demo")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing System Components...")

    # Create calibration system
    calibration = AdaptiveCalibration()
    calibration.load_calibration("demo_device", "demo_batch")

    # Create writer with anomaly detection
    writer = LaserWriter(
        grid_size=(16, 16, 4),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=8,
        polarization_states=8,
        intensity_range=(0.2, 1.0),
        polarization_range=(0.0, 3.14159),
        error_correction=NoErrorCorrection(),
        calibration=calibration
    )

    print("✓ Writer initialized with anomaly detection")
    print("✓ Calibration system loaded")

    # Create test data
    print("\n2. Creating Test Data...")
    payload = b"5D optical storage with anomaly detection and retry capabilities!"
    pattern = writer.write(payload)
    print(f"✓ Created storage pattern with {len(pattern.voxels)} voxels")

    # Simulate various types of noise/anomalies
    print("\n3. Simulating Real-World Anomalies...")

    # Add Gaussian noise (simulates optical noise)
    noisy_voxels = apply_gaussian_noise(pattern, intensity_std=0.03, polarization_std=0.05, seed=123)

    # Add some extreme outliers (simulates hardware glitches)
    glitch_indices = [10, 25, 40, 55]  # Random voxel indices to corrupt
    for idx in glitch_indices:
        if idx < len(noisy_voxels):
            voxel = noisy_voxels[idx]
            # Create extreme intensity spike
            from optical_storage.voxel import Voxel
            noisy_voxels[idx] = Voxel(
                x=voxel.x, y=voxel.y, z=voxel.z,
                intensity=min(2.0, voxel.intensity * 3.0),
                polarization=voxel.polarization
            )

    # Detect anomalies from simulated measurements
    print("\n4. Detecting Anomalies...")
    write_anomalies = writer.detect_anomalies_write_feedback(pattern, noisy_voxels)

    anomaly_types = {}
    for anomaly in write_anomalies:
        anomaly_types[anomaly.anomaly_type] = anomaly_types.get(anomaly.anomaly_type, 0) + 1

    print(f"✓ Detected {len(write_anomalies)} total anomalies:")
    for anomaly_type, count in anomaly_types.items():
        print(f"  - {anomaly_type}: {count}")

    # Show some specific anomalies
    print("\n  Sample anomalies:")
    for i, anomaly in enumerate(write_anomalies[:3]):
        print(f"    {i+1}. {anomaly.anomaly_type} at {anomaly.voxel_coordinates}")
        print(f"       Expected: {anomaly.expected_value:.3f}, Measured: {anomaly.measured_value:.3f}, Deviation: {anomaly.deviation:.3f}")
    # Implement retry strategies
    print("\n5. Implementing Retry Strategies...")
    if write_anomalies:
        updated_pattern, remaining_anomalies = writer.retry_failed_regions(pattern, write_anomalies)
        print(f"✓ Retry operations completed")
        print(f"  - Regions retried: {len(writer.retry_manager.retry_history)}")
        print(f"  - Remaining anomalies: {len(remaining_anomalies)}")

    # Test reading with anomaly detection
    print("\n6. Testing Read Operations with Anomaly Detection...")
    reader = LaserReader(updated_pattern if 'updated_pattern' in locals() else pattern, calibration=calibration)

    # Read with noisy data
    result = reader.read(noisy_voxels)
    print(f"✓ Read operation completed")
    print(f"  - Data recovered: {len(result.data)} bytes")
    print(f"  - Hash validation: {'✓' if result.hash_valid else '✗'}")

    # Detect read anomalies
    read_anomalies = reader.detect_anomalies_read_feedback(noisy_voxels)
    print(f"✓ Detected {len(read_anomalies)} read anomalies")

    # Implement read retry strategies
    if read_anomalies:
        corrected_voxels, remaining_read_anomalies = reader.retry_failed_read_regions(read_anomalies, noisy_voxels)
        print(f"✓ Read retry operations completed")
        print(f"  - Remaining anomalies: {len(remaining_read_anomalies)}")

    # Demonstrate hardware drift detection
    print("\n7. Demonstrating Hardware Drift Detection...")

    # Simulate multiple read cycles with increasing drift
    from optical_storage.voxel import Voxel
    drift_anomalies = []
    for cycle in range(5):
        # Add progressive drift
        drift_factor = 1.0 + (cycle * 0.01)  # 1% drift per cycle
        drifted_voxels = []
        for voxel in noisy_voxels:
            drifted_voxels.append(Voxel(
                x=voxel.x,
                y=voxel.y,
                z=voxel.z,
                intensity=voxel.intensity * drift_factor,
                polarization=voxel.polarization + (cycle * 0.01)
            ))

        # Detect drift
        cycle_anomalies = reader.detect_anomalies_read_feedback(drifted_voxels)
        drift_anomalies.extend(cycle_anomalies)
        print(f"  Cycle {cycle + 1}: {len(cycle_anomalies)} drift-related anomalies")

    # Show anomaly logging
    print("\n8. Anomaly Logging for Retraining...")
    log_files = [f for f in os.listdir('anomaly_logs') if f.endswith('.jsonl')] if os.path.exists('anomaly_logs') else []
    if log_files:
        print(f"✓ Anomalies logged to {len(log_files)} files for ML retraining")
        for log_file in log_files[:2]:  # Show first 2
            with open(f'anomaly_logs/{log_file}', 'r') as f:
                lines = f.readlines()
                print(f"  - {log_file}: {len(lines)} anomaly records")
    else:
        print("ℹ No anomaly logs created (expected in simulation)")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Original payload: {len(payload)} bytes")
    print(f"Storage pattern: {len(pattern.voxels)} voxels")
    print(f"Write anomalies detected: {len(write_anomalies)}")
    print(f"Read anomalies detected: {len(read_anomalies)}")
    print(f"Drift anomalies detected: {len(drift_anomalies)}")
    print(f"Data integrity: {'✓ MAINTAINED' if result.hash_valid else '✗ COMPROMISED'}")
    print("\n✅ Anomaly detection and retry strategies successfully demonstrated!")


if __name__ == "__main__":
    demonstrate_anomaly_detection()