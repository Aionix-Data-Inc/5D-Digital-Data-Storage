"""Anomaly detection and retry strategies for optical data storage."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .storage_pattern import StoragePattern
from .voxel import Voxel


@dataclass
class AnomalyLog:
    """Log entry for detected anomalies."""
    timestamp: datetime = field(default_factory=datetime.now)
    anomaly_type: str = ""
    voxel_coordinates: Tuple[int, int, int] = (0, 0, 0)
    expected_value: float = 0.0
    measured_value: float = 0.0
    deviation: float = 0.0
    region_bounds: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((0, 0, 0), (0, 0, 0))
    retry_count: int = 0
    device_id: str = ""
    media_batch_id: str = ""


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    corrected_voxels: List[Voxel] = field(default_factory=list)
    remaining_anomalies: List[AnomalyLog] = field(default_factory=list)
    retry_count: int = 0


class AnomalyDetector:
    """Detects various types of anomalies in optical data storage."""

    def __init__(self, log_dir: str = "anomaly_logs", device_id: str = "", media_batch_id: str = ""):
        self.log_dir = log_dir
        self.device_id = device_id
        self.media_batch_id = media_batch_id
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Statistical tracking for drift detection
        self.intensity_history: List[float] = []
        self.polarization_history: List[float] = []
        self.drift_threshold = 0.05  # 5% change threshold

    def detect_miswritten_voxels(
        self,
        pattern: StoragePattern,
        measured_voxels: List[Voxel],
        tolerance: float = 0.1
    ) -> List[AnomalyLog]:
        """Detect voxels that were not written correctly."""
        anomalies = []

        # Create lookup for measured voxels by coordinates
        measured_lookup = {(v.x, v.y, v.z): v for v in measured_voxels}

        for expected_voxel in pattern.voxels:
            coords = (expected_voxel.x, expected_voxel.y, expected_voxel.z)
            measured = measured_lookup.get(coords)

            if measured is None:
                # Missing voxel
                anomaly = AnomalyLog(
                    anomaly_type="missing_voxel",
                    voxel_coordinates=coords,
                    expected_value=expected_voxel.intensity,
                    measured_value=0.0,
                    deviation=1.0,
                    device_id=self.device_id,
                    media_batch_id=self.media_batch_id
                )
                anomalies.append(anomaly)
                continue

            # Check intensity deviation
            intensity_dev = abs(expected_voxel.intensity - measured.intensity) / expected_voxel.intensity
            if intensity_dev > tolerance:
                anomaly = AnomalyLog(
                    anomaly_type="intensity_mismatch",
                    voxel_coordinates=coords,
                    expected_value=expected_voxel.intensity,
                    measured_value=measured.intensity,
                    deviation=intensity_dev,
                    device_id=self.device_id,
                    media_batch_id=self.media_batch_id
                )
                anomalies.append(anomaly)

            # Check polarization deviation (normalized to 0-1 range)
            pol_dev = abs(expected_voxel.polarization - measured.polarization) / (2 * math.pi)
            if pol_dev > tolerance:
                anomaly = AnomalyLog(
                    anomaly_type="polarization_mismatch",
                    voxel_coordinates=coords,
                    expected_value=expected_voxel.polarization,
                    measured_value=measured.polarization,
                    deviation=pol_dev,
                    device_id=self.device_id,
                    media_batch_id=self.media_batch_id
                )
                anomalies.append(anomaly)

        return anomalies

    def detect_optical_noise_spikes(
        self,
        voxels: List[Voxel],
        intensity_threshold: float = 3.0,  # Standard deviations
        polarization_threshold: float = 3.0
    ) -> List[AnomalyLog]:
        """Detect optical noise spikes using statistical outlier detection."""
        anomalies = []

        if not voxels:
            return anomalies

        # Calculate statistics for intensity
        intensities = [v.intensity for v in voxels]
        intensity_mean = sum(intensities) / len(intensities)
        intensity_std = math.sqrt(sum((x - intensity_mean) ** 2 for x in intensities) / len(intensities))

        # Calculate statistics for polarization
        polarizations = [v.polarization for v in voxels]
        pol_mean = sum(polarizations) / len(polarizations)
        pol_std = math.sqrt(sum((x - pol_mean) ** 2 for x in polarizations) / len(polarizations))

        for voxel in voxels:
            # Check intensity outliers
            intensity_z_score = abs(voxel.intensity - intensity_mean) / intensity_std if intensity_std > 0 else 0
            if intensity_z_score > intensity_threshold:
                anomaly = AnomalyLog(
                    anomaly_type="intensity_noise_spike",
                    voxel_coordinates=(voxel.x, voxel.y, voxel.z),
                    expected_value=intensity_mean,
                    measured_value=voxel.intensity,
                    deviation=intensity_z_score,
                    device_id=self.device_id,
                    media_batch_id=self.media_batch_id
                )
                anomalies.append(anomaly)

            # Check polarization outliers
            pol_z_score = abs(voxel.polarization - pol_mean) / pol_std if pol_std > 0 else 0
            if pol_z_score > polarization_threshold:
                anomaly = AnomalyLog(
                    anomaly_type="polarization_noise_spike",
                    voxel_coordinates=(voxel.x, voxel.y, voxel.z),
                    expected_value=pol_mean,
                    measured_value=voxel.polarization,
                    deviation=pol_z_score,
                    device_id=self.device_id,
                    media_batch_id=self.media_batch_id
                )
                anomalies.append(anomaly)

        return anomalies

    def detect_hardware_drift(
        self,
        current_measurements: List[Voxel],
        window_size: int = 100
    ) -> List[AnomalyLog]:
        """Detect hardware drift by analyzing trends over time."""
        anomalies = []

        if not current_measurements:
            return anomalies

        # Update history
        current_intensity_avg = sum(v.intensity for v in current_measurements) / len(current_measurements)
        current_pol_avg = sum(v.polarization for v in current_measurements) / len(current_measurements)

        self.intensity_history.append(current_intensity_avg)
        self.polarization_history.append(current_pol_avg)

        # Keep only recent history
        if len(self.intensity_history) > window_size:
            self.intensity_history = self.intensity_history[-window_size:]
            self.polarization_history = self.polarization_history[-window_size:]

        # Need minimum history for drift detection
        if len(self.intensity_history) < 10:
            return anomalies

        # Calculate trend for intensity
        intensity_trend = self._calculate_trend(self.intensity_history)
        if abs(intensity_trend) > self.drift_threshold:
            anomaly = AnomalyLog(
                anomaly_type="intensity_drift",
                expected_value=self.intensity_history[0],  # Baseline
                measured_value=current_intensity_avg,
                deviation=intensity_trend,
                device_id=self.device_id,
                media_batch_id=self.media_batch_id
            )
            anomalies.append(anomaly)

        # Calculate trend for polarization
        pol_trend = self._calculate_trend(self.polarization_history)
        if abs(pol_trend) > self.drift_threshold:
            anomaly = AnomalyLog(
                anomaly_type="polarization_drift",
                expected_value=self.polarization_history[0],  # Baseline
                measured_value=current_pol_avg,
                deviation=pol_trend,
                device_id=self.device_id,
                media_batch_id=self.media_batch_id
            )
            anomalies.append(anomaly)

        return anomalies

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend (slope) of a series."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        # Normalize by mean to get relative trend
        return slope / y_mean if y_mean != 0 else slope

    def log_anomalies(self, anomalies: List[AnomalyLog]) -> None:
        """Log anomalies to file for retraining."""
        if not anomalies:
            return

        log_file = os.path.join(self.log_dir, f"{self.device_id}_{self.media_batch_id}_anomalies.jsonl")

        with open(log_file, 'a') as f:
            for anomaly in anomalies:
                log_entry = {
                    "timestamp": anomaly.timestamp.isoformat(),
                    "anomaly_type": anomaly.anomaly_type,
                    "voxel_coordinates": anomaly.voxel_coordinates,
                    "expected_value": anomaly.expected_value,
                    "measured_value": anomaly.measured_value,
                    "deviation": anomaly.deviation,
                    "region_bounds": anomaly.region_bounds,
                    "retry_count": anomaly.retry_count,
                    "device_id": anomaly.device_id,
                    "media_batch_id": anomaly.media_batch_id,
                }
                f.write(json.dumps(log_entry) + '\n')

        self.logger.info(f"Logged {len(anomalies)} anomalies to {log_file}")


class RetryManager:
    """Manages retry strategies for failed regions."""

    def __init__(self, max_retries: int = 3, region_size: int = 5):
        self.max_retries = max_retries
        self.region_size = region_size
        self.retry_history: Dict[Tuple[int, int, int], int] = {}

    def identify_retry_regions(self, anomalies: List[AnomalyLog]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Identify regions that need retry based on anomaly clustering."""
        if not anomalies:
            return []

        # Group anomalies by proximity
        regions = []
        processed_coords: Set[Tuple[int, int, int]] = set()

        for anomaly in anomalies:
            if anomaly.voxel_coordinates in processed_coords:
                continue

            # Find all anomalies within region_size of this one
            region_anomalies = [a for a in anomalies
                              if self._distance(anomaly.voxel_coordinates, a.voxel_coordinates) <= self.region_size]

            if region_anomalies:
                # Calculate bounding box
                coords = [a.voxel_coordinates for a in region_anomalies]
                min_x = min(c[0] for c in coords)
                max_x = max(c[0] for c in coords)
                min_y = min(c[1] for c in coords)
                max_y = max(c[1] for c in coords)
                min_z = min(c[2] for c in coords)
                max_z = max(c[2] for c in coords)

                region = ((min_x, min_y, min_z), (max_x, max_y, max_z))
                regions.append(region)

                # Mark as processed
                for a in region_anomalies:
                    processed_coords.add(a.voxel_coordinates)

        return regions

    def _distance(self, coord1: Tuple[int, int, int], coord2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between coordinates."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    def should_retry_region(self, region: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> bool:
        """Check if a region should be retried based on retry history."""
        region_key = region[0]  # Use min corner as key
        current_retries = self.retry_history.get(region_key, 0)
        return current_retries < self.max_retries

    def record_retry_attempt(self, region: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> None:
        """Record a retry attempt for a region."""
        region_key = region[0]
        self.retry_history[region_key] = self.retry_history.get(region_key, 0) + 1

    def get_voxels_in_region(
        self,
        region: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        all_voxels: List[Voxel]
    ) -> List[Voxel]:
        """Get all voxels within a region."""
        (min_x, min_y, min_z), (max_x, max_y, max_z) = region
        return [v for v in all_voxels
                if min_x <= v.x <= max_x and min_y <= v.y <= max_y and min_z <= v.z <= max_z]

    def retry_write_region(
        self,
        region: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        pattern: StoragePattern,
        writer  # LaserWriter instance
    ) -> RetryResult:
        """Retry writing a specific region."""
        if not self.should_retry_region(region):
            return RetryResult(success=False, remaining_anomalies=[])

        # Get target voxels for this region
        region_voxels = self.get_voxels_in_region(region, pattern.voxels)

        if not region_voxels:
            return RetryResult(success=True, corrected_voxels=[])

        # Generate new laser instructions for retry
        retry_instructions = []
        for voxel in region_voxels:
            physical_x = voxel.x * pattern.voxel_pitch[0]
            physical_y = voxel.y * pattern.voxel_pitch[1]
            physical_z = voxel.z * pattern.voxel_pitch[2]

            # Apply increased power for retry (compensate for potential issues)
            retry_intensity = min(voxel.intensity * 1.1, pattern.intensity_range[1])

            retry_instructions.append({
                'x': physical_x,
                'y': physical_y,
                'z': physical_z,
                'intensity': retry_intensity,
                'polarization': voxel.polarization
            })

        # Record retry attempt
        self.record_retry_attempt(region)

        # In a real implementation, this would execute the laser instructions
        # For simulation, we'll assume success with some probability
        success_probability = 0.8 ** (self.retry_history[region[0]] - 1)  # Decreasing success rate
        success = True  # For simulation, assume success

        corrected_voxels = region_voxels if success else []

        return RetryResult(
            success=success,
            corrected_voxels=corrected_voxels,
            retry_count=self.retry_history[region[0]]
        )

    def retry_read_region(
        self,
        region: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        reader  # LaserReader instance
    ) -> RetryResult:
        """Retry reading a specific region."""
        if not self.should_retry_region(region):
            return RetryResult(success=False, remaining_anomalies=[])

        # Record retry attempt
        self.record_retry_attempt(region)

        # In a real implementation, this would perform additional imaging/scanning
        # For simulation, we'll assume partial success
        success = True
        corrected_voxels = []  # Would be populated with re-read data

        return RetryResult(
            success=success,
            corrected_voxels=corrected_voxels,
            retry_count=self.retry_history[region[0]]
        )