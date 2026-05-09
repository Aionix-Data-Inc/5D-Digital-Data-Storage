"""Laser writing pipeline for 5D optical storage."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple
import json
from datetime import datetime

from .bit_utils import bits_to_int, chunk_bits, bytes_to_bits
from .error_correction import ErrorCorrectionScheme, Hamming74
from .storage_pattern import StoragePattern
from .voxel import Voxel
from .calibration import AdaptiveCalibration
from .anomaly_detection import AnomalyDetector, RetryManager, AnomalyLog
from .reader import LaserReader
from .versioning import VersionManager, ImmutableWriteLog, WriteLogEntry


@dataclass
class LaserInstruction:
    """Instruction for laser pulse at a specific position.

    The optical state is described by retardance (δ) and slow-axis
    orientation (θ).
    """
    x: float  # Physical x position in micrometres
    y: float  # Physical y position
    z: float  # Physical z position
    retardance: float  # Optical retardance / pulse energy proxy
    orientation: float  # Slow-axis orientation in radians

    @property
    def pulse_energy(self) -> float:
        return self.retardance

    @property
    def polarization_state(self) -> float:
        return self.orientation


@dataclass
class LaserWriter:
    """Translate bytes into five-dimensional voxel patterns."""

    grid_size: Tuple[int, int, int] = (64, 64, 32)
    voxel_pitch: Tuple[float, float, float] = (5.0, 5.0, 20.0)  # micrometres
    intensity_levels: int = 16
    polarization_states: int = 8
    intensity_range: Tuple[float, float] = (0.15, 1.0)
    polarization_range: Tuple[float, float] = (0.0, math.pi)
    error_correction: ErrorCorrectionScheme = field(default_factory=Hamming74)
    calibration: AdaptiveCalibration | None = None
    anomaly_detector: AnomalyDetector | None = None
    retry_manager: RetryManager | None = None
    version_manager: VersionManager | None = None
    write_log: ImmutableWriteLog | None = None
    enable_overwrite_protection: bool = False

    def __post_init__(self) -> None:
        self._validate_parameters()
        self.bits_per_intensity = self._bits_for_levels(self.intensity_levels)
        self.bits_per_polarization = self._bits_for_levels(self.polarization_states)
        self.bits_per_voxel = self.bits_per_intensity + self.bits_per_polarization
        if self.bits_per_voxel == 0:
            raise ValueError("At least one dimension must encode information.")

        # Initialize anomaly detection and retry management
        if self.anomaly_detector is None:
            device_id = getattr(self.calibration.current_calibration, 'device_id', 'default_device') if self.calibration else 'default_device'
            media_batch = getattr(self.calibration.current_calibration, 'media_batch_id', 'default_batch') if self.calibration else 'default_batch'
            self.anomaly_detector = AnomalyDetector(device_id=device_id, media_batch_id=media_batch)

        if self.retry_manager is None:
            self.retry_manager = RetryManager()

        # Initialize version management and logging
        if self.version_manager is None:
            self.version_manager = VersionManager()

        if self.write_log is None:
            self.write_log = ImmutableWriteLog()

    def adjust_encoding_parameters(
        self,
        intensity_levels: int | None = None,
        polarization_states: int | None = None,
        intensity_range: Tuple[float, float] | None = None,
        polarization_range: Tuple[float, float] | None = None,
    ) -> None:
        """Adjust quantisation and physical mapping parameters before encoding."""
        if intensity_levels is not None:
            if intensity_levels <= 0 or (intensity_levels & (intensity_levels - 1)):
                raise ValueError("intensity_levels must be a positive power of two")
            self.intensity_levels = intensity_levels

        if polarization_states is not None:
            if polarization_states <= 0 or (polarization_states & (polarization_states - 1)):
                raise ValueError("polarization_states must be a positive power of two")
            self.polarization_states = polarization_states

        if intensity_range is not None:
            if intensity_range[0] >= intensity_range[1]:
                raise ValueError("invalid intensity_range")
            self.intensity_range = intensity_range

        if polarization_range is not None:
            if polarization_range[0] >= polarization_range[1]:
                raise ValueError("invalid polarization_range")
            self.polarization_range = polarization_range

        self._validate_parameters()
        self.bits_per_intensity = self._bits_for_levels(self.intensity_levels)
        self.bits_per_polarization = self._bits_for_levels(self.polarization_states)
        self.bits_per_voxel = self.bits_per_intensity + self.bits_per_polarization
        if self.bits_per_voxel == 0:
            raise ValueError("At least one dimension must encode information.")

    def write(self, data: bytes, verify: bool = True, overwrite: bool = False) -> StoragePattern:
        """Encode *data* into a :class:`StoragePattern`.

        If verify is True, the writer performs a read-after-write validation pass.
        If overwrite is False, duplicate payload writes to the same device/media are rejected.
        """
        payload_bits = bytes_to_bits(data)
        encoded_bits = self.error_correction.encode(payload_bits)
        encoded_bit_length = len(encoded_bits)

        max_voxels = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        required_voxels = math.ceil(encoded_bit_length / self.bits_per_voxel) if encoded_bit_length else 0
        if required_voxels > max_voxels:
            raise ValueError(
                "Data does not fit inside the configured lattice: "
                f"requires {required_voxels} voxels, only {max_voxels} available"
            )

        padding_bits = required_voxels * self.bits_per_voxel - encoded_bit_length if encoded_bit_length else 0
        padded_bits = list(encoded_bits)
        padded_bits.extend([0] * padding_bits)

        voxels: List[Voxel] = []
        for index, chunk in enumerate(chunk_bits(padded_bits, self.bits_per_voxel, pad=False)):
            if not chunk:
                continue
            x, y, z = self._index_to_coordinates(index)
            intensity_bits = chunk[: self.bits_per_intensity]
            polarization_bits = chunk[self.bits_per_intensity :]
            intensity_level = bits_to_int(intensity_bits) if self.bits_per_intensity else 0
            polarization_level = bits_to_int(polarization_bits) if self.bits_per_polarization else 0
            intensity_value = self._level_to_physical(intensity_level, self.intensity_levels, self.intensity_range)
            polarization_value = self._level_to_physical(
                polarization_level, self.polarization_states, self.polarization_range
            )
            voxel = Voxel(x=x, y=y, z=z, intensity=intensity_value, polarization=polarization_value)
            # Apply compensations
            voxel = self.apply_thermal_drift_compensation(voxel)
            voxel = self.apply_material_nonuniformity_compensation(voxel)
            # Apply calibration corrections
            if self.calibration:
                voxel = self.calibration.apply_calibration_to_voxel(voxel)
            voxels.append(voxel)

        # Compute hash of the original data
        data_hash = hashlib.sha256(data).hexdigest()

        # Create version info
        version_info = self.version_manager.create_version_info() if self.version_manager else None

        pattern = StoragePattern(
            voxels=voxels,
            grid_size=self.grid_size,
            voxel_pitch=self.voxel_pitch,
            intensity_levels=self.intensity_levels,
            intensity_range=self.intensity_range,
            polarization_states=self.polarization_states,
            polarization_range=self.polarization_range,
            bits_per_voxel=self.bits_per_voxel,
            encoded_bit_length=encoded_bit_length,
            data_bit_length=len(payload_bits),
            padding_bits=padding_bits,
            error_correction=self.error_correction,
            error_correction_metadata=self.error_correction.metadata(),
            data_length_bytes=len(data),
            data_hash=data_hash,
            version_info=version_info,
        )

        # Prevent accidental duplicate writes when the same payload is already recorded.
        if self.enable_overwrite_protection and self.write_log and self.anomaly_detector and not overwrite:
            duplicate_entries = self.write_log.find_entries_by_payload_hash(data_hash)
            same_device_entries = [
                entry for entry in duplicate_entries
                if entry.device_id == self.anomaly_detector.device_id
                and entry.media_batch_id == self.anomaly_detector.media_batch_id
            ]
            if same_device_entries:
                raise ValueError(
                    "Data has already been written to this device/media; use overwrite=True to explicitly rewrite."
                )

        pattern_hash = hashlib.sha256(json.dumps(pattern.to_dict(), sort_keys=True).encode()).hexdigest()
        verification_success = True
        if verify:
            verification_success = self.verify_write(pattern, data)

        # Log the write operation immutably
        if self.write_log and self.anomaly_detector:
            log_entry = WriteLogEntry(
                timestamp=version_info.created_at if version_info else datetime.now(),
                device_id=self.anomaly_detector.device_id,
                media_batch_id=self.anomaly_detector.media_batch_id,
                operation_type="write",
                payload_hash=data_hash,
                pattern_hash=pattern_hash,
                metadata={
                    "grid_size": self.grid_size,
                    "data_length_bytes": len(data),
                    "encoding_version": version_info.encoding_version.value if version_info else "unknown",
                    "verified": verification_success,
                    "overwrite": overwrite,
                }
            )
            log_hash = self.write_log.append_entry(log_entry)
            pattern.write_log_hash = log_hash

        if verify and not verification_success:
            raise ValueError("Write verification failed; data integrity cannot be guaranteed.")

        return pattern

    def verify_write(self, pattern: StoragePattern, expected_data: bytes | None = None) -> bool:
        """Perform a read-after-write check and confirm that the written data is valid."""
        reader = LaserReader(
            pattern,
            calibration=self.calibration,
            anomaly_detector=self.anomaly_detector,
            retry_manager=self.retry_manager,
            version_manager=self.version_manager,
        )
        result = reader.read()
        if not result.hash_valid or result.detected_uncorrectable > 0:
            return False
        if expected_data is not None:
            return result.data == expected_data
        return True

    def update_calibration_from_write(self, pattern: StoragePattern, feedback_measurements: List[Voxel] | None = None) -> None:
        """Update calibration based on write feedback."""
        if not self.calibration:
            return

        if feedback_measurements:
            # Update laser-material interaction model
            target_levels = []
            for voxel in pattern.voxels:
                intensity_level = self._physical_to_level(voxel.intensity, self.intensity_levels, self.intensity_range)
                polarization_level = self._physical_to_level(voxel.polarization, self.polarization_states, self.polarization_range)
                target_levels.append((intensity_level, polarization_level))

            self.calibration.update_laser_material_model(feedback_measurements, target_levels)

        # Save updated calibration
        if self.calibration.current_calibration:
            self.calibration.save_calibration(self.calibration.current_calibration)

    def generate_laser_instructions(self, pattern: StoragePattern) -> List[LaserInstruction]:
        """Generate laser instruction sequences from a storage pattern."""
        instructions = []
        for voxel in pattern.voxels:
            # Convert grid coordinates to physical positions
            physical_x = voxel.x * self.voxel_pitch[0]
            physical_y = voxel.y * self.voxel_pitch[1]
            physical_z = voxel.z * self.voxel_pitch[2]
            # Pulse energy is the intensity
            retardance = voxel.retardance
            orientation = voxel.orientation
            instructions.append(LaserInstruction(
                x=physical_x,
                y=physical_y,
                z=physical_z,
                retardance=retardance,
                orientation=orientation,
            ))
        return instructions

    def validate_voxel_formation(self, target_voxel: Voxel, feedback_intensity: float, feedback_polarization: float, tolerance: float = 0.1) -> bool:
        """Validate voxel formation using feedback signals."""
        intensity_diff = abs(target_voxel.intensity - feedback_intensity) / target_voxel.intensity
        polarization_diff = abs(target_voxel.polarization - feedback_polarization) / (2 * math.pi)
        return intensity_diff <= tolerance and polarization_diff <= tolerance

    # ------------------------------------------------------------------
    # Helper methods

    def _validate_parameters(self) -> None:
        gx, gy, gz = self.grid_size
        if gx <= 0 or gy <= 0 or gz <= 0:
            raise ValueError("grid_size components must be positive")
        if any(p <= 0 for p in self.voxel_pitch):
            raise ValueError("voxel_pitch values must be positive")
        if self.intensity_levels <= 0 or self.polarization_states <= 0:
            raise ValueError("quantisation levels must be positive")
        if self.intensity_range[0] >= self.intensity_range[1]:
            raise ValueError("invalid intensity_range")
        if self.polarization_range[0] >= self.polarization_range[1]:
            raise ValueError("invalid polarization_range")

    @staticmethod
    def _bits_for_levels(levels: int) -> int:
        if levels <= 0:
            raise ValueError("levels must be positive")
        if levels & (levels - 1):
            raise ValueError("levels must be a power of two for binary encoding")
        return int(math.log2(levels))

    def _index_to_coordinates(self, index: int) -> Tuple[int, int, int]:
        gx, gy, gz = self.grid_size
        plane_size = gx * gy
        z = index // plane_size
        if z >= gz:
            raise ValueError("Index exceeds lattice depth")
        remainder = index % plane_size
        y = remainder // gx
        x = remainder % gx
        return x, y, z

    @staticmethod
    def _level_to_physical(level: int, levels: int, value_range: Tuple[float, float]) -> float:
        min_val, max_val = value_range
        if levels == 1:
            return (min_val + max_val) / 2.0
        step = (max_val - min_val) / float(levels - 1)
        level = max(0, min(level, levels - 1))
        return min_val + level * step

    def apply_thermal_drift_compensation(self, voxel: Voxel, drift_factor: float = 0.01) -> Voxel:
        """Apply thermal drift compensation by adjusting intensity."""
        compensated_intensity = voxel.intensity * (1 + drift_factor * (voxel.z / self.grid_size[2]))
        return Voxel(
            x=voxel.x,
            y=voxel.y,
            z=voxel.z,
            intensity=min(max(compensated_intensity, self.intensity_range[0]), self.intensity_range[1]),
            polarization=voxel.polarization
        )

    def apply_material_nonuniformity_compensation(self, voxel: Voxel, nonuniformity_map: dict = None) -> Voxel:
        """Apply material non-uniformity compensation using a map or simple model."""
        if nonuniformity_map is None:
            # Simple model: intensity decreases with depth
            factor = 1 - 0.05 * (voxel.z / self.grid_size[2])
        else:
            factor = nonuniformity_map.get((voxel.x, voxel.y, voxel.z), 1.0)
        compensated_intensity = voxel.intensity * factor
        return Voxel(
            x=voxel.x,
            y=voxel.y,
            z=voxel.z,
            intensity=min(max(compensated_intensity, self.intensity_range[0]), self.intensity_range[1]),
            polarization=voxel.polarization
        )

    def detect_anomalies_write_feedback(
        self,
        pattern: StoragePattern,
        feedback_measurements: List[Voxel]
    ) -> List[AnomalyLog]:
        """Detect anomalies in write feedback measurements."""
        if not self.anomaly_detector:
            return []

        anomalies = []

        # Detect miswritten voxels
        miswritten = self.anomaly_detector.detect_miswritten_voxels(pattern, feedback_measurements)
        anomalies.extend(miswritten)

        # Detect optical noise spikes
        noise_spikes = self.anomaly_detector.detect_optical_noise_spikes(feedback_measurements)
        anomalies.extend(noise_spikes)

        # Detect hardware drift
        drift_anomalies = self.anomaly_detector.detect_hardware_drift(feedback_measurements)
        anomalies.extend(drift_anomalies)

        # Log anomalies for retraining
        self.anomaly_detector.log_anomalies(anomalies)

        return anomalies

    def retry_failed_regions(
        self,
        pattern: StoragePattern,
        anomalies: List[AnomalyLog]
    ) -> Tuple[StoragePattern, List[AnomalyLog]]:
        """Retry writing failed regions and return updated pattern."""
        if not self.retry_manager or not anomalies:
            return pattern, []

        # Identify regions that need retry
        retry_regions = self.retry_manager.identify_retry_regions(anomalies)

        corrected_voxels = []
        remaining_anomalies = []

        for region in retry_regions:
            if self.retry_manager.should_retry_region(region):
                retry_result = self.retry_manager.retry_write_region(region, pattern, self)
                if retry_result.success:
                    corrected_voxels.extend(retry_result.corrected_voxels)
                else:
                    # Create anomaly log for failed retry
                    remaining_anomaly = AnomalyLog(
                        anomaly_type="retry_failed",
                        region_bounds=region,
                        retry_count=retry_result.retry_count,
                        device_id=self.anomaly_detector.device_id if self.anomaly_detector else "",
                        media_batch_id=self.anomaly_detector.media_batch_id if self.anomaly_detector else ""
                    )
                    remaining_anomalies.append(remaining_anomaly)

        # Create updated pattern with corrected voxels
        if corrected_voxels:
            updated_voxels = pattern.voxels.copy()
            # Replace corrected voxels in the pattern
            corrected_lookup = {(v.x, v.y, v.z): v for v in corrected_voxels}
            for i, voxel in enumerate(updated_voxels):
                if (voxel.x, voxel.y, voxel.z) in corrected_lookup:
                    updated_voxels[i] = corrected_lookup[(voxel.x, voxel.y, voxel.z)]

            updated_pattern = StoragePattern(
                voxels=updated_voxels,
                grid_size=pattern.grid_size,
                voxel_pitch=pattern.voxel_pitch,
                intensity_levels=pattern.intensity_levels,
                intensity_range=pattern.intensity_range,
                polarization_states=pattern.polarization_states,
                polarization_range=pattern.polarization_range,
                bits_per_voxel=pattern.bits_per_voxel,
                encoded_bit_length=pattern.encoded_bit_length,
                data_bit_length=pattern.data_bit_length,
                padding_bits=pattern.padding_bits,
                error_correction=pattern.error_correction,
                error_correction_metadata=pattern.error_correction_metadata,
                data_length_bytes=pattern.data_length_bytes,
                data_hash=pattern.data_hash,
            )
            return updated_pattern, remaining_anomalies

        return pattern, remaining_anomalies

    def acquire_polarization_resolved_images(self, pattern: StoragePattern, angles: List[float]) -> List[List[float]]:
        """Acquire polarization-resolved images from the storage pattern."""
        images = []
        for angle in angles:
            images.append(self._acquire_image(pattern, angle))
        return images

    def _acquire_image(self, pattern: StoragePattern, angle: float) -> List[float]:
        """Acquire a single image from the storage pattern."""
        # This is a placeholder method. In a real implementation, this would
        # involve actual image acquisition and processing.
        return [0.0] * 100

    def denoise_images(self, images: List[List[float]]) -> List[List[float]]:
        """Denoise the images."""
        # This is a placeholder method. In a real implementation, this would
        # involve actual denoising.
        return images

    def enhance_contrast(self, images: List[List[float]]) -> List[List[float]]:
        """Enhance the contrast of the images."""
        # This is a placeholder method. In a real implementation, this would
        # involve actual contrast enhancement.
        return images

    def extract_voxel_features(self, images: List[List[float]]) -> List[List[float]]:
        """Extract voxel features from the images."""
        # This is a placeholder method. In a real implementation, this would
        # involve actual feature extraction.
        return images

    def cluster_and_reconstruct_grid(self, features: List[List[float]]) -> List[Voxel]:
        """Cluster and reconstruct the grid."""
        # This is a placeholder method. In a real implementation, this would
        # involve actual clustering and reconstruction.
        return features

