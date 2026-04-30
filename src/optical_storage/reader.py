"""Optical reader that reconstructs data from voxel measurements."""

from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from .bit_utils import bits_to_bytes, int_to_bits
from .error_correction import DecodingResult
from .storage_pattern import StoragePattern
from .voxel import Voxel
from .calibration import AdaptiveCalibration
from .anomaly_detection import AnomalyDetector, RetryManager, AnomalyLog
from .versioning import VersionManager


@dataclass
class ReadResult:
    """Result returned after reading a :class:`StoragePattern`."""

    data: bytes
    corrected_errors: int
    detected_uncorrectable: int
    voxels_used: int
    raw_bitstream: List[int] = field(repr=False)
    decoded_payload_bits: List[int] = field(repr=False)
    hash_valid: bool = True
    uncertain_voxels: List[Voxel] = field(default_factory=list)
    block_confidence_scores: List[float] = field(default_factory=list, repr=False)
    confidence_summary: Dict[str, float] = field(default_factory=dict, repr=False)


class LaserReader:
    """Recover the original bytes from a 5D voxel lattice."""

    def __init__(self, pattern: StoragePattern, calibration: AdaptiveCalibration | None = None,
                 anomaly_detector: AnomalyDetector | None = None, retry_manager: RetryManager | None = None,
                 version_manager: VersionManager | None = None) -> None:
        self.pattern = pattern
        self.calibration = calibration
        self.anomaly_detector = anomaly_detector
        self.retry_manager = retry_manager
        self.version_manager = version_manager
        self._validate_pattern()

        self._override_intensity_levels = pattern.intensity_levels
        self._override_polarization_states = pattern.polarization_states
        self._override_intensity_range = pattern.intensity_range
        self._override_polarization_range = pattern.polarization_range
        self._uncertainty_threshold = 0.1
        self._update_dynamic_parameters()

        self._region_cache = {}  # Cache for processed voxel regions
        # Initialize anomaly detection and retry management if not provided
        if self.anomaly_detector is None:
            device_id = getattr(self.calibration.current_calibration, 'device_id', 'default_device') if self.calibration else 'default_device'
            media_batch = getattr(self.calibration.current_calibration, 'media_batch_id', 'default_batch') if self.calibration else 'default_batch'
            self.anomaly_detector = AnomalyDetector(device_id=device_id, media_batch_id=media_batch)

        if self.retry_manager is None:
            self.retry_manager = RetryManager()

        if self.version_manager is None:
            self.version_manager = VersionManager()

        self.logger = logging.getLogger(__name__)

    def read(self, voxels: Sequence[Voxel] | None = None, parallel: bool = False, optimize_scan: bool = True, use_ml: bool = False) -> ReadResult:
        # Check version compatibility
        if not self.validate_version_compatibility():
            raise ValueError(f"Incompatible encoding version: {self.get_version_info()['encoding_version']}")

        voxels = list(voxels) if voxels is not None else list(self.pattern.voxels)
        if not voxels and self.pattern.encoded_bit_length:
            raise ValueError("No voxels provided for decoding")

        # Apply calibration corrections to input voxels
        if self.calibration:
            voxels = [self.calibration.apply_calibration_to_voxel(v) for v in voxels]

        if optimize_scan:
            voxels = self._optimize_scan_path(voxels)

        required_bits = self.pattern.encoded_bit_length + self.pattern.padding_bits

        if parallel and len(voxels) > 100:  # Only parallel for large datasets
            collected_bits, voxels_used, uncertain_voxels, block_confidence_scores = self._read_parallel(voxels, required_bits, use_ml)
        else:
            collected_bits, voxels_used, uncertain_voxels, block_confidence_scores = self._read_sequential(voxels, required_bits, use_ml)

        if required_bits and len(collected_bits) < required_bits:
            raise ValueError("Insufficient voxel data to reconstruct payload")

        collected_bits = collected_bits[:required_bits] if required_bits else collected_bits
        if self.pattern.padding_bits:
            collected_bits = collected_bits[: -self.pattern.padding_bits]

        # Trim confidence scores if there were extra voxels read beyond required_bits.
        block_confidence_scores = block_confidence_scores[:voxels_used]

        ecc_result: DecodingResult = self.pattern.error_correction.decode(collected_bits)
        payload_bits = ecc_result.bits[: self.pattern.data_bit_length]
        data = bits_to_bytes(payload_bits)
        data = data[: self.pattern.data_length_bytes]

        # Validate hash
        computed_hash = hashlib.sha256(data).hexdigest()
        hash_valid = computed_hash == self.pattern.data_hash

        if ecc_result.detected_uncorrectable or not hash_valid:
            self.logger.warning(
                "Data integrity check failed during read: uncorrectable=%s hash_valid=%s",
                ecc_result.detected_uncorrectable,
                hash_valid,
            )
            data = b""
            hash_valid = False

        confidence_summary: Dict[str, float] = {}
        if block_confidence_scores:
            confidence_summary = {
                "min": min(block_confidence_scores),
                "max": max(block_confidence_scores),
                "mean": sum(block_confidence_scores) / len(block_confidence_scores),
            }

        return ReadResult(
            data=data,
            corrected_errors=ecc_result.corrected_errors,
            detected_uncorrectable=ecc_result.detected_uncorrectable,
            voxels_used=voxels_used,
            raw_bitstream=collected_bits,
            decoded_payload_bits=payload_bits,
            hash_valid=hash_valid,
            uncertain_voxels=uncertain_voxels,
            block_confidence_scores=block_confidence_scores,
            confidence_summary=confidence_summary,
        )

    def compare_voxel_states(
        self,
        observed_voxels: Sequence[Voxel],
        intensity_tolerance: float = 0.05,
        polarization_tolerance: float = 0.05,
        max_differences: int = 50,
    ) -> Dict[str, Any]:
        """Compare pattern expectations against observed voxel measurements."""
        return self.pattern.compare_voxel_states(
            observed_voxels,
            intensity_tolerance=intensity_tolerance,
            polarization_tolerance=polarization_tolerance,
            max_differences=max_differences,
        )

    def adjust_read_parameters(
        self,
        intensity_range: tuple[float, float] | None = None,
        polarization_range: tuple[float, float] | None = None,
        uncertainty_threshold: float | None = None,
    ) -> None:
        """Adjust decoding parameters dynamically during read operations."""
        if intensity_range is not None:
            if intensity_range[0] >= intensity_range[1]:
                raise ValueError("invalid intensity_range")
            self._override_intensity_range = intensity_range

        if polarization_range is not None:
            if polarization_range[0] >= polarization_range[1]:
                raise ValueError("invalid polarization_range")
            self._override_polarization_range = polarization_range

        if uncertainty_threshold is not None:
            if uncertainty_threshold < 0:
                raise ValueError("uncertainty_threshold must be non-negative")
            self._uncertainty_threshold = uncertainty_threshold

        self._update_dynamic_parameters()

    def update_calibration_from_read(self, result: ReadResult, original_voxels: List[Voxel]) -> None:
        """Update calibration based on read results."""
        if not self.calibration or not self.calibration.current_calibration:
            return

        # Update performance metrics
        read_accuracy = 1.0 - (result.corrected_errors + result.detected_uncorrectable) / len(result.raw_bitstream)
        error_rate = (result.corrected_errors + result.detected_uncorrectable) / len(result.raw_bitstream)

        self.calibration.update_performance_metrics(
            write_success=True,  # Assuming write was successful if we got here
            read_accuracy=read_accuracy,
            error_rate=error_rate
        )

        # Retrain ML model with uncertain voxels as training samples
        if result.uncertain_voxels:
            # For ML retraining, we need target levels - this is simplified
            training_samples = []
            for voxel in result.uncertain_voxels[:10]:  # Limit to avoid too much computation
                # In practice, we'd need the expected levels from the pattern
                target_intensity = self._physical_to_level(voxel.intensity, self.pattern.intensity_levels, self.pattern.intensity_range)
                target_polarization = self._physical_to_level(voxel.polarization, self.pattern.polarization_states, self.pattern.polarization_range)
                training_samples.append((voxel, (target_intensity, target_polarization)))

            if training_samples:
                self.calibration.retrain_ml_model(training_samples)

        # Save updated calibration
        self.calibration.save_calibration(self.calibration.current_calibration)

    # ------------------------------------------------------------------
    # Helpers

    def _validate_pattern(self) -> None:
        if self.pattern.bits_per_voxel <= 0 and self.pattern.encoded_bit_length:
            raise ValueError("Pattern does not contain encodable information")

    def _update_dynamic_parameters(self) -> None:
        self.bits_per_intensity = self._bits_for_levels(self._override_intensity_levels)
        self.bits_per_polarization = self._bits_for_levels(self._override_polarization_states)
        self.bits_per_voxel = self.bits_per_intensity + self.bits_per_polarization

    @staticmethod
    def _bits_for_levels(levels: int) -> int:
        if levels <= 0:
            raise ValueError("levels must be positive")
        if levels & (levels - 1):
            raise ValueError("levels must be a power of two for binary encoding")
        return 0 if levels == 1 else levels.bit_length() - 1

    @staticmethod
    def _physical_to_level(value: float, levels: int, value_range: tuple[float, float]) -> int:
        min_val, max_val = value_range
        if levels == 1:
            return 0
        value_clamped = max(min(value, max_val), min_val)
        step = (max_val - min_val) / float(levels - 1)
        if step == 0:
            return 0
        normalized = (value_clamped - min_val) / step
        level = int(round(normalized))
        return max(0, min(level, levels - 1))

    @staticmethod
    def _physical_to_level_with_uncertainty(
        value: float,
        levels: int,
        value_range: tuple[float, float],
        uncertainty_threshold: float = 0.1,
    ) -> tuple[int, bool]:
        """Return level and whether it's uncertain (close to boundary)."""
        min_val, max_val = value_range
        if levels == 1:
            return 0, False
        value_clamped = max(min(value, max_val), min_val)
        step = (max_val - min_val) / float(levels - 1)
        if step == 0:
            return 0, False
        normalized = (value_clamped - min_val) / step
        level = int(round(normalized))
        level = max(0, min(level, levels - 1))
        # Check if close to boundary
        distance_to_level = abs(normalized - level)
        uncertain = distance_to_level > uncertainty_threshold
        return level, uncertain

    def _compute_confidence(
        self,
        value: float,
        levels: int,
        value_range: tuple[float, float],
    ) -> float:
        if levels == 1:
            return 1.0
        min_val, max_val = value_range
        value_clamped = max(min(value, max_val), min_val)
        step = (max_val - min_val) / float(levels - 1)
        if step == 0:
            return 1.0
        normalized = (value_clamped - min_val) / step
        distance = abs(normalized - round(normalized))
        return max(0.0, 1.0 - min(distance, 0.5) * 2.0)

    @staticmethod
    def _aggregate_confidences(intensity_confidence: float, polarization_confidence: float) -> float:
        if intensity_confidence is None:
            return polarization_confidence
        if polarization_confidence is None:
            return intensity_confidence
        return (intensity_confidence + polarization_confidence) / 2.0

    def _optimize_scan_path(self, voxels: List[Voxel]) -> List[Voxel]:
        """Sort voxels to minimize motion latency (z-major order)."""
        return sorted(voxels, key=lambda v: (v.z, v.y, v.x))

    def _read_sequential(self, voxels: List[Voxel], required_bits: int, use_ml: bool = False) -> tuple[List[int], int, List[Voxel], List[float]]:
        collected_bits: List[int] = []
        voxels_used = 0
        uncertain_voxels: List[Voxel] = []
        block_confidence_scores: List[float] = []
        for voxel in voxels:
            if required_bits and len(collected_bits) >= required_bits:
                break
            if use_ml:
                intensity_level, polarization_level = self.classify_voxel_with_ml(voxel)
                intensity_confidence = self._compute_confidence(
                    voxel.intensity,
                    self._override_intensity_levels,
                    self._override_intensity_range,
                )
                polarization_confidence = self._compute_confidence(
                    voxel.polarization,
                    self._override_polarization_states,
                    self._override_polarization_range,
                )
                intensity_uncertain = intensity_confidence < 0.5
                polarization_uncertain = polarization_confidence < 0.5
            else:
                intensity_level, intensity_uncertain = self._physical_to_level_with_uncertainty(
                    voxel.intensity,
                    self._override_intensity_levels,
                    self._override_intensity_range,
                    self._uncertainty_threshold,
                )
                polarization_level, polarization_uncertain = self._physical_to_level_with_uncertainty(
                    voxel.polarization,
                    self._override_polarization_states,
                    self._override_polarization_range,
                    self._uncertainty_threshold,
                )
                intensity_confidence = self._compute_confidence(
                    voxel.intensity,
                    self._override_intensity_levels,
                    self._override_intensity_range,
                )
                polarization_confidence = self._compute_confidence(
                    voxel.polarization,
                    self._override_polarization_states,
                    self._override_polarization_range,
                )
            if intensity_uncertain or polarization_uncertain:
                uncertain_voxels.append(voxel)
            if self.bits_per_intensity:
                collected_bits.extend(int_to_bits(intensity_level, self.bits_per_intensity))
            if self.bits_per_polarization:
                collected_bits.extend(int_to_bits(polarization_level, self.bits_per_polarization))
            block_confidence_scores.append(
                self._aggregate_confidences(intensity_confidence, polarization_confidence)
            )
            voxels_used += 1
        return collected_bits, voxels_used, uncertain_voxels, block_confidence_scores

    def _read_parallel(self, voxels: List[Voxel], required_bits: int, use_ml: bool = False) -> tuple[List[int], int, List[Voxel], List[float]]:
        """Process voxels in parallel chunks."""
        num_chunks = min(4, len(voxels) // 50 + 1)  # Up to 4 processes
        chunk_size = max(1, len(voxels) // num_chunks)
        chunks = [voxels[i:i + chunk_size] for i in range(0, len(voxels), chunk_size)]
        max_voxels = math.ceil(required_bits / self.bits_per_voxel) if required_bits else len(voxels)

        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            futures = [executor.submit(self._process_chunk, chunk, required_bits, use_ml) for chunk in chunks]
            results = [f.result() for f in futures]

        # Combine results
        all_bits = []
        total_used = 0
        all_uncertain = []
        all_confidences: List[float] = []
        for bits, used, uncertain, confidences in results:
            all_bits.extend(bits)
            total_used += used
            all_uncertain.extend(uncertain)
            all_confidences.extend(confidences)
            if required_bits and len(all_bits) >= required_bits:
                all_bits = all_bits[:required_bits]
                all_uncertain = all_uncertain[:max_voxels]
                all_confidences = all_confidences[:max_voxels]
                total_used = min(total_used, max_voxels)
                break

        if required_bits and len(all_bits) > required_bits:
            all_bits = all_bits[:required_bits]

        return all_bits, total_used, all_uncertain, all_confidences

    def _process_chunk(self, chunk: List[Voxel], required_bits: int, use_ml: bool = False) -> tuple[List[int], int, List[Voxel], List[float]]:
        """Process a chunk of voxels."""
        return self._read_sequential(chunk, required_bits, use_ml)

    @lru_cache(maxsize=1024)
    def _cached_level_conversion(self, value: float, levels: int, min_val: float, max_val: float) -> int:
        """Cached level conversion for frequently accessed values."""
        return self._physical_to_level(value, levels, (min_val, max_val))

    def classify_voxel_with_ml(self, voxel: Voxel) -> tuple[int, int]:
        """Use ML model (CNN) to classify voxel intensity and polarization levels.
        
        Placeholder: Currently uses simple thresholding. 
        In production, integrate with TensorFlow/PyTorch for CNN classification.
        """
        # Placeholder implementation uses current dynamic parameter mapping
        intensity_level = self._physical_to_level(
            voxel.intensity,
            self._override_intensity_levels,
            self._override_intensity_range,
        )
        polarization_level = self._physical_to_level(
            voxel.polarization,
            self._override_polarization_states,
            self._override_polarization_range,
        )
        return intensity_level, polarization_level

    def detect_anomalies_read_feedback(
        self,
        measured_voxels: List[Voxel]
    ) -> List[AnomalyLog]:
        """Detect anomalies in read measurements."""
        if not self.anomaly_detector:
            return []

        anomalies = []

        # Detect optical noise spikes
        noise_spikes = self.anomaly_detector.detect_optical_noise_spikes(measured_voxels)
        anomalies.extend(noise_spikes)

        # Detect hardware drift
        drift_anomalies = self.anomaly_detector.detect_hardware_drift(measured_voxels)
        anomalies.extend(drift_anomalies)

        # Log anomalies for retraining
        self.anomaly_detector.log_anomalies(anomalies)

        return anomalies

    def retry_failed_read_regions(
        self,
        anomalies: List[AnomalyLog],
        measured_voxels: List[Voxel]
    ) -> Tuple[List[Voxel], List[AnomalyLog]]:
        """Retry reading failed regions and return corrected voxels."""
        if not self.retry_manager or not anomalies:
            return measured_voxels, []

        # Identify regions that need retry
        retry_regions = self.retry_manager.identify_retry_regions(anomalies)

        corrected_voxels = measured_voxels.copy()
        remaining_anomalies = []

        for region in retry_regions:
            if self.retry_manager.should_retry_region(region):
                retry_result = self.retry_manager.retry_read_region(region, self)
                if retry_result.success:
                    # In a real implementation, this would merge newly measured voxels
                    # For simulation, we'll mark the region as corrected
                    pass
                else:
                    # Create anomaly log for failed retry
                    remaining_anomaly = AnomalyLog(
                        anomaly_type="read_retry_failed",
                        region_bounds=region,
                        retry_count=retry_result.retry_count,
                        device_id=self.anomaly_detector.device_id if self.anomaly_detector else "",
                        media_batch_id=self.anomaly_detector.media_batch_id if self.anomaly_detector else ""
                    )
                    remaining_anomalies.append(remaining_anomaly)

        return corrected_voxels, remaining_anomalies

    def validate_version_compatibility(self) -> bool:
        """Validate that the pattern version is compatible with current reader."""
        if not self.pattern.version_info:
            # No version info - assume V1.0 compatibility
            return True

        current_version_info = self.version_manager.create_version_info()
        pattern_version = self.pattern.version_info.encoding_version.value
        return (
            current_version_info.is_compatible(pattern_version)
            or self.pattern.version_info.is_compatible(current_version_info.encoding_version.value)
        )

    def get_version_info(self) -> dict:
        """Get version information for the pattern."""
        if not self.pattern.version_info:
            return {"encoding_version": "unknown", "firmware_version": "unknown"}

        return {
            "encoding_version": self.pattern.version_info.encoding_version.value,
            "firmware_version": self.pattern.version_info.firmware_version,
            "created_at": self.pattern.version_info.created_at.isoformat(),
            "compatible_versions": self.pattern.version_info.compatible_versions,
        }
