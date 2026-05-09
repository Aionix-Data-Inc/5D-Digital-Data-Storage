"""Adaptive calibration system for 5D optical data storage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .storage_pattern import StoragePattern
from .voxel import Voxel


@dataclass
class CalibrationData:
    """Calibration data for a specific device and media batch."""

    device_id: str
    media_batch_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Laser-material interaction parameters
    intensity_calibration: Dict[int, float] = field(default_factory=dict)  # level -> actual_intensity
    polarization_calibration: Dict[int, float] = field(default_factory=dict)  # level -> actual_polarization

    # Optical distortion corrections
    distortion_matrix: List[List[float]] = field(default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    aberration_corrections: Dict[str, float] = field(default_factory=dict)

    # ML model versions and performance metrics
    ml_model_version: str = ""
    ml_accuracy_history: List[float] = field(default_factory=list)
    ml_training_samples: int = 0

    # Performance metrics
    write_success_rate: float = 1.0
    read_accuracy: float = 1.0
    error_rate_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "media_batch_id": self.media_batch_id,
            "timestamp": self.timestamp.isoformat(),
            "intensity_calibration": self.intensity_calibration,
            "polarization_calibration": self.polarization_calibration,
            "distortion_matrix": self.distortion_matrix,
            "aberration_corrections": self.aberration_corrections,
            "ml_model_version": self.ml_model_version,
            "ml_accuracy_history": self.ml_accuracy_history,
            "ml_training_samples": self.ml_training_samples,
            "write_success_rate": self.write_success_rate,
            "read_accuracy": self.read_accuracy,
            "error_rate_history": self.error_rate_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationData":
        """Deserialize from dictionary."""
        return cls(
            device_id=data["device_id"],
            media_batch_id=data["media_batch_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            intensity_calibration=data.get("intensity_calibration", {}),
            polarization_calibration=data.get("polarization_calibration", {}),
            distortion_matrix=data.get("distortion_matrix", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            aberration_corrections=data.get("aberration_corrections", {}),
            ml_model_version=data.get("ml_model_version", ""),
            ml_accuracy_history=data.get("ml_accuracy_history", []),
            ml_training_samples=data.get("ml_training_samples", 0),
            write_success_rate=data.get("write_success_rate", 1.0),
            read_accuracy=data.get("read_accuracy", 1.0),
            error_rate_history=data.get("error_rate_history", []),
        )


class AdaptiveCalibration:
    """Manages continuous calibration updates for optical storage devices."""

    def __init__(self, calibration_dir: str = "calibrations"):
        self.calibration_dir = calibration_dir
        os.makedirs(calibration_dir, exist_ok=True)
        self.current_calibration: Optional[CalibrationData] = None

    def load_calibration(self, device_id: str, media_batch_id: str) -> CalibrationData:
        """Load calibration data for a specific device and media batch."""
        filename = f"{device_id}_{media_batch_id}.json"
        filepath = os.path.join(self.calibration_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.current_calibration = CalibrationData.from_dict(data)
        else:
            self.current_calibration = CalibrationData(device_id, media_batch_id)

        return self.current_calibration

    def save_calibration(self, calibration: CalibrationData) -> None:
        """Save calibration data."""
        filename = f"{calibration.device_id}_{calibration.media_batch_id}.json"
        filepath = os.path.join(self.calibration_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(calibration.to_dict(), f, indent=2)

    def update_laser_material_model(self, measured_voxels: List[Voxel], target_levels: List[Tuple[int, int]]) -> None:
        """Update laser-material interaction model based on measurements."""
        if not self.current_calibration:
            return

        intensity_errors = {}
        polarization_errors = {}

        for voxel, (target_intensity, target_polarization) in zip(measured_voxels, target_levels):
            # Calculate deltas
            intensity_delta = voxel.intensity - target_intensity
            polarization_delta = voxel.polarization - target_polarization

            # Accumulate errors per level
            if target_intensity not in intensity_errors:
                intensity_errors[target_intensity] = []
            intensity_errors[target_intensity].append(intensity_delta)

            if target_polarization not in polarization_errors:
                polarization_errors[target_polarization] = []
            polarization_errors[target_polarization].append(polarization_delta)

        # Update calibration with average corrections
        for level, errors in intensity_errors.items():
            avg_correction = sum(errors) / len(errors)
            self.current_calibration.intensity_calibration[level] = avg_correction

        for level, errors in polarization_errors.items():
            avg_correction = sum(errors) / len(errors)
            self.current_calibration.polarization_calibration[level] = avg_correction

    def update_optical_distortions(self, distortion_measurements: Dict[str, float]) -> None:
        """Update optical distortion corrections."""
        if not self.current_calibration:
            return

        for key, value in distortion_measurements.items():
            self.current_calibration.aberration_corrections[key] = value

    def retrain_ml_model(self, new_samples: List[Tuple[Voxel, Tuple[int, int]]]) -> None:
        """Retrain ML models with new media samples.

        Placeholder: In production, this would integrate with TensorFlow/PyTorch
        to retrain CNN models on new calibration data.
        """
        if not self.current_calibration:
            return

        # Placeholder logic
        self.current_calibration.ml_training_samples += len(new_samples)
        self.current_calibration.ml_model_version = f"v{self.current_calibration.ml_training_samples // 100 + 1}"

        # Simulate accuracy improvement
        if self.current_calibration.ml_accuracy_history:
            last_accuracy = self.current_calibration.ml_accuracy_history[-1]
            new_accuracy = min(1.0, last_accuracy + 0.01)  # Small improvement
        else:
            new_accuracy = 0.95

        self.current_calibration.ml_accuracy_history.append(new_accuracy)

    def update_performance_metrics(self, write_success: bool, read_accuracy: float, error_rate: float) -> None:
        """Update performance metrics for continuous monitoring."""
        if not self.current_calibration:
            return

        # Update write success rate (rolling average)
        alpha = 0.1
        self.current_calibration.write_success_rate = (
            alpha * (1.0 if write_success else 0.0) +
            (1 - alpha) * self.current_calibration.write_success_rate
        )

        # Update read accuracy
        self.current_calibration.read_accuracy = read_accuracy

        # Update error rate history
        self.current_calibration.error_rate_history.append(error_rate)
        if len(self.current_calibration.error_rate_history) > 100:
            self.current_calibration.error_rate_history = self.current_calibration.error_rate_history[-100:]

    def apply_calibration_to_voxel(self, voxel: Voxel) -> Voxel:
        """Apply calibration corrections to a voxel."""
        if not self.current_calibration:
            return voxel

        # Apply intensity calibration
        intensity_correction = self.current_calibration.intensity_calibration.get(
            int(voxel.intensity), 0.0
        )
        corrected_intensity = voxel.intensity + intensity_correction

        # Apply polarization calibration
        polarization_correction = self.current_calibration.polarization_calibration.get(
            int(voxel.polarization), 0.0
        )
        corrected_polarization = voxel.polarization + polarization_correction

        # Apply optical distortion corrections (simplified)
        # In practice, this would involve matrix transformations
        distortion_factor = 1.0
        for correction in self.current_calibration.aberration_corrections.values():
            distortion_factor *= (1.0 + correction * 0.01)

        corrected_intensity *= distortion_factor

        return Voxel(
            x=voxel.x,
            y=voxel.y,
            z=voxel.z,
            intensity=corrected_intensity,
            polarization=corrected_polarization,
        )

    def get_calibration_summary(self) -> Dict:
        """Get a summary of current calibration status."""
        if not self.current_calibration:
            return {"status": "no_calibration_loaded"}

        return {
            "device_id": self.current_calibration.device_id,
            "media_batch_id": self.current_calibration.media_batch_id,
            "timestamp": self.current_calibration.timestamp.isoformat(),
            "ml_model_version": self.current_calibration.ml_model_version,
            "ml_accuracy": self.current_calibration.ml_accuracy_history[-1] if self.current_calibration.ml_accuracy_history else 0.0,
            "write_success_rate": self.current_calibration.write_success_rate,
            "read_accuracy": self.current_calibration.read_accuracy,
            "recent_error_rate": self.current_calibration.error_rate_history[-1] if self.current_calibration.error_rate_history else 0.0,
            "intensity_corrections": len(self.current_calibration.intensity_calibration),
            "polarization_corrections": len(self.current_calibration.polarization_calibration),
        }