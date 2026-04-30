"""Optical storage simulation package for 5D femtosecond laser writing."""

from .bit_utils import bytes_to_bits, bits_to_bytes
from .error_correction import ErrorCorrectionScheme, Hamming74, NoErrorCorrection, Parity8
from .reader import LaserReader, ReadResult
from .storage_pattern import StoragePattern
from .voxel import Voxel
from .writer import LaserWriter, LaserInstruction
from .noise import apply_gaussian_noise
from .calibration import AdaptiveCalibration, CalibrationData
from .anomaly_detection import AnomalyDetector, RetryManager, AnomalyLog, RetryResult
from .versioning import VersionManager, ImmutableWriteLog, WriteLogEntry, VersionInfo, EncodingVersion
from .image_processing import (
    acquire_polarization_resolved_images,
    denoise_images,
    enhance_contrast,
    correct_alignment,
    extract_voxel_features,
    cluster_and_reconstruct_grid,
    PolarizationImage,
    VoxelFeature,
)
from .pipeline import OpticalStoragePipeline, StoragePipelineResult

__all__ = [
    "LaserWriter",
    "LaserInstruction",
    "LaserReader",
    "ReadResult",
    "StoragePattern",
    "Voxel",
    "bytes_to_bits",
    "bits_to_bytes",
    "ErrorCorrectionScheme",
    "NoErrorCorrection",
    "Hamming74",
    "Parity8",
    "apply_gaussian_noise",
    "AdaptiveCalibration",
    "CalibrationData",
    "AnomalyDetector",
    "RetryManager",
    "AnomalyLog",
    "RetryResult",
    "OpticalStoragePipeline",
    "StoragePipelineResult",
    "acquire_polarization_resolved_images",
    "denoise_images",
    "enhance_contrast",
    "correct_alignment",
    "extract_voxel_features",
    "cluster_and_reconstruct_grid",
    "PolarizationImage",
    "VoxelFeature",
]
