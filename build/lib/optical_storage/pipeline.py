"""End-to-end pipeline orchestration for optical storage write and read cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .bit_utils import bytes_to_bits
from .image_processing import (
    acquire_polarization_resolved_images,
    correct_alignment,
    denoise_images,
    enhance_contrast,
    extract_voxel_features,
    cluster_and_reconstruct_grid,
)
from .noise import apply_gaussian_noise
from .reader import LaserReader, ReadResult
from .storage_pattern import StoragePattern
from .voxel import Voxel
from .writer import LaserInstruction, LaserWriter


@dataclass
class StoragePipelineResult:
    pattern: StoragePattern
    instructions: List[LaserInstruction]
    measured_voxels: List[Voxel]
    read_result: ReadResult
    voxel_state_comparison: Dict[str, object]


class OpticalStoragePipeline:
    """Orchestrate an end-to-end write/read pipeline for 5D optical storage."""

    def __init__(self, writer: LaserWriter, reader: Optional[LaserReader] = None) -> None:
        self.writer = writer
        self.reader = reader

    def write(self, data: bytes, verify: bool = True, overwrite: bool = False) -> StoragePattern:
        return self.writer.write(data, verify=verify, overwrite=overwrite)

    def generate_instructions(self, pattern: StoragePattern) -> List[LaserInstruction]:
        return self.writer.generate_laser_instructions(pattern)

    def simulate_measurements(
        self,
        pattern: StoragePattern,
        intensity_std: float = 0.01,
        polarization_std: float | None = None,
        orientation_std: float | None = None,
        seed: int | None = None,
    ) -> List[Voxel]:
        if orientation_std is None:
            orientation_std = polarization_std if polarization_std is not None else 0.01
        return apply_gaussian_noise(
            pattern,
            intensity_std=intensity_std,
            polarization_std=orientation_std,
            seed=seed,
        )

    def simulate_image_readout(
        self,
        pattern: StoragePattern,
        angles: List[float],
        noise_std: float = 0.01,
        seed: int | None = None,
        kernel_size: int = 3,
        threshold: float = 0.1,
    ) -> List[Voxel]:
        images = acquire_polarization_resolved_images(pattern, angles, noise_std=noise_std, seed=seed)
        images = denoise_images(images, kernel_size=kernel_size)
        images = enhance_contrast(images)
        images = correct_alignment(images)
        features = extract_voxel_features(images)
        return cluster_and_reconstruct_grid(features, threshold=threshold)

    def read(
        self,
        pattern: StoragePattern,
        measured_voxels: List[Voxel],
        parallel: bool = False,
        optimize_scan: bool = True,
        use_ml: bool = False,
        adjust_read_args: Optional[Dict[str, object]] = None,
    ) -> ReadResult:
        reader = self.reader or LaserReader(
            pattern,
            calibration=self.writer.calibration,
            anomaly_detector=self.writer.anomaly_detector,
            retry_manager=self.writer.retry_manager,
            version_manager=self.writer.version_manager,
        )

        if adjust_read_args:
            reader.adjust_read_parameters(
                intensity_range=adjust_read_args.get("intensity_range"),
                polarization_range=adjust_read_args.get("polarization_range"),
                uncertainty_threshold=adjust_read_args.get("uncertainty_threshold"),
            )

        return reader.read(
            measured_voxels,
            parallel=parallel,
            optimize_scan=optimize_scan,
            use_ml=use_ml,
        )

    def run(
        self,
        data: bytes,
        verify: bool = True,
        overwrite: bool = False,
        intensity_std: float = 0.01,
        polarization_std: float = 0.01,
        seed: int | None = None,
        read_kwargs: Optional[Dict[str, object]] = None,
    ) -> StoragePipelineResult:
        pattern = self.write(data, verify=verify, overwrite=overwrite)
        instructions = self.generate_instructions(pattern)
        measured_voxels = self.simulate_measurements(
            pattern,
            intensity_std=intensity_std,
            polarization_std=polarization_std,
            seed=seed,
        )
        read_result = self.read(
            pattern,
            measured_voxels,
            parallel=bool(read_kwargs.get("parallel") if read_kwargs else False),
            optimize_scan=bool(read_kwargs.get("optimize_scan") if read_kwargs else True),
            use_ml=bool(read_kwargs.get("use_ml") if read_kwargs else False),
            adjust_read_args=read_kwargs,
        )
        comparison = pattern.compare_voxel_states(measured_voxels)

        return StoragePipelineResult(
            pattern=pattern,
            instructions=instructions,
            measured_voxels=measured_voxels,
            read_result=read_result,
            voxel_state_comparison=comparison,
        )
