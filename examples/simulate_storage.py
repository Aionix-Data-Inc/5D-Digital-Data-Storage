"""Example demonstrating 5D optical write/read simulation."""

from __future__ import annotations

import pathlib
import sys
import math

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from optical_storage import (
    Hamming74, 
    LaserReader, 
    LaserWriter,
    acquire_polarization_resolved_images,
    denoise_images,
    enhance_contrast,
    correct_alignment,
    extract_voxel_features,
    cluster_and_reconstruct_grid,
)
from optical_storage.noise import apply_gaussian_noise


def main() -> None:
    payload = b"5D optical storage with femtosecond lasers!"

    writer = LaserWriter(
        grid_size=(64, 64, 8),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=16,
        polarization_states=8,
        intensity_range=(0.2, 1.0),
        polarization_range=(0.0, 3.14159),
        error_correction=Hamming74(),
    )

    pattern = writer.write(payload)
    print("--- Write phase ---")
    for key, value in pattern.summary().items():
        print(f"{key:24s}: {value}")

    # Simulate image acquisition with polarization-resolved imaging
    angles = [i * math.pi / 4 for i in range(8)]  # 8 angles
    images = acquire_polarization_resolved_images(pattern, angles, noise_std=0.02, seed=7)
    print(f"\n--- Image acquisition ---")
    print(f"Acquired {len(images)} polarization-resolved images")

    # Preprocess images
    images = denoise_images(images, kernel_size=3)
    images = enhance_contrast(images)
    images = correct_alignment(images)
    print("Applied preprocessing: denoising, contrast enhancement, alignment correction")

    # Extract voxel features
    features = extract_voxel_features(images)
    print(f"Extracted {len(features)} voxel features")

    # Cluster and reconstruct voxel grid
    reconstructed_voxels = cluster_and_reconstruct_grid(features, threshold=0.1)
    print(f"Reconstructed {len(reconstructed_voxels)} voxels from features")

    # Read from reconstructed voxels
    reader = LaserReader(pattern)
    result = reader.read(reconstructed_voxels)

    print("\n--- Read phase ---")
    print(f"Recovered payload       : {result.data!r}")
    print(f"Corrected single-bit errs: {result.corrected_errors}")
    print(f"Detected uncorrectable  : {result.detected_uncorrectable}")
    print(f"Voxels processed        : {result.voxels_used}")


if __name__ == "__main__":
    main()
