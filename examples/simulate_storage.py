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
    cluster_and_reconstruct_grid,
    process_polarization_image_pipeline,
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

    # Simulate and process polarization-resolved image data in one pipeline
    angles = [i * math.pi / 4 for i in range(8)]  # 8 angles
    pipeline_result = process_polarization_image_pipeline(
        pattern,
        angles,
        noise_std=0.02,
        seed=7,
        kernel_size=3,
        threshold=0.1,
        projection="sum",
    )

    print(f"\n--- Image pipeline ---")
    print(f"Acquired {len(pipeline_result.images)} polarization-resolved images")
    print(f"Denoised images      : {len(pipeline_result.denoised_images)}")
    print(f"Enhanced images      : {len(pipeline_result.enhanced_images)}")
    print(f"Aligned images       : {len(pipeline_result.aligned_images)}")
    print(f"Extracted features   : {len(pipeline_result.features)}")
    print(f"Reconstructed voxels : {len(pipeline_result.reconstructed_voxels)}")
    print("Note: reconstructed voxels are estimated from image data and may not always recover the full payload.")

    # Read from reconstructed voxels
    reader = LaserReader(pattern)
    result = reader.read(pipeline_result.reconstructed_voxels)

    print("\n--- Read phase ---")
    print(f"Recovered payload       : {result.data!r}")
    print(f"Corrected single-bit errs: {result.corrected_errors}")
    print(f"Detected uncorrectable  : {result.detected_uncorrectable}")
    print(f"Voxels processed        : {result.voxels_used}")


if __name__ == "__main__":
    main()
