"""Example demonstrating 5D optical write/read simulation."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from optical_storage import Hamming74, LaserReader, LaserWriter
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

    # Simulate measurement noise
    noisy_voxels = apply_gaussian_noise(pattern, intensity_std=0.005, polarization_std=0.005, seed=7)

    reader = LaserReader(pattern)
    result = reader.read(noisy_voxels)

    print("\n--- Read phase ---")
    print(f"Recovered payload       : {result.data!r}")
    print(f"Corrected single-bit errs: {result.corrected_errors}")
    print(f"Detected uncorrectable  : {result.detected_uncorrectable}")
    print(f"Voxels processed        : {result.voxels_used}")


if __name__ == "__main__":
    main()
