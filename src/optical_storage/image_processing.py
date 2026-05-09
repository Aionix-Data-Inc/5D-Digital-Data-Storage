"""Image processing pipeline for polarization-resolved optical data storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from secrets import SystemRandom
from typing import List, Optional, Tuple
import math
import random

from .storage_pattern import StoragePattern


class ImageProcessingError(ValueError):
    """Raised when image processing inputs or integrity checks fail."""


@dataclass
class PolarizationImage:
    """Simulated polarization-resolved image data."""
    angle: float  # Polarization angle in radians
    data: List[List[float]]  # 2D intensity data
    integrity_hash: Optional[str] = None
    processing_steps: List[str] = field(default_factory=list)


@dataclass
class VoxelFeature:
    """Extracted voxel features."""
    x: float
    y: float
    z: float
    delta: float  # Depolarization or intensity measure
    theta: float  # Polarization angle


@dataclass
class ImageProcessingResult:
    """Structured result for the full image processing pipeline."""
    images: List[PolarizationImage]
    denoised_images: List[PolarizationImage]
    enhanced_images: List[PolarizationImage]
    aligned_images: List[PolarizationImage]
    features: List[VoxelFeature]
    reconstructed_voxels: List["Voxel"]


def _create_noise_rng(seed: Optional[int]) -> random.Random:
    if seed is None:
        secure_seed = SystemRandom().randbits(64)
        return random.Random(secure_seed)
    if not isinstance(seed, int):
        raise ImageProcessingError("Seed must be an integer or None.")
    return random.Random(seed)


def _validate_angles(angles: List[float]) -> None:
    if not angles:
        raise ImageProcessingError("At least one polarization angle is required.")
    for angle in angles:
        if not isinstance(angle, (float, int)):
            raise ImageProcessingError("Polarization angles must be numeric.")
        if math.isnan(angle) or math.isinf(angle):
            raise ImageProcessingError("Polarization angles must be finite values.")


def _validate_noise_std(noise_std: float) -> None:
    if not isinstance(noise_std, (float, int)) or noise_std < 0:
        raise ImageProcessingError("Noise standard deviation must be a non-negative number.")


def _validate_kernel_size(kernel_size: int) -> None:
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ImageProcessingError("Kernel size must be a positive odd integer.")


def _validate_threshold(threshold: float) -> None:
    if not isinstance(threshold, (float, int)) or threshold < 0:
        raise ImageProcessingError("Threshold must be a non-negative number.")


def _normalize_angle(angle: float) -> float:
    normalized = angle % (2 * math.pi)
    return normalized if normalized >= 0 else normalized + 2 * math.pi


def _compute_image_integrity_hash(angle: float, data: List[List[float]]) -> str:
    hasher = sha256()
    hasher.update(f"{angle:.12g}".encode("utf-8"))
    for row in data:
        for value in row:
            hasher.update(f"{value:.12g}".encode("utf-8"))
    return hasher.hexdigest()


def compute_image_integrity_hash(image: PolarizationImage) -> str:
    """Compute a stable hash for a polarization image and its angle."""
    return _compute_image_integrity_hash(image.angle, image.data)


def verify_image_integrity(image: PolarizationImage, expected_hash: str) -> bool:
    """Verify that the image data matches the expected integrity hash."""
    if not expected_hash:
        raise ImageProcessingError("Expected hash must be a non-empty string.")
    return compute_image_integrity_hash(image) == expected_hash


def _project_intensity(data: List[List[float]], x: int, y: int, value: float, projection: str) -> None:
    if projection == "sum":
        data[x][y] += value
    elif projection == "max":
        data[x][y] = max(data[x][y], value)
    elif projection == "average":
        data[x][y] = (data[x][y] + value) / 2
    else:
        raise ImageProcessingError("Projection must be 'sum', 'max', or 'average'.")


def acquire_polarization_resolved_images(
    pattern: StoragePattern,
    angles: List[float],
    noise_std: float = 0.01,
    seed: Optional[int] = None,
    projection: str = "sum",
) -> List[PolarizationImage]:
    """Simulate acquisition of polarization-resolved images.

    Algorithm:
      1. Validate pattern and acquisition parameters.
      2. Create a deterministic or secure RNG.
      3. For each polarization angle, build a 2D projection.
      4. Apply polarization modulation and sensor noise.
      5. Compute integrity hashes for each generated image.
    """
    if not isinstance(pattern, StoragePattern):
        raise ImageProcessingError("pattern must be a StoragePattern instance.")
    _validate_angles(angles)
    _validate_noise_std(noise_std)

    rng = _create_noise_rng(seed)
    grid_x, grid_y, _ = pattern.grid_size

    images: List[PolarizationImage] = []

    for angle in angles:
        angle = _normalize_angle(float(angle))
        data = [[0.0 for _ in range(grid_y)] for _ in range(grid_x)]
        processing_steps = ["allocate image grid", "project voxels", "apply noise"]

        for voxel in pattern.voxels:
            pol_diff = abs(voxel.polarization - angle) % (2 * math.pi)
            if pol_diff > math.pi:
                pol_diff = 2 * math.pi - pol_diff
            modulation = math.cos(pol_diff / 2) ** 2
            intensity = voxel.intensity * modulation
            intensity += rng.gauss(0, noise_std)
            intensity = max(0.0, intensity)

            if 0 <= voxel.x < grid_x and 0 <= voxel.y < grid_y:
                _project_intensity(data, voxel.x, voxel.y, intensity, projection)

        integrity_hash = _compute_image_integrity_hash(angle, data)
        images.append(
            PolarizationImage(
                angle=angle,
                data=data,
                integrity_hash=integrity_hash,
                processing_steps=processing_steps,
            )
        )

    return images


def denoise_images(images: List[PolarizationImage], kernel_size: int = 3) -> List[PolarizationImage]:
    """Apply mean filter denoising to each image."""
    _validate_kernel_size(kernel_size)

    def mean_filter(data: List[List[float]], k: int) -> List[List[float]]:
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                total = 0.0
                count = 0
                for di in range(-k // 2, k // 2 + 1):
                    for dj in range(-k // 2, k // 2 + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            total += data[ni][nj]
                            count += 1
                result[i][j] = total / count if count > 0 else 0.0
        return result

    output: List[PolarizationImage] = []
    for img in images:
        data = mean_filter(img.data, kernel_size)
        integrity_hash = _compute_image_integrity_hash(img.angle, data)
        output.append(
            PolarizationImage(
                angle=img.angle,
                data=data,
                integrity_hash=integrity_hash,
                processing_steps=img.processing_steps + ["denoise"],
            )
        )
    return output


def enhance_contrast(images: List[PolarizationImage]) -> List[PolarizationImage]:
    """Normalize each image to the [0,1] range to enhance contrast."""
    def normalize(data: List[List[float]]) -> List[List[float]]:
        flat = [value for row in data for value in row]
        if not flat:
            return data
        min_val = min(flat)
        max_val = max(flat)
        if max_val == min_val:
            return [[0.0 for _ in row] for row in data]
        return [[(value - min_val) / (max_val - min_val) for value in row] for row in data]

    output: List[PolarizationImage] = []
    for img in images:
        data = normalize(img.data)
        integrity_hash = _compute_image_integrity_hash(img.angle, data)
        output.append(
            PolarizationImage(
                angle=img.angle,
                data=data,
                integrity_hash=integrity_hash,
                processing_steps=img.processing_steps + ["enhance_contrast"],
            )
        )
    return output


def _compute_centroid(data: List[List[float]]) -> Tuple[float, float]:
    total = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if value <= 0:
                continue
            total += value
            sum_x += i * value
            sum_y += j * value
    if total == 0.0:
        return 0.0, 0.0
    return sum_x / total, sum_y / total


def _translate_image(data: List[List[float]], dx: int, dy: int) -> List[List[float]]:
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0
    shifted = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            ni = i + dx
            nj = j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                shifted[ni][nj] = data[i][j]
    return shifted


def correct_alignment(images: List[PolarizationImage]) -> List[PolarizationImage]:
    """Align images by centroid registration to reduce acquisition drift."""
    if not images:
        return []

    reference_centroid = _compute_centroid(images[0].data)
    output: List[PolarizationImage] = []

    for img in images:
        current_centroid = _compute_centroid(img.data)
        dx = int(round(reference_centroid[0] - current_centroid[0]))
        dy = int(round(reference_centroid[1] - current_centroid[1]))
        data = _translate_image(img.data, dx, dy)
        integrity_hash = _compute_image_integrity_hash(img.angle, data)
        output.append(
            PolarizationImage(
                angle=img.angle,
                data=data,
                integrity_hash=integrity_hash,
                processing_steps=img.processing_steps + ["correct_alignment"],
            )
        )

    return output


def extract_voxel_features(images: List[PolarizationImage]) -> List[VoxelFeature]:
    """Extract voxel features from polarization-resolved images.

    Algorithm:
      1. Validate the input image stack.
      2. For each pixel, build the intensity response vector.
      3. Compute delta as the intensity range and theta as the weighted polarization.
      4. Emit a feature only when the intensity signal is significant.
    """
    if not images:
        return []

    grid_x = len(images[0].data)
    grid_y = len(images[0].data[0]) if grid_x > 0 else 0
    features: List[VoxelFeature] = []

    for i in range(grid_x):
        for j in range(grid_y):
            intensities = [img.data[i][j] for img in images]
            if not intensities:
                continue
            max_intensity = max(intensities)
            min_intensity = min(intensities)
            delta = max_intensity - min_intensity
            if delta <= 0:
                continue

            sum_sin = 0.0
            sum_cos = 0.0
            for intensity, img in zip(intensities, images):
                angle = _normalize_angle(img.angle)
                sum_sin += intensity * math.sin(angle)
                sum_cos += intensity * math.cos(angle)

            if sum_sin == 0 and sum_cos == 0:
                theta = images[intensities.index(max_intensity)].angle
            else:
                theta = math.atan2(sum_sin, sum_cos)
                theta = _normalize_angle(theta)

            features.append(
                VoxelFeature(
                    x=float(i),
                    y=float(j),
                    z=0.0,
                    delta=float(delta),
                    theta=theta,
                )
            )

    return features


def cluster_and_reconstruct_grid(features: List[VoxelFeature], threshold: float = 0.1) -> List["Voxel"]:
    """Cluster features and reconstruct voxel grid with a threshold.

    Algorithm:
      1. Validate the feature list and threshold.
      2. Convert feature coordinates back to integer voxel locations.
      3. Preserve polarization and intensity metadata.
    """
    from .voxel import Voxel

    _validate_threshold(threshold)
    voxels: List[Voxel] = []
    for feature in features:
        if feature.delta <= threshold:
            continue
        voxels.append(
            Voxel(
                x=int(round(feature.x)),
                y=int(round(feature.y)),
                z=int(round(feature.z)),
                intensity=feature.delta,
                polarization=_normalize_angle(feature.theta),
            )
        )
    return voxels


def process_polarization_image_pipeline(
    pattern: StoragePattern,
    angles: List[float],
    noise_std: float = 0.01,
    seed: Optional[int] = None,
    kernel_size: int = 3,
    threshold: float = 0.1,
    projection: str = "sum",
) -> ImageProcessingResult:
    """Execute the full image processing pipeline end to end."""
    images = acquire_polarization_resolved_images(
        pattern=pattern,
        angles=angles,
        noise_std=noise_std,
        seed=seed,
        projection=projection,
    )
    denoised = denoise_images(images, kernel_size=kernel_size)
    enhanced = enhance_contrast(denoised)
    aligned = correct_alignment(enhanced)
    features = extract_voxel_features(aligned)
    voxels = cluster_and_reconstruct_grid(features, threshold=threshold)

    return ImageProcessingResult(
        images=images,
        denoised_images=denoised,
        enhanced_images=enhanced,
        aligned_images=aligned,
        features=features,
        reconstructed_voxels=voxels,
    )