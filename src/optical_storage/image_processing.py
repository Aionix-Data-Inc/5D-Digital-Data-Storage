"""Image processing pipeline for polarization-resolved optical data storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import random

from .storage_pattern import StoragePattern


@dataclass
class PolarizationImage:
    """Simulated polarization-resolved image data."""
    angle: float  # Polarization angle in radians
    data: List[List[float]]  # 2D intensity data


@dataclass
class VoxelFeature:
    """Extracted voxel features."""
    x: float
    y: float
    z: float
    delta: float  # Depolarization or intensity measure
    theta: float  # Polarization angle


def acquire_polarization_resolved_images(
    pattern: StoragePattern, 
    angles: List[float], 
    noise_std: float = 0.01, 
    seed: int | None = None
) -> List[PolarizationImage]:
    """Simulate acquisition of polarization-resolved images from storage pattern.
    
    For each polarization angle, generate a 2D image where voxels appear
    based on their polarization matching the angle.
    """
    rng = random.Random(seed)
    images = []
    
    grid_x, grid_y, grid_z = pattern.grid_size
    
    for angle in angles:
        # Create a 2D projection or slice; for simplicity, sum over z
        data = [[0.0 for _ in range(grid_y)] for _ in range(grid_x)]
        
        for voxel in pattern.voxels:
            # Simulate polarization-dependent intensity
            # Intensity modulated by polarization difference
            pol_diff = abs(voxel.polarization - angle) % (2 * math.pi)
            if pol_diff > math.pi:
                pol_diff = 2 * math.pi - pol_diff
            modulation = math.cos(pol_diff / 2) ** 2  # Malus' law approximation
            
            intensity = voxel.intensity * modulation
            intensity += rng.gauss(0, noise_std)
            intensity = max(0, intensity)
            
            # For simplicity, project to 2D by taking max or sum
            # Here, just add to the position
            if 0 <= voxel.x < grid_x and 0 <= voxel.y < grid_y:
                data[voxel.x][voxel.y] += intensity
        
        images.append(PolarizationImage(angle=angle, data=data))
    
    return images


def denoise_images(images: List[PolarizationImage], kernel_size: int = 3) -> List[PolarizationImage]:
    """Apply simple mean denoising to images."""
    def mean_filter(data: List[List[float]], k: int) -> List[List[float]]:
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                total = 0.0
                count = 0
                for di in range(-k//2, k//2 + 1):
                    for dj in range(-k//2, k//2 + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            total += data[ni][nj]
                            count += 1
                result[i][j] = total / count if count > 0 else 0.0
        return result
    
    return [
        PolarizationImage(angle=img.angle, data=mean_filter(img.data, kernel_size))
        for img in images
    ]


def enhance_contrast(images: List[PolarizationImage]) -> List[PolarizationImage]:
    """Apply contrast enhancement by normalizing to [0,1] range."""
    def normalize(data: List[List[float]]) -> List[List[float]]:
        flat = [val for row in data for val in row]
        if not flat:
            return data
        min_val = min(flat)
        max_val = max(flat)
        if max_val == min_val:
            return [[0.0 for _ in row] for row in data]
        return [[(val - min_val) / (max_val - min_val) for val in row] for row in data]
    
    return [
        PolarizationImage(angle=img.angle, data=normalize(img.data))
        for img in images
    ]


def correct_alignment(images: List[PolarizationImage]) -> List[PolarizationImage]:
    """Placeholder for alignment correction. For now, return as is."""
    # In a real implementation, this would involve image registration
    return images


def extract_voxel_features(images: List[PolarizationImage]) -> List[VoxelFeature]:
    """Extract voxel features from polarization-resolved images.
    
    For each position, analyze the polarization response to extract
    delta (max intensity) and theta (angle of max response).
    """
    if not images:
        return []
    
    grid_x = len(images[0].data)
    grid_y = len(images[0].data[0]) if grid_x > 0 else 0
    
    features = []
    for i in range(grid_x):
        for j in range(grid_y):
            intensities = [img.data[i][j] for img in images]
            if not intensities:
                continue
            
            max_intensity = max(intensities)
            if max_intensity == 0:
                continue
            
            # Find angle with max intensity
            max_angle = images[intensities.index(max_intensity)].angle
            
            # For delta, use the max intensity as depolarization measure
            delta = max_intensity
            
            # Theta is the polarization angle
            theta = max_angle
            
            # For z, since we projected, set to 0 or estimate
            z = 0.0  # Placeholder
            
            features.append(VoxelFeature(x=float(i), y=float(j), z=z, delta=delta, theta=theta))
    
    return features


def cluster_and_reconstruct_grid(features: List[VoxelFeature], threshold: float = 0.1) -> List[Voxel]:
    """Cluster features and reconstruct voxel grid.
    
    Simple thresholding: keep features above threshold.
    """
    from .voxel import Voxel  # Import here to avoid circular
    
    voxels = []
    for feat in features:
        if feat.delta > threshold:
            # Convert to integer coordinates, assume unit pitch
            x = int(round(feat.x))
            y = int(round(feat.y))
            z = int(round(feat.z))
            # Intensity from delta, polarization from theta
            intensity = feat.delta
            polarization = feat.theta
            voxels.append(Voxel(x=x, y=y, z=z, intensity=intensity, polarization=polarization))
    
    return voxels