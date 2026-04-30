"""Tests for image processing pipeline."""

from __future__ import annotations

import math
import pytest

from optical_storage import (
    Hamming74,
    LaserWriter,
    acquire_polarization_resolved_images,
    denoise_images,
    enhance_contrast,
    extract_voxel_features,
    cluster_and_reconstruct_grid,
)


def test_image_acquisition() -> None:
    """Test acquiring polarization-resolved images from a pattern."""
    writer = LaserWriter(
        grid_size=(8, 8, 2),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=4,
        polarization_states=4,
        intensity_range=(0.1, 0.9),
        polarization_range=(0.0, math.pi),
        error_correction=Hamming74(),
    )
    
    payload = b"test"
    pattern = writer.write(payload)
    
    angles = [0, math.pi/2, math.pi, 3*math.pi/2]
    images = acquire_polarization_resolved_images(pattern, angles, noise_std=0.0, seed=42)
    
    assert len(images) == 4
    for img in images:
        assert len(img.data) == 8  # grid_x
        assert len(img.data[0]) == 8  # grid_y


def test_preprocessing() -> None:
    """Test preprocessing steps."""
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=2,
        polarization_states=2,
        intensity_range=(0.0, 1.0),
        polarization_range=(0.0, math.pi),
        error_correction=Hamming74(),
    )
    
    pattern = writer.write(b"t")
    angles = [0, math.pi/2]
    images = acquire_polarization_resolved_images(pattern, angles, noise_std=0.1, seed=1)
    
    # Denoise
    denoised = denoise_images(images)
    assert len(denoised) == len(images)
    
    # Enhance contrast
    enhanced = enhance_contrast(denoised)
    assert len(enhanced) == len(images)
    
    # Check that values are in [0,1]
    for img in enhanced:
        for row in img.data:
            for val in row:
                assert 0 <= val <= 1


def test_feature_extraction_and_clustering() -> None:
    """Test extracting features and reconstructing voxels."""
    writer = LaserWriter(
        grid_size=(4, 4, 1),
        voxel_pitch=(5.0, 5.0, 15.0),
        intensity_levels=2,
        polarization_states=2,
        intensity_range=(0.0, 1.0),
        polarization_range=(0.0, math.pi),
        error_correction=Hamming74(),
    )
    
    pattern = writer.write(b"t")
    angles = [0, math.pi/2]
    images = acquire_polarization_resolved_images(pattern, angles, noise_std=0.0, seed=42)
    
    features = extract_voxel_features(images)
    assert isinstance(features, list)
    
    voxels = cluster_and_reconstruct_grid(features, threshold=0.0)
    assert isinstance(voxels, list)
    # Should reconstruct some voxels
    assert len(voxels) > 0