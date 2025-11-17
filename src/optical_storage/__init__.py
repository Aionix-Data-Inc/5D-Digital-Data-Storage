"""Optical storage simulation package for 5D femtosecond laser writing."""

from .bit_utils import bytes_to_bits, bits_to_bytes
from .error_correction import ErrorCorrectionScheme, Hamming74, NoErrorCorrection, Parity8
from .reader import LaserReader, ReadResult
from .storage_pattern import StoragePattern
from .voxel import Voxel
from .writer import LaserWriter
from .noise import apply_gaussian_noise

__all__ = [
    "LaserWriter",
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
]
