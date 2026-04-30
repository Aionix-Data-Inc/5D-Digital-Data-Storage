"""Security best practices and input validation helpers."""

from __future__ import annotations


def validate_bytes_input(data: bytes, max_size: int = 1_000_000, param_name: str = "data") -> None:
    """Validate that input is bytes and within size limits.

    Args:
        data: The input to validate.
        max_size: Maximum allowed size in bytes (default 1 MB).
        param_name: Parameter name for error messages.

    Raises:
        TypeError: If data is not bytes.
        ValueError: If data exceeds max_size.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"{param_name} must be bytes, not {type(data).__name__}")
    if len(data) > max_size:
        raise ValueError(f"{param_name} exceeds maximum size ({max_size} bytes): {len(data)} bytes provided")


def validate_grid_dimensions(grid_size: tuple[int, int, int]) -> None:
    """Validate grid dimensions are positive and reasonable.

    Args:
        grid_size: (x, y, z) grid dimensions.

    Raises:
        ValueError: If dimensions are invalid or unreasonably large.
    """
    if not isinstance(grid_size, tuple) or len(grid_size) != 3:
        raise ValueError("grid_size must be a tuple of 3 integers")
    x, y, z = grid_size
    if not all(isinstance(d, int) for d in grid_size):
        raise ValueError("grid_size components must be integers")
    if any(d <= 0 for d in grid_size):
        raise ValueError(f"grid_size components must be positive: {grid_size}")
    max_dim = 10_000  # Sanity check: 10k x 10k x 10k is very large
    if any(d > max_dim for d in grid_size):
        raise ValueError(f"grid_size dimensions exceed maximum ({max_dim}): {grid_size}")
