# 5D Optical Data Storage

This repository contains a Python simulation of a five-dimensional (3D position, light intensity, and polarization) data storage platform inspired by femtosecond-laser writing in nanostructured glass. The codebase models the **write** and **read** pipelines, includes error-correction, and exposes utilities for injecting realistic measurement noise.

## Features

* `LaserWriter` – maps byte streams onto a 3D voxel lattice using configurable intensity/polarization quantisation and forward-error correction.
* `LaserReader` – reconstructs the original payload from measured voxels, applying quantisation and decoding.
* Hamming(7,4) and passthrough error-correction implementations.
* Noise helpers to simulate laser writing/reading imperfections.
* Example script demonstrating an end-to-end round-trip with noise.

## Getting Started

Create a virtual environment (optional) and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> The project has no third-party runtime dependencies; installing in editable mode simply exposes the `optical_storage` package.

## Example: Writing and Reading Data

```bash
python examples/simulate_storage.py
```

Sample output:

```
--- Write phase ---
grid_size               : (64, 64, 8)
voxel_pitch             : (5.0, 5.0, 15.0)
intensity_levels        : 16
polarization_states     : 8
bits_per_voxel          : 7
encoded_bit_length      : 392
data_bit_length         : 320
padding_bits            : 0
error_correction        : hamming74
data_length_bytes       : 40
voxel_count             : 56

--- Read phase ---
Recovered payload       : b'5D optical storage with femtosecond lasers!'
Corrected single-bit errs: 0
Detected uncorrectable  : 0
Voxels processed        : 56
```

## Running Tests

```bash
pytest
```

The test-suite exercises noise-free and noisy readbacks as well as capacity validation for oversized payloads.
