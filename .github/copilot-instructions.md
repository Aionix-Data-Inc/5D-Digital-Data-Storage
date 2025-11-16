# Copilot instructions for 5D-Digital-Data-Storage

Short, actionable guidance for AI coding agents working on this repository.

1) Big picture
- Purpose: a pure-Python simulation of a 5D optical data-storage pipeline (3D voxel lattice + intensity + polarization).
- Two main pipelines: write (bytes -> bits -> ECC -> quantised levels -> `Voxel` list) and read (measured voxels -> levels -> bits -> ECC decode -> bytes).
- Key modules: `src/optical_storage/writer.py` (write pipeline), `reader.py` (read pipeline), `storage_pattern.py` (metadata + voxels), `voxel.py` (data model), `error_correction.py` (ECC interface/impl), `bit_utils.py` (bit packing), `noise.py` (measurement noise helpers).

2) Important implementation details agents must follow
- Bits are MSB-first everywhere: `bytes_to_bits` / `bits_to_bytes` (see `bit_utils.py`). Preserve this ordering when adding conversion code.
- Quantisation levels MUST be powers of two. Both `LaserWriter` and `LaserReader` validate this. Use `_bits_for_levels` or equivalent logic.
- Coordinate ordering: `LaserWriter._index_to_coordinates` maps index -> (x, y, z) with x being the fastest axis (x fastest, then y, then z). Keep new indexing logic consistent with this mapping.
- Level-to-physical mapping: `LaserWriter._level_to_physical` uses linear steps across `intensity_range` / `polarization_range`. The reader reverses this via `_physical_to_level` in `reader.py`.
- ECC interface: subclass `ErrorCorrectionScheme` and implement `encode(bits)` and `decode(bits)` returning `DecodingResult`. See `Hamming74` and `NoErrorCorrection` in `error_correction.py` for examples.

3) Developer workflows & commands
- Install in editable mode (recommended):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```
- Run the example end-to-end: `python examples/simulate_storage.py`.
- Run tests: from repo root run `pytest`. pyproject config sets `pythonpath = ["src"]` so tests import `optical_storage` directly.

4) Project-specific conventions and patterns
- Dataclasses are used for core domain types (`Voxel`, `StoragePattern`, `ReadResult`). Prefer returning dataclasses for structured results rather than tuples.
- There are no runtime third-party dependencies; keep new runtime deps justified and minimal.
- Validation is explicit and raises `ValueError` for invalid parameters (see `_validate_parameters` in `writer.py` and `Voxel.__post_init__`). Follow this pattern when adding checks.
- Padding behaviour: `LaserWriter.write` pads ECC-encoded bitstreams to fill the lattice; `LaserReader.read` trims padding using `pattern.padding_bits`. When changing payload-layout logic, ensure padding and `encoded_bit_length` remain consistent.

5) Integration points & extension guides
- Add new ECC: implement `ErrorCorrectionScheme` in `src/optical_storage/error_correction.py`, update `__init__.py` exports, and add tests under `tests/` mirroring `test_roundtrip.py`.
- Add alternative noise models: create helpers in `src/optical_storage/noise.py` that accept and return `List[Voxel]` (pattern-level helpers use `StoragePattern`). Keep deterministic seeding support (use `seed` param) for reproducible tests.
- If adding new IO/serialization formats for `StoragePattern`, prefer small utilities that operate on `StoragePattern` and keep `StoragePattern`’s fields stable: `voxels`, `grid_size`, `voxel_pitch`, ranges, ECC metadata and `data_length_bytes`.

6) Tests and examples to reference
- `examples/simulate_storage.py` — end-to-end usage pattern and printing of `StoragePattern.summary()`.
- `tests/test_roundtrip.py` — canonical tests for noise-free and noisy roundtrips and capacity guard checks. New features should add analogous tests.

7) Quick FAQs for agents
- Q: How is capacity calculated? A: `StoragePattern.capacity_bits()` multiplies grid dimensions by `bits_per_voxel`. Writer computes required voxels from `encoded_bit_length` and `bits_per_voxel`.
- Q: What happens when levels == 1? A: That dimension encodes zero bits (handled in `_bits_for_levels` / `_bits_for_levels` variant in reader/writer).

If anything above is unclear or you'd like more examples (new ECC implementation, serialization helpers, or a CLI wrapper), tell me which area to expand and I will iterate.
