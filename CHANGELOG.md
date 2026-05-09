# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2] - Unreleased
### Added
- New polarization image processing pipeline with acquisition, denoising, contrast enhancement, alignment correction, feature extraction, and voxel reconstruction.
- Image integrity hashing and verification support for polarization-resolved images.
- New example usage in `examples/simulate_storage.py` demonstrating the full image pipeline.

### Changed
- Updated package exports to expose image-processing utilities from `optical_storage` root.
- Added focused tests for image processing, integrity hashing, and end-to-end pipeline coverage.

## [0.1.1] - 2025-11-17
### Added
- Harden `Voxel` validation to reject non-finite values (NaN/Inf).
- `Parity8` example ECC implementation and tests.
- `StoragePattern` JSON serialization helpers: `to_dict` / `from_dict`.
- Security tooling: Bandit and CodeQL workflows; `scripts/run_security_checks.sh` and `scripts/audit_security.sh`.
- `SECURITY.md` and `docs/SECURITY_PRACTICES.md` documentation.
- `src/optical_storage/security.py` input validation helpers.
- Pinned dev dependencies and `tox.ini` for reproducible dev/test/security runs.

### Changed
- Updated `README.md` with CI badges and security notes.

### Notes
- Release prepared for publication to PyPI as `optical-storage==0.1.1`.
