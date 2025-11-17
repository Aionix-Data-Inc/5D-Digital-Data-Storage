"""Security best practices guide for the 5D Storage project."""

# Security Best Practices

## Input Validation
- All user-provided inputs (bytes, dimensions, ranges) are validated before use.
- Use `security.validate_bytes_input()` for payload validation.
- Use `security.validate_grid_dimensions()` for grid size validation.
- `Voxel` rejects NaN/Inf coordinates and values via `__post_init__`.

## Dependency Management
- No runtime dependencies; all dev tools are pinned in `requirements-dev.txt`.
- Dependabot monitors PyPI for security updates.
- Run `pip-audit` regularly to check for known vulnerabilities.

## Code Analysis
- Bandit scans source code for security issues on every push (CI).
- CodeQL performs deeper static analysis via GitHub Actions.
- Use `scripts/audit_security.sh` to run comprehensive security audits locally.

## Data Handling
- Payload size is limited to 1 MB for safety (see `security.py`).
- Grid dimensions are validated and capped at 10,000 per axis.
- Serialization helpers (`StoragePattern.to_dict/from_dict`) handle untrusted input safely.

## Secrets & Credentials
- Never commit API tokens, passwords, or keys to the repository.
- Use GitHub Secrets for CI/CD credentials (PyPI tokens, etc.).
- Check `.gitignore` for patterns that exclude sensitive files.

## Release & Deployment
- Tag releases with semantic versioning (e.g., `v0.1.1`).
- Publish packages to PyPI only after security scans pass.
- Review CI/CodeQL results before merging PRs.

## Reporting Vulnerabilities
- See `SECURITY.md` for responsible disclosure instructions.
- Report security issues privately to maintainers.

## Local Security Checks
Run these before committing:

```bash
# Full security audit
bash scripts/audit_security.sh

# Or individual checks
bandit -r src/optical_storage -ll
pip-audit --progress off
pytest -q
```

## References
- Bandit: https://bandit.readthedocs.io/
- CodeQL: https://codeql.github.com/
- OWASP Top 10: https://owasp.org/Top10/
