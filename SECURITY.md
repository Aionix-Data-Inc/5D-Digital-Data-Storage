# Security policy

If you discover a security vulnerability in this project, please report it responsibly by opening an issue or contacting the maintainers. For serious vulnerabilities, contact the repository owner directly.

Local checks you can run:

- Run the test-suite:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest bandit
pytest
```

- Run Bandit for a quick static security scan:

```bash
bandit -r src/optical_storage -ll
```

The CI workflow runs these checks on push/PR. Dependabot is enabled to watch for Python dependency updates.

If you need a private disclosure channel please request a maintainer contact via an issue and mark it private.
