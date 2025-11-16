#!/usr/bin/env bash
set -euo pipefail

# scripts/run_security_checks.sh
# Create a virtual environment, install dev dependencies and security tools,
# run the test-suite, Bandit, and pip-audit.
#
# Usage:
#   bash scripts/run_security_checks.sh
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"

echo "[security-checks] workspace: $ROOT_DIR"

if [ -d "$VENV_DIR" ]; then
  echo "[security-checks] reusing existing virtualenv at $VENV_DIR"
else
  echo "[security-checks] creating virtualenv at $VENV_DIR"
  python -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[security-checks] upgrading pip/tools"
python -m pip install --upgrade pip setuptools wheel

if [ -f requirements-dev.txt ]; then
  pip install -r requirements-dev.txt || pip install pytest bandit
else
  pip install pytest bandit || true
fi

# pip-audit may not be available on older pip; install separately if needed
pip install pip-audit || true

echo "[security-checks] installing package in editable mode"
pip install -e . || true

echo "[security-checks] running pytest"
pytest -q

echo "[security-checks] running Bandit (static scan)"
bandit -r src/optical_storage -ll || true

echo "[security-checks] running pip-audit"
pip-audit --progress off || true

echo "[security-checks] done"
