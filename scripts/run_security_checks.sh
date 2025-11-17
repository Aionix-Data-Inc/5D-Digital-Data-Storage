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

# Prefer python3 when available
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
  echo "[security-checks] no python interpreter found in PATH"
  exit 1
fi

if [ -d "$VENV_DIR" ]; then
  echo "[security-checks] reusing existing virtualenv at $VENV_DIR"
else
  echo "[security-checks] creating virtualenv at $VENV_DIR using $PYTHON_CMD"
  "$PYTHON_CMD" -m venv "$VENV_DIR" || { echo "[security-checks] venv creation failed"; exit 1; }
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[security-checks] upgrading pip/tools"
"$PYTHON_CMD" -m pip install --upgrade pip setuptools wheel || echo "[security-checks] pip upgrade failed"

if [ -f requirements-dev.txt ]; then
  "$PYTHON_CMD" -m pip install -r requirements-dev.txt || "$PYTHON_CMD" -m pip install pytest bandit || echo "[security-checks] installing dev deps failed"
else
  "$PYTHON_CMD" -m pip install pytest bandit || echo "[security-checks] installing pytest/bandit failed"
fi

# pip-audit may not be available on older pip; install separately if needed
"$PYTHON_CMD" -m pip install pip-audit || echo "[security-checks] pip-audit install failed"

echo "[security-checks] installing package in editable mode"
"$PYTHON_CMD" -m pip install -e . || echo "[security-checks] editable install failed"

echo "[security-checks] running pytest"
"$PYTHON_CMD" -m pytest -q || echo "[security-checks] pytest failed"

echo "[security-checks] running Bandit (static scan)"
"$PYTHON_CMD" -m bandit -r src/optical_storage -ll || echo "[security-checks] bandit found issues or failed"

echo "[security-checks] running pip-audit"
"$PYTHON_CMD" -m pip_audit --progress off || echo "[security-checks] pip-audit found vulnerabilities or failed"

echo "[security-checks] done"
