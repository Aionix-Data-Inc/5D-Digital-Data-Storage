#!/usr/bin/env bash
set -euo pipefail

# scripts/audit_security.sh
# Comprehensive security audit: checks for common vulnerabilities,
# outdated dependencies, and code quality issues.
#
# Usage:
#   bash scripts/audit_security.sh
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[audit] Security audit started at $(date)"
echo "[audit] workspace: $ROOT_DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
  echo "[audit] activating virtualenv"
  source .venv/bin/activate || true
fi

echo ""
echo "=== 1. Running Bandit (security scan) ==="
bandit -r src/optical_storage -ll -c .bandit 2>&1 || echo "[audit] Bandit found issues (review above)"

echo ""
echo "=== 2. Running pip-audit (dependency vulnerabilities) ==="
pip-audit --progress off 2>&1 || echo "[audit] pip-audit found vulnerabilities (review above)"

echo ""
echo "=== 3. Checking for hardcoded secrets ==="
# Simple regex check for common secret patterns (not foolproof)
if grep -r "password\|secret\|api_key\|token" src/ --include="*.py" | grep -v "# " | head -5; then
  echo "[audit] WARNING: Potential hardcoded secrets detected (review above)"
else
  echo "[audit] No obvious hardcoded secrets detected"
fi

echo ""
echo "=== 4. File permissions audit ==="
# Check for world-writable files
if find . -perm -002 -type f 2>/dev/null | head -5; then
  echo "[audit] WARNING: World-writable files detected (review above)"
else
  echo "[audit] No world-writable files detected"
fi

echo ""
echo "=== 5. Checking for debug/test code ==="
# Check for common debug patterns
if grep -r "pdb\|breakpoint\|import ipdb" src/ --include="*.py" | head -5; then
  echo "[audit] WARNING: Debug code detected in src/ (review above)"
else
  echo "[audit] No debug statements in src/"
fi

echo ""
echo "[audit] Security audit completed at $(date)"
echo "[audit] Review findings above. Address any HIGH severity issues."
