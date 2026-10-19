#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  elif command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    PYTHON_BIN="$(command -v python)"
  else
    echo 'Python 3.12 is required. Install it, then rerun ./run.sh.' >&2
    exit 1
  fi
  "$PYTHON_BIN" -m venv .venv
fi

if [[ ! -f .venv/.requirements-installed ]] || ! cmp -s requirements-lock.txt .venv/.requirements-installed; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements-lock.txt
  cp requirements-lock.txt .venv/.requirements-installed
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p data/matplotlib
export MPLCONFIGDIR="$PWD/data/matplotlib"
exec .venv/bin/python demo.py "$@"
