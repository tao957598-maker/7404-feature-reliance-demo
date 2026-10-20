#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3.12 >/dev/null 2>&1; then
  exec python3.12 run.py "$@"
elif command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  exec python run.py "$@"
else
  echo 'Python 3.12 is required. Install it, then rerun ./run.sh.' >&2
  exit 1
fi
