#!/usr/bin/env python3
"""Create the local environment and run the Experiment I demo on macOS or Windows."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
LOCK = ROOT / "requirements-lock.txt"
INSTALLED = VENV / ".requirements-installed"


def main() -> None:
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise SystemExit("64-bit Python 3.12 is required. Install it and rerun this command.")

    os.chdir(ROOT)
    if not PYTHON.is_file():
        print("Creating Python 3.12 virtual environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)

    lock_text = LOCK.read_bytes()
    if not INSTALLED.is_file() or INSTALLED.read_bytes() != lock_text:
        print("Installing pinned Python packages...", flush=True)
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(LOCK)],
            check=True,
        )
        INSTALLED.write_bytes(lock_text)

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["MPLCONFIGDIR"] = str(ROOT / "data" / "matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call([str(PYTHON), str(ROOT / "demo.py"), *sys.argv[1:]], env=env))


if __name__ == "__main__":
    main()
