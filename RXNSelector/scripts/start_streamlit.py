from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "ui" / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
