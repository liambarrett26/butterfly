#!/usr/bin/env python3
"""Compatibility shim: `python reel.py …` is the same as the `butterfly` command.

Works without installing the package by putting src/ on the import path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from butterfly.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
