#!/usr/bin/env python3
"""Repository-local wrapper for the InteractMove--InterMimic contract CLI."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.interactmove_intermimic.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
