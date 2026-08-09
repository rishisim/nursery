#!/usr/bin/env python3
"""Count-only wrapper for the frozen Gemma worker on the v1.8 extension.

The prompt, schema, parser, frame offsets, audio slicing, model path handling,
and checkpoint format are inherited byte-for-byte from the validated v1.3
worker. Only the predeclared extension window count changes from 137 to 135.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


BASE = Path(__file__).with_name("nursery_gemma4_referential_worker_v1_3.py")
SPEC = importlib.util.spec_from_file_location("nursery_gemma4_worker_extension_base_v18", BASE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("E_MODULE")
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)
worker.WINDOW_COUNT = 135
worker.legacy.WINDOW_COUNT = 135


if __name__ == "__main__":
    raise SystemExit(worker.main())
