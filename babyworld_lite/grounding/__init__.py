"""Calibration-ready synthetic language-grounding dataset tools.

This package is intentionally separate from the original BabyWorld-Lite demo.
The demo renderer includes explanatory overlays; grounding renders never do.
"""

from babyworld_lite.grounding.audit import audit_records
from babyworld_lite.grounding.config import load_grounding_config, validate_grounding_config
from babyworld_lite.grounding.pipeline import (
    RAW_FRAME_POLICY,
    build_records,
    generate_grounding_dataset,
    observable_leakage_paths,
    render_raw_frame,
)

__all__ = [
    "RAW_FRAME_POLICY",
    "audit_records",
    "build_records",
    "generate_grounding_dataset",
    "load_grounding_config",
    "observable_leakage_paths",
    "render_raw_frame",
    "validate_grounding_config",
]
