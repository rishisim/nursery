"""Public/dummy EgoBabyVLM compatibility preflight.

EgoBabyVLM remains copyright Meta Platforms, Inc. and is used at the pinned
upstream revision under CC BY-NC 4.0. This package is an attributed adapter,
not a vendored copy and not a statement of commercial clearance.
"""

from .bridge import (
    ConversionRecord,
    convert_hf_dinov2_base,
    strict_state_equality,
)
from .contract import canonical_json_sha256, lexical_macro_wiring, schedule_cycle
from .synthetic_video_pilot import compile_prompt, compile_work_order, validate_config

__all__ = [
    "ConversionRecord",
    "canonical_json_sha256",
    "compile_prompt",
    "compile_work_order",
    "convert_hf_dinov2_base",
    "lexical_macro_wiring",
    "schedule_cycle",
    "strict_state_equality",
    "validate_config",
]
