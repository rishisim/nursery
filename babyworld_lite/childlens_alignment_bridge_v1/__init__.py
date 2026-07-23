"""Fail-closed ChildLens continuous alignment bridge preflight."""

from .preflight import (
    BridgeError,
    build_control_assignment,
    character_similarity,
    interval_boundary_f1,
    participant_bootstrap_interval,
)
from .alignment import (
    effect_summary,
    normalized_diagonal_cosine,
    shifted_window,
    sign_flip_pvalue,
)

__all__ = [
    "BridgeError",
    "build_control_assignment",
    "character_similarity",
    "interval_boundary_f1",
    "participant_bootstrap_interval",
    "effect_summary",
    "normalized_diagonal_cosine",
    "shifted_window",
    "sign_flip_pvalue",
]
