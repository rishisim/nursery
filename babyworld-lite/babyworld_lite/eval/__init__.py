"""Leakage-aware evaluation utilities for BabyWorld-Lite."""

from babyworld_lite.eval.features import (
    ARM_MODALITIES,
    extract_windowed_feature_groups,
    flatten_feature_groups,
    hidden_impulse,
    prediction_frame,
    regression_targets,
)

__all__ = [
    "ARM_MODALITIES",
    "extract_windowed_feature_groups",
    "flatten_feature_groups",
    "hidden_impulse",
    "prediction_frame",
    "regression_targets",
]
