"""Canonical Unity-native procedural clothed embodiment scene gate."""

from .contracts import (
    CONFIG_PATH,
    OUTPUT_ROOT,
    compile_contract_matrix,
    load_frozen_config,
    validate_frozen_config,
)

__all__ = [
    "CONFIG_PATH",
    "OUTPUT_ROOT",
    "compile_contract_matrix",
    "load_frozen_config",
    "validate_frozen_config",
]
