"""Contracts for the HOIDiNi + InterMimic research protocol."""

from .stage12 import (
    ContractError,
    bind_shared_scene,
    resolve_scene_manifest,
    validate_task_contract,
)

__all__ = [
    "ContractError",
    "bind_shared_scene",
    "resolve_scene_manifest",
    "validate_task_contract",
]
