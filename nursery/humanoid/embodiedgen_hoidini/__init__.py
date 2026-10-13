"""Public API for deterministic EmbodiedGen to HOIDiNi preparation."""

from .prepare_scene import (
    BridgeValidationError,
    PreparedSceneBundle,
    prepare_scene,
)
from .collision import (
    CollisionGeometry,
    CollisionSample,
    build_static_collision_scene,
    is_watertight,
    signed_distance,
    validate_trajectories,
)

__all__ = [
    "BridgeValidationError",
    "PreparedSceneBundle",
    "prepare_scene",
    "CollisionGeometry",
    "CollisionSample",
    "build_static_collision_scene",
    "is_watertight",
    "signed_distance",
    "validate_trajectories",
]
