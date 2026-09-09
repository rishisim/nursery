import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from nursery.humanoid.embodiedgen_hoidini.collision import CollisionSample
from nursery.humanoid.embodiedgen_hoidini.run_motion import alignment_transform
from nursery.humanoid.embodiedgen_hoidini.validate_motion import (
    evaluate_scene_collisions,
)
from nursery.humanoid.embodiedgen_hoidini import BridgeValidationError


def test_prefix_alignment_keeps_feet_grounded_and_body_rigid():
    source_object = np.array([2.0, 3.0, 0.9])
    target_object = np.array([4.0, 1.0, 1.1])
    transform = alignment_transform(source_object, target_object, 90)
    aligned_object = transform[:3, :3] @ source_object + transform[:3, 3]
    np.testing.assert_allclose(aligned_object, [4, 1, 0.9])
    # Object height is supplied separately from the scene; raising the table
    # must not also raise the human's feet.
    body = np.array([[2, 2, 0], [2, 2.2, 0], [2, 2.1, 1.6]])
    aligned_body = body @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(aligned_body[:, 2], body[:, 2])
    np.testing.assert_allclose(np.linalg.norm(aligned_body[2] - aligned_body[0]),
                               np.linalg.norm(body[2] - body[0]))
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1)


@pytest.mark.parametrize("source,target,yaw", [
    ([0, 0], [1, 2, 3], 0),
    ([0, 0, np.nan], [1, 2, 3], 0),
    ([0, 0, 0], [1, 2, 3], np.inf),
])
def test_invalid_prefix_alignment_fails(source, target, yaw):
    with pytest.raises(BridgeValidationError):
        alignment_transform(source, target, yaw)


def test_collision_report_preserves_support_contact_and_groups_failures(monkeypatch):
    validator = importlib.import_module(
        "nursery.humanoid.embodiedgen_hoidini.validate_motion"
    )
    active = SimpleNamespace(
        target_name="mug",
        support_name="table",
        canonical_target_mesh=SimpleNamespace(faces=np.array([[0, 1, 2]])),
    )
    bundle = SimpleNamespace(active_interaction=active)
    rows = (
        CollisionSample(0, "object", "table", -0.5, 0.5, 1),
        CollisionSample(0, "body", "chair", -0.03, 0.03, 1),
        CollisionSample(0, "object", "chair", -0.021, 0.021, 1),
        CollisionSample(0, "body", "mug", -0.01, 0.01, 1),
    )
    monkeypatch.setattr(
        validator,
        "build_static_collision_scene",
        lambda _: (
            SimpleNamespace(enclosing_boundary=False),
            SimpleNamespace(enclosing_boundary=True),
        ),
    )
    monkeypatch.setattr(
        validator, "validate_trajectories", lambda *args, **kwargs: rows
    )

    report = evaluate_scene_collisions(
        bundle,
        np.zeros((1, 1, 3)),
        np.zeros((1, 3, 3)),
    )

    assert report["maximum_body_room_penetration_m"] == 0.03
    assert report["enclosing_boundary_meshes"] == 1
    assert report["maximum_object_room_penetration_m"] == 0.021
    assert report["maximum_body_object_penetration_m"] == 0.01
    assert report["checks"] == {
        "body_not_penetrating_room": False,
        "object_not_penetrating_room": False,
        "body_not_penetrating_target": True,
    }
