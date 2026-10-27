import copy
import json

import numpy as np
import pytest

from nursery.humanoid.hoidini_intermimic.execute import (
    ExecutionError,
    checkpoint_signature,
    diagnose_result,
    humanoid_body_states,
    load_execution_config,
    measure_execution,
    room_collision_mesh,
    validate_controller,
    validate_execution_reference,
)
from nursery.humanoid.hoidini_intermimic.schema import FIELDS, WIDTH


def scene(tmp_path):
    path = tmp_path / "triangle.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    mesh_transform = np.eye(4)
    mesh_transform[:3, :3] *= 2
    mesh_transform[0, 3] = 1
    node = {"pose": [10, 20, 3, 0, 0, 2**-0.5, 2**-0.5],
            "collision_geometry": [{"mesh_path": str(path), "mesh_to_asset": mesh_transform.tolist()}],
            "visual_geometry": []}
    return {"active_interaction": {"target_name": "mug"},
            "nodes": {"desk": node, "mug": copy.deepcopy(node),
                      "root": {"collision_geometry": [], "visual_geometry": []}}}


def test_room_geometry_preserves_scale_local_origin_and_world_pose(tmp_path):
    receipt = scene(tmp_path)
    mesh, sources = room_collision_mesh(receipt)
    np.testing.assert_allclose(mesh.vertices, [[10, 21, 3], [10, 23, 3], [8, 21, 3]])
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])
    assert [source["node"] for source in sources] == ["desk"]


def test_target_is_not_duplicated_as_static_collision(tmp_path):
    receipt = scene(tmp_path)
    receipt["nodes"]["mug"]["collision_geometry"][0]["mesh_path"] = "missing-target.obj"
    assert len(room_collision_mesh(receipt)[0].faces) == 1


def test_missing_room_collision_mesh_is_asset_mismatch(tmp_path):
    receipt = scene(tmp_path)
    receipt["nodes"]["desk"]["collision_geometry"][0]["mesh_path"] = str(tmp_path / "missing.obj")
    with pytest.raises(ExecutionError, match="room collision mesh") as error:
        room_collision_mesh(receipt)
    assert error.value.category == "asset_mismatch"


def test_visible_room_geometry_without_collision_is_rejected(tmp_path):
    receipt = scene(tmp_path)
    desk = receipt["nodes"]["desk"]
    desk["visual_geometry"], desk["collision_geometry"] = desk["collision_geometry"], []
    with pytest.raises(ExecutionError, match="no collision geometry") as error:
        room_collision_mesh(receipt)
    assert error.value.category == "asset_mismatch"


class TensorShape:
    def __init__(self, *shape):
        self.shape = shape


def compatible_checkpoint():
    return {
        "model": {
            "a2c_network.actor_mlp.0.weight": TensorShape(1024, 3198),
            "a2c_network.mu.weight": TensorShape(153, 512),
        },
        "running_mean_std": {"running_mean": TensorShape(3198)},
    }


def test_canonical_config_is_explicit_and_starts_with_one_environment():
    config = load_execution_config()
    assert config["num_environments"] == 1
    assert config["rates_hz"] == {"physics": 60, "control": 30}
    assert config["device"] == {"type": "cuda", "id": 0, "gpu_pipeline": True}
    assert config["controller"]["role"] == "general_student"
    assert config["termination"]["root_height_m"] == 0.3
    assert config["termination"]["stop_on_any_environment"] is True
    assert config["object"]["friction"] == 0.6
    assert config["object"]["rest_offset_m"] == 0.002
    assert config["recording"]["video"] is False


@pytest.mark.parametrize("change, message", [
    (("rates_hz", "control", 20), "requires 30 Hz"),
    (("termination", "stop_on_any_environment", False), "requires stop_on_any_environment=true"),
])
def test_unsupported_execution_modes_are_rejected(tmp_path, change, message):
    config = load_execution_config()
    group, key, value = change
    config[group][key] = value
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        load_execution_config(path)


def test_controller_selection_uses_artifact_metadata_not_activity_name(tmp_path):
    checkpoint_path = tmp_path / "arbitrary-name.pth"
    checkpoint_path.write_bytes(b"general controller")
    config = load_execution_config()
    import hashlib
    config["controller"]["checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()).hexdigest()
    manifest = {
        "units": "metres", "up_axis": "+Z",
        "body_names": [f"body-{index}" for index in range(52)],
        "conversion": {"fps": 30},
        "prompt": "an activity name must not select the controller",
    }
    result = validate_controller(
        checkpoint_path, compatible_checkpoint(), manifest, config)
    assert result["role"] == "general_student"
    assert checkpoint_signature(compatible_checkpoint()) == {
        "observation_dimension": 3198,
        "action_dimension": 153,
        "normalization_dimension": 3198,
    }


def test_incompatible_controller_dimensions_are_rejected(tmp_path):
    checkpoint_path = tmp_path / "controller.pth"
    checkpoint_path.write_bytes(b"controller")
    config = load_execution_config()
    import hashlib
    config["controller"]["checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()).hexdigest()
    checkpoint = compatible_checkpoint()
    checkpoint["model"]["a2c_network.mu.weight"] = TensorShape(12, 512)
    manifest = {"units": "metres", "up_axis": "+Z", "body_names": [None] * 52}
    with pytest.raises(ExecutionError, match="network contract") as error:
        validate_controller(checkpoint_path, checkpoint, manifest, config)
    assert error.value.category == "controller_mismatch"


def execution_arrays():
    steps, environments, bodies_count = 3, 1, 52
    reference = np.zeros((steps + 1, WIDTH), dtype=np.float32)
    reference[:, FIELDS["root_rot"]] = [0, 0, 0, 1]
    reference[:, FIELDS["obj_rot"]] = [0, 0, 0, 1]
    reference[:, FIELDS["body_rot"]] = np.tile([0, 0, 0, 1], bodies_count)
    reference[:, FIELDS["body_pos"]] = 0
    reference[1:, FIELDS["contact_human"].start] = 1
    reference[1:, FIELDS["contact_obj"]] = 1
    initial = np.zeros((environments, 2, 13), dtype=np.float32)
    initial[..., 6] = 1
    states = np.repeat(initial[None], steps, axis=0)
    initial_dofs = np.zeros((environments, 153), dtype=np.float32)
    body_states = np.zeros((steps, environments, bodies_count, 13), dtype=np.float32)
    body_states[..., 2] = 1
    body_states[..., 6] = 1
    contacts = np.zeros((steps, environments, bodies_count, 3), dtype=np.float32)
    contacts[:, :, 0, 2] = 1
    object_contacts = np.zeros((steps, environments, 1, 3), dtype=np.float32)
    object_contacts[..., 2] = 1
    actions = np.zeros((steps, environments, 153), dtype=np.float32)
    return (reference, initial, initial_dofs, states, body_states, contacts,
            object_contacts, actions)


def test_general_measurements_cover_tracking_contact_movement_and_sync():
    arrays = execution_arrays()
    measurements = measure_execution(
        *arrays, load_execution_config(), False, False, [0, 0])
    assert measurements["initialization"]["within_tolerance"] is True
    assert measurements["finite"] is True
    assert measurements["completion_fraction"] == 1
    assert measurements["human_tracking"]["mean_body_position_error_m"] == pytest.approx(1)
    assert measurements["object_tracking"]["mean_position_error_m"] == 0
    assert measurements["expected_contact"]["desired_agreement_per_environment"] == [1]
    assert measurements["expected_contact"]["any_matched_duration_s_per_environment"] == pytest.approx([0.1])
    assert measurements["object_contact_force_proxy"]["pair_specific_agreement"] is None
    assert measurements["termination"]["fall"] is False
    assert measurements["object_movement"]["maximum_displacement_m"] == 0
    assert measurements["synchronization"]["aligned"] is True


def test_humanoid_body_trace_excludes_dynamic_object_body():
    all_actor_bodies = np.arange(53 * 13).reshape(53, 13)
    selected = humanoid_body_states(all_actor_bodies, 1, 52)
    assert selected.shape == (1, 52, 13)
    np.testing.assert_array_equal(selected[0, -1], all_actor_bodies[51])


def test_finger_contact_satisfies_hand_intent_and_durations_are_per_environment():
    arrays = list(execution_arrays())
    reference, initial, initial_dofs, states, bodies, contacts, object_contacts, actions = arrays
    reference[:, FIELDS["contact_human"]] = 0
    reference[1:, FIELDS["contact_human"].start + 17] = 1
    states = np.repeat(states, 2, axis=1)
    initial = np.repeat(initial, 2, axis=0)
    initial_dofs = np.repeat(initial_dofs, 2, axis=0)
    bodies = np.repeat(bodies, 2, axis=1)
    contacts = np.repeat(contacts * 0, 2, axis=1)
    contacts[:, 0, 20, 0] = -0.11  # L_Index3; native absolute-component threshold.
    object_contacts = np.repeat(object_contacts, 2, axis=1)
    actions = np.repeat(actions, 2, axis=1)
    measurements = measure_execution(
        reference, initial, initial_dofs, states, bodies, contacts,
        object_contacts, actions, load_execution_config(), False, False, [0, 0])
    assert measurements["expected_contact"]["desired_agreement_per_environment"] == [1, 0]
    assert measurements["expected_contact"]["expected_duration_s_per_environment"] == pytest.approx([0.1, 0.1])
    assert measurements["expected_contact"]["any_matched_duration_s_per_environment"] == pytest.approx([0.1, 0])


def test_contact_threshold_and_forbidden_unconstrained_semantics():
    arrays = list(execution_arrays())
    reference, initial, initial_dofs, states, bodies, contacts, object_contacts, actions = arrays
    reference[:, FIELDS["contact_human"]] = 0
    reference[1:, FIELDS["contact_human"].start] = -1
    contacts[:] = 0
    contacts[:, :, 0, :2] = 0.08  # Norm exceeds 0.1; no component does.
    contacts[:, :, 1, 0] = 1.0  # Zero label is unconstrained and ignored.
    measurements = measure_execution(
        reference, initial, initial_dofs, states, bodies, contacts,
        object_contacts, actions, load_execution_config(), False, False, [0, 0])
    expected = measurements["expected_contact"]
    assert expected["desired_expected_entries_per_environment"] == [0]
    assert expected["forbidden_expected_entries_per_environment"] == [3]
    assert expected["forbidden_agreement_per_environment"] == [1]


def test_dof_mismatch_fails_full_initialization_check():
    arrays = list(execution_arrays())
    arrays[2][0, 42] = 0.01
    measurements = measure_execution(
        *arrays, load_execution_config(), False, False, [0, 0])
    assert measurements["initialization"]["within_tolerance"] is False
    assert measurements["initialization"]["maximum_dof_position_error_rad"] == pytest.approx(0.01)


def test_reference_timebase_is_validated_before_execution():
    reference = execution_arrays()[0]
    config = load_execution_config()
    manifest = {"conversion": {"fps": 30, "frames": len(reference)}}
    assert validate_execution_reference(reference, manifest, config) is reference
    manifest["conversion"]["fps"] = 24
    with pytest.raises(ExecutionError, match="reference fps") as error:
        validate_execution_reference(reference, manifest, config)
    assert error.value.category == "invalid_reference"


def test_incomplete_valid_attempt_is_diagnosed_as_interaction_failure():
    arrays = execution_arrays()
    measurements = measure_execution(
        *arrays, load_execution_config(), True, False, [0, 11])
    category, evidence = diagnose_result(measurements, completed_reference=False)
    assert category == "interaction_failure"
    assert "terminated=True" in evidence
