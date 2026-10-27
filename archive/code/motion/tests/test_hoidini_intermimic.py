import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nursery.humanoid.hoidini_intermimic.contacts import map_contacts
from nursery.humanoid.hoidini_intermimic.convert import convert_arrays
from nursery.humanoid.hoidini_intermimic.coordinates import resample_motion
from nursery.humanoid.hoidini_intermimic.schema import FIELDS, validate_reference
from nursery.humanoid.hoidini_intermimic.skeleton import (
    BASIS, SOURCE_NAMES, SOURCE_PARENTS, read_skeleton,
)


@pytest.fixture
def humanoid(tmp_path):
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    bodies = []
    for i, (name, parent) in enumerate(zip(SOURCE_NAMES, SOURCE_PARENTS)):
        body = ET.SubElement(world if parent < 0 else bodies[parent], "body", name=name,
                             pos="0 0 0" if i == 0 else "0.1 0.02 0.03")
        bodies.append(body)
        if parent < 0:
            ET.SubElement(body, "freejoint", name=name)
        else:
            for axis, vector in zip("xyz", ("1 0 0", "0 1 0", "0 0 1")):
                ET.SubElement(body, "joint", name=f"{name}_{axis}", type="hinge", axis=vector)
    path = tmp_path / "humanoid.xml"
    ET.ElementTree(root).write(path)
    return path


def source(frames=100):
    poses = np.zeros((frames, 156))
    poses[:, :3] = Rotation.from_matrix(BASIS).as_rotvec()
    root = np.tile([2, -3, -0.12], (frames, 1))
    return {"poses": poses, "trans": root, "joints": np.repeat(root[:, None], 52, axis=1),
            "poses_obj": np.tile([0, 0, np.pi / 2], (frames, 1)),
            "trans_obj": np.tile([4, -2, 0.7], (frames, 1)), "contact": np.zeros((frames, 60))}


def test_upstream_layout_timing_and_unchanged_world(humanoid):
    arrays = source()
    arrays["trans"][:, 0] += np.arange(100) / 20
    arrays["trans_obj"][:, 1] += np.arange(100) / 20
    data, stats = convert_arrays(arrays, 20, read_skeleton(humanoid))
    assert data.shape == (150, 591)
    assert stats["duration_s"] == stats["source_duration_s"] == 5
    np.testing.assert_allclose(data[::3, :3], arrays["trans"][::2], atol=5e-7)
    np.testing.assert_allclose(data[::3, 318:321], arrays["trans_obj"][::2], atol=5e-7)
    np.testing.assert_allclose(data[-1, :3], arrays["trans"][-1], atol=5e-7)
    np.testing.assert_allclose(data[-1, 318:321], arrays["trans_obj"][-1], atol=5e-7)
    np.testing.assert_allclose(data[:, 2], -0.12, atol=1e-7)  # Never repair the floor.
    np.testing.assert_allclose(data[:, 3:7], np.tile([0, 0, 0, 1], (150, 1)), atol=1e-7)
    np.testing.assert_allclose(data[:, 321:325], np.tile([0, 0, 2**-0.5, 2**-0.5], (150, 1)), atol=1e-7)
    assert not data[:, 7:9].any() and not data[:, 325:330].any()


def test_joint_mapping_basis_and_target_forward_kinematics(humanoid):
    arrays = source(4)
    rotation_vector = np.array([0.3, 0.4, 0.2])
    arrays["poses"][:, 3:6] = rotation_vector  # L_Hip, not R_Hip.
    names, parents, offsets, order = skeleton = read_skeleton(humanoid)
    data, _ = convert_arrays(arrays, 20, skeleton)
    assert names.index("L_Knee") == 2 and order[2] == 4
    assert order[names.index("R_Wrist")] == 21
    assert order[names.index("L_Thumb3")] == 36
    np.testing.assert_allclose(data[0, 9:12], BASIS @ rotation_vector, atol=1e-7)
    dofs = data[:, 9:162].reshape(len(data), 51, 3)
    other = names.index("R_Hip") - 1
    np.testing.assert_allclose(dofs[:, other], 0, atol=1e-7)
    # Independent rest-chain case: hip translated in the pelvis frame;
    # knee offset rotated by the target hip, not by its own rotation.
    positions = data[:, 162:318].reshape(len(data), 52, 3)
    expected_hip = arrays["trans"][0] + offsets[1]
    target_hip_rotation = Rotation.from_rotvec(BASIS @ rotation_vector).as_matrix()
    expected_knee = expected_hip + target_hip_rotation @ offsets[2]
    np.testing.assert_allclose(positions[0, 1], expected_hip, atol=2e-7)
    np.testing.assert_allclose(positions[0, 2], expected_knee, atol=2e-7)
    body_rot = Rotation.from_quat(data[0, 383:591].reshape(52, 4)).as_matrix()
    np.testing.assert_allclose(body_rot[2], target_hip_rotation, atol=2e-7)


def test_rotation_resampling_uses_short_arc_and_holds_tail():
    arrays = source(2)
    arrays["poses_obj"] = np.deg2rad([[0, 0, 170], [0, 0, -170]])
    sampled, times = resample_motion(arrays, 20)
    assert len(times) == 3
    matrices = sampled["poses_obj"][:, 0]
    delta = Rotation.from_matrix(matrices[0].T @ matrices[1]).magnitude()
    assert np.rad2deg(delta) == pytest.approx(20 * 2 / 3)
    np.testing.assert_allclose(matrices[-1], Rotation.from_rotvec(arrays["poses_obj"][-1]).as_matrix())


def test_contacts_preserve_hand_intent_without_inventing_body_contacts(humanoid):
    names = read_skeleton(humanoid)[0]
    scores = np.zeros((3, 60))
    scores[0, 29] = 1.6
    scores[1, 30] = 0.41
    scores[2] = -0.3
    human, obj = map_contacts(scores, names)
    np.testing.assert_array_equal(obj, [1, 1, 0])
    np.testing.assert_array_equal(human[:, names.index("L_Wrist")], [1, 0, 0])
    np.testing.assert_array_equal(human[:, names.index("R_Wrist")], [0, 1, 0])
    assert not human[:, [i for i, name in enumerate(names) if "Wrist" not in name]].any()
    assert (human >= 0).all()


@pytest.mark.parametrize("key", ["poses", "trans", "joints", "poses_obj", "trans_obj", "contact"])
def test_reject_invalid_source_channels(key, humanoid):
    arrays = source(2)
    arrays[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        convert_arrays(arrays, 20, read_skeleton(humanoid))
    arrays[key] = arrays[key][:1]
    with pytest.raises(ValueError):
        convert_arrays(arrays, 20, read_skeleton(humanoid))


@pytest.mark.parametrize("failure", ["axis", "missing_body", "rest_rotation", "parent"])
def test_reject_incompatible_target_skeleton(humanoid, failure):
    tree = ET.parse(humanoid)
    root = tree.getroot()
    if failure == "axis":
        root.find(".//joint").set("axis", "0 0 1")
    elif failure == "rest_rotation":
        root.find(".//body").set("quat", "0 1 0 0")
    elif failure == "missing_body":
        root.find(".//body[@name='L_Hip']").remove(root.find(".//body[@name='L_Knee']"))
    else:
        knee = root.find(".//body[@name='L_Knee']")
        root.find(".//body[@name='L_Hip']").remove(knee)
        root.find(".//body[@name='R_Hip']").append(knee)
    tree.write(humanoid)
    with pytest.raises(ValueError):
        read_skeleton(humanoid)


@pytest.mark.parametrize("column,value", [(3, 2), (7, 1), (325, 1), (330, -1), (331, 2), (0, np.inf)])
def test_reject_corrupted_reference(humanoid, column, value):
    data, _ = convert_arrays(source(2), 20, read_skeleton(humanoid))
    data[0, column] = value
    with pytest.raises(ValueError):
        validate_reference(data)
