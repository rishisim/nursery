import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from nursery.egocentric_replay.build import load_config, room_bank
from nursery.egocentric_replay.grab import mesh_path, rigid_matrix
from nursery.egocentric_replay.schema import FIELDS, table, visible_objects
from nursery.egocentric_replay.sessions import chain_clips, contact_rule, sample_chain, subject_split


def standin_rooms(output):
    import trimesh

    for name, color in (("train", [180, 190, 200, 255]), ("heldout", [150, 180, 150, 255])):
        directory = output / name / "urdf/export_scene"
        directory.mkdir(parents=True, exist_ok=True)
        root = ET.Element("robot", name=name)
        for index, (size, position) in enumerate([
                ([8, 8, 0.1], [0, 0, -0.05]), ([8, 0.1, 3], [0, -4, 1.5]),
                ([8, 0.1, 3], [0, 4, 1.5]), ([0.1, 8, 3], [-4, 0, 1.5]),
                ([0.1, 8, 3], [4, 0, 1.5])]):
            mesh = trimesh.creation.box(extents=size)
            mesh.visual = trimesh.visual.texture.TextureVisuals(
                material=trimesh.visual.material.SimpleMaterial(diffuse=color))
            mesh.export(directory / f"mesh{index}.obj")
            link = ET.SubElement(root, "link", name=f"{name}_floor" if index == 0 else f"wall{index}")
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", xyz=" ".join(map(str, position)))
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "mesh", filename=f"mesh{index}.obj")
        ET.ElementTree(root).write(directory / "scene.urdf")
    return output


@pytest.mark.parametrize("name", FIELDS)
def test_contract_schema(name):
    row = {key: (True if kind == "bool" else 1 if kind == "int64" else "x")
           for key, kind in FIELDS[name]}
    if name == "frames":
        row["held_category"] = None
    actual = table([row], name)
    assert actual.column_names == [key for key, _ in FIELDS[name]]
    assert actual.num_rows == 1
    for field in actual.schema:
        assert field.nullable == (field.name == "held_category")
    with pytest.raises(ValueError):
        table([{**row, "extra": "x"}], name)
    invalid = dict(row)
    invalid[next(key for key in row if key != "held_category")] = None
    with pytest.raises(ValueError):
        table([invalid], name)


def test_visibility_exact_pixels_and_odd_image_grid():
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[0, 0] = mask[6, 8] = mask[3, 4] = 1
    mask[1, 7] = 2
    objects = json.loads(visible_objects(mask, {1: "mug", 2: "apple", 3: "pear"}))
    assert objects == [{"category": "mug", "visible_px": 3, "cells": [0, 5, 15]},
                       {"category": "apple", "visible_px": 1, "cells": [3]}]
    assert json.loads(visible_objects(mask * 0, {1: "mug"})) == []


def test_chaining_exact_duration_source_timebase_and_hard_cuts():
    clips = [("a", 25, 10), ("b", 7, 2)]
    chain = chain_clips(clips, 13, 7)
    assert chain == chain_clips(clips, 13, 7)
    assert chain[0]["start"] == 0 and chain[-1]["stop"] == 13
    assert all(first["stop"] == second["start"] for first, second in zip(chain, chain[1:]))
    samples = sample_chain(chain, 13)
    assert len(samples) == 13
    for second, (name, frame) in enumerate(samples):
        segment = next(item for item in chain if item["start"] <= second < item["stop"])
        assert name == segment["clip"]
        assert frame == int((second - segment["start"]) * segment["fps"])
    assert all(len(sample_chain(chain_clips(clips, 120, seed), 120)) == 120 for seed in range(4))


@pytest.mark.parametrize("subject,split", [(f"s{i}", "train" if i < 9 else "dev" if i == 9 else "test")
                                            for i in range(1, 11)])
def test_subject_split(subject, split):
    assert subject_split(subject) == split


@pytest.mark.parametrize("subject", ["s0", "s11", "s01", "other"])
def test_invalid_subject(subject):
    with pytest.raises(ValueError):
        subject_split(subject)


def test_contact_nonzero_on_any_hand_vertex_only():
    labels = np.zeros((4, 8), dtype=np.int32)
    labels[0, 1] = 42
    labels[1, 3] = -1
    labels[2, 0] = 22
    labels[3, [1, 3]] = [41, 55]
    left, right = contact_rule(labels, {"left_hand": [1, 2], "right_hand": [3, 4]})
    np.testing.assert_array_equal(left, [1, 0, 0, 1])
    np.testing.assert_array_equal(right, [0, 1, 0, 1])


def test_mesh_remapping_and_rigid_pose(tmp_path):
    (tmp_path / "contact_meshes").mkdir()
    (tmp_path / "contact_meshes/mug.ply").touch()
    assert mesh_path(tmp_path, "tools/object_meshes/contact_meshes/mug.ply") == tmp_path / "contact_meshes/mug.ply"
    with pytest.raises(ValueError):
        mesh_path(tmp_path, "../outside.ply")
    matrix = rigid_matrix({"global_orient": [[0, 0, np.pi / 2]], "transl": [[1, 2, 3]]}, 0)
    np.testing.assert_allclose(matrix @ [1, 0, 0, 1], [1, 1, 3, 1], atol=1e-7)


def test_standin_format_and_room_separation(tmp_path):
    rooms = room_bank(standin_rooms(tmp_path / "rooms"))
    assert len(rooms) == 2
    assert rooms[0].parents[2].name != rooms[1].parents[2].name
    assert load_config()["duration_s"] == 120


def test_build_contract_with_fake_sessions_and_probes(tmp_path, monkeypatch):
    import pickle
    import pyarrow.parquet as pq
    from nursery.egocentric_replay import build as implementation

    config = load_config()
    config.update(subjects=["s1", "s9", "s10"], duration_s=2, probes_per_category_per_split=1)
    ids = tmp_path / "ids.pkl"
    ids.write_bytes(pickle.dumps({"left_hand": [0], "right_hand": [1]}))
    config["hand_vertex_ids"] = str(ids)
    config["body_models"] = str(tmp_path / "models")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    rooms = standin_rooms(tmp_path / "rooms")
    config["session_rooms"] = ["train"]
    config["probe_rooms"] = ["heldout"]
    config_path.write_text(json.dumps(config))
    (rooms / "room_bank_report.json").write_text(json.dumps({"rooms": [
        {"room_id": name, "geometry": {"clear_2_5m_square": True, "square_vertices_m": [],
                                      "square_orientation_degrees": 0, "floor_height_m": 0}}
        for name in ("train", "heldout")]}))
    monkeypatch.setattr(implementation, "_catalog", lambda config: (
        {subject: [("clip", 100, 10)] for subject in config["subjects"]}, ["mug", "apple"]))
    monkeypatch.setattr(implementation, "_output_directory", lambda path: _new_directory(path))
    monkeypatch.setattr(implementation, "load_clip", lambda path: {"table": {"params": {
        "global_orient": [[0, 0, 0]], "transl": [[0, 0, 0]]}}})

    def fake_session(config, root, ids, subject, number, rooms, clips, categories, seed, output):
        video = f"{subject}_{number:04d}"
        room = rooms[0]
        rows = [{"video_id": video, "frame_idx": index, "image": f"frames/{video}/{index:05d}.jpg",
                 "subject": subject, "room": room.parents[2].name, "clip": "mug_lift",
                 "contact_left": True, "contact_right": False, "held_category": "mug",
                 "objects": '[{"category":"mug","visible_px":10,"cells":[1]}, {"category":"apple","visible_px":5,"cells":[2]}]'} for index in range(2)]
        session = {"video_id": video, "subject": subject, "room": room.parents[2].name,
                   "split": subject_split(subject), "clips": '["mug_lift"]',
                   "distractors": '["apple"]', "duration_s": 2}
        (output / "work").mkdir(exist_ok=True)
        (output / "render_stats.json").write_text(json.dumps({"blender_version": "synthetic"}))
        return rows, session, [0.001], [0.002], {"clip": "sha"}

    monkeypatch.setattr(implementation, "_session", fake_session)
    rendered = []
    monkeypatch.setattr(implementation, "render_job", lambda job, config, output: rendered.extend(job["probes"]))
    output = tmp_path / "episodes"
    manifest = implementation.build(config_path, rooms, output, 2, 7)
    assert manifest["statistics"]["frames"] == 12
    assert manifest["statistics"]["probes"] == 4
    assert len(rendered) == 4
    sessions = pq.read_table(output / "sessions.parquet").to_pylist()
    probes = pq.read_table(output / "probes.parquet").to_pylist()
    train_rooms = {row["room"] for row in sessions if row["split"] == "train"}
    assert train_rooms.isdisjoint({row["room"] for row in probes})
    assert {row["split"] for row in sessions} == {"train", "dev", "test"}
    assert {row["duration_s"] for row in sessions} == {2}
    rendered.clear()
    worker = implementation.build(config_path, rooms, tmp_path / "worker", 2, 7, session_index=3)
    assert worker["statistics"]["sessions"] == 1
    assert worker["statistics"]["frames"] == 2
    assert not rendered
    assert pq.read_table(tmp_path / "worker/sessions.parquet").to_pylist() == [sessions[3]]
    assert worker["tables_sha256"]["frames"] == implementation._sha256(tmp_path / "worker/frames.parquet")

    from types import SimpleNamespace
    packed_counts = []

    class Pool:
        def __init__(self, workers):
            packed_counts.append(workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starmap(self, function, arguments):
            return [function(*args) for args in arguments]

    monkeypatch.setattr(implementation.multiprocessing, "get_context",
                        lambda name: SimpleNamespace(Pool=Pool))
    packed = implementation.build(config_path, rooms, tmp_path / "packed", 2, 7, workers=2)
    assert packed_counts == [2, 1]
    assert packed["workers"] == 2
    assert pq.read_table(tmp_path / "packed/frames.parquet").to_pylist() == pq.read_table(
        output / "frames.parquet").to_pylist()

    def invisible_session(*args):
        result = fake_session(*args)
        for row in result[0]:
            row["objects"] = "[]"
        return result

    monkeypatch.setattr(implementation, "_session", invisible_session)
    with pytest.raises(ValueError, match="visibility gate failed"):
        implementation.build(config_path, rooms, tmp_path / "failed_visibility", 2, 7)
    failed = json.loads((tmp_path / "failed_visibility/manifest.json").read_text())
    assert failed["status"] == "referential_ambiguity_failed"
    assert failed["statistics"]["frames"] == 12
    assert failed["statistics"]["frames_with_two_visible_categories"] == 0


def _new_directory(path):
    from pathlib import Path

    path = Path(path)
    path.mkdir()
    return path


def test_renderer_refuses_unignored_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from nursery.egocentric_replay import render

    monkeypatch.setattr(render.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(render, "_run_native", lambda *args, **kwargs: pytest.fail("Renderer must not launch"))
    with pytest.raises(ValueError, match="ignored"):
        render.render_job({"room_urdf": "unused"}, load_config(), tmp_path / "unignored")
    assert not (tmp_path / "unignored").exists()


def test_contact_gate_rejects_before_render_and_removes_intermediates(tmp_path, monkeypatch):
    from nursery.egocentric_replay import build as implementation

    config = load_config()
    config.update(duration_s=2, clear_squares={"room0": {}})
    source = tmp_path / "clip.npz"
    source.touch()
    data = {"obj_name": "mug", "object": {"object_mesh": "mug.ply", "params": {
        "transl": np.zeros((2, 3)), "global_orient": np.zeros((2, 3))}},
        "table": {"table_mesh": "table.ply", "params": {
            "transl": np.zeros((2, 3)), "global_orient": np.zeros((2, 3))}}}
    monkeypatch.setattr(implementation, "_category", lambda path: "mug")
    monkeypatch.setattr(implementation, "load_clip", lambda path: data)
    monkeypatch.setattr(implementation, "_distractors", lambda *args: ["apple", "pear", "dice"])
    monkeypatch.setattr(implementation, "mesh_path", lambda root, name: tmp_path / name)
    monkeypatch.setattr(implementation, "_trajectory", lambda *args: [[0, 0, 0], [1, 1, 1]])
    monkeypatch.setattr(implementation, "reconstruct", lambda *args: (
        np.zeros((2, 4, 3)), np.array([[0, 1, 2]]), np.tile(np.eye(4), (2, 1, 1)),
        np.ones(2, dtype=bool), np.zeros(2, dtype=bool)))
    # Labelled vertices must pass independently of the whole-hand diagnostic.
    monkeypatch.setattr(implementation, "contact_distances", lambda *args: ([0.011], [0.03]))
    monkeypatch.setattr(implementation, "render_job", lambda *args: pytest.fail("Must gate before rendering"))
    rooms = [tmp_path / f"room{i}/urdf/export_scene/scene.urdf" for i in range(2)]
    output = _new_directory(tmp_path / "rejected")
    with pytest.raises(ValueError, match="Contact reconstruction"):
        implementation._session(config, tmp_path, {}, "s1", 0, rooms,
                                [("clip.npz", 2, 1)], ["mug", "apple", "pear", "dice"], 7, output)
    assert not (output / "work").exists()
    assert not list(output.rglob("*.npz"))
    diagnostic = json.loads((output / "failure.json").read_text())
    assert diagnostic["whole_contacting_hand_median_m"] == 0.03


def test_row_vector_rotation_round_trip_and_contact_distance(tmp_path):
    import trimesh
    from nursery.egocentric_replay.grab import contact_distances

    mesh = trimesh.creation.box(extents=[0.2, 0.4, 0.6])
    mesh.export(tmp_path / "box.ply")
    params = {"global_orient": np.array([[0.3, -0.6, 0.8]]),
              "transl": np.array([[1, 2, 3]])}
    from scipy.spatial.transform import Rotation
    row_rotation = Rotation.from_rotvec(params["global_orient"][0]).as_matrix()
    world = mesh.vertices @ row_rotation + params["transl"][0]
    matrix = rigid_matrix(params, 0)
    np.testing.assert_allclose(world, mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3])
    np.testing.assert_allclose((world - matrix[:3, 3]) @ matrix[:3, :3], mesh.vertices)
    data = {"object": {"object_mesh": "box.ply", "params": params},
            "contact": {"body": np.ones((1, len(world)))}}
    labelled, whole = contact_distances(data, [0], world[None],
                                       {"left_hand": np.arange(len(world))}, tmp_path)
    assert max(labelled + whole) < 1e-7
    from nursery.egocentric_replay.build import _trajectory
    swept = np.asarray(_trajectory(data, tmp_path))
    np.testing.assert_allclose(swept[0], [world.min(0), world.max(0)], atol=1e-7)


def test_retargeted_body_ignores_stale_template_gender_and_pca(monkeypatch):
    import smplx
    import torch
    from types import SimpleNamespace
    from nursery.egocentric_replay.grab import reconstruct

    captured = {}

    class Model:
        parents = [-1] * 55
        faces = np.array([[0, 1, 2]])

        def __call__(self, **params):
            captured.update(params)
            return SimpleNamespace(full_pose=torch.zeros((1, 165)),
                                   vertices=torch.zeros((1, 3, 3)),
                                   joints=torch.zeros((1, 55, 3)))

    def create(path, **kwargs):
        captured.update(kwargs)
        return Model()

    monkeypatch.setattr(smplx, "create", create)
    data = {"gender": "female", "body": {"vtemp": "missing.ply", "params": {
        "global_orient": np.zeros((1, 3)), "body_pose": np.zeros((1, 63)),
        "transl": np.zeros((1, 3)), "left_hand_pose": np.ones((1, 24)),
        "right_hand_pose": np.ones((1, 24)), "betas": np.ones((1, 10))}},
        "lhand": {"params": {"fullpose": np.full((1, 45), 0.2)}},
        "rhand": {"params": {"fullpose": np.full((1, 45), 0.3)}},
        "contact": {"body": np.zeros((1, 3))}}
    reconstruct(data, np.array([0]), {"body_models": "unused"},
                {"left_hand": [0], "right_hand": [1]})
    assert captured["gender"] == "neutral" and captured["use_pca"] is False
    assert "v_template" not in captured and "betas" not in captured
    assert captured["left_hand_pose"].shape == (1, 45)
    np.testing.assert_allclose(captured["right_hand_pose"], 0.3)


def test_certified_square_contains_actor_and_table_and_rejects_oversize():
    import ast
    import math
    from pathlib import Path
    from types import SimpleNamespace

    source = Path("nursery/egocentric_replay/blender.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "open_area")
    namespace = {"np": np, "math": math, "Vector": np.asarray}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "placement", "exec"), namespace)
    distance = 2.5 / 2**0.5
    square = {"square_orientation_degrees": 45, "floor_height_m": 0.1,
              "square_vertices_m": [[1 + distance, 1], [1, 1 + distance],
                                    [1 - distance, 1], [1, 1 - distance]]}
    points = np.array([[0, 0, -0.2], [1, 1, 1.8]])
    table = SimpleNamespace(matrix_world=np.eye(3), bound_box=[[0, 0, 0], [0.5, 0.5, 0.8]])
    shift = namespace["open_area"](points, [table], square)
    rotation = np.array([[2**-0.5, -2**-0.5], [2**-0.5, 2**-0.5]])
    local = (points[:, :2] + shift[:2] - [1, 1]) @ rotation
    assert np.max(np.abs(local)) <= 1.25
    assert (points + shift)[:, 2].min() == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="does not fit"):
        namespace["open_area"](points * 4, [table], square)
