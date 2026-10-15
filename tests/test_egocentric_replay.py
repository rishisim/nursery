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
                       {"category": "apple", "visible_px": 1, "cells": [3]},
                       {"category": "pear", "visible_px": 0, "cells": []}]
    assert json.loads(visible_objects(mask * 0, {1: "mug"})) == [
        {"category": "mug", "visible_px": 0, "cells": []}]


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
    vocabulary = implementation.load_vocabulary()
    words = set(vocabulary["words"]) | set(vocabulary["extras"].values())
    assert manifest["statistics"]["frames"] == 12
    assert manifest["statistics"]["probes"] == 2 * len(words)
    assert len(rendered) == 2 * len(words)
    # Konkle-word probes show held-out instances only; extras keep their single GRAB mesh.
    assets = json.loads(implementation.MANIFEST.read_text())["words"]
    train_names = {f"a_{item['id']}" for items in assets.values() for item in items if item["split"] == "train"}
    assert not train_names & {probe["name"] for probe in rendered if "name" in probe}
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

        def apply_async(self, function, arguments):
            result = function(*arguments)
            return SimpleNamespace(get=lambda: result)

    monkeypatch.setattr(implementation.multiprocessing, "get_context",
                        lambda name: SimpleNamespace(Pool=Pool))
    packed = implementation.build(config_path, rooms, tmp_path / "packed", 2, 7, workers=2)
    assert packed_counts == [2]
    assert packed["workers"] == 2
    assert pq.read_table(tmp_path / "packed/frames.parquet").to_pylist() == pq.read_table(
        output / "frames.parquet").to_pylist()

    selected = implementation.build(config_path, rooms, tmp_path / "selected", 2, 7, subjects=["s1"])
    selected_sessions = pq.read_table(tmp_path / "selected/sessions.parquet").to_pylist()
    assert {row["subject"] for row in selected_sessions} == {"s1"}
    assert selected["statistics"]["probes"] == 2 * len(words)
    with pytest.raises(ValueError, match="distinct subjects"):
        implementation.build(config_path, rooms, tmp_path / "duplicates", subjects=["s1", "s1"])

    def sometimes_one_category(*args):
        result = fake_session(*args)
        result[0][0]["objects"] = '[{"category":"mug","visible_px":10,"cells":[1]}]'
        result[0][1]["objects"] = '[{"category":"mug","visible_px":10,"cells":[1]}, {"category":"apple","visible_px":5,"cells":[2]}, {"category":"pear","visible_px":5,"cells":[3]}]'
        return result

    monkeypatch.setattr(implementation, "_session", sometimes_one_category)
    accepted = implementation.build(config_path, rooms, tmp_path / "sometimes_one", 2, 7)
    assert accepted["status"] == "complete"
    assert accepted["statistics"]["two_visible_categories_fraction"] == 0.5
    assert accepted["statistics"]["mean_visible_categories"] == 2

    def held_invisible(*args):
        result = fake_session(*args)
        for row in result[0]:
            row["objects"] = '[{"category":"mug","visible_px":0,"cells":[]}, {"category":"pear","visible_px":10,"cells":[1]}, {"category":"apple","visible_px":5,"cells":[2]}]'
        return result

    monkeypatch.setattr(implementation, "_session", held_invisible)
    with pytest.raises(ValueError, match="visibility gate failed"):
        implementation.build(config_path, rooms, tmp_path / "held_invisible", 2, 7)

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
    (tmp_path / "swaps.json").write_text(json.dumps({"sources": {}}))
    config["swaps"] = str(tmp_path / "swaps.json")
    source = tmp_path / "clip.npz"
    source.touch()
    data = {"obj_name": "mug", "contact": {"body": np.zeros((2, 4))}, "object": {"object_mesh": "mug.ply", "params": {
        "transl": np.zeros((2, 3)), "global_orient": np.zeros((2, 3))}},
        "table": {"table_mesh": "table.ply", "params": {
            "transl": np.zeros((2, 3)), "global_orient": np.zeros((2, 3))}}}
    monkeypatch.setattr(implementation, "_category", lambda path: "mug")
    monkeypatch.setattr(implementation, "load_clip", lambda path: data)
    monkeypatch.setattr(implementation, "_distractors", lambda *args: [
        {"category": name, "mesh": f"{name}.ply"} for name in ("apple", "pear", "dice")])
    monkeypatch.setattr(implementation, "mesh_path", lambda root, name: tmp_path / name)
    monkeypatch.setattr(implementation, "_trajectory", lambda *args: ([[0, 0, 0], [1, 1, 1]], 0))
    monkeypatch.setattr(implementation, "contact_indices", lambda *args: np.arange(2))
    monkeypatch.setattr(implementation, "hand_geometry", lambda *args: (np.zeros((2, 4, 3)), np.arange(4), [], 0))
    monkeypatch.setattr(implementation, "_body_model", lambda *args: None)
    monkeypatch.setattr(implementation, "_object_surface", lambda *args: None)
    monkeypatch.setattr(implementation, "_render_samples", lambda *args: (
        np.zeros((2, 4, 3)), np.array([[0, 1, 2]]), np.tile(np.eye(4), (2, 1, 1)),
        np.ones(2, dtype=bool), np.zeros(2, dtype=bool), np.zeros((2, 55, 3))))
    # Labelled vertices must pass independently of the whole-hand diagnostic.
    monkeypatch.setattr(implementation, "contact_distances", lambda *args, **kwargs: ([0.011], [0.03]))
    monkeypatch.setattr(implementation, "render_job", lambda *args: pytest.fail("Must gate before rendering"))
    rooms = [tmp_path / f"room{i}/urdf/export_scene/scene.urdf" for i in range(2)]
    output = _new_directory(tmp_path / "rejected")
    with pytest.raises(ValueError, match="Contact reconstruction"):
        implementation._session(config, tmp_path, {"left_hand": [0, 1], "right_hand": [2, 3]}, "s1", 0, rooms,
                                [("clip.npz", 2, 1)], {"table": [{"category": name, "mesh": f"{name}.ply"}
                                for name in ("mug", "apple", "pear", "dice")], "floor": []}, 7, output)
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
    data["framerate"] = 120
    swept = np.asarray(_trajectory(data, tmp_path)[0])
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


def test_distractor_candidates_include_real_objects_and_exclude_only_current_clip():
    from nursery.egocentric_replay.build import _distractors

    pool = [{"category": name, "name": f"{name}_{i}"} for name in ["mug", "camera", "apple", "cheese", "pen"]
            for i in range(3)]
    order = _distractors(pool, "mug", np.random.default_rng(7))
    assert {entry["category"] for entry in order} == {"camera", "apple", "cheese", "pen"}
    assert order == _distractors(pool, "mug", np.random.default_rng(7))
    assert "mug" in {entry["category"] for entry in _distractors(pool, "camera", np.random.default_rng(7))}
    assert len(_distractors(pool, "mug", np.random.default_rng(7), limit=5)) == 5


def test_pools_hold_train_instances_by_surface_and_extras_on_table(tmp_path):
    from pathlib import Path
    from nursery.egocentric_replay.assets import MANIFEST, load_vocabulary
    from nursery.egocentric_replay.build import _pools

    vocabulary = load_vocabulary()
    assets = json.loads(MANIFEST.read_text())["words"]
    pools = _pools({"library": str(tmp_path)}, vocabulary, assets, Path("grab"))
    floor = {entry["category"] for entry in pools["floor"]}
    assert floor == {w for w, spec in vocabulary["words"].items() if spec["place"] == "floor"}
    held_out = {f"a_{item['id']}" for items in assets.values() for item in items if item["split"] != "train"}
    assert not held_out & {entry.get("name") for entry in pools["table"] + pools["floor"]}
    assert set(vocabulary["extras"].values()) <= {entry["category"] for entry in pools["table"] if "mesh" in entry}


def test_swap_alignment_recovers_rotation_and_scale():
    import trimesh
    from scipy.spatial.transform import Rotation
    from nursery.egocentric_replay.swaps import align

    target = trimesh.creation.box(extents=[0.3, 0.1, 0.05])
    rotation = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
    source = trimesh.Trimesh(target.vertices @ rotation.T * 0.5 + [0.1, 0, 0.2], target.faces)
    transform, chamfer, scale = align(source, target)
    assert abs(scale - 0.5) < 0.01 and chamfer < 0.002
    moved = target.vertices @ transform[:3, :3].T + transform[:3, 3]
    assert np.abs(np.sort(moved, 0) - np.sort(source.vertices, 0)).max() < 0.01


def test_floor_objects_stay_in_square_and_clear_of_occupied():
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    function = next(node for node in ast.parse(Path("nursery/egocentric_replay/blender.py").read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_place_floor")

    class Obj:
        def __init__(self, size):
            self.size, self.location, self.rotation_euler = size, (0, 0, 0), (0, 0, 0)
            self.bound_box = [(x, y, z) for x in (-size / 2, size / 2) for y in (-size / 2, size / 2) for z in (0, size)]

        @property
        def matrix_world(self):
            return World(self)

    class World:
        def __init__(self, obj):
            self.obj = obj

        def __matmul__(self, corner):
            return np.asarray(corner) + np.asarray(self.obj.location)

    removed = []
    namespace = {"np": np, "math": __import__("math"), "Vector": np.asarray, "remove_object": removed.append,
                 "bpy": SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None)))}
    sizes = {"big": 3.0, "rug": 0.6, "ball": 0.2}
    objects = []

    def load_entry(entry, instance):
        obj = Obj(sizes[entry["category"]])
        obj.instance = instance
        objects.append(obj)
        return obj

    namespace["load_entry"] = load_entry
    exec(compile(ast.Module(body=[function], type_ignores=[]), "floor", "exec"), namespace)
    square = {"square_orientation_degrees": 0, "square_vertices_m": [[-1.25, -1.25], [1.25, -1.25], [1.25, 1.25], [-1.25, 1.25]],
              "floor_height_m": 0.0}
    occupied = [np.array([[-0.5, -0.5], [0.5, 0.5]])]
    placed, names = namespace["_place_floor"]([{"category": c} for c in ("big", "rug", "ball")], 2, occupied,
                                              square, 5, np.random.default_rng(3))
    assert names == ["rug", "ball"] and removed and removed[0].size == 3.0
    assert [obj.instance for obj in placed] == [5, 6]
    for obj in placed:
        low = np.asarray(obj.location)[:2] - obj.size / 2
        high = np.asarray(obj.location)[:2] + obj.size / 2
        assert np.all(low >= -1.25 - 1e-9) and np.all(high <= 1.25 + 1e-9)
        assert not (np.all(high > -0.55) and np.all(low < 0.55))


def test_contact_subsample_is_deterministic_and_keeps_rendered_frames():
    from nursery.egocentric_replay.grab import contact_indices

    data = {"contact": {"body": np.ones((2400, 4))}, "framerate": 120}
    rendered = np.array([0, 100, 2399])
    ids = {"left_hand": [0, 1], "right_hand": [2, 3]}
    selected = contact_indices(data, rendered, ids)
    assert set(rendered) <= set(selected)
    assert set(np.arange(0, 2400, 120)) <= set(selected)
    assert len(selected) <= 223
    np.testing.assert_array_equal(selected, contact_indices(data, rendered, ids))


def test_surface_search_matches_exact_triangle_queries():
    import trimesh
    from nursery.egocentric_replay.grab import _surface_distances, _surface_index

    mesh = trimesh.creation.icosphere(subdivisions=2)
    points = np.random.default_rng(7).normal(size=(30, 3)) * 2
    points = np.r_[points, [[0, 0, 0], [10, 0, 0]]]
    triangles = np.asarray(mesh.triangles)
    actual = _surface_distances(triangles, _surface_index(triangles), points)
    expected = trimesh.proximity.closest_point(mesh, points)[1]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("zero_fit", [False, True])
def test_placement_resamples_category_after_bounded_positions(zero_fit):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    function = next(node for node in ast.parse(Path("nursery/egocentric_replay/blender.py").read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_place_distractors")
    table = SimpleNamespace(low=np.array([0, 0, 0]), high=np.array([1, 1, 0]), location=np.zeros(3))
    removed, loaded = [], []

    def load(path, color, instance):
        size = 2 if path == "oversized" else 0.03
        obj = SimpleNamespace(low=np.array([-size / 2, -size / 2, 0]),
                              high=np.array([size / 2, size / 2, size]),
                              location=np.zeros(3), instance=instance)
        loaded.append(obj)
        return obj

    class Rng:
        calls = 0

        def uniform(self, low, high):
            self.calls += 1
            # The first small-object proposal intersects the swept region.
            return np.array([0.5, 0.5]) if self.calls == 257 else np.array([0.2, 0.2])

    rng = Rng()
    namespace = {"np": np, "bounds": lambda obj: (obj.low + obj.location, obj.high + obj.location),
                 "load_ply": load, "remove_object": removed.append,
                 "load_entry": lambda entry, instance: load(entry["mesh"], entry["color"], instance),
                 "bpy": SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None))),
                 "table_surface": lambda table, x, y: (np.array([x, y, 0]), None),
                 "rest_position": lambda obj, point, normal: point}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "placement", "exec"), namespace)
    occupied = np.array([[[-1, -1, -1], [2, 2, 2]]]) if zero_fit else np.array([[[0.45, 0.45, 0], [0.55, 0.55, 0.2]]])
    objects, names = namespace["_place_distractors"](table,
        [{"mesh": name, "category": name, "color": []} for name in ["oversized", "mug"]], 1,
        occupied, rng)
    if zero_fit:
        assert objects == names == []
        assert rng.calls == 512 and removed == loaded
        return
    assert rng.calls == 258
    assert removed == [loaded[0]]
    assert names == ["mug"] and objects[0].instance == 2
    np.testing.assert_allclose(objects[0].location, [0.2, 0.2, 0])


def test_hand_only_skinning_matches_upstream_lbs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import smplx
    import torch
    from smplx.lbs import lbs
    from nursery.egocentric_replay.grab import hand_geometry

    template = torch.tensor([[0.1, 0, 0], [0.2, 0, 0], [0, 0.1, 0], [0, 0.2, 0]])
    parents = torch.tensor([-1] + [0] * 54)
    weights = torch.zeros(4, 55)
    weights[:2, 25] = 1
    weights[2:, 40] = 1
    posedirs = torch.randn(486, 12, generator=torch.Generator().manual_seed(7)) * 0.001
    model = SimpleNamespace(v_template=template, posedirs=posedirs,
                            J_regressor=torch.zeros(55, 4), parents=parents, lbs_weights=weights)
    monkeypatch.setattr(smplx, "create", lambda *args, **kwargs: model)
    pose = np.random.default_rng(7).normal(size=(3, 165)).astype(np.float32) * 0.2
    pose[:, 66:75] = 0
    translation = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=np.float32)
    data = {"framerate": 120, "body": {"params": {"global_orient": pose[:, :3],
            "body_pose": pose[:, 3:66], "transl": translation}},
            "lhand": {"params": {"fullpose": pose[:, 75:120]}},
            "rhand": {"params": {"fullpose": pose[:, 120:]}}}
    expected = lbs(torch.zeros(3, 1), torch.from_numpy(pose), template, torch.zeros(4, 3, 1),
                   posedirs, model.J_regressor, parents, weights)[0].numpy() + translation[:, None]
    actual, ids, bounds, margin = hand_geometry(data, {"left_hand": [0, 1], "right_hand": [2, 3]},
                                               np.arange(3), {"body_models": str(tmp_path)})
    np.testing.assert_allclose(actual, expected[:, ids], atol=1e-7)
    assert len(bounds) == 6 and margin == 0
    expected_bounds = []
    for hand in ([0, 1], [2, 3]):
        points = expected[:, hand]
        expected_bounds.extend(np.stack([points.min(axis=1), points.max(axis=1)], axis=1))
    np.testing.assert_allclose(bounds, expected_bounds, atol=1e-7)


def test_local_hand_path_does_not_block_free_tabletop(tmp_path, monkeypatch):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from nursery.egocentric_replay.grab import hand_geometry
    import smplx
    import torch

    template = torch.tensor([[0.05, 0.05, 0], [0.07, 0.07, 0.1]])
    weights = torch.zeros(2, 55); weights[:, 0] = 1
    model = SimpleNamespace(v_template=template, posedirs=torch.zeros(486, 6),
        J_regressor=torch.zeros(55, 2), parents=torch.tensor([-1] + [0] * 54), lbs_weights=weights)
    monkeypatch.setattr(smplx, "create", lambda *args, **kwargs: model)
    translation = np.zeros((25, 3)); translation[:12, 0] = np.linspace(0, .15, 12)
    data = {"framerate":120,"body":{"params":{"global_orient":np.zeros((25,3)),
        "body_pose":np.zeros((25,63)),"transl":translation}},
        "lhand":{"params":{"fullpose":np.zeros((25,45))}},
        "rhand":{"params":{"fullpose":np.zeros((25,45))}}}
    _,_,occupied,margin = hand_geometry(data,{"left_hand":[0,1],"right_hand":[0,1]},
                                         np.array([0]),{"body_models":str(tmp_path)})
    assert margin == 0 and np.max(np.array(occupied)[:,:,0]) < .23
    table = SimpleNamespace(low=np.array([0,0,0]),high=np.array([1,1,0]),location=np.zeros(3))
    function = next(n for n in ast.parse(Path("nursery/egocentric_replay/blender.py").read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name=="_place_distractors")
    namespace = {"np":np,"bounds":lambda o:(o.low+o.location,o.high+o.location),
        "load_ply":lambda *args:SimpleNamespace(low=np.array([-.03,-.03,0]),high=np.array([.03,.03,.06]),location=np.zeros(3)),
        "load_entry":lambda *args:SimpleNamespace(low=np.array([-.03,-.03,0]),high=np.array([.03,.03,.06]),location=np.zeros(3)),
        "remove_object":lambda o:None, "bpy":SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda:None))),
        "table_surface":lambda t,x,y:(np.array([x,y,0]),None),
        "rest_position":lambda o,p,n:p}
    exec(compile(ast.Module(body=[function],type_ignores=[]),"placement","exec"),namespace)
    objects,names = namespace["_place_distractors"](table,[{"mesh":"mug","category":"mug","color":[]}],
        1,np.array(occupied),SimpleNamespace(uniform=lambda *args:np.array([.3,.1])))
    assert len(objects)==1 and names==["mug"]
