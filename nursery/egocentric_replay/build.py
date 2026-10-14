"""Build equal-length egocentric replay sessions and held-out object probes."""

import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from ..humanoid.embodiedgen_hoidini.run_motion import _output_directory
from .grab import (_body_model, _object_surface, clearance_indices, contact_distances, contact_indices, hand_geometry,
                   inflated_bounds, load_clip, mesh_path, reconstruct, rigid_matrix)
from .render import render_job
from .schema import table, visible_objects
from .sessions import chain_clips, sample_chain, subject_split


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported replay configuration")
    if not isinstance(config["duration_s"], int) or config["duration_s"] < 1:
        raise ValueError("duration_s must be a positive integer")
    if not 0 < config["horizontal_fov_deg"] < 180:
        raise ValueError("Invalid horizontal field of view")
    if len(config["resolution"]) != 2 or min(config["resolution"]) < 4:
        raise ValueError("Invalid resolution")
    if config["probes_per_category_per_split"] < 1:
        raise ValueError("Positive render and probe counts required")
    session_rooms, probe_rooms = config["session_rooms"], config["probe_rooms"]
    if (not session_rooms or not probe_rooms or set(session_rooms) & set(probe_rooms) or
            len(set(session_rooms + probe_rooms)) != len(session_rooms + probe_rooms)):
        raise ValueError("Session and probe rooms must be distinct nonempty lists")
    for subject in config["subjects"]:
        subject_split(subject)
    if len(set(config["subjects"])) != len(config["subjects"]):
        raise ValueError("Duplicate subjects")
    return config


def room_bank(path):
    rooms = sorted(Path(path).resolve().glob("*/urdf/export_scene/scene.urdf"))
    if len(rooms) < 2:
        raise ValueError("At least two exported rooms required for train/probe separation")
    return rooms


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _catalog(config):
    root = Path(config["grab_root"])
    catalog = {}
    categories = set()
    # Read the actual object key, rather than splitting filenames at underscores.
    for subject in config["subjects"]:
        clips = []
        for path in sorted((root / subject).glob("*.npz")):
            with np.load(path, allow_pickle=True) as data:
                if data["sbj_id"].item() != subject:
                    raise ValueError("GRAB subject mismatch")
                categories.add(str(data["obj_name"].item()))
                clips.append((str(path.relative_to(root)), len(data["body"].item()["params"]["transl"]),
                              float(data["framerate"])))
        if not clips:
            raise ValueError(f"No clips for {subject}")
        catalog[subject] = clips
    # Vocabulary includes every object category, even when selecting pilot subjects.
    for path in root.glob("s*/*.npz"):
        with np.load(path, allow_pickle=True) as data:
            categories.add(str(data["obj_name"].item()))
    return catalog, sorted(categories)


def _category(path):
    with np.load(path, allow_pickle=True) as data:
        return str(data["obj_name"].item())


def _distractors(categories, active, rng):
    candidates = [category for category in categories if category != active]
    if len(candidates) < 3:
        raise ValueError("Need at least three distractor categories")
    return rng.permutation(candidates).tolist()


def _trajectory(data, root):
    import trimesh
    from scipy.spatial.transform import Rotation

    mesh = trimesh.load(mesh_path(root, data["object"]["object_mesh"]), process=False)
    corners = trimesh.bounds.corners(mesh.bounds)
    params = data["object"]["params"]
    points = np.einsum("nij,kj->nki", Rotation.from_rotvec(params["global_orient"]).as_matrix().transpose(0, 2, 1),
                       corners) + params["transl"][:, None]
    indices = clearance_indices(len(points), data["framerate"])
    bounds, margin = inflated_bounds(points, indices)
    return bounds.tolist(), margin



def _render_samples(data, samples, config, hand_ids):
    combined = {"body": {"params": {}}, "lhand": {"params": {}},
                "rhand": {"params": {}}, "contact": {}}
    for part, keys in (("body", ("global_orient", "body_pose", "transl")),
                       ("lhand", ("fullpose",)), ("rhand", ("fullpose",))):
        for key in keys:
            combined[part]["params"][key] = np.stack([data[clip][part]["params"][key][frame]
                                                       for clip, frame in samples])
    combined["contact"]["body"] = np.stack([data[clip]["contact"]["body"][frame] for clip, frame in samples])
    return reconstruct(combined, np.arange(len(samples)), config, hand_ids)

def _session(config, root, hand_ids, subject, number, rooms, clips, categories, seed, output):
    from PIL import Image

    started = time.monotonic()
    video_id = f"{subject}_{number:04d}"
    rng = np.random.default_rng(seed)
    split = subject_split(subject)
    room = rooms[((int(subject[1:]) - 1) + number) % len(rooms)]
    room_id = room.parents[2].name
    segments = chain_clips(clips, config["duration_s"], seed)
    samples = sample_chain(segments, config["duration_s"])
    data_by_clip = {item["clip"]: load_clip(root / item["clip"]) for item in segments}
    clip_categories = {name: str(data["obj_name"]) for name, data in data_by_clip.items()}
    all_vertices, faces, all_camera, all_left, all_right = _render_samples(data_by_clip, samples, config, hand_ids)
    hand_model = _body_model(config)
    colors = {category: rng.uniform(0.15, 0.85, 3).tolist() for category in categories}
    job = {"room_urdf": str(room), "clear_square": config["clear_squares"][room_id], "job_id": video_id, "seed": seed, "segments": []}
    rows, labelled, whole = [], [], []
    sources = {}
    for segment_index, segment in enumerate(segments):
        selected = [(second, source) for second, (clip, source) in enumerate(samples)
                    if clip == segment["clip"] and segment["start"] <= second < segment["stop"]]
        if not selected:
            continue
        data = data_by_clip[segment["clip"]]
        category = str(data["obj_name"])
        indices = np.array([source for _, source in selected])
        seconds = np.array([second for second, _ in selected])
        vertices, camera = all_vertices[seconds], all_camera[seconds]
        left, right = all_left[seconds], all_right[seconds]
        gate_indices = contact_indices(data, indices, hand_ids)
        gate_vertices, vertex_ids, hand_bounds, hand_margin = hand_geometry(data, hand_ids, gate_indices, config, hand_model)
        surface = _object_surface(data, root)
        distances, _ = contact_distances(data, gate_indices, gate_vertices, hand_ids, root, vertex_ids, labelled_only=True, surface=surface)
        _, all_distances = contact_distances(data, indices, vertices, hand_ids, root, surface=surface)
        trajectory_bounds, object_margin = _trajectory(data, root)
        labelled.extend(distances)
        whole.extend(all_distances)
        sources[segment["clip"]] = _sha256(root / segment["clip"])
        array_path = output / "work" / f"{video_id}_{segment_index:04d}.npz"
        array_path.parent.mkdir(exist_ok=True)
        np.savez(array_path, vertices=vertices, faces=faces, camera=camera,
                 object=np.array([rigid_matrix(data["object"]["params"], i) for i in indices]),
                 table=np.array([rigid_matrix(data["table"]["params"], i) for i in indices]))
        frames = []
        for index, (second, _) in enumerate(selected):
            image = f"frames/{video_id}/{second:05d}.jpg"
            mask = f"work/{video_id}_{second:05d}.png"
            frames.append({"image": image, "mask": mask})
            rows.append({"video_id": video_id, "frame_idx": second, "image": image,
                         "subject": subject, "room": room_id, "clip": segment["clip"],
                         "contact_left": bool(left[index]), "contact_right": bool(right[index]),
                         "held_category": category if left[index] or right[index] else None,
                         "objects": "[]"})
        job["segments"].append({"arrays": str(array_path), "clip": segment["clip"], "segment_index": segment_index, "table_mesh": str(mesh_path(
            root, data["table"]["table_mesh"])), "object_mesh": str(mesh_path(
            root, data["object"]["object_mesh"])), "color": colors[category],
            "trajectory_bounds": trajectory_bounds,
            "hand_bounds": hand_bounds, "hand_displacement_m": hand_margin,
            "object_displacement_m": object_margin, "frames": frames,
            "distractor_count": int(rng.integers(3, min(5, len(categories) - 1) + 1)),
            "distractors": [{"mesh": str(root / "contact_meshes" / f"{name}.ply"),
                             "category": name, "color": colors[name]}
                            for name in _distractors(categories, category, rng)]})
    # Do not render geometry that has already failed the required contact construction gate.
    if not labelled or not whole or np.median(labelled) >= 0.01:
        for segment in job["segments"]:
            Path(segment["arrays"]).unlink()
        if not any((output / "work").iterdir()):
            (output / "work").rmdir()
        diagnostic = {"status": "contact_reconstruction_failed", "video_id": video_id,
                      "labelled_contact_median_m": float(np.median(labelled)) if labelled else None,
                      "labelled_contact_p90_m": float(np.percentile(labelled, 90)) if labelled else None,
                      "whole_contacting_hand_median_m": float(np.median(whole)) if whole else None,
                      "source_sha256": sources, "config": config}
        (output / "failure.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        raise ValueError(f"Contact reconstruction gate failed: labelled median={np.median(labelled) if labelled else None} m; "
                         f"whole contacting-hand median={np.median(whole) if whole else None} m")
    print(json.dumps({"video_id": video_id, "labelled_contact_median_m": float(np.median(labelled)),
                      "labelled_contact_p90_m": float(np.percentile(labelled, 90)),
                      "whole_contacting_hand_median_m": float(np.median(whole))}), flush=True)
    prepared = time.monotonic()
    render_job(job, config, output)
    receipt = json.loads((output / f"render_stats_{video_id}.json").read_text())
    receipt["preparation_seconds"] = prepared - started
    receipt["clearance_displacements"] = [{"clip": s["clip"], "hand_m": s["hand_displacement_m"],
                                            "object_m": s["object_displacement_m"]} for s in job["segments"]]
    (output / f"render_stats_{video_id}.json").write_text(json.dumps(receipt) + "\n")
    realized = receipt["distractors"]
    rows.sort(key=lambda row: row["frame_idx"])
    for row in rows:
        mask_path = output / f"work/{video_id}_{row['frame_idx']:05d}.png"
        mask = np.asarray(Image.open(mask_path))[:, :, 0]
        category = clip_categories[row["clip"]]
        row["objects"] = visible_objects(mask, {1: category, **dict(enumerate(realized[row["image"]], 2))})
        mask_path.unlink()
    for segment in job["segments"]:
        Path(segment["arrays"]).unlink()
    session = {"video_id": video_id, "subject": subject, "room": room_id, "split": split,
               "clips": json.dumps([item["clip"] for item in segments]),
               "distractors": json.dumps(sorted({name for names in realized.values() for name in names})), "duration_s": config["duration_s"]}
    return rows, session, labelled, whole, sources


def build(config_path, rooms_path, output_path, sessions_per_subject=1, seed=7, session_index=None, workers=1, subjects=None):
    import pyarrow.parquet as pq

    config = load_config(config_path)
    if subjects is not None:
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("Select distinct subjects")
        for subject in subjects:
            subject_split(subject)
        config["subjects"] = list(subjects)
    if not 1 <= workers <= 64:
        raise ValueError("workers must be between 1 and 64")
    if sessions_per_subject < 1:
        raise ValueError("Positive sessions-per-subject required")
    available = {room.parents[2].name: room for room in room_bank(rooms_path)}
    rooms = [available[name] for name in config["session_rooms"]]
    probe_rooms = [available[name] for name in config["probe_rooms"]]
    report = json.loads((Path(rooms_path) / "room_bank_report.json").read_text())
    square_keys = ("clear_2_5m_square", "square_vertices_m", "square_orientation_degrees",
                   "floor_height_m")
    config["clear_squares"] = {entry["room_id"]: {key: entry["geometry"][key] for key in square_keys}
                               for entry in report["rooms"]
                               if entry["room_id"] in config["session_rooms"] + config["probe_rooms"]}
    for name in config["session_rooms"] + config["probe_rooms"]:
        if not config["clear_squares"][name]["clear_2_5m_square"]:
            raise ValueError(f"No certified square for {name}")
    catalog, categories = _catalog(config)
    schedule = [(subject, number) for subject in config["subjects"]
                for number in range(sessions_per_subject)]
    if session_index is not None:
        if session_index not in range(len(schedule)):
            raise ValueError("Session index outside schedule")
        schedule = [schedule[session_index]]
    with Path(config["hand_vertex_ids"]).open("rb") as stream:
        hand_ids = pickle.load(stream)
    output = _output_directory(output_path)
    started = time.monotonic()
    frames, sessions, labelled, whole, sources = [], [], [], [], {}
    arguments = [(config, Path(config["grab_root"]), hand_ids, subject, number, rooms,
                  catalog[subject], categories,
                  seed + config["subjects"].index(subject) * sessions_per_subject + number, output)
                 for subject, number in schedule]
    if workers == 1:
        results = [_session(*args) for args in arguments]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(_session, arguments)
    for result in results:
        frames.extend(result[0]); sessions.append(result[1])
        labelled.extend(result[2]); whole.extend(result[3]); sources.update(result[4])
    probes = []
    if session_index is None:
        rng = np.random.default_rng(seed + 100000)
        table_pose = rigid_matrix(load_clip(Path(config["grab_root"]) /
            catalog[config["subjects"][0]][0][0])["table"]["params"], 0).tolist()
        jobs = {str(room): {"room_urdf": str(room), "clear_square": config["clear_squares"][room.parents[2].name],
                            "job_id": "probe_" + room.parents[2].name, "segments": [], "probes": []} for room in probe_rooms}
        for split in ("dev", "test"):
            for category in categories:
                for number in range(config["probes_per_category_per_split"]):
                    room = probe_rooms[number % len(probe_rooms)]
                    probe_id = f"{split}_{category}_{number:04d}"
                    probes.append({"probe_id": probe_id, "image": f"probes/{probe_id}.jpg",
                                   "category": category, "room": room.parents[2].name, "split": split})
                    jobs[str(room)]["probes"].append({
                        "mesh": str(Path(config["grab_root"]) / "contact_meshes" / f"{category}.ply"),
                        "table_mesh": str(Path(config["grab_root"]) / "contact_meshes/table.ply"),
                        "table_pose": table_pose,
                        "color": rng.uniform(0.15, 0.85, 3).tolist(), "image": probes[-1]["image"],
                        "yaw": float(rng.uniform(0, 2 * np.pi)), "height": float(rng.uniform(0.3, 0.6))})
        if workers == 1:
            for job in jobs.values():
                render_job(job, config, output)
        else:
            with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
                pool.starmap(render_job, [(job, config, output) for job in jobs.values()])
    receipts = [json.loads(path.read_text()) for path in sorted(output.glob("render_stats_*.json"))]
    if receipts:
        (output / "render_stats.json").write_text(json.dumps({
            "seconds": sum(item["seconds"] for item in receipts),
            "image_seconds": [second for item in receipts for second in item["image_seconds"]],
            "blender_version": receipts[0]["blender_version"], "jobs": receipts}))
    for name, rows in (("frames", frames), ("sessions", sessions), ("probes", probes)):
        pq.write_table(table(rows, name), output / f"{name}.parquet")
    seconds = time.monotonic() - started
    contacted = [row for row in frames if row["held_category"] is not None]
    statistics = {"frames": len(frames), "sessions": len(sessions), "probes": len(probes),
                  "wall_seconds": seconds, "total_images_per_minute": (len(frames) + len(probes)) * 60 / seconds,
                  "contact_fraction": len(contacted) / len(frames),
                  "mean_visible_categories": float(np.mean([sum(obj["visible_px"] > 0 for obj in json.loads(row["objects"])) for row in frames])),
                  "held_visible_fraction": sum(any(obj["category"] == row["held_category"] and obj["visible_px"] > 0
                      for obj in json.loads(row["objects"])) for row in contacted) / len(contacted) if contacted else None,
                  "labelled_contact_median_m": float(np.median(labelled)),
                  "labelled_contact_p90_m": float(np.percentile(labelled, 90)),
                  "whole_contacting_hand_median_m": float(np.median(whole))}
    statistics["realized_distractor_counts"] = {str(n): sum(c["realized"] == n for r in receipts
        for c in r.get("clip_counts", [])) for n in range(6)}
    statistics["frames_with_two_visible_categories"] = sum(
        sum(obj["visible_px"] > 0 for obj in json.loads(row["objects"])) >= 2 for row in frames)
    statistics["two_visible_categories_fraction"] = statistics["frames_with_two_visible_categories"] / len(frames)
    status = ("complete" if statistics["held_visible_fraction"] is not None and
              statistics["held_visible_fraction"] >= 0.7 and statistics["mean_visible_categories"] >= 2
              else "referential_ambiguity_failed")
    manifest = {"schema_version": 1, "status": status, "config": config, "seed": seed,
                "sessions_per_subject": sessions_per_subject, "session_index": session_index, "workers": workers,
                "renderer": {"engine": "BLENDER_WORKBENCH", "threads": 1,
                    "passes": "Workbench studio/texture RGB; Workbench flat/material raw ID, antialiasing off",
                    "blender_version": json.loads((output / "render_stats.json").read_text())["blender_version"]},
                "sources": sources,
                "rooms": {str(room): _sha256(room) for room in rooms + probe_rooms},
                "hand_vertex_ids_sha256": _sha256(config["hand_vertex_ids"]),
                "body_models_sha256": {path.name: _sha256(path) for path in
                    sorted((Path(config["body_models"]) / "smplx").glob("SMPLX_*.npz"))},
                "software": {name: importlib.metadata.version(name) for name in
                    ("numpy", "smplx", "torch", "trimesh", "pyarrow")},
                "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                    text=True, capture_output=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip(),
                "tables_sha256": {name: _sha256(output / f"{name}.parquet") for name in FIELDS_NAMES},
                "statistics": statistics}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "work").rmdir()
    print(json.dumps(statistics, indent=2), flush=True)
    if status != "complete":
        raise ValueError("Held-object visibility or mean category visibility gate failed; pilot saved")
    return manifest


FIELDS_NAMES = ("frames", "sessions", "probes")
