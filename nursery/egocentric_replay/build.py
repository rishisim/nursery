"""Build equal-length egocentric replay sessions and held-out object probes."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from ..humanoid.embodiedgen_hoidini.run_motion import _output_directory
from .grab import contact_distances, load_clip, mesh_path, reconstruct, rigid_matrix
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
    if config["device"] not in ("CPU", "GPU"):
        raise ValueError("device must be CPU or GPU")
    if len(config["resolution"]) != 2 or min(config["resolution"]) < 4:
        raise ValueError("Invalid resolution")
    if config["samples"] < 1 or config["probes_per_category_per_split"] < 1:
        raise ValueError("Positive render and probe counts required")
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


def _distractors(categories, active, config, rng):
    import trimesh

    candidates = []
    for category in categories:
        if category in active:
            continue
        mesh = trimesh.load(Path(config["grab_root"]) / "contact_meshes" / f"{category}.ply",
                            process=False)
        size = mesh.extents
        if max(size[:2]) < 0.16:
            candidates.append(category)
    if len(candidates) < 3:
        raise ValueError("Need three distinct distractor categories fitting on the GRAB table")
    return rng.choice(candidates, size=3, replace=False).tolist()


def _trajectory(data, root):
    import trimesh
    from scipy.spatial.transform import Rotation

    mesh = trimesh.load(mesh_path(root, data["object"]["object_mesh"]), process=False)
    corners = trimesh.bounds.corners(mesh.bounds)
    params = data["object"]["params"]
    points = np.einsum("nij,kj->nki", Rotation.from_rotvec(params["global_orient"]).as_matrix(),
                       corners) + params["transl"][:, None]
    return [points.min(axis=(0, 1)).tolist(), points.max(axis=(0, 1)).tolist()]


def _session(config, root, hand_ids, subject, number, rooms, clips, categories, seed, output):
    from PIL import Image

    video_id = f"{subject}_{number:04d}"
    rng = np.random.default_rng(seed)
    split = subject_split(subject)
    room = rooms[0] if split == "train" else rooms[1 + number % (len(rooms) - 1)]
    room_id = room.parents[2].name
    segments = chain_clips(clips, config["duration_s"], seed)
    samples = sample_chain(segments, config["duration_s"])
    clip_categories = {item["clip"]: _category(root / item["clip"]) for item in segments}
    active = set(clip_categories.values())
    distractors = _distractors(categories, active, config, rng)
    colors = {category: rng.uniform(0.15, 0.85, 3).tolist() for category in categories}
    job = {"room_urdf": str(room), "segments": []}
    rows, labelled, whole = [], [], []
    sources = {}
    for segment_index, segment in enumerate(segments):
        selected = [(second, source) for second, (clip, source) in enumerate(samples)
                    if clip == segment["clip"] and segment["start"] <= second < segment["stop"]]
        if not selected:
            continue
        data = load_clip(root / segment["clip"])
        category = str(data["obj_name"])
        indices = np.array([source for _, source in selected])
        vertices, faces, camera, left, right = reconstruct(data, indices, config, hand_ids)
        distances, all_distances = contact_distances(data, indices, vertices, hand_ids, root)
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
        job["segments"].append({"arrays": str(array_path), "table_mesh": str(mesh_path(
            root, data["table"]["table_mesh"])), "object_mesh": str(mesh_path(
            root, data["object"]["object_mesh"])), "color": colors[category],
            "trajectory_bounds": _trajectory(data, root), "frames": frames,
            "distractors": [{"mesh": str(root / "contact_meshes" / f"{name}.ply"),
                             "color": colors[name]} for name in distractors]})
    # Do not render geometry that has already failed the required contact construction gate.
    if not labelled or not whole or np.median(labelled) >= 0.01 or np.median(whole) >= 0.01:
        for segment in job["segments"]:
            Path(segment["arrays"]).unlink()
        (output / "work").rmdir()
        diagnostic = {"status": "contact_reconstruction_failed", "video_id": video_id,
                      "labelled_contact_median_m": float(np.median(labelled)) if labelled else None,
                      "whole_contacting_hand_median_m": float(np.median(whole)) if whole else None,
                      "source_sha256": sources, "config": config}
        (output / "failure.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        raise ValueError(f"Contact reconstruction gate failed: labelled median={np.median(labelled) if labelled else None} m; "
                         f"whole contacting-hand median={np.median(whole) if whole else None} m")
    render_job(job, config, output)
    rows.sort(key=lambda row: row["frame_idx"])
    for row in rows:
        mask_path = output / f"work/{video_id}_{row['frame_idx']:05d}.png"
        mask = np.asarray(Image.open(mask_path))[:, :, 0]
        category = clip_categories[row["clip"]]
        row["objects"] = visible_objects(mask, {1: category, **dict(enumerate(distractors, 2))})
        mask_path.unlink()
    for segment in job["segments"]:
        Path(segment["arrays"]).unlink()
    ambiguous = sum(len(json.loads(row["objects"])) >= 2 for row in rows)
    if ambiguous != len(rows):
        (output / "failure.json").write_text(json.dumps({
            "status": "referential_ambiguity_failed", "video_id": video_id,
            "rendered_frames": len(rows), "frames_with_two_visible_categories": ambiguous,
        }, indent=2) + "\n")
        raise ValueError(f"Only {ambiguous}/{len(rows)} frames have two visible categories")
    session = {"video_id": video_id, "subject": subject, "room": room_id, "split": split,
               "clips": json.dumps([item["clip"] for item in segments]),
               "distractors": json.dumps(distractors), "duration_s": config["duration_s"]}
    return rows, session, labelled, whole, sources


def build(config_path, rooms_path, output_path, sessions_per_subject=1, seed=7, session_index=None):
    import pyarrow.parquet as pq

    config = load_config(config_path)
    if sessions_per_subject < 1:
        raise ValueError("Positive sessions-per-subject required")
    rooms = room_bank(rooms_path)
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
    for subject, number in schedule:
        index = config["subjects"].index(subject) * sessions_per_subject + number
        result = _session(config, Path(config["grab_root"]), hand_ids, subject, number, rooms,
                          catalog[subject], categories, seed + index, output)
        frames.extend(result[0]); sessions.append(result[1])
        labelled.extend(result[2]); whole.extend(result[3]); sources.update(result[4])
    probes = []
    if session_index is None:
        rng = np.random.default_rng(seed + 100000)
        table_pose = rigid_matrix(load_clip(Path(config["grab_root"]) /
            catalog[config["subjects"][0]][0][0])["table"]["params"], 0).tolist()
        jobs = {str(room): {"room_urdf": str(room), "segments": [], "probes": []} for room in rooms[1:]}
        for split in ("dev", "test"):
            for category in categories:
                for number in range(config["probes_per_category_per_split"]):
                    room = rooms[1 + number % (len(rooms) - 1)]
                    probe_id = f"{split}_{category}_{number:04d}"
                    probes.append({"probe_id": probe_id, "image": f"probes/{probe_id}.jpg",
                                   "category": category, "room": room.parents[2].name, "split": split})
                    jobs[str(room)]["probes"].append({
                        "mesh": str(Path(config["grab_root"]) / "contact_meshes" / f"{category}.ply"),
                        "table_mesh": str(Path(config["grab_root"]) / "contact_meshes/table.ply"),
                        "table_pose": table_pose,
                        "color": rng.uniform(0.15, 0.85, 3).tolist(), "image": probes[-1]["image"],
                        "yaw": float(rng.uniform(0, 2 * np.pi)), "height": float(rng.uniform(0.3, 0.6))})
        for job in jobs.values():
            render_job(job, config, output)
    for name, rows in (("frames", frames), ("sessions", sessions), ("probes", probes)):
        pq.write_table(table(rows, name), output / f"{name}.parquet")
    seconds = time.monotonic() - started
    contacted = [row for row in frames if row["held_category"] is not None]
    statistics = {"frames": len(frames), "sessions": len(sessions), "probes": len(probes),
                  "wall_seconds": seconds, "total_images_per_minute": (len(frames) + len(probes)) * 60 / seconds,
                  "contact_fraction": len(contacted) / len(frames),
                  "mean_visible_categories": float(np.mean([len(json.loads(row["objects"])) for row in frames])),
                  "held_visible_fraction": sum(any(obj["category"] == row["held_category"]
                      for obj in json.loads(row["objects"])) for row in contacted) / len(contacted) if contacted else None,
                  "labelled_contact_median_m": float(np.median(labelled)),
                  "whole_contacting_hand_median_m": float(np.median(whole))}
    manifest = {"schema_version": 1, "config": config, "seed": seed,
                "sessions_per_subject": sessions_per_subject, "session_index": session_index,
                "renderer": {"engine": "CYCLES", "device": config["device"],
                    "passes": "RGB plus raw emission instance-ID",
                    "blender_version": json.loads((output / "render_stats.json").read_text())["blender_version"]},
                "sources": sources,
                "rooms": {str(room): _sha256(room) for room in rooms},
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
    print(json.dumps(statistics, indent=2))
    return manifest


FIELDS_NAMES = ("frames", "sessions", "probes")
