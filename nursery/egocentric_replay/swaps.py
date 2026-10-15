"""Fit word instances onto GRAB grasp objects and measure how well the recorded hands fit them."""

import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle

import numpy as np

from .assets import MANIFEST, load_vocabulary
from .grab import hand_geometry, load_clip, rigid_matrix


HERE = Path(__file__).parent
SWAPS = HERE / "swaps.json"
CONTACT_M = 0.005
POOL = 6


def geometry(library, identifier):
    data = json.loads((Path(library) / "geometry" / f"{identifier}.json").read_text())
    return np.asarray(data["vertices"], dtype=np.float64), np.asarray(data["faces"], dtype=np.int64)


def _mesh(vertices, faces):
    import trimesh

    return trimesh.Trimesh(vertices, faces, process=False)


def _frame(mesh, count=4000, seed=0):
    import trimesh

    points = trimesh.sample.sample_surface(mesh, count, seed=seed)[0]
    center = points.mean(0)
    values, vectors = np.linalg.eigh(np.cov((points - center).T))
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] *= -1
    return points, center, vectors


def align(source, target):
    """Rigid+uniform-scale map from target (instance) coordinates onto the source GRAB mesh."""
    from scipy.spatial import cKDTree

    source_points, source_center, source_axes = _frame(source)
    target_points, target_center, target_axes = _frame(target)
    source_extent = np.ptp((source_points - source_center) @ source_axes, axis=0)
    target_extent = np.ptp((target_points - target_center) @ target_axes, axis=0)
    scale = source_extent[0] / target_extent[0]
    tree = cKDTree(source_points)
    best = None
    for signs in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
        rotation = source_axes @ np.diag(signs) @ target_axes.T
        moved = (target_points - target_center) @ rotation.T * scale + source_center
        forward = tree.query(moved)[0]
        backward = cKDTree(moved).query(source_points)[0]
        chamfer = (forward.mean() + backward.mean()) / 2
        if best is None or chamfer < best[0]:
            best = (chamfer, rotation)
    chamfer, rotation = best
    transform = np.eye(4)
    transform[:3, :3] = rotation * scale
    transform[:3, 3] = source_center - rotation @ target_center * scale
    return transform, float(chamfer), float(scale)


def signed_distances(mesh, points):
    """Unsigned distance with an outward-normal sign; works on open meshes with consistent winding."""
    import trimesh

    if not len(points):
        return np.zeros(0)
    closest, distance, triangle = trimesh.proximity.closest_point(mesh, points)
    sign = np.sign(np.einsum("ij,ij->i", points - closest, mesh.face_normals[triangle]))
    sign[sign == 0] = 1
    return distance * sign


def _clips(config, source):
    root = Path(config["grab_root"])
    clips = []
    for subject in config["subjects"][:8]:
        for path in sorted((root / subject).glob(f"{source}_*.npz")):
            clips.append(path)
    return clips


def hand_fit(config, hand_ids, model, clips, mesh, frames_per_clip=24):
    """Gap on labelled-contact vertices, penetration, and contact agreement with the GRAB labels."""
    ids = np.unique(np.concatenate(list(hand_ids.values())))
    gaps, depths, agree, labelled_total, recomputed_total, near_total = [], [], [], 0, 0, 0
    low, high = mesh.bounds[0] - 0.02, mesh.bounds[1] + 0.02
    for path in clips:
        data = load_clip(path)
        labels = data["contact"]["body"][:, ids] != 0
        frames = np.flatnonzero(labels.any(1))
        if not len(frames):
            continue
        frames = frames[np.unique(np.linspace(0, len(frames) - 1, min(frames_per_clip, len(frames))).astype(int))]
        hands, vertex_ids, _, _ = hand_geometry(data, hand_ids, frames, config, model)
        for position, frame in enumerate(frames):
            pose = rigid_matrix(data["object"]["params"], frame)
            local = (hands[position] - pose[:3, 3]) @ pose[:3, :3]
            near = np.all((local > low) & (local < high), axis=1)
            distance = np.full(len(local), np.inf)
            distance[near] = signed_distances(mesh, local[near])
            label = labels[frame]
            gaps.extend(np.abs(distance[label & near]).tolist())
            gaps.extend([np.inf] * int((label & ~near).sum()))
            depths.extend((-distance[near & (distance < 0)]).tolist())
            near_total += int(near.sum())
            recomputed = distance < CONTACT_M
            agree.append((recomputed & label).sum() / max(1, (recomputed | label).sum()))
            labelled_total += int(label.sum())
            recomputed_total += int(recomputed.sum())
    gaps = np.asarray(gaps)
    finite = gaps[np.isfinite(gaps)]
    return {
        "frames": len(agree),
        "gap_median_mm": round(1000 * float(np.median(gaps)), 2) if len(gaps) else None,
        "gap_p90_mm": round(1000 * float(np.percentile(gaps, 90)), 2) if len(gaps) and np.isfinite(np.percentile(gaps, 90)) else None,
        "gap_within_5mm": round(float((gaps < CONTACT_M).mean()), 3) if len(gaps) else None,
        "penetrating_fraction": round(len(depths) / max(1, near_total), 4),
        "penetration_p95_mm": round(1000 * float(np.percentile(depths, 95)), 2) if depths else 0.0,
        "contact_iou": round(float(np.mean(agree)), 3) if agree else None,
        "contact_vertices_ratio": round(recomputed_total / max(1, labelled_total), 3),
        "unbounded_gap_fraction": round(float(1 - len(finite) / len(gaps)), 3) if len(gaps) else None,
    }


def _fit_word(args):
    import trimesh

    config, hand_ids, library, source, word, instances = args
    from .grab import _body_model

    model = _body_model(config)
    root = Path(config["grab_root"])
    grab = trimesh.load(root / "contact_meshes" / f"{source}.ply", process=False)
    clips = _clips(config, source)
    # Fit on a fixed subset of clips; the per-instance ranking is what matters.
    sample = clips[:: max(1, len(clips) // 6)][:6]
    baseline = hand_fit(config, hand_ids, model, sample, grab)
    fits = []
    for item in instances:
        vertices, faces = geometry(library, item["id"])
        mesh = _mesh(vertices, faces)
        transform, chamfer, scale = align(grab, mesh)
        moved = _mesh(vertices @ transform[:3, :3].T + transform[:3, 3], faces)
        fit = hand_fit(config, hand_ids, model, sample, moved)
        fits.append({"id": item["id"], "transform": np.round(transform, 6).tolist(), "scale": round(scale, 5),
                     "chamfer_mm": round(1000 * chamfer, 2), **fit})
    fits.sort(key=lambda f: (f["gap_median_mm"] if f["gap_median_mm"] is not None else 1e9) + f["penetration_p95_mm"])
    return source, word, {"baseline_grab": baseline, "clips_sampled": [p.name for p in sample],
                          "candidates": fits, "pool": [f["id"] for f in fits[:POOL]]}


def build(library, workers=10, config_path=None):
    from .build import DEFAULT_CONFIG, load_config

    vocabulary = load_vocabulary()
    manifest = json.loads(MANIFEST.read_text())["words"]
    config = load_config(config_path or DEFAULT_CONFIG)
    with Path(config["hand_vertex_ids"]).open("rb") as stream:
        hand_ids = pickle.load(stream)
    jobs = []
    for kind, sources in vocabulary["held"].items():
        for source, words in sources.items():
            for word in words:
                train = [item for item in manifest[word] if item["split"] == "train"]
                jobs.append((config, hand_ids, library, source, word, train))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_fit_word, jobs)
    kinds = {source: kind for kind, sources in vocabulary["held"].items() for source in sources}
    swaps = {"schema_version": 1, "contact_threshold_m": CONTACT_M, "pool_size": POOL, "sources": {}}
    for source, word, result in results:
        entry = swaps["sources"].setdefault(source, {"kind": kinds[source], "words": {}})
        entry["words"][word] = result
    SWAPS.write_text(json.dumps(swaps, indent=1) + "\n")
    return swaps


def assignment(swaps, clip):
    """Deterministic word and instance for one GRAB clip of a held source object."""
    source = Path(clip).stem.split("_")[0]
    entry = swaps["sources"].get(source)
    if entry is None:
        return None
    digest = int(hashlib.sha256(Path(clip).name.encode()).hexdigest(), 16)
    words = sorted(entry["words"])
    word = words[digest % len(words)]
    pool = entry["words"][word]["pool"]
    instance = pool[(digest // len(words)) % len(pool)]
    transform = next(c["transform"] for c in entry["words"][word]["candidates"] if c["id"] == instance)
    return {"word": word, "instance": instance, "transform": transform, "kind": entry["kind"]}


def summary(swaps):
    rows = []
    for source, entry in swaps["sources"].items():
        for word, result in entry["words"].items():
            pool = [c for c in result["candidates"] if c["id"] in result["pool"]]
            rows.append({"source": source, "word": word, "kind": entry["kind"],
                         "grab_gap_median_mm": result["baseline_grab"]["gap_median_mm"],
                         "grab_penetration_p95_mm": result["baseline_grab"]["penetration_p95_mm"],
                         **{key: round(float(np.median([c[key] for c in pool if c[key] is not None])), 2)
                            for key in ("chamfer_mm", "gap_median_mm", "gap_p90_mm", "gap_within_5mm",
                                        "penetrating_fraction", "penetration_p95_mm", "contact_iou")}})
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    for row in summary(build(args.library, args.workers)):
        print(json.dumps(row))
