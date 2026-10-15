"""Fit word instances onto GRAB grasp objects and measure how well the recorded hands fit them."""

import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import time

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


def _rotations():
    """The 24 proper signed axis permutations."""
    import itertools

    out = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            if np.linalg.det(matrix) > 0:
                out.append(matrix)
    return out


def _umeyama(points, targets):
    """Similarity (scale, rotation, translation) with targets ~ scale * R @ points + t."""
    mu_p, mu_q = points.mean(0), targets.mean(0)
    u, d, vt = np.linalg.svd((targets - mu_q).T @ (points - mu_p) / len(points))
    flip = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(d) @ flip) / (points - mu_p).var(0).sum())
    return scale, rotation, mu_q - scale * rotation @ mu_p


def _icp(source_tree, source_points, points, scale, rotation, translation, anisotropic, iterations=20, trim=0.9):
    """Refine x -> rotation @ diag(axis) @ x + translation; axis stays within +-20% of the uniform scale."""
    from scipy.spatial import cKDTree
    from scipy.optimize import lsq_linear

    axis = np.full(3, scale)
    base = scale
    for _ in range(iterations):
        moved = points @ (rotation * axis).T + translation
        forward, index = source_tree.query(moved)
        keep = forward <= np.quantile(forward, trim)
        pairs_p, pairs_q = points[keep], source_points[index[keep]]
        back, back_index = cKDTree(moved).query(source_points)
        keep_back = back <= np.quantile(back, trim)
        pairs_p = np.concatenate([pairs_p, points[back_index[keep_back]]])
        pairs_q = np.concatenate([pairs_q, source_points[keep_back]])
        if anisotropic:
            _, rotation, _ = _umeyama(pairs_p * axis, pairs_q)
            design = np.concatenate([rotation[None] * pairs_p[:, None, :],
                                     np.broadcast_to(np.eye(3), (len(pairs_p), 3, 3))], axis=2).reshape(-1, 6)
            result = lsq_linear(design, pairs_q.reshape(-1),
                                bounds=(np.r_[np.full(3, 0.8 * base), np.full(3, -np.inf)],
                                        np.r_[np.full(3, 1.2 * base), np.full(3, np.inf)]))
            axis, translation = result.x[:3], result.x[3:]
        else:
            axis_scale, rotation, translation = _umeyama(pairs_p, pairs_q)
            axis = np.full(3, axis_scale)
            base = axis_scale
    return rotation, axis, translation


def _chamfer(tree, source_points, moved):
    from scipy.spatial import cKDTree

    return float((tree.query(moved)[0].mean() + cKDTree(moved).query(source_points)[0].mean()) / 2)


def candidates_v2(source, target, keep=6):
    """Chamfer-ranked ICP candidates (rigid+uniform and bounded per-axis scale) from all 24 axis permutations.

    Each transform is x -> R @ diag(axis) @ x + t in instance coordinates, so Blender can decompose it exactly.
    """
    from scipy.spatial import cKDTree

    source_points, source_center, source_axes = _frame(source)
    target_points, target_center, target_axes = _frame(target)
    centered = target_points - target_center
    source_extent = np.ptp((source_points - source_center) @ source_axes, axis=0)
    target_extent = np.ptp(centered @ target_axes, axis=0)
    tree = cKDTree(source_points)
    starts = []
    for permutation in _rotations():
        rotation = source_axes @ permutation @ target_axes.T
        extent = np.abs(permutation) @ target_extent
        for scale in (source_extent[0] / target_extent[0], float(np.exp(np.mean(np.log(source_extent / extent))))):
            moved = centered @ rotation.T * scale + source_center
            starts.append((_chamfer(tree, source_points, moved), rotation, scale))
    starts.sort(key=lambda item: item[0])
    out, seen = [], []
    for _, rotation, scale in starts:
        if any(np.allclose(rotation, other, atol=1e-6) and abs(scale - s) / s < 0.02 for other, s in seen):
            continue
        seen.append((rotation, scale))
        if len(seen) > keep:
            break
        for anisotropic in (False, True):
            rot, axis, trans = _icp(tree, source_points, centered, scale, rotation, source_center, anisotropic)
            linear = rot * axis
            moved = centered @ linear.T + trans
            matrix = np.eye(4)
            matrix[:3, :3] = linear
            matrix[:3, 3] = trans - linear @ target_center
            out.append((_chamfer(tree, source_points, moved), matrix, float(axis.mean())))
    return out


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


def hand_frames(config, hand_ids, model, clips, frames_per_clip=24):
    """Recorded hand vertices in object coordinates with their contact labels, per sampled frame."""
    ids = np.unique(np.concatenate(list(hand_ids.values())))
    frames_out = []
    for path in clips:
        data = load_clip(path)
        labels = data["contact"]["body"][:, ids] != 0
        frames = np.flatnonzero(labels.any(1))
        if not len(frames):
            continue
        frames = frames[np.unique(np.linspace(0, len(frames) - 1, min(frames_per_clip, len(frames))).astype(int))]
        hands, _, _, _ = hand_geometry(data, hand_ids, frames, config, model)
        for position, frame in enumerate(frames):
            pose = rigid_matrix(data["object"]["params"], frame)
            frames_out.append(((hands[position] - pose[:3, 3]) @ pose[:3, :3], labels[frame]))
    return frames_out


def hand_fit(config, hand_ids, model, clips, mesh, frames_per_clip=24, frames=None):
    """Gap on labelled-contact vertices, penetration, and contact agreement with the GRAB labels."""
    gaps, depths, agree, labelled_total, recomputed_total, near_total = [], [], [], 0, 0, 0
    low, high = mesh.bounds[0] - 0.02, mesh.bounds[1] + 0.02
    if frames is None:
        frames = hand_frames(config, hand_ids, model, clips, frames_per_clip)
    for local, label in frames:
        near = np.all((local > low) & (local < high), axis=1)
        distance = np.full(len(local), np.inf)
        distance[near] = signed_distances(mesh, local[near])
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


def hand_score(frames, mesh):
    """Lower is better: mean labelled-contact gap (capped at 30 mm) plus the 95th-percentile penetration.

    Distances come from a dense surface sample (fast, ~2 mm accurate); reported metrics use hand_fit.
    """
    from scipy.spatial import cKDTree
    import trimesh

    points, faces = trimesh.sample.sample_surface(mesh, 30000, seed=0)
    normals = mesh.face_normals[faces]
    tree = cKDTree(points)
    gaps, depths = [], []
    low, high = mesh.bounds[0] - 0.02, mesh.bounds[1] + 0.02
    for local, label in frames:
        near = np.all((local > low) & (local < high), axis=1)
        distance = np.full(len(local), np.inf)
        if near.any():
            unsigned, index = tree.query(local[near])
            sign = np.sign(np.einsum("ij,ij->i", local[near] - points[index], normals[index]))
            distance[near] = unsigned * np.where(sign == 0, 1, sign)
        gaps.extend(np.minimum(np.abs(distance[label]), 0.03).tolist())
        depths.extend((-distance[near & (distance < 0)]).tolist())
    gap = float(np.mean(gaps)) if gaps else 0.03
    return gap + (float(np.percentile(depths, 95)) if depths else 0.0)


def _moved(vertices, faces, transform):
    return _mesh(vertices @ transform[:3, :3].T + transform[:3, 3], faces)


def _nudge(vertices, faces, transform, frames):
    """Coordinate search over a small translation in GRAB coordinates, scored on the hands."""
    best, best_score = transform, hand_score(frames, _moved(vertices, faces, transform))
    for step in (0.008, 0.004, 0.002):
        for axis in range(3):
            for sign in (1, -1):
                trial = best.copy()
                trial[axis, 3] += sign * step
                score = hand_score(frames, _moved(vertices, faces, trial))
                if score < best_score - 1e-6:
                    best, best_score = trial, score
    return best, best_score


def align_v2(source, vertices, faces, frames):
    """Pick among 24-permutation ICP candidates (and the legacy fit) by the hand score, then nudge the translation."""
    target = _mesh(vertices, faces)
    legacy, legacy_chamfer, _ = align(source, target)
    options = [(legacy_chamfer, legacy, float(np.cbrt(abs(np.linalg.det(legacy[:3, :3])))))]
    options.extend(candidates_v2(source, target))
    light = frames[::6]
    scored = [(hand_score(light, _moved(vertices, faces, matrix)), chamfer, matrix, scale)
              for chamfer, matrix, scale in options]
    scored.sort(key=lambda item: item[0])
    return scored, light


def _fit_word(args):
    import trimesh

    config, hand_ids, library, source, word, instances, method = args
    from .grab import _body_model

    model = _body_model(config)
    root = Path(config["grab_root"])
    grab = trimesh.load(root / "contact_meshes" / f"{source}.ply", process=False)
    clips = _clips(config, source)
    # Fit on a fixed subset of clips; the per-instance ranking is what matters.
    sample = clips[:: max(1, len(clips) // 6)][:6]
    frames = hand_frames(config, hand_ids, model, sample)
    baseline = hand_fit(config, hand_ids, model, sample, grab, frames=frames)
    fitted = []
    for item in instances:
        vertices, faces = geometry(library, item["id"])
        if method == "v2":
            scored, light = align_v2(grab, vertices, faces, frames)
            score, chamfer, transform, scale = scored[0]
        else:
            transform, chamfer, scale = align(grab, _mesh(vertices, faces))
            score = 0.0
        fitted.append([score, item, vertices, faces, transform, chamfer, scale])
    if method == "v2":
        fitted.sort(key=lambda f: f[0])
        for entry in fitted[:POOL + 2]:
            entry[4], entry[0] = _nudge(entry[2], entry[3], entry[4], frames[::6])
    fits = []
    for score, item, vertices, faces, transform, chamfer, scale in fitted:
        fit = hand_fit(config, hand_ids, model, sample, _moved(vertices, faces, transform), frames=frames)
        axis = np.linalg.norm(transform[:3, :3], axis=0)
        fits.append({"id": item["id"], "transform": np.round(transform, 6).tolist(), "scale": round(scale, 5),
                     "axis_scale": np.round(axis, 5).tolist(), "chamfer_mm": round(1000 * chamfer, 2), **fit})
    fits.sort(key=lambda f: (f["gap_median_mm"] if f["gap_median_mm"] is not None else 1e9) + f["penetration_p95_mm"])
    return source, word, {"baseline_grab": baseline, "clips_sampled": [p.name for p in sample],
                          "candidates": fits, "pool": [f["id"] for f in fits[:POOL]]}


def build(library, workers=10, config_path=None, output=None, method="legacy", words=None):
    from .build import DEFAULT_CONFIG, load_config

    vocabulary = load_vocabulary()
    manifest = json.loads(MANIFEST.read_text())["words"]
    config = load_config(config_path or DEFAULT_CONFIG)
    with Path(config["hand_vertex_ids"]).open("rb") as stream:
        hand_ids = pickle.load(stream)
    jobs = []
    for kind, sources in vocabulary["held"].items():
        for source, source_words in sources.items():
            for word in source_words:
                if words and word not in words:
                    continue
                train = [item for item in manifest[word] if item["split"] == "train"]
                jobs.append((config, hand_ids, library, source, word, train, method))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = []
        started = time.monotonic()
        for result in pool.imap_unordered(_fit_word, jobs):
            results.append(result)
            elapsed = time.monotonic() - started
            print(f"{len(results)}/{len(jobs)} {result[1]} elapsed {elapsed / 60:.1f} min, "
                  f"eta {elapsed / len(results) * (len(jobs) - len(results)) / 60:.1f} min", flush=True)
    kinds = {source: kind for kind, sources in vocabulary["held"].items() for source in sources}
    swaps = {"schema_version": 1, "contact_threshold_m": CONTACT_M, "pool_size": POOL, "align": method, "sources": {}}
    for source, word, result in results:
        entry = swaps["sources"].setdefault(source, {"kind": kinds[source], "words": {}})
        entry["words"][word] = result
    Path(output or SWAPS).write_text(json.dumps(swaps, indent=1) + "\n")
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
    parser.add_argument("--output", help="write here instead of swaps.json")
    parser.add_argument("--align", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--words", nargs="*", help="fit only these words")
    args = parser.parse_args()
    for row in summary(build(args.library, args.workers, output=args.output, method=args.align, words=args.words)):
        print(json.dumps(row))
