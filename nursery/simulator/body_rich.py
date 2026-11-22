"""CPU export of clip-isolated 30 Hz bodily summaries (Step 3).

No labels or object identifiers are features. Full/leading/trailing summaries
are stored in that order; trailing includes its centre as specified for +5s.
Run with the simulator environment, PYTHONPATH=. and bytecode disabled.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

from .grab import _body_model, load_clip, reconstruct, rigid_matrix
from .build import _swapped_contact
from .swaps import assignment, geometry

ARMS = (18, 19, 20, 21)
VARIANTS = ('full', 'leading', 'trailing')


def bin_frames(centre, length, variant=0):
    """30Hz lattice anchored at the existing 120Hz centre; never leaves source."""
    offsets = (np.arange(-60, 60, 4), np.arange(0, 60, 4), np.arange(-60, 1, 4))[variant]
    indices = centre + offsets
    return indices[(indices >= 0) & (indices < length)]


def region_maps(config, hand_ids):
    """Fixed rest geometry/skinning map, independent of every data partition."""
    from smplx.lbs import vertices2joints
    import torch

    model = _body_model(config)
    vertices = model.v_template.detach().numpy()
    weights = model.lbs_weights.detach().numpy()
    joints = vertices2joints(model.J_regressor, model.v_template[None])[0].detach().numpy()
    patches = json.loads((Path(__file__).parents[1] / 'learner/hand_patches.json').read_text())
    normals = vertex_normals(vertices, model.faces)
    maps = {22: [], 32: [], 64: []}
    palm_normals = []
    for side, wrist, first, sign in [('left', 20, 25, 1), ('right', 21, 40, -1)]:
        ids = hand_ids[f'{side}_hand']
        labels = weights[ids][:, [wrist] + list(range(first, first + 15))].argmax(1)
        # Left cross(index,pinky), right reversed: anatomical palmar direction.
        palm = sign * np.cross(joints[first] - joints[wrist], joints[first + 6] - joints[wrist])
        palm /= np.linalg.norm(palm)
        dorsal = (normals[ids] @ palm < 0).astype(int)
        offset = 0 if side == 'left' else 1
        maps[22].extend(np.array(patches[side]) + 11 * offset)
        maps[32].extend(labels + 16 * offset)
        maps[64].extend(2 * labels + dorsal + 32 * offset)
        palm_normals.append(palm.tolist())
    return {g: np.array(x, dtype=np.int16) for g, x in maps.items()}, model.parents.detach().numpy(), palm_normals


def vertex_normals(vertices, faces):
    """Area-weighted outward vertex normals from the recorded body's own surface."""
    tri = vertices[faces]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)


def contact_geometry(points, normals):
    """Centroid, sqrt covariance eigenvalues descending, mean normal, opposition, span."""
    from scipy.spatial.distance import pdist

    if not len(points):
        return np.zeros(11, dtype=np.float32)
    centre = points.mean(0)
    delta = points - centre
    extents = np.sqrt(np.maximum(np.linalg.eigvalsh(delta.T @ delta / len(points)), 0))[::-1]
    mean_normal = normals.mean(0)
    span = pdist(points).max() if len(points) > 1 else 0.
    return np.r_[centre, extents, mean_normal, np.clip(1 - np.linalg.norm(mean_normal), 0, 1), span].astype(np.float32)


def correspondences(episodes):
    """Recover the exact sample_chain rule from retained ordered segment names."""
    import pandas as pd

    episodes = Path(episodes)
    config = dict(json.loads((episodes / 'manifest.json').read_text())['config'])
    config['library'] = str(Path('outputs/assets/objaverse/library').resolve())
    config['swaps'] = str(Path('outputs/assets/objaverse/swaps_v2.json').resolve())
    frames = pd.read_parquet(episodes / 'frames.parquet').sort_values(['video_id', 'frame_idx'])
    sessions = pd.read_parquet(episodes / 'sessions.parquet')
    meta = {}
    for clip in sorted(frames['clip'].unique()):
        with np.load(Path(config['grab_root']) / clip) as data:
            rate, count = float(data['framerate']), int(data['n_frames'])
        if rate != 120:
            raise ValueError(f'{clip}: expected 120Hz, got {rate}')
        meta[clip] = {'length': count, 'centres': set(), 'rows': []}
    for row in sessions.itertuples():
        group = frames[frames.video_id == row.video_id]
        elapsed = 0.
        for clip in json.loads(row.clips):
            item = meta[clip]
            stop = min(120., elapsed + item['length'] / 120.)
            selected = group[(group.frame_idx >= elapsed) & (group.frame_idx < stop)]
            if not (selected['clip'] == clip).all():
                raise ValueError('Session clip correspondence mismatch')
            for second in selected.frame_idx:
                source = min(int((second - elapsed) * 120), item['length'] - 1)
                item['centres'].add(source)
                item['rows'].append((row.video_id, int(second), source))
            elapsed = stop
    for item in meta.values():
        item['centres'] = sorted(item['centres'])
    return config, meta


def _clip_export(args):
    """One source per process job; repeated sessions reuse source-frame calculations."""
    clip, item, config, output = args
    started = time.monotonic()
    output = Path(output)
    destination = output / 'clips' / clip
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + json.dumps(item['centres']).encode()).hexdigest()
    if destination.exists():
        with np.load(destination) as saved:
            receipt = json.loads(str(saved['receipt']))
        if receipt.get('fingerprint') == fingerprint:
            return receipt
    with open(config['hand_vertex_ids'], 'rb') as stream:
        hands = pickle.load(stream)
    maps, parents, _ = region_maps(config, hands)
    ids = np.r_[hands['left_hand'], hands['right_hand']]
    data = load_clip(Path(config['grab_root']) / clip)
    length = item['length']
    centres = np.array(item['centres'])
    required = np.unique(np.concatenate([bin_frames(f, length) for f in centres]))
    # Neighbouring same-lattice frames for central differences and onset context.
    sampled = np.unique(np.clip(np.r_[required, required - 4, required + 4], 0, length - 1))
    lookup = {int(f): i for i, f in enumerate(sampled)}
    a = assignment(json.loads(Path(config['swaps']).read_text()), clip)
    mesh = None
    if a:
        import trimesh
        v, faces = geometry(config['library'], a['instance'])
        transform = np.array(a['transform'])
        mesh = trimesh.Trimesh(v @ transform[:3, :3].T + transform[:3, 3], faces, process=False)
    contact = np.zeros((len(sampled), 1556), bool)
    geom = np.zeros((len(sampled), 22), np.float32)
    positions = np.empty((len(sampled), 12), np.float32)
    gravity = np.empty((len(sampled), 3), np.float32)
    joints_all = np.empty((len(sampled), 55, 3), np.float32)
    timings = {'reconstruction_s': 0., 'contact_s': 0., 'geometry_s': 0.}
    for start in range(0, len(sampled), 64):
        stop = min(start + 64, len(sampled))
        indices = sampled[start:stop]
        tick = time.monotonic()
        vertices, faces, camera, _, _, joints = reconstruct(data, indices, config, hands)
        timings['reconstruction_s'] += time.monotonic() - tick
        joints_all[start:stop] = joints
        # Differentiating these coordinates preserves the current head-relative convention.
        positions[start:stop] = (np.einsum('tji,tkj->tki', camera[:, :3, :3],
                                joints[:, ARMS] - camera[:, None, :3, 3])).reshape(-1, 12)
        gravity[start:stop] = np.einsum('tji,j->ti', camera[:, :3, :3], [0., 0., 1.])
        tick = time.monotonic()
        if a:
            poses = np.array([rigid_matrix(data['object']['params'], f) for f in indices])
            contact[start:stop] = _swapped_contact(vertices, poses, mesh, hands)
        else:
            contact[start:stop] = data['contact']['body'][indices][:, ids] != 0
        timings['contact_s'] += time.monotonic() - tick
        tick = time.monotonic()
        from scipy.spatial.transform import Rotation
        pose = np.c_[data['body']['params']['global_orient'][indices],
                     data['body']['params']['body_pose'][indices], np.zeros((len(indices), 9)),
                     data['lhand']['params']['fullpose'][indices], data['rhand']['params']['fullpose'][indices]]
        rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 55, 3, 3)
        for joint in range(1, 55):
            rotations[:, joint] = rotations[:, parents[joint]] @ rotations[:, joint]
        for i in range(len(indices)):
            normals = vertex_normals(vertices[i], faces)
            for h, (hand, wrist) in enumerate(zip((hands['left_hand'], hands['right_hand']), (20, 21))):
                touching = contact[start + i, h * 778:(h + 1) * 778]
                points = (vertices[i, hand[touching]] - joints[i, wrist]) @ rotations[i, wrist]
                local_normals = normals[hand[touching]] @ rotations[i, wrist]
                geom[start + i, h * 11:(h + 1) * 11] = contact_geometry(points, local_normals)
        timings['geometry_s'] += time.monotonic() - tick
    fingers = np.c_[data['lhand']['params']['fullpose'][sampled], data['rhand']['params']['fullpose'][sampled]]
    velocity = np.zeros_like(positions)
    angular = np.zeros((len(sampled), 10), np.float32)
    onsets = np.zeros((len(sampled), 2), bool)
    any_contact = contact.reshape(-1, 2, 778).any(2)
    from scipy.spatial.transform import Rotation
    for frame in required:
        i = lookup[int(frame)]
        before = int(frame) - 4 if frame >= 4 else int(frame)
        after = int(frame) + 4 if frame + 4 < length else int(frame)
        lo, hi = lookup[before], lookup[after]
        dt = (after - before) / 120.
        if dt:
            velocity[i] = (positions[hi] - positions[lo]) / dt
            lower = Rotation.from_rotvec(fingers[lo].reshape(-1, 3))
            upper = Rotation.from_rotvec(fingers[hi].reshape(-1, 3))
            angular[i] = (upper * lower.inv()).magnitude().reshape(10, 3).mean(1) / dt
        # No invented onset at source boundary. Previous same-lattice state otherwise.
        if frame >= 4:
            onsets[i] = any_contact[i] & ~any_contact[lookup[int(frame) - 4]]
    arrays = {f'regions{g}': np.zeros((len(centres), 3, g), np.float32) for g in maps}
    arrays.update(events=np.zeros((len(centres), 3, 4), np.float32),
                  geometry=np.zeros((len(centres), 3, 22), np.float32),
                  fingers=np.zeros((len(centres), 3, 100), np.float32),
                  arm=np.zeros((len(centres), 3, 31), np.float32),
                  touching=np.zeros((len(centres), 3, 2), bool),
                  contact_bits=np.zeros((len(centres), 30, 195), np.uint8),
                  source_frames=np.full((len(centres), 30), -1, np.int32),
                  centre_joints=joints_all[[lookup[int(f)] for f in centres]],
                  centre_hand_pose=fingers[[lookup[int(f)] for f in centres]].astype(np.float32),
                  centres=centres)
    for row, centre in enumerate(centres):
        frames = bin_frames(centre, length)
        slots = ((frames - centre + 60) // 4).astype(int)
        ix = np.array([lookup[int(f)] for f in frames])
        arrays['source_frames'][row, slots] = frames
        arrays['contact_bits'][row, slots] = np.packbits(contact[ix], axis=1)
        for variant in range(3):
            ix = np.array([lookup[int(f)] for f in bin_frames(centre, length, variant)])
            bits = contact[ix]
            for g, labels in maps.items():
                arrays[f'regions{g}'][row, variant] = [bits[:, labels == label].mean() if (labels == label).any() else 0. for label in range(g)]
            event_count = onsets[ix].sum(0)
            if variant == 1:  # no onset context before the outer window start
                event_count -= onsets[ix[0]]
            arrays['events'][row, variant] = np.r_[any_contact[ix].mean(0), event_count]
            for hand in range(2):
                mask = any_contact[ix, hand]
                arrays['touching'][row, variant, hand] = mask.any()
                if mask.any():
                    arrays['geometry'][row, variant, hand * 11:(hand + 1) * 11] = geom[ix[mask], hand * 11:(hand + 1) * 11].mean(0)
            bin_velocity, bin_angular = velocity[ix].copy(), angular[ix].copy()
            if variant and len(ix) > 1:
                edge, neighbour = (0, 1) if variant == 1 else (-1, -2)
                dt = (sampled[ix[neighbour]] - sampled[ix[edge]]) / 120.
                bin_velocity[edge] = (positions[ix[neighbour]] - positions[ix[edge]]) / dt
                lower = Rotation.from_rotvec(fingers[ix[edge]].reshape(-1, 3))
                upper = Rotation.from_rotvec(fingers[ix[neighbour]].reshape(-1, 3))
                bin_angular[edge] = (upper * lower.inv()).magnitude().reshape(10, 3).mean(1) / abs(dt)
            elif variant:
                bin_velocity[:] = 0
                bin_angular[:] = 0
            arrays['fingers'][row, variant] = np.r_[fingers[ix].mean(0), bin_angular.mean(0)]
            speed_std = np.linalg.norm(bin_velocity.reshape(-1, 4, 3), axis=2).std(0)
            arrays['arm'][row, variant] = np.r_[positions[ix].mean(0), bin_velocity.mean(0), speed_std, gravity[ix].mean(0)]
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f'{clip}: nonfinite {key}')
    receipt = {'clip': clip, 'swapped': bool(a), 'centres': len(centres), 'sampled_frames': len(sampled),
               'seconds': time.monotonic() - started, 'fingerprint': fingerprint, **timings}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays, receipt=json.dumps(receipt))
    return receipt


def assemble(episodes, meta, output):
    """One canonical 120-row session file; source chunks retained for diagnostics."""
    by_video = {}
    for clip, item in meta.items():
        for video, second, centre in item['rows']:
            by_video.setdefault(video, []).append((second, clip, centre))
    for video, rows in sorted(by_video.items()):
        path = Path(output) / f'{video}.npz'
        if path.exists():
            continue
        arrays = {}
        for second, clip, centre in rows:
            with np.load(Path(output) / 'clips' / clip) as chunk:
                index = int(np.searchsorted(chunk['centres'], centre))
                for key in chunk.files:
                    if key in ('centres', 'receipt'):
                        continue
                    value = chunk[key][index]
                    if key not in arrays:
                        arrays[key] = np.zeros((120,) + value.shape, value.dtype)
                    arrays[key][second] = value
        arrays['frame_idx'] = np.arange(120)
        np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('timing', 'export'))
    parser.add_argument('--episodes', default='outputs/renders/clean_856')
    parser.add_argument('--output', default='outputs/renders/clean_856/body_rich')
    parser.add_argument('--analysis', default='outputs/analysis/step3_body/build')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('CPU worker count must be 1..8')
    # This command never writes outside ignored roots.
    import subprocess
    for path in (args.output, args.analysis):
        if subprocess.run(['git', 'check-ignore', '--quiet', str(Path(path).resolve())]).returncode:
            raise ValueError(f'Output must be ignored: {path}')
        Path(path).mkdir(parents=True, exist_ok=True)
    config, meta = correspondences(args.episodes)
    with open(config['hand_vertex_ids'], 'rb') as stream:
        hands = pickle.load(stream)
    maps, _, palm = region_maps(config, hands)
    swaps = json.loads(Path(config['swaps']).read_text())
    jobs = [(clip, item, config, args.output) for clip, item in sorted(meta.items())]
    if args.action == 'timing':
        # Selection is strictly train-only, using already frozen partitions.
        partitions = json.loads(Path('outputs/runs/child_gaze/matrix/frame_partitions.json').read_text())
        chosen = []
        for swapped in (False, True):
            eligible = [job for job in jobs if bool(assignment(swaps, job[0])) == swapped and
                        all(partitions[f'{v}|{s}'] == 'train' for v, s, _ in job[1]['rows'])]
            eligible.sort(key=lambda job: len(job[1]['centres']))
            chosen.append(eligible[len(eligible) // 2])
        receipts = [_clip_export(job) for job in chosen]
        projection = 0.
        for swapped, receipt in zip((False, True), receipts):
            total_centres = sum(len(item['centres']) for clip, item in meta.items() if bool(assignment(swaps, clip)) == swapped)
            projection += total_centres / receipt['centres'] * receipt['seconds']
        projection = projection / args.workers * 1.5  # 50% process/IO/contention headroom
        result = {'receipts': receipts, 'workers': args.workers, 'projection_s': projection, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'headroom_factor': 1.5, 'passed': projection <= 4 * 3600}
        Path(args.analysis, 'timing.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    timing = json.loads(Path(args.analysis, 'timing.json').read_text())
    if timing.get('code_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Exporter changed since timing gate; rerun timing')
    if not timing['passed']:
        raise ValueError('Timing gate failed')
    started = time.monotonic()
    receipts = []
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for receipt in pool.imap_unordered(_clip_export, jobs):
            receipts.append(receipt)
            print(json.dumps({'completed': len(receipts), 'total': len(jobs), **receipt}), flush=True)
    assemble(args.episodes, meta, args.output)
    Path(args.output, 'regions.json').write_text(json.dumps({'maps': {str(g): x.tolist() for g, x in maps.items()},
              'palm_normals': palm, 'G64_order': 'joint then palmar(0)/dorsal(1)', 'variants': VARIANTS}, indent=2) + '\n')
    Path(args.analysis, 'export.json').write_text(json.dumps({'seconds': time.monotonic() - started, 'receipts': receipts,
                      'source_clips': len(meta), 'sessions': 856, 'rows': sum(len(x['rows']) for x in meta.values())}, indent=2) + '\n')


if __name__ == '__main__':
    main()
