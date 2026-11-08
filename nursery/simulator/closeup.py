"""Re-export saved scenes and render sharp gaze views without changing source video."""

import argparse
import json
import multiprocessing
from functools import lru_cache
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from . import build as replay
from .assets import MANIFEST, load_vocabulary
from .render import render_job


def validate_geometry(original, exported):
    """Retain distinct instance IDs, including same-category distractors."""
    objects = json.loads(original)
    if len(objects) != len(exported):
        raise ValueError("Replayed object count differs from saved scene")
    result = []
    for instance, (old, new) in enumerate(zip(objects, exported), 1):
        if new['id'] != instance or old['visible_px'] != new['visible_px']:
            raise ValueError("Replayed instance visibility differs from saved scene")
        point = new['image_xy']
        if bool(old['visible_px']) != (point is not None):
            raise ValueError("Missing visible point or invented point on invisible object")
        result.append({**old, **new})
    return json.dumps(result)


def _ignored_output(output):
    output = Path(output).resolve()
    if subprocess.run(['git', 'check-ignore', '--quiet', str(output)], check=False).returncode:
        raise ValueError('Close-up output must be ignored')
    return output


def _configuration(episodes):
    manifest = json.loads((episodes / 'manifest.json').read_text())
    config = dict(manifest['config'])
    # Recorded paths predate archive consolidation. These are the retained files,
    # not replacement assets; exact rendered-ID matching gates every exported row.
    for key, retained in [('library', 'outputs/assets/objaverse/library'),
                          ('swaps', 'outputs/assets/objaverse/swaps_v2.json')]:
        if not Path(config[key]).exists():
            config[key] = str(Path(retained).resolve())
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])
    return manifest, config


@lru_cache(maxsize=1)
def _replay_inputs(config_json):
    config = json.loads(config_json)
    catalog, _ = replay._catalog(config)
    pools = replay._pools(config, load_vocabulary(), json.loads(MANIFEST.read_text())['words'],
                          Path(config['grab_root']))
    with Path(config['hand_vertex_ids']).open('rb') as stream:
        hand_ids = pickle.load(stream)
    return catalog, pools, hand_ids


def _work(arguments):
    episodes, output, video, mode, gaze, selected = arguments
    import pyarrow.parquet as pq
    episodes, output = Path(episodes), Path(output)
    destination = output / 'sessions' / video
    destination.mkdir(parents=True, exist_ok=True)
    manifest, config = _configuration(episodes)
    job_path = destination / 'job.json'
    receipt = destination / f'{mode}_receipt.json'
    if receipt.exists():
        raise FileExistsError(f'Completed {mode} already exists: {receipt}')
    provenance_path = destination / 'replay_provenance.json'
    provenance = {'episodes': str(episodes), 'config': config,
                  'input_sha256': {str(path): replay._sha256(path) for path in
                      [episodes / 'manifest.json', episodes / 'frames.parquet', MANIFEST,
                       Path(config['swaps']), Path(replay.__file__),
                       Path(__file__).with_name('blender.py')]}}
    started = time.monotonic()
    prep_seconds = 0.0
    if job_path.exists():
        if not provenance_path.exists() or json.loads(provenance_path.read_text()) != provenance:
            raise ValueError('Saved replay provenance no longer matches inputs or implementation')
        job = json.loads(job_path.read_text())
    else:
        if mode == 'closeup':
            raise ValueError('Geometry must be exported before close-ups')
        subject, number = video.rsplit('_', 1)
        catalog, pools, hand_ids = _replay_inputs(json.dumps(config, sort_keys=True))
        rooms = [Path(next(p for p in manifest['rooms'] if Path(p).parents[2].name == name))
                 for name in config['session_rooms']]
        job = replay._session(config, Path(config['grab_root']), hand_ids, subject, int(number),
                              rooms, catalog[subject], pools,
                              manifest['seed'] + config['subjects'].index(subject) * manifest['sessions_per_subject'] + int(number),
                              destination, replay_only=True)
        job_path.write_text(json.dumps(job))
        provenance_path.write_text(json.dumps(provenance, indent=2))
        prep_seconds = time.monotonic() - started
    job['mode'] = mode
    if mode == 'sample':
        job['frame_images'] = [s['frames'][len(s['frames']) // 2]['image'] for s in job['segments']]
    elif mode == 'closeup':
        job['gaze'] = gaze
        job['frame_images'] = sorted(gaze if selected is None else set(gaze) & set(selected))
    # A completed export is never appended to or silently overwritten.
    before_render = time.monotonic()
    render_job(job, config, destination)
    stats = json.loads((destination / f'render_stats_{video}.json').read_text())
    report = {'video_id': video, 'mode': mode, 'preparation_seconds': prep_seconds,
              'render_wall_seconds': time.monotonic() - before_render,
              'wall_seconds': time.monotonic() - started,
              'closeup_seconds': stats['closeup_seconds']}
    rows = pq.read_table(episodes / 'frames.parquet', filters=[('video_id', '=', video)]).to_pylist()
    if mode in ('geometry', 'sample'):
        geometry = json.loads((destination / f'geometry_{video}.json').read_text())
        rows = [{**row, 'objects': validate_geometry(row['objects'], geometry[row['image']])}
                for row in rows if row['image'] in geometry]
    else:
        rows = [{ 'video_id': video, 'frame_idx': row['frame_idx'],
                  'image': str((destination / 'closeups' / row['image']).relative_to(output))}
                for row in rows if row['image'] in job['frame_images']]
    receipt.write_text(json.dumps(report, indent=2))
    return rows, report


def export(episodes, output, mode, video_ids=None, workers=1, gaze_path=None, frame_keys=None):
    """Export geometry first; a later closeup pass uses the same retained replay jobs."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    episodes, output = Path(episodes).resolve(), _ignored_output(output)
    if output == episodes or episodes in output.parents:
        raise ValueError('Never write into the retained source render root')
    if workers not in (1, 2):
        raise ValueError('Locally measured worker choices are 1 and 2')
    if mode not in ('geometry', 'sample', 'closeup'):
        raise ValueError('Unknown export mode')
    destination = output / ('frames.parquet' if mode == 'closeup' else 'geometry.parquet')
    if destination.exists():
        raise FileExistsError(destination)
    output.mkdir(parents=True, exist_ok=True)
    sessions = pq.read_table(episodes / 'sessions.parquet').to_pylist()
    available = {row['video_id'] for row in sessions}
    videos = sorted(available if video_ids is None else set(video_ids))
    if not videos or set(videos) - available:
        raise ValueError('Unknown or empty session selection')
    gazes, gaze_keys = {}, set()
    if mode == 'closeup':
        if gaze_path is None:
            raise ValueError('Close-ups require scene-only gaze')
        for row in pq.read_table(gaze_path).to_pylist():
            key = (row['video_id'], row['frame_idx'])
            if key in gaze_keys:
                raise ValueError('Duplicate gaze frame key')
            gaze_keys.add(key)
            if row.get('gaze_object_id') is not None:
                x, y = row['gaze_x'], row['gaze_y']
                if not (np.isfinite([x,y]).all() and 0 <= x < 1 and 0 <= y < 1):
                    raise ValueError('Invalid gaze coordinates')
                gazes.setdefault(row['video_id'], {})[f"frames/{row['video_id']}/{row['frame_idx']:05d}.jpg"] = [x,y]
    selected = None
    if frame_keys is not None:
        selected = {}
        for row in pq.read_table(frame_keys, columns=['video_id', 'frame_idx']).to_pylist():
            if mode == 'closeup' and (row['video_id'], row['frame_idx']) not in gaze_keys:
                raise ValueError('Requested frame key is missing from the gaze table')
            selected.setdefault(row['video_id'], []).append(f"frames/{row['video_id']}/{row['frame_idx']:05d}.jpg")
    elif mode == 'closeup':
        expected = pq.read_table(episodes / 'frames.parquet', columns=['video_id', 'frame_idx']).to_pylist()
        if any((r['video_id'], r['frame_idx']) not in gaze_keys for r in expected if r['video_id'] in videos):
            raise ValueError('Gaze table does not cover all requested source frames')
    arguments = [(str(episodes), str(output), video, mode, gazes.get(video, {}),
                  None if selected is None else selected.get(video, [])) for video in videos]
    started = time.monotonic()
    if workers == 1:
        results = [_work(args) for args in arguments]
    else:
        with multiprocessing.get_context('spawn').Pool(workers) as pool:
            results = pool.map(_work, arguments)
    rows = sorted([row for batch, _ in results for row in batch], key=lambda r: (r['video_id'],r['frame_idx']))
    pq.write_table(pa.Table.from_pylist(rows), destination)
    report = {'mode': mode, 'frames': len(rows), 'workers': workers, 'wall_seconds': time.monotonic() - started,
              'source_frames_sha256': replay._sha256(episodes / 'frames.parquet'),
              'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'source_sha256': {str(path): replay._sha256(path) for path in [Path(__file__),Path(__file__).with_name('blender.py'),Path(replay.__file__)]},
              'resolution': [224,224], 'horizontal_fov_degrees': 24.0,
              'sessions': [report for _, report in results]}
    (output / f'{mode}_manifest.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['geometry','closeup','sample'], required=True)
    parser.add_argument('--video-ids', nargs='+')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--gaze')
    parser.add_argument('--frame-keys')
    args = parser.parse_args()
    print(json.dumps(export(args.episodes,args.output,args.mode,args.video_ids,args.workers,args.gaze,args.frame_keys),indent=2))


if __name__ == '__main__':
    main()
