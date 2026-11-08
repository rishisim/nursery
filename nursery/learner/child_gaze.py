"""Step 2 preparation, development-only readout selection and timing gate.

The study directory is the canonical prepared experience. Pilot directories
retain two explicitly requested alternatives, sharing its immutable panels and
subsets. This entry point never launches rentals or the Phase C matrix.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from nursery.common import REPO, _output_directory
from . import acquisition as runner
from . import acquisition_data as data


CONDITIONS = ('video_only',) + runner.GAZE_CONDITIONS
OPTIONS = ('additive', 'steering')
SELECTION_RULE = {
    'options': list(OPTIONS), 'seed': 0, 'clips': 630, 'epochs': 60,
    'score': 'gaze-removed clips_dev accuracy at earliest maximum with-gaze clips_dev checkpoint',
    'tie_break': 'additive', 'test_panels_permitted': False,
    'budget_seconds': 4 * 3600, 'timing_headroom': 1.25,
    'settings': 'Step 1 batch/optimizer/epochs/window; gaze dropout 0.5; no parameter sweep',
}


def _write(path, value):
    runner._write(Path(path), value)


def _compact(source, destination):
    """Save exactly the float32 patch mean used by GazeData, excluding CLS."""
    source, destination = Path(source), _output_directory(destination, exist_ok=False)
    index = pd.read_parquet(source / 'index.parquet')
    values = np.load(source / 'emb.f16.npy', mmap_mode='r', allow_pickle=False)
    if values.shape != (len(index), 257, 768):
        raise ValueError('Expected complete full-token close-up cache')
    result = np.lib.format.open_memmap(destination / 'emb.f16.npy', mode='w+',
                                     dtype=np.float32, shape=(len(index), 768))
    for start in range(0, len(index), 128):
        block = np.asarray(values[start:start+128, 1:], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite close-up cache')
        result[start:start+len(block)] = block.mean(axis=1)
    result.flush()
    del result
    index.to_parquet(destination / 'index.parquet', index=False)
    _write(destination / 'compact.json', {
        'operation': 'float32 mean of 256 frozen DINO patches, excluding CLS',
        'source': str(source), 'source_sha256': {
            name: runner._sha(source / name) for name in ('index.parquet', 'emb.f16.npy')},
        'rows': len(index), 'dtype': 'float32',
        'files_sha256': {name: runner._sha(destination / name)
                         for name in ('index.parquet', 'emb.f16.npy')},
    })


def _input_view(destination, visual, speech):
    destination = _output_directory(destination, exist_ok=False)
    visual, speech = Path(visual).resolve(), Path(speech).resolve()
    names = ('train_cache', 'probes_cache', 'konkle_cache', 'probes_frames.parquet',
             'konkle_frames.parquet', 'embedding.json')
    for name in names:
        (destination / name).symlink_to(visual / name)
    for name in ('manifest.parquet', 'generator.json'):
        (destination / name).symlink_to(speech / name)
    # Preserve the original feature provenance separately. Its run_input_sha256
    # describes old speech and must not be applied to the intentional new world.
    _write(destination / 'preserved_inputs.json', {
        'source': str(visual), 'source_record_sha256': runner._sha(visual / 'preserved_inputs.json'),
        'sha256': {str(visual / name): runner._sha(visual / name)
                   for name in ('preserved_inputs.json', 'embedding.json')},
        'new_speech_sha256': {str(speech / name): runner._sha(speech / name)
                              for name in ('manifest.parquet', 'generator.json')},
    })
    return destination


def _coverage(inputs, gaze, closeups, scene_frames):
    manifest = pd.read_parquet(inputs / 'manifest.parquet')
    index = pd.read_parquet(inputs / 'train_cache/index.parquet')
    points = pd.read_parquet(gaze)
    sharp = pd.read_parquet(closeups / 'index.parquet')
    scene = pd.read_parquet(scene_frames)
    keys = lambda table: set(zip(table.video_id, table.frame_idx.astype(int)))
    available, point_keys, sharp_keys = keys(index), keys(points), keys(sharp)
    scene_keys = keys(scene)
    needed = {(r.video_id, frame) for r in manifest.itertuples()
              for frame in range(max(0, int(r.frame_idx)-2), int(r.frame_idx)+3)
              if (r.video_id, frame) in scene_keys}
    missing_centers = keys(manifest) - available
    missing_video = needed - available
    valid = keys(points[points.gaze_x.notna()])
    missing_points, missing_closeups = needed - point_keys, (needed & valid) - sharp_keys
    if missing_centers or missing_video or missing_points or missing_closeups:
        raise ValueError(f'Cache coverage gap: centers={len(missing_centers)}, '
                         f'video={len(missing_video)}, points={len(missing_points)}, closeups={len(missing_closeups)}')
    return {'utterances': len(manifest), 'accessible_window_frames': len(needed),
            'valid_gaze_window_frames': len(needed & valid),
            'masked_null_gaze_frames': len(needed - valid),
            'missing_centers': 0, 'missing_video': 0, 'missing_points': 0, 'missing_closeups': 0}


def prepare(root, *, visual, speech, episodes, points, closeups, features):
    root = _output_directory(root)
    if (root / 'study').exists():
        raise FileExistsError('Study preparation already exists; inspect it before resuming')
    inputs = _input_view(Path(features) / 'world16', visual, speech)
    compact = Path(features).resolve() / 'compact'
    _compact(closeups, compact)
    coverage = _coverage(inputs, points, compact, Path(episodes) / 'frames.parquet')
    study = root / 'study'
    protocol = data.prepare(study, inputs, episodes, fractions=[1.0], conditions=['video_only'])
    data.audit(study)
    if protocol['subsets']['1.0']['selected_clip_count'] != 630:
        raise ValueError('The matched study must use exactly 630 independent training clips')
    from .train import clip_partition
    frames = pd.read_parquet(Path(episodes) / 'frames.parquet')
    part, _ = clip_partition(frames, 's8', {'seed': 0, 'dev': .15, 'test': .15})
    partitions = {f'{r.video_id}|{int(r.frame_idx)}':
                  's8' if r.subject == 's8' else ('train', 'dev', 'test')[part[r.video_id][int(r.frame_idx)]]
                  for r in frames.itertuples()}
    partition_path = study / 'frame_partitions.json'
    _write(partition_path, partitions)
    protocol.update(conditions=list(CONDITIONS), streams=[], body_dropout=0.0,
                    video_batch_loading='vectorized cached-dtype gather; transfer then float32 cast',
                    selection_status='pending development-only pilot',
                    gaze={'option': 'additive', 'points': str(Path(points).resolve()),
                          'partitions': str(partition_path), 'closeup_cache': str(compact),
                          'dropout': .5, 'shuffle_seed': 0, 'weight': 1.0, 'steering_sigma': .15})
    for path in (Path(points).resolve(), partition_path, compact / 'index.parquet', compact / 'emb.f16.npy'):
        protocol['hashes'][str(path)] = runner._sha(path)
    _write(study / 'protocol.json', protocol)
    _write(root / 'cache_coverage.json', coverage)
    _write(root / 'selection_rule.json', SELECTION_RULE)
    data.audit(study)
    return {'study': str(study), 'coverage': coverage,
            'training': protocol['subsets']['1.0'], 'selection_rule': SELECTION_RULE}


def _pilot_root(root, option):
    study = root / 'study'
    destination = root / 'pilot' / option
    if destination.exists():
        return destination
    destination = _output_directory(destination, exist_ok=False)
    for name in ('panels', 'subsets'):
        shutil.copytree(study / name, destination / name, copy_function=os.link)
    for name in ('splits.json', 'signal_hashes.json'):
        os.link(study / name, destination / name)
    protocol = copy.deepcopy(runner._read(study / 'protocol.json'))
    protocol['gaze']['option'] = option
    protocol.update(seeds=[0], conditions=['video_gaze'], development_only=True)
    _write(destination / 'protocol.json', protocol)
    runner._freeze_inputs(destination)
    return destination


def choose(scores):
    """The frozen selection uses only one held-out development panel."""
    if set(scores) != set(OPTIONS) or any(not np.isfinite(v) for v in scores.values()):
        raise ValueError('Selection requires finite development scores for both options')
    return 'steering' if scores['steering'] > scores['additive'] else 'additive'


def pilot(root, *, device='mps'):
    root = Path(root).resolve()
    if runner._read(root / 'selection_rule.json') != SELECTION_RULE:
        raise ValueError('Selection rule changed before pilot')
    state_path = root / 'pilot.json'
    if state_path.exists():
        raise FileExistsError('Pilot has started; preserve and inspect partial results')
    start = time.time()
    state = {'status': 'timing', 'started': start, 'deadline': start + SELECTION_RULE['budget_seconds'],
             'rule_sha256': runner._sha(root / 'selection_rule.json'), 'device': device}
    _write(state_path, state)
    try:
        candidates = {option: _pilot_root(root, option) for option in OPTIONS}
        timing = {}
        for option, candidate in candidates.items():
            metric = runner.train_one(candidate, 1.0, 0, 'video_gaze', smoke=True,
                                      deadline=state['deadline'], device=device)
            timing[option] = {k: metric[k] for k in ('setup_seconds', 'epoch_seconds', 'final_evaluation_seconds')}
            for name in ('model.pt', 'model_bodypresent.pt'):
                (candidate / 'timing/video_gaze' / name).unlink(missing_ok=True)
        projected = SELECTION_RULE['timing_headroom'] * sum(
            m['setup_seconds'] + SELECTION_RULE['epochs'] * m['epoch_seconds'] + m['final_evaluation_seconds']
            for m in timing.values())
        state.update(timing=timing, projected_remaining_seconds=projected)
        if time.time() + projected > state['deadline']:
            raise RuntimeError('Both matched 60-epoch development pilots project beyond the four-hour cap')
        state['status'] = 'running'
        _write(state_path, state)
        scores, selected_epochs = {}, {}
        for option, candidate in candidates.items():
            metric = runner.train_one(candidate, 1.0, 0, 'video_gaze', dev_only=True,
                                      deadline=state['deadline'], device=device)
            scores[option] = metric['timing_dev']['gaze_removed']['accuracy']
            selected_epochs[option] = metric['selection']['body_present']['epoch']
            state.update(scores=scores, selected_epochs=selected_epochs)
            _write(state_path, state)
        state.update(status='complete', selected=choose(scores), elapsed_seconds=time.time()-start)
    except Exception as exc:
        state.update(status='blocked', error=str(exc), elapsed_seconds=time.time()-start)
        _write(state_path, state)
        raise
    _write(state_path, state)
    return state


def freeze(root):
    root = Path(root).resolve()
    state = runner._read(root / 'pilot.json')
    if state['status'] != 'complete' or state['selected'] != choose(state['scores']):
        raise ValueError('Complete development-only pilot required')
    if state['rule_sha256'] != runner._sha(root / 'selection_rule.json'):
        raise ValueError('Selection rule changed after pilot')
    study = root / 'study'
    protocol = runner._read(study / 'protocol.json')
    if protocol.get('selection_status') == 'frozen from development-only pilot' or (study / 'frozen_inputs.json').exists():
        raise FileExistsError('Study protocol already frozen; preserve its execution identity')
    protocol['gaze']['option'] = state['selected']
    protocol['selection_status'] = 'frozen from development-only pilot'
    protocol['selection'] = {'rule_sha256': state['rule_sha256'], 'pilot_sha256': runner._sha(root / 'pilot.json')}
    _write(study / 'protocol.json', protocol)
    runner._freeze_inputs(study)
    return protocol


def timing(root, *, device='mps'):
    """Measure development-only epochs and stop; never execute Phase C."""
    root = Path(root).resolve()
    study = root / 'study'
    protocol = runner._read(study / 'protocol.json')
    if protocol.get('selection_status') != 'frozen from development-only pilot':
        raise ValueError('Freeze the pilot-selected protocol before timing')
    runner._verify_frozen(study)
    result = {'status': 'timing', 'device': device, 'started': time.time(), 'conditions': {}}
    path = root / 'timing_gate.json'
    if path.exists():
        raise FileExistsError('Timing gate already started; inspect retained record')
    _write(path, result)
    for condition in CONDITIONS:
        metric = runner.train_one(study, 1.0, 0, condition, smoke=True, device=device)
        result['conditions'][condition] = {k: metric[k] for k in
            ('runtime_seconds', 'setup_seconds', 'epoch_seconds', 'final_evaluation_seconds')}
        for name in ('model.pt', 'model_bodypresent.pt'):
            (study / 'timing' / condition / name).unlink(missing_ok=True)
        _write(path, result)
    projected = 1.25 * 3 * sum(m['setup_seconds'] + 60*m['epoch_seconds'] + m['final_evaluation_seconds']
                              for m in result['conditions'].values())
    result.update(status='blocked', projected_serial_matrix_seconds=projected,
                  reason='Phase C requires an approved complete rental/transfer/timing plan within four hours',
                  elapsed_seconds=time.time()-result['started'], rentals_launched=False)
    _write(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'pilot', 'freeze', 'timing'))
    parser.add_argument('--root', default='outputs/runs/child_gaze')
    parser.add_argument('--device', choices=('mps', 'cuda'), default='mps')
    parser.add_argument('--visual', default=str(REPO / 'outputs/features/visual_grid16'))
    parser.add_argument('--speech', default=str(REPO / 'outputs/speech/child_gaze'))
    parser.add_argument('--episodes', default=str(REPO / 'outputs/renders/clean_856'))
    parser.add_argument('--points', default=str(REPO / 'outputs/renders/child_gaze_closeup/gaze.parquet'))
    parser.add_argument('--closeups', default=str(REPO / 'outputs/features/child_gaze/hybrid/full/closeup_cache'))
    parser.add_argument('--features', default=str(REPO / 'outputs/features/child_gaze'))
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.root, **{k: getattr(args, k) for k in
                         ('visual', 'speech', 'episodes', 'points', 'closeups', 'features')})
    elif args.action == 'freeze':
        result = freeze(args.root)
    else:
        result = (pilot if args.action == 'pilot' else timing)(args.root, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
