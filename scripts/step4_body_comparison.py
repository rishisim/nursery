"""Freeze Step 4 or run its small, real-input end-to-end smoke. Never rent hosts."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nursery.learner import acquisition as run

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = REPO / 'outputs/runs/child_gaze/steering/frozen'
DEFAULT_ROOT = REPO / 'outputs/runs/step4_body_comparison'
ARMS = {'video': {'body_representation': None},
        'video_gaze': {'body_representation': None},
        'video_gaze_body_current': {'body_representation': 'current'},
        'video_gaze_body_rich': {'body_representation': 'rich'}}


def comparison_protocol(parent):
    child = copy.deepcopy(parent)
    if parent['subsets']['1.0']['selected_clip_count'] != 630:
        raise ValueError('Step 4 requires the matched 630-clip parent')
    child.update(conditions=list(ARMS), arms=copy.deepcopy(ARMS), seeds=[0, 1, 2],
                 fractions=[1.0], epochs=60, batch=256, optimizer='AdamW', lr=3e-4,
                 weight_decay=.1, body_dropout=.5, auxiliary_losses=0,
                 body_window=5, window=2, visual_grid=16, device='cuda', runtime_cap_hours=4,
                 streams=['contact', 'fingers', 'arm', 'load', 'skin'])
    child['gaze'].update(option='steering', dropout=.5, steering_sigma=.15)
    child['selection'] = {'panel': 'clips_dev', 'body_weight_grid': list(run.BODY_WEIGHT_GRID),
                          'body_rule': 'highest accuracy, then smaller weight, then earlier epoch',
                          'non_body_rule': 'earliest epoch at maximum clips_dev accuracy',
                          'picture_rule': 'separate earliest epoch at maximum picture_dev accuracy',
                          'test_panels_select': False}
    child['rng'] = {'example_order': 'seed', 'visual_init': 'seed', 'gaze_dropout': 'seed+1000',
                    'body_init': 'forked CPU seed+3000', 'body_dropout': 'seed+2000'}
    child['body_input_dims'] = {'current': 153, 'rich': 193}
    return child


def freeze(parent=DEFAULT_PARENT, destination=DEFAULT_ROOT / 'frozen', *, overwrite=False):
    parent, destination = Path(parent).resolve(), Path(destination).resolve()
    subprocess.run(['git', '-C', str(REPO), 'check-ignore', str(destination)], check=True,
                   stdout=subprocess.DEVNULL)
    if destination.exists() and not overwrite:
        raise FileExistsError(f'Frozen destination already exists: {destination}; use --overwrite explicitly')
    protocol = comparison_protocol(run._read(parent / 'protocol.json'))
    rich = Path(protocol['episodes']) / 'body_rich'
    if run._read(rich / 'selection.json')['granularity'] != 22:
        raise ValueError('Expected frozen Step 3 G=22 (193 dimensions)')
    protocol['rich_body_manifest'] = {str(p): run._sha(p) for p in sorted(rich.glob('*')) if p.is_file()}
    protocol['hashes'].update(protocol['rich_body_manifest'])
    protocol.update(run._source_snapshot())
    protocol['code_hashes'][str(Path(__file__).resolve())] = run._sha(__file__)
    destination.mkdir(parents=True, exist_ok=overwrite)
    for name in ('panels', 'subsets'):
        shutil.copytree(parent / name, destination / name, dirs_exist_ok=overwrite)
    for name in ('splits.json', 'signal_hashes.json'):
        shutil.copyfile(parent / name, destination / name)
    run._write(destination / 'protocol.json', protocol)
    return run._freeze_inputs(destination)


def smoke(frozen=DEFAULT_ROOT / 'frozen', destination=DEFAULT_ROOT / 'smoke', *, device='cpu', clips=8, epochs=1):
    """Use selected real clips; retain the full real dev/test and picture panels."""
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    frozen, destination = Path(frozen).resolve(), Path(destination).resolve()
    subprocess.run(['git', '-C', str(REPO), 'check-ignore', str(destination)], check=True,
                   stdout=subprocess.DEVNULL)
    if clips < 1 or epochs not in (1, 2):
        raise ValueError('Smoke requires positive clips and 1-2 epochs')
    protocol = run._read(frozen / 'protocol.json')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('panels', 'subsets'):
        shutil.copytree(frozen / name, destination / name)
    subset = destination / 'subsets/1.0'
    stats = run._read(subset / 'stats.json')
    selected = set(stats['selected_clips'][:clips])
    frames = pd.read_parquet(Path(protocol['episodes']) / 'frames.parquet')
    with np.load(subset / 'fit_mask.npz') as fit:
        ids, include = fit['video_ids'].copy(), fit['include'].copy()
    lookup = frames.set_index(['video_id', 'frame_idx'])['clip'].to_dict()
    for s, video in enumerate(ids):
        for t in range(include.shape[1]):
            include[s, t] &= lookup.get((video, t)) in selected
    np.savez(subset / 'fit_mask.npz', video_ids=ids, include=include)
    manifest = pd.read_parquet(subset / 'manifest.parquet')
    manifest = manifest[[all(lookup.get((r.video_id, t)) in selected for t in range(max(0, r.frame_idx-2), min(include.shape[1], r.frame_idx+3)))
                         for r in manifest.itertuples()]].reset_index(drop=True)
    if len(manifest) < 2:
        raise ValueError('Too few smoke examples')
    manifest.to_parquet(subset / 'manifest.parquet')
    stats.update(selected_clips=sorted(selected), smoke_utterances=len(manifest),
                 normalization_note='Smoke body normalization refit only on selected smoke clips')
    run._write(subset / 'stats.json', stats)
    protocol.update(epochs=epochs, seeds=[0], smoke_evaluate_all_panels=True,
                    smoke_selected_clips=sorted(selected), smoke_parent=str(frozen))
    run._write(destination / 'protocol.json', protocol)
    # Bind smoke inputs/source too, allowing the same runner on CPU or MPS.
    prepared = {str(p): run._sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    run._write(destination / 'frozen_inputs.json', dict(run._source_snapshot(), prepared_hashes=prepared,
                external_hashes=protocol['hashes']))
    return {arm: run.train_one(destination, 1., 0, arm, smoke=True, epochs=epochs, device=device) for arm in ARMS}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'smoke'])
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing existing frozen preparation')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--clips', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.parent, args.root / 'frozen', overwrite=args.overwrite)
    else:
        smoke(args.root / 'frozen', args.root / 'smoke', device=args.device, clips=args.clips, epochs=args.epochs)
