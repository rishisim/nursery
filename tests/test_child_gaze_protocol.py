"""Phase B gates and input preparation, without scientific training."""
import json
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.learner import acquisition as runner
from nursery.learner import child_gaze
from nursery.learner.gaze_data import collate_frame_indices


def test_selection_is_fixed_dev_only_and_ties_choose_additive():
    assert child_gaze.SELECTION_RULE['epochs'] == 60
    assert child_gaze.SELECTION_RULE['clips'] == 630
    assert not child_gaze.SELECTION_RULE['test_panels_permitted']
    assert child_gaze.choose({'additive': 50., 'steering': 50.}) == 'additive'
    assert child_gaze.choose({'additive': 49., 'steering': 50.}) == 'steering'
    with pytest.raises(ValueError):
        child_gaze.choose({'additive': 40., 'steering': float('nan')})


def test_explicit_gaze_matrix_does_not_change_historical_defaults():
    protocol = {'fractions': [1.0], 'seeds': [0, 1, 2], 'conditions': list(child_gaze.CONDITIONS),
                'gaze': {'option': 'steering', 'points': 'points', 'partitions': 'parts',
                         'dropout': .5, 'shuffle_seed': 0}}
    assert runner.protocol_matrix(protocol) == ([1.0], [0, 1, 2], list(child_gaze.CONDITIONS))
    del protocol['gaze']
    with pytest.raises(ValueError):
        runner.protocol_matrix(protocol)


def test_compact_cache_is_exact_patch_mean_and_keeps_key_mapping(tmp_path, monkeypatch):
    source, destination = tmp_path / 'full', tmp_path / 'compact'
    source.mkdir()
    monkeypatch.setattr(child_gaze, '_output_directory', lambda p, **kw: p.mkdir(**kw) or p)
    values = np.random.default_rng(4).normal(size=(2, 257, 768)).astype(np.float16)
    values[:, 0] = 100  # CLS must have no contribution.
    np.save(source / 'emb.f16.npy', values)
    index = pd.DataFrame({'video_id': ['b', 'a'], 'frame_idx': [3, 5], 'row': [0, 1]})
    index.to_parquet(source / 'index.parquet', index=False)
    child_gaze._compact(source, destination)
    assert np.array_equal(np.load(destination / 'emb.f16.npy'), values[:, 1:].astype(np.float32).mean(1))
    pd.testing.assert_frame_equal(pd.read_parquet(destination / 'index.parquet'), index)
    assert json.loads((destination / 'compact.json').read_text())['rows'] == 2


def test_coverage_rejects_missing_neighbor_and_masks_null_gaze(tmp_path):
    inputs, sharp = tmp_path / 'inputs', tmp_path / 'sharp'
    (inputs / 'train_cache').mkdir(parents=True)
    sharp.mkdir()
    pd.DataFrame({'video_id': ['v'], 'frame_idx': [1]}).to_parquet(inputs / 'manifest.parquet')
    points = pd.DataFrame({'video_id': ['v'] * 3, 'frame_idx': [0, 1, 2],
                           'gaze_x': [.2, np.nan, .4], 'gaze_y': [.2, np.nan, .4]})
    path = tmp_path / 'points.parquet'
    points.to_parquet(path)
    scene_frames = tmp_path / 'frames.parquet'
    points[['video_id', 'frame_idx']].to_parquet(scene_frames)
    points.iloc[[0, 2]][['video_id', 'frame_idx']].to_parquet(sharp / 'index.parquet')
    points.iloc[[0, 1]][['video_id', 'frame_idx']].to_parquet(inputs / 'train_cache/index.parquet')
    with pytest.raises(ValueError, match='video=1'):
        child_gaze._coverage(inputs, path, sharp, scene_frames)
    points[['video_id', 'frame_idx']].to_parquet(inputs / 'train_cache/index.parquet')
    coverage = child_gaze._coverage(inputs, path, sharp, scene_frames)
    assert coverage['valid_gaze_window_frames'] == 2
    assert coverage['masked_null_gaze_frames'] == 1
    points.iloc[[0, 1]].to_parquet(path)
    with pytest.raises(ValueError, match='points=1'):
        child_gaze._coverage(inputs, path, sharp, scene_frames)


def test_pilot_projection_stops_before_full_runs(tmp_path, monkeypatch):
    (tmp_path / 'selection_rule.json').write_text(json.dumps(child_gaze.SELECTION_RULE))
    monkeypatch.setattr(child_gaze, '_pilot_root', lambda root, option: root / option)
    calls = []
    def train(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs['smoke'] is True
        return {'setup_seconds': 10., 'epoch_seconds': 120., 'final_evaluation_seconds': 10.}
    monkeypatch.setattr(runner, 'train_one', train)
    with pytest.raises(RuntimeError, match='four-hour'):
        child_gaze.pilot(tmp_path)
    assert len(calls) == 2
    assert json.loads((tmp_path / 'pilot.json').read_text())['status'] == 'blocked'


def test_development_only_protocol_refuses_heldout_execution_before_outputs(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'epochs': 60, 'development_only': True}))
    with pytest.raises(ValueError, match='development-only'):
        runner.train_one(tmp_path, 1., 0, 'video_only', device='cpu')
    assert not (tmp_path / 'models').exists()


@pytest.mark.usefixtures('offline')
def test_vectorized_visual_batches_match_original_values_order_and_optimizer_step(upstream):
    from nursery.learner.video_only import VideoOnlyMIL
    cache = np.random.default_rng(7).normal(size=(6, 17, 8)).astype(np.float16)
    lut = {f'v|{i}': (0, i) for i in range(6)}
    manifest = pd.DataFrame({'video_id': ['v'] * 3, 'frame_idx': [0, 2, 5],
                             'text': ['ball cup', 'cup', 'ball']})
    ds = upstream.FramePairs([cache], lut, manifest, {'<pad>': 0, 'ball': 1, 'cup': 2}, 2, False)
    old_loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=upstream.collate,
                                           generator=torch.Generator().manual_seed(12))
    from functools import partial
    new_loader = torch.utils.data.DataLoader(range(len(ds)), batch_size=2, shuffle=True,
                    collate_fn=partial(collate_frame_indices, ds), generator=torch.Generator().manual_seed(12))
    for old, new in zip(old_loader, new_loader):
        assert new[1].dtype == torch.float16
        assert all(torch.equal(a, b.float() if i == 1 else b) for i, (a, b) in enumerate(zip(old, new)))
    ids = [2, 0, 1]  # Includes both boundary-padding patterns, in a non-source order.
    old = upstream.collate([ds[i] for i in ids])
    new = collate_frame_indices(ds, ids)
    model = VideoOnlyMIL(3, 8)
    alternate = copy.deepcopy(model)
    for learner, batch in ((model, old), (alternate, new)):
        torch.manual_seed(99)
        optimizer = torch.optim.AdamW(learner.parameters(), lr=.0003, weight_decay=.1)
        _, visual, text, count = batch
        loss = learner.forward_loss(visual.float(), text, count, torch.ones(3))
        loss.backward()
        optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), alternate.parameters()))


def test_design_freeze_preserves_preparation_and_needs_no_pilot(tmp_path, monkeypatch):
    study = tmp_path / 'study'
    study.mkdir()
    for name in ('panels', 'subsets'):
        (study / name).mkdir()
        (study / name / 'input.json').write_text('{}')
    for name in ('splits.json', 'signal_hashes.json', 'frame_partitions.json'):
        (study / name).write_text('{}')
    protocol = {'subsets': {'1.0': {'selected_clip_count': 630}},
                'seeds': [0, 1, 2], 'epochs': 60, 'conditions': list(child_gaze.CONDITIONS),
                'gaze': {'option': 'additive', 'dropout': .5, 'weight': 1.0},
                'selection_status': 'pending development-only pilot'}
    (study / 'protocol.json').write_text(json.dumps(protocol))
    original = (study / 'protocol.json').read_bytes()
    monkeypatch.setattr(child_gaze, '_output_directory', lambda p, **kw: p.mkdir(**kw) or p)
    frozen = []
    monkeypatch.setattr(runner, '_freeze_inputs', lambda root: frozen.append(root))
    result = child_gaze.freeze(tmp_path, readout='additive')
    assert frozen == [tmp_path / 'matrix']
    assert result['selection_status'] == 'frozen by user design decision'
    assert result['gaze'] == protocol['gaze']
    assert (study / 'protocol.json').read_bytes() == original
    assert not (tmp_path / 'pilot.json').exists()
    assert (tmp_path / 'matrix' / 'panels' / 'input.json').read_bytes() == b'{}'
    with pytest.raises(FileExistsError):
        child_gaze.freeze(tmp_path, readout='additive')


def test_matrix_timing_is_two_epochs_development_only_with_serial_headroom(tmp_path, monkeypatch):
    matrix = tmp_path / 'matrix'
    matrix.mkdir()
    (matrix / 'protocol.json').write_text(json.dumps({'selection_status': 'frozen by user design decision'}))
    monkeypatch.setattr(runner, '_verify_frozen', lambda root: None)
    calls = []
    def train(root, fraction, seed, condition, **kwargs):
        calls.append((root, fraction, seed, condition, kwargs))
        assert kwargs == {'epochs': 2, 'smoke': True, 'device': 'mps'}
        return {'runtime_seconds': 25., 'setup_seconds': 1., 'epoch_seconds': 10.,
                'epoch_durations_seconds': [9., 11.], 'final_evaluation_seconds': 4.}
    monkeypatch.setattr(runner, 'train_one', train)
    result = child_gaze.timing(tmp_path)
    assert [c[3] for c in calls] == list(child_gaze.CONDITIONS)
    assert all(c[:3] == (matrix, 1., 0) for c in calls)
    assert result['projected_serial_matrix_seconds'] == 1.25 * 3 * 4 * (1 + 60 * 10 + 4)
    assert result['status'] == 'blocked' and not result['rentals_launched']
