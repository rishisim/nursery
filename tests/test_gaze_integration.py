"""Gaze orchestration checks: synthetic inputs only, no pilot or training run."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.learner import acquisition as runner
from nursery.learner.acquisition_evaluate import StudyEvaluation
from nursery.learner.gaze import GazeMIL

pytestmark = pytest.mark.usefixtures('offline', 'upstream')


def test_gaze_hooks_do_not_activate_historical_matrix():
    assert runner.CONDITIONS == ('video_only', 'additive', 'joint_interaction')
    with pytest.raises(ValueError):
        runner.matrix_selection([1.0], list(runner.GAZE_CONDITIONS))
    for value in ({}, {'gaze': {'option': 'additive'}}, {'gaze': {'option': 'oracle'}}):
        with pytest.raises(ValueError):
            runner._gaze_settings(value)


def test_gaze_scientific_inputs_must_be_frozen(tmp_path):
    points, partitions = tmp_path / 'points.parquet', tmp_path / 'partitions.json'
    points.write_bytes(b'points')
    partitions.write_text('{}')
    protocol = {'gaze': {'option': 'steering', 'points': str(points), 'partitions': str(partitions),
                         'dropout': .5, 'shuffle_seed': 0}}
    frozen = tmp_path / 'frozen_inputs.json'
    frozen.write_text(json.dumps({'external_hashes': {}}))
    with pytest.raises(ValueError, match='absent'):
        runner._verify_gaze_frozen(tmp_path, protocol, 'video_gaze_point')
    frozen.write_text(json.dumps({'external_hashes': {str(p): runner._sha(p) for p in (points, partitions)}}))
    runner._verify_gaze_frozen(tmp_path, protocol, 'video_gaze_point')
    points.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        runner._verify_gaze_frozen(tmp_path, protocol, 'video_gaze_point')


def test_new_world_baseline_refuses_changed_visual_window_before_outputs(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'gaze': {}, 'window': 1, 'epochs': 60}))
    with pytest.raises(ValueError, match='five-frame'):
        runner.train_one(tmp_path, 1.0, 0, 'video_only', device='cpu')
    assert not (tmp_path / 'models').exists()


@pytest.mark.parametrize('option', ['additive', 'steering'])
@pytest.mark.parametrize('condition', runner.GAZE_CONDITIONS)
def test_panel_modes_use_same_candidates_and_removed_bypasses_adapter(tmp_path, option, condition):
    (tmp_path / 'protocol.json').write_text(json.dumps({'picture_panels': {}}))
    directory = tmp_path / 'panels' / 'clips_dev'
    directory.mkdir(parents=True)
    rows = pd.DataFrame({'video_id': [f'v{i}' for i in range(4)], 'frame_idx': [0] * 4,
                         'text': ['unrelated'] * 4, 'referent': ['mug', 'unknown', 'mug', 'unknown']})
    rows.to_parquet(directory / 'manifest.parquet')
    trials = {'trial_rows': [0, 1], 'candidate_rows': [[0, 1, 2, 3], [3, 2, 1, 0]],
              'correct_slots': [0, 3], 'donor_rows': [1, 2, 3, 0]}
    (directory / 'trials.json').write_text(json.dumps(trials))
    cache = np.random.default_rng(0).normal(size=(4, 257, 8)).astype(np.float32)
    lut = {f'v{i}|0': (0, i) for i in range(4)}
    evaluator = StudyEvaluation(tmp_path, tmp_path, [cache], lut, {'<pad>': 0, 'mug': 1}, None, 'cpu')
    model = GazeMIL(2, 8, option=option, point_only=condition == 'video_gaze_point').eval()
    modes = []

    class Data:
        donor_indices = np.array([1, 2, 3, 0])

        def batch(self, ids, *, mode, device):
            modes.append(mode)
            return {'gaze_point': torch.full((len(ids), 5, 2), .5),
                    'gaze_valid': torch.tensor([[True, False, False, False, False]] * len(ids)),
                    'closeup': torch.ones(len(ids), 5, 8), 'keep': torch.ones(len(ids), dtype=torch.bool)}

    calls = []
    evaluator.gaze_condition = condition
    evaluator.gaze_factory = lambda rows: calls.append(len(rows)) or Data()
    absent = evaluator.evaluate(model, 'clips_dev', gaze_mode='removed')
    assert calls == modes == []
    present = evaluator.evaluate(model, 'clips_dev')
    assert calls == [4]
    assert modes == ['shuffled' if condition == 'video_shuffled_gaze' else 'real']
    assert present['candidate_rows'] == absent['candidate_rows']
    assert present['queries'] == absent['queries'] == ['mug', 'unknown']
    assert present['n_trials'] == absent['n_trials'] == 2
    assert absent['gaze_mode'] == 'removed'
    assert present['credits'][1] == absent['credits'][1] == .25
    # Removing input uses this very checkpoint's original video score.
    evaluator.gaze_factory = None
    assert evaluator.evaluate(model, 'clips_dev')['scores'] == absent['scores']


@pytest.mark.parametrize('option', ['additive', 'steering'])
@pytest.mark.parametrize('dev_only', [False, True])
def test_tiny_gaze_train_dispatch_and_dev_only_checkpoint_modes(tmp_path, monkeypatch, upstream, option, dev_only):
    """Four synthetic pairs exercise dispatch; this is not a scientific run."""
    from nursery.learner import gaze, video_only

    monkeypatch.setattr(runner, '_output_directory', lambda path: path.mkdir(parents=True, exist_ok=True) or path)
    words = ['apple', 'cup', 'book', 'ball']
    vocab = dict(zip(['<pad>'] + words, range(5)))
    rows = pd.DataFrame({'video_id': ['tiny'] * 4, 'frame_idx': range(4), 'text': words, 'referent': words})
    cache = np.random.default_rng(0).normal(size=(4, 257, 8)).astype(np.float32)
    lut = {f'tiny|{i}': (0, i) for i in range(4)}
    monkeypatch.setattr(upstream, 'load_multi', lambda paths: ([cache], lut))
    subset = tmp_path / 'subsets' / '1.0'
    subset.mkdir(parents=True)
    rows.to_parquet(subset / 'manifest.parquet')
    (subset / 'vocab.json').write_text(json.dumps(vocab))
    (subset / 'stats.json').write_text('{}')
    np.savez(subset / 'fit_mask.npz', video_ids=np.array(['tiny']), include=np.ones((1, 4), bool))
    points = rows[['video_id', 'frame_idx']].assign(gaze_x=[.2, .4, .6, .8], gaze_y=.5)
    points.to_parquet(tmp_path / 'points.parquet')
    (tmp_path / 'partitions.json').write_text(json.dumps({key: 'synthetic' for key in lut}))
    closeup = tmp_path / 'closeup'
    closeup.mkdir()
    rows[['video_id', 'frame_idx']].assign(row=range(4)).to_parquet(closeup / 'index.parquet')
    np.save(closeup / 'emb.f16.npy', cache[:, 1:].mean(1).astype(np.float16))
    picture = tmp_path / 'pictures'
    picture.mkdir()
    np.save(picture / 'emb.f16.npy', cache)
    epochs = 2 if dev_only else 1
    protocol = {'inputs': str(tmp_path), 'episodes': str(tmp_path), 'epochs': epochs, 'window': 2,
                'development_only': dev_only,
                'batch': 2, 'tau': .1, 'lr': .0003, 'weight_decay': .1,
                'picture_panels': {'picture_dev': {'cache': str(picture)}},
                'gaze': {'option': option, 'points': str(tmp_path / 'points.parquet'),
                         'partitions': str(tmp_path / 'partitions.json'), 'closeup_cache': str(closeup),
                         'dropout': .5, 'shuffle_seed': 0}}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    panel = tmp_path / 'panels' / 'clips_dev'
    panel.mkdir(parents=True)
    rows.to_parquet(panel / 'manifest.parquet')
    (panel / 'trials.json').write_text(json.dumps({'trial_rows': list(range(4)),
        'candidate_rows': [list(range(4))] * 4, 'correct_slots': list(range(4)), 'donor_rows': [1, 2, 3, 0]}))
    panel = tmp_path / 'panels' / 'picture_dev'
    panel.mkdir()
    (panel / 'trials.json').write_text(json.dumps({'trials': [
        {'query': word, 'candidate_rows': list(range(4)), 'correct_slot': i} for i, word in enumerate(words)]}))
    initials, orders, contexts, evaluation_states = [], [], [], []
    for module, attr in ((video_only, 'VideoOnlyMIL'), (gaze, 'GazeMIL')):
        original = getattr(module, attr)
        def construct(*args, _original=original, **kwargs):
            model = _original(*args, **kwargs)
            initials.append(model.word.weight.detach().clone())
            batches, seen = [], []
            orders.append(batches)
            contexts.append(seen)
            forward = model.forward_loss
            def loss(v, t, n, w, ctx):
                batches.append(t.tolist())
                seen.append(ctx)
                return forward(v, t, n, w, ctx)
            model.forward_loss = loss
            return model
        monkeypatch.setattr(module, attr, construct)
    original_evaluate = StudyEvaluation.evaluate
    def evaluate(self, model, panel, **kwargs):
        assert panel == 'clips_dev'  # No held-out panels even exist in this fixture.
        evaluation_states.append((kwargs.get('gaze_mode'), {k: v.clone() for k, v in model.state_dict().items()}))
        return original_evaluate(self, model, panel, **kwargs)
    monkeypatch.setattr(StudyEvaluation, 'evaluate', evaluate)
    for condition in ('video_only',) + runner.GAZE_CONDITIONS:
        evaluation_states.clear()
        metrics = runner.train_one(tmp_path, 1., 7, condition, smoke=not dev_only,
                                   dev_only=dev_only, device='cpu')
        assert metrics['status'] == 'complete'
        assert 1 <= metrics['selection']['body_present']['epoch'] <= epochs
        assert len(metrics['history']) == epochs
        assert metrics['development_only']
        assert metrics['primary'] == metrics['picture'] == {}
        if condition != 'video_only':
            assert metrics['gaze_removed'] == {}
            assert metrics['timing_dev']['gaze_removed']['gaze_mode'] == 'removed'
            assert evaluation_states[-1][0] == 'removed'
            for key, value in evaluation_states[-2][1].items():
                assert torch.equal(value, evaluation_states[-1][1][key])
            expected_mode = 'shuffled' if condition == 'video_shuffled_gaze' else 'real'
            assert metrics['timing_dev']['real']['gaze_mode'] == expected_mode
    assert all(torch.equal(initials[0], value) for value in initials[1:])
    assert all(order == orders[0] for order in orders[1:])
    assert all(ctx is None for ctx in contexts[0])
    assert all('gaze_point' in ctx for batch in contexts[1:] for ctx in batch)
    assert all('closeup' not in ctx for ctx in contexts[-1])
