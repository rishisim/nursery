"""Small orchestration checks; never launch scientific training."""
import hashlib
import json
import time

import pytest
import torch

from nursery.learner import acquisition as runner


def test_selection_strictly_improves_and_preserves_earliest_tie(tmp_path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / 'model.pt'
    assert runner._select(50.0, -1, model, path)
    first = torch.load(path, weights_only=True)
    with torch.no_grad():
        model.weight.add_(10)
    assert not runner._select(50.0, 50.0, model, path)
    assert torch.equal(torch.load(path, weights_only=True)['weight'], first['weight'])
    with pytest.raises(ValueError, match='Non-finite'):
        runner._select(float('nan'), 50.0, model, path)


def test_independent_body_rng_preserves_example_order():
    def order(body):
        examples = torch.Generator().manual_seed(7)
        masks = torch.Generator().manual_seed(1007)
        result = []
        for _ in range(3):
            result.append(torch.randperm(25, generator=examples))
            if body:
                torch.rand(25, generator=masks)
        return torch.cat(result)
    assert torch.equal(order(False), order(True))


def test_deadline_rejects_before_work():
    with pytest.raises(TimeoutError, match='cap'):
        runner._check_deadline(time.time() - 1)
    runner._check_deadline(None)


def test_matrix_order_and_completed_exact_skip(tmp_path, monkeypatch):
    protocol = {'fractions': [0.1, 1.0], 'seeds': [0, 1], 'epochs': 60}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    (tmp_path / 'execution.json').write_text(json.dumps({'status': 'ready', 'deadline_unix': time.time() + 60, 'completed': [], 'protocol_sha256': hashlib.sha256((tmp_path / 'protocol.json').read_bytes()).hexdigest()}))
    monkeypatch.setattr(runner, '_output_directory', lambda p: p)
    monkeypatch.setattr(runner, '_verify_frozen', lambda *a, **k: None)
    calls = []
    def train(root, fraction, seed, condition, deadline):
        calls.append((fraction, seed, condition))
        return {'runtime_seconds': 1}
    monkeypatch.setattr(runner, 'train_one', train)
    existing = runner._output(tmp_path, 0.1, 0, 'video_only')
    existing.mkdir(parents=True)
    record = {**runner._identity(tmp_path, 0.1, 0, 'video_only', 60, 'mps'), 'status': 'complete', 'runtime_seconds': 1}
    (existing / 'metrics.json').write_text(json.dumps(record))
    state = runner.run(tmp_path)
    expected = [(f, s, c) for f in protocol['fractions'] for s in protocol['seeds'] for c in runner.CONDITIONS]
    assert calls == expected[1:]
    assert state['status'] == 'complete'
    assert len(state['completed']) == len(expected)


def test_mismatched_existing_output_stops(tmp_path, monkeypatch):
    (tmp_path / 'protocol.json').write_text(json.dumps({'fractions': [1.0], 'seeds': [0], 'epochs': 60}))
    (tmp_path / 'execution.json').write_text(json.dumps({'status': 'ready', 'deadline_unix': time.time() + 60, 'completed': [], 'protocol_sha256': hashlib.sha256((tmp_path / 'protocol.json').read_bytes()).hexdigest()}))
    existing = runner._output(tmp_path, 1.0, 0, 'video_only')
    existing.mkdir(parents=True)
    (existing / 'metrics.json').write_text(json.dumps({'status': 'incomplete'}))
    monkeypatch.setattr(runner, '_output_directory', lambda p: p)
    monkeypatch.setattr(runner, '_verify_frozen', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'train_one', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='match frozen'):
        runner.run(tmp_path)
    assert json.loads((tmp_path / 'execution.json').read_text())['status'] == 'incomplete'


def test_timing_over_budget_stops_before_scientific_work(tmp_path, monkeypatch):
    (tmp_path / 'protocol.json').write_text(json.dumps({'fractions': [0.1, .25, .5, .75, 1.0], 'seeds': [0, 1, 2], 'epochs': 60}))
    monkeypatch.setattr(runner, '_output_directory', lambda p: p)
    monkeypatch.setattr(runner, '_freeze_inputs', lambda *a: None)
    calls = []
    def train(*args, **kwargs):
        calls.append(kwargs)
        return {'runtime_seconds': 100, 'setup_seconds': 0, 'epoch_seconds': 100, 'final_evaluation_seconds': 0}
    monkeypatch.setattr(runner, 'train_one', train)
    with pytest.raises(RuntimeError, match='projection exceeds'):
        runner.timing(tmp_path)
    assert all(c['smoke'] for c in calls)
    assert json.loads((tmp_path / 'execution.json').read_text())['status'] == 'stopped'


def test_actual_train_one_pairs_models_order_masks_and_dev_only_selection(tmp_path, monkeypatch):
    """Exercise real FramePairs, learners, gradients, BodyData and panel evaluator."""
    import numpy as np
    import pandas as pd
    from nursery.learner import body, additive, interaction, video_only
    from nursery.learner.acquisition_evaluate import StudyEvaluation

    def temporary_output(path):
        path.mkdir(parents=True, exist_ok=True)
        return path
    monkeypatch.setattr(runner, '_output_directory', temporary_output)
    upstream = runner._upstream_module('train_frame_mil')
    vocab = {'<pad>': 0, 'apple': 1, 'cup': 2, 'book': 3, 'ball': 4}
    words = ['apple', 'cup', 'book', 'ball']
    manifest = pd.DataFrame({'video_id': ['tiny'] * 4, 'frame_idx': list(range(4)), 'text': words, 'referent': words})
    rng = np.random.default_rng(0)
    cache = rng.normal(size=(4, 17, 8)).astype(np.float32)
    lut = {f'tiny|{i}': (0, i) for i in range(4)}
    subset = tmp_path / 'subsets' / '1.0'
    subset.mkdir(parents=True)
    manifest.to_parquet(subset / 'manifest.parquet')
    (subset / 'vocab.json').write_text(json.dumps(vocab))
    (subset / 'stats.json').write_text(json.dumps({'normalization': {}}))
    np.savez(subset / 'fit_mask.npz', video_ids=np.array(['tiny']), include=np.ones((1, 4), bool))
    episodes = tmp_path / 'episodes'
    episodes.mkdir()
    manifest.to_parquet(episodes / 'frames.parquet')
    cache_root = tmp_path / 'picture_cache'
    cache_root.mkdir()
    np.save(cache_root / 'emb.f16.npy', cache)
    picture = {name: {'cache': str(cache_root)} for name in ('picture_dev', 'picture_test', 'konkle_test')}
    protocol = {'inputs': str(tmp_path), 'episodes': str(episodes), 'physical': str(episodes), 'epochs': 1,
                'window': 2, 'body_window': 5, 'batch': 2, 'tau': .1, 'body_dropout': .5,
                'lr': .0003, 'weight_decay': .1, 'picture_panels': picture}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    for panel in ('clips_dev', 'clips_test', 's8'):
        directory = tmp_path / 'panels' / panel
        directory.mkdir(parents=True)
        manifest.to_parquet(directory / 'manifest.parquet')
        trials = {'trial_rows': [0, 1, 2, 3], 'candidate_rows': [[0, 1, 2, 3]] * 4,
                  'correct_slots': [0, 1, 2, 3], 'donor_rows': [1, 2, 3, 0]}
        (directory / 'trials.json').write_text(json.dumps(trials))
    for panel in picture:
        directory = tmp_path / 'panels' / panel
        directory.mkdir(parents=True)
        (directory / 'trials.json').write_text(json.dumps({'trials': [
            {'query': word, 'candidate_rows': [0, 1, 2, 3], 'correct_slot': i} for i, word in enumerate(words)]}))
    store = {'groups': {k: rng.normal(size=(1, 4, 2)).astype(np.float32) for k in ('contact', 'fingers', 'arm', 'load', 'skin')},
             'targets': {}, 'clip': np.zeros((1, 4), np.int64), 'present': np.ones((1, 4), bool),
             'next_valid': np.ones((1, 4), bool), 'row': {'tiny': 0}, 'stats': {}}
    monkeypatch.setattr(body, 'load_store', lambda *a, **k: store)
    monkeypatch.setattr(upstream, 'load_multi', lambda *a: ([cache], lut))
    initial, batches, masks, events = {}, {}, {}, []
    for module, name, condition in ((video_only, 'VideoOnlyMIL', 'video_only'), (additive, 'AdditiveMIL', 'additive'),
                                    (interaction, 'JointInteractionMIL', 'joint_interaction')):
        constructor = getattr(module, name)
        def wrapped(*args, _constructor=constructor, _condition=condition, **kwargs):
            model = _constructor(*args, **kwargs)
            initial[_condition] = model.word.weight.detach().clone()
            forward = model.forward_loss
            batches[_condition], masks[_condition] = [], []
            def record(v, t, n, w, ctx):
                batches[_condition].append(t[:, 0].tolist())
                if ctx is not None:
                    masks[_condition].append(ctx['keep'].tolist())
                    assert ctx['visual_valid'].shape == v.shape[:2]
                events.append(('train', _condition))
                return forward(v, t, n, w, ctx)
            model.forward_loss = record
            return model
        monkeypatch.setattr(module, name, wrapped)
    evaluate, evaluate_picture = StudyEvaluation.evaluate, StudyEvaluation.evaluate_picture
    def record_eval(self, model, panel, **kwargs):
        events.append(('evaluate', panel))
        return evaluate(self, model, panel, **kwargs)
    def record_picture(self, model, panel):
        events.append(('evaluate', panel))
        return evaluate_picture(self, model, panel)
    monkeypatch.setattr(StudyEvaluation, 'evaluate', record_eval)
    monkeypatch.setattr(StudyEvaluation, 'evaluate_picture', record_picture)
    for condition in runner.CONDITIONS:
        events.clear()
        result = runner.train_one(tmp_path, 1.0, 7, condition, device='cpu')
        assert result['status'] == 'complete'
        assert result['selection']['body_present']['epoch'] == 1
        panels = [event[1] for event in events if event[0] == 'evaluate']
        assert panels[:2] == ['clips_dev', 'picture_dev']
        assert 'clips_test' not in panels[:2]
        assert result['primary']['clips_test']['n_trials'] == 4
        assert result['setup_seconds'] >= 0 and result['epoch_seconds'] > 0
        checkpoint = torch.load(runner._output(tmp_path, 1.0, 7, condition) / 'model.pt', weights_only=True)
        assert not torch.equal(checkpoint['word.weight'], initial[condition])
        with pytest.raises(FileExistsError):
            runner.train_one(tmp_path, 1.0, 7, condition, device='cpu')
    events.clear()
    smoke = runner.train_one(tmp_path, 1.0, 7, 'additive', smoke=True, device='cpu')
    panels = [event[1] for event in events if event[0] == 'evaluate']
    assert set(panels) == {'clips_dev', 'picture_dev'}
    assert smoke['primary'] == smoke['shuffled'] == smoke['picture'] == {}
    assert set(smoke['timing_dev']) == {'real', 'shuffled', 'picture'}
    assert smoke['final_evaluation_estimate_rule'].startswith('4x development')
    assert torch.equal(initial['video_only'], initial['additive'])
    assert torch.equal(initial['additive'], initial['joint_interaction'])
    assert batches['video_only'] == batches['additive'] == batches['joint_interaction']
    assert masks['additive'] == masks['joint_interaction']


def test_frozen_preparation_and_code_changes_are_rejected(tmp_path, monkeypatch):
    artifact = tmp_path / 'manifest.parquet'
    artifact.write_bytes(b'original')
    source = {'revision': 'test-revision', 'code_hashes': {'runner.py': 'code-before'}}
    monkeypatch.setattr(runner, '_source_snapshot', lambda: source)
    snapshot = {**source, 'prepared_hashes': {str(artifact): runner._sha(artifact)}, 'external_hashes': {}}
    (tmp_path / 'frozen_inputs.json').write_text(json.dumps(snapshot))
    runner._verify_frozen(tmp_path)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        runner._verify_frozen(tmp_path)
    artifact.write_bytes(b'original')
    source['code_hashes'] = {'runner.py': 'code-after'}
    with pytest.raises(ValueError, match='Source changed'):
        runner._verify_frozen(tmp_path)
