"""Paired Step 4 software checks, on tiny CPU inputs only."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.learner import acquisition as run
from nursery.learner.gaze import GazeMIL
from nursery.learner.acquisition_evaluate import StudyEvaluation

pytestmark = pytest.mark.usefixtures('offline', 'upstream')
CURRENT = dict(contact=22, fingers=90, arm=27, load=10, skin=4)
RICH = dict(contact=48, fingers=100, arm=31, load=10, skin=4)


def context(dims, batch=3):
    return {'gaze_point': torch.rand(batch, 5, 2, requires_grad=True),
            'gaze_valid': torch.ones(batch, 5, dtype=torch.bool),
            'closeup': torch.randn(batch, 5, 8, requires_grad=True),
            'keep': torch.ones(batch, dtype=torch.bool),
            'body': {'groups': {k: torch.randn(batch, 11, d) for k, d in dims.items()},
                     'missing': torch.zeros(batch, 11), 'seg': torch.zeros(batch, 11, dtype=torch.long),
                     'keep': torch.ones(batch)}}


@pytest.mark.parametrize('dims', [CURRENT, RICH])
def test_combined_gradients_and_exact_absent_body(dims):
    torch.manual_seed(7)
    gaze = GazeMIL(5, 8, option='steering')
    rng = torch.random.get_rng_state()
    torch.manual_seed(7)
    combined = GazeMIL(5, 8, option='steering', dims=dims)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in gaze.state_dict().items():
        assert torch.equal(value, combined.state_dict()[key])
    v, t, n = torch.randn(3, 5*257, 8), torch.tensor([[1], [2], [3]]), torch.ones(3, dtype=torch.long)
    ctx = context(dims)
    gaze_ctx = {k: v for k, v in ctx.items() if k != 'body'}
    for _ in range(2):
        state = torch.random.get_rng_state()
        expected = gaze.forward_loss(v, t, n, torch.ones(3), gaze_ctx)
        torch.random.set_rng_state(state)
        actual = combined.forward_loss(v, t, n, torch.ones(3), gaze_ctx)
        assert torch.equal(expected, actual)
        # Identical loss gradients as well, without changing the shared input graph.
        a = torch.autograd.grad(expected, tuple(gaze.parameters()), retain_graph=True)
        b = torch.autograd.grad(actual, tuple(p for name, p in combined.named_parameters()
                                            if name in dict(gaze.named_parameters())), retain_graph=True)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    loss = combined.forward_loss(v, t, n, torch.ones(3), ctx)
    loss.backward()
    assert ctx['gaze_point'].grad.abs().sum() > 0
    assert ctx['closeup'].grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in combined.encoder.parameters())
    assert combined.fuse.weight.grad.abs().sum() > 0 and combined.beta.grad.abs() > 0
    assert combined.vproj[2].weight.grad.abs().sum() > 0


def test_grid_and_cross_epoch_tie_rule():
    assert run.BODY_WEIGHT_GRID == tuple(i/20 for i in range(31))
    assert 0 in run.BODY_WEIGHT_GRID and 1 in run.BODY_WEIGHT_GRID
    best = dict(dev_score=50., weight=.5, epoch=1)
    assert run.calibration_better(50., .05, 2, best)
    assert not run.calibration_better(50., .5, 2, best)
    assert not run.calibration_better(50., 1., 1, best)
    assert run.calibration_better(51., 1.5, 2, best)


@pytest.fixture
def tiny(tmp_path, monkeypatch, upstream):
    monkeypatch.setattr(run, '_output_directory', lambda p: p.mkdir(parents=True, exist_ok=True) or p)
    words = ['apple', 'cup', 'book', 'ball']
    vocab = dict(zip(['<pad>'] + words, range(5)))
    rows = pd.DataFrame(dict(video_id=['tiny']*4, frame_idx=range(4), text=words, referent=words))
    rng = np.random.default_rng(0)
    cache = rng.normal(size=(4, 257, 8)).astype(np.float32)
    lut = {f'tiny|{i}': (0, i) for i in range(4)}
    monkeypatch.setattr(upstream, 'load_multi', lambda paths: ([cache], lut))
    subset = tmp_path / 'subsets/1.0'
    subset.mkdir(parents=True)
    rows.to_parquet(subset / 'manifest.parquet')
    run._write(subset / 'vocab.json', vocab)
    run._write(subset / 'stats.json', {})
    np.savez(subset / 'fit_mask.npz', video_ids=np.array(['tiny']), include=np.array([[True, True, False, False]]))
    rows.to_parquet(tmp_path / 'frames.parquet')
    rows[['video_id', 'frame_idx']].assign(gaze_x=.4, gaze_y=.6).to_parquet(tmp_path / 'points.parquet')
    run._write(tmp_path / 'partitions.json', {key: 'tiny' for key in lut})
    closeup = tmp_path / 'closeup'
    closeup.mkdir()
    rows[['video_id', 'frame_idx']].assign(row=range(4)).to_parquet(closeup / 'index.parquet')
    np.save(closeup / 'emb.f16.npy', cache[:, 1:].mean(1))
    pictures = tmp_path / 'pictures'
    pictures.mkdir()
    np.save(pictures / 'emb.f16.npy', cache)
    protocol = dict(inputs=str(tmp_path), episodes=str(tmp_path), physical=str(tmp_path), window=2,
                    body_window=5, epochs=2, batch=2, lr=3e-4, weight_decay=.1, tau=.1, body_dropout=.5,
                    arms={c: {'body_representation': c.rsplit('_', 1)[-1]} for c in run.BODY_GAZE_CONDITIONS},
                    gaze=dict(option='steering', points=str(tmp_path/'points.parquet'),
                              partitions=str(tmp_path/'partitions.json'), closeup_cache=str(closeup),
                              dropout=.5, shuffle_seed=0),
                    picture_panels={p: {'cache': str(pictures)} for p in ['picture_dev', 'picture_test', 'konkle_test']})
    run._write(tmp_path / 'protocol.json', protocol)
    trials = dict(trial_rows=list(range(4)), candidate_rows=[list(range(4))]*4,
                  correct_slots=list(range(4)), donor_rows=[1, 2, 3, 0])
    for panel in ['clips_dev', 'clips_test', 's8']:
        path = tmp_path / 'panels' / panel
        path.mkdir(parents=True)
        rows.to_parquet(path / 'manifest.parquet')
        run._write(path / 'trials.json', trials)
    for panel in protocol['picture_panels']:
        path = tmp_path / 'panels' / panel
        path.mkdir(parents=True)
        run._write(path / 'trials.json', {'trials': [dict(query=w, candidate_rows=list(range(4)), correct_slot=i)
                                                    for i, w in enumerate(words)]})
    stores = {}
    for rep, dims in [('current', CURRENT), ('rich', RICH)]:
        groups = {k: rng.normal(size=(1, 4, d)).astype(np.float32) for k, d in dims.items()}
        stores[rep] = dict(groups=groups, targets={}, clip=np.zeros((1, 4), np.int64),
                           present=np.ones((1, 4), bool), next_valid=np.ones((1, 4), bool), row={'tiny': 0}, stats={})
    def load(*args, representation='current', exclude=None, **kwargs):
        assert np.array_equal(exclude, [[False, False, True, True]])
        return stores[representation]
    from nursery.learner import body
    monkeypatch.setattr(body, 'load_store', load)
    # The real adapter should still enforce fit keys. These tiny rows deliberately
    # use all four moments, so permit all in this fixture only.
    original = run._gaze_data
    monkeypatch.setattr(run, '_gaze_data', lambda *a, **k: original(*a, **{key: val for key, val in k.items() if key != 'allowed_keys'}))
    return tmp_path, cache, lut, vocab, stores


def test_runner_paired_streams_dimensions_and_all_evaluations(tiny, monkeypatch):
    from nursery.learner import gaze, video_only
    root, *_ = tiny
    initial, orders, draws = [], [], []
    for module, attr in [(gaze, 'GazeMIL'), (video_only, 'VideoOnlyMIL')]:
        original = getattr(module, attr)
        def construct(*a, _original=original, **kw):
            model = _original(*a, **kw)
            initial.append({k: v.clone() for k, v in model.state_dict().items() if k in ['word.weight', 'vproj.2.weight']})
            order, masks = [], []
            orders.append(order)
            draws.append(masks)
            forward = model.forward_loss
            def loss(v, t, n, w, ctx):
                order.append(t.tolist())
                if ctx is not None:
                    masks.append(ctx['keep'].tolist())
                return forward(v, t, n, w, ctx)
            model.forward_loss = loss
            return model
        monkeypatch.setattr(module, attr, construct)
    for arm, dim in [('video', 0), ('video_gaze', 0), ('video_gaze_body_current', 153), ('video_gaze_body_rich', 193)]:
        metrics = run.train_one(root, 1., 7, arm, device='cpu')
        assert metrics['body_input_dim'] == dim
        assert metrics['trainable_parameters'] > 0
        assert set(metrics['primary']) == set(metrics['all_extra_inputs_removed']) == {'clips_test', 's8'}
        assert set(metrics['picture']) == {'picture_test', 'konkle_test'}
        assert len(metrics['epoch_durations_seconds']) == 2
        if dim:
            assert len(metrics['dev_body_weight_table']) == 62
            assert set(metrics['secondary_as_trained_body_weight']) == set(metrics['shuffled']) == {'clips_test', 's8'}
            assert metrics['selection']['body_present']['weight'] in run.BODY_WEIGHT_GRID
            assert metrics['shuffled']['s8']['gaze_mode'] == 'real'
            assert metrics['all_extra_inputs_removed']['s8']['body_mode'] == 'none'
    for state in initial[1:]:
        assert all(torch.equal(value, state[key]) for key, value in initial[0].items())
    assert orders[0] == orders[1] == orders[2] == orders[3]
    assert draws[1] == draws[2] == draws[3]


def test_shuffled_whole_context_keeps_gaze_and_sweep_matches_direct(tiny, monkeypatch):
    root, cache, lut, vocab, stores = tiny
    evaluator = StudyEvaluation(root, root, [cache], lut, vocab, stores['current'], 'cpu')
    evaluator.gaze_condition = 'video_gaze_body_current'
    evaluator.gaze_factory = lambda rows: run._gaze_data(run._read(root / 'protocol.json'), rows, lut, 'video_gaze')
    model = GazeMIL(5, 8, option='steering', dims=CURRENT)
    captured = []
    score = model.score_encoded
    def record(R, T, ctx):
        captured.append(ctx)
        return score(R, T, ctx)
    monkeypatch.setattr(model, 'score_encoded', record)
    evaluator.evaluate(model, 'clips_dev')
    evaluator.evaluate(model, 'clips_dev', body_mode='shuffled')
    real, shuffled = captured
    donors = torch.tensor([1, 2, 3, 0])
    for key in ('gaze_point', 'gaze_valid', 'closeup', 'keep'):
        assert torch.equal(real[key], shuffled[key])
    for key in ('missing', 'seg', 'same', 'keep', 'frame_valid', 'frame_next_valid'):
        assert torch.equal(real['body'][key][donors], shuffled['body'][key])
    for key in CURRENT:
        assert torch.equal(real['body']['groups'][key][donors], shuffled['body']['groups'][key])
    sweep = evaluator.evaluate(model, 'clips_dev', body_weights=run.BODY_WEIGHT_GRID)
    for result in sweep:
        model.body_weight = result['body_weight']
        direct = evaluator.evaluate(model, 'clips_dev')
        assert direct['accuracy'] == result['accuracy']
        np.testing.assert_allclose(direct['scores'], result['scores'], atol=1e-7)


def test_frozen_protocol_settings():
    spec = importlib.util.spec_from_file_location('step4', Path(__file__).parents[1] / 'scripts/step4_body_comparison.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    parent = dict(subsets={'1.0': {'selected_clip_count': 630}}, gaze={'points': 'points', 'partitions': 'partitions', 'shuffle_seed': 0})
    saved = copy.deepcopy(parent)
    protocol = script.comparison_protocol(parent)
    assert parent == saved
    assert protocol['seeds'] == [0, 1, 2] and protocol['epochs'] == 60
    assert protocol['conditions'] == list(script.ARMS)
    assert protocol['selection']['body_weight_grid'] == list(run.BODY_WEIGHT_GRID)
    assert protocol['batch'] == 256 and protocol['body_dropout'] == .5
    assert protocol['gaze']['option'] == 'steering'
    assert protocol['device'] == 'cuda' and protocol['runtime_cap_hours'] == 4
    assert protocol['rng']['body_init'] == 'forked CPU seed+3000'
    assert run.protocol_matrix(protocol)[2] == list(script.ARMS)


@pytest.mark.parametrize('dims', [CURRENT, RICH])
def test_body_initialization_uses_run_seed_without_changing_visual_rng(dims):
    bodies = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        gaze = GazeMIL(5, 8, option='steering')
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(seed)
        combined = GazeMIL(5, 8, option='steering', dims=dims, seed=seed)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for key, value in gaze.state_dict().items():
            assert torch.equal(value, combined.state_dict()[key])
        bodies.append(combined.encoder.state_dict())
    assert any(not torch.equal(bodies[0][key], bodies[1][key]) for key in bodies[0])


def test_freeze_requires_explicit_overwrite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('step4', Path(__file__).parents[1] / 'scripts/step4_body_comparison.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    parent, destination = tmp_path / 'parent', tmp_path / 'frozen'
    destination.mkdir()
    existing = destination / 'protocol.json'
    existing.write_text('original')
    monkeypatch.setattr(script.subprocess, 'run', lambda *a, **k: None)
    with pytest.raises(FileExistsError, match='--overwrite'):
        script.freeze(parent, destination)
    assert existing.read_text() == 'original'
    parent.mkdir()
    rich = tmp_path / 'body_rich'
    rich.mkdir()
    run._write(rich / 'selection.json', {'granularity': 22})
    run._write(parent / 'protocol.json', dict(episodes=str(tmp_path), hashes={},
               subsets={'1.0': {'selected_clip_count': 630}}, gaze={}))
    for name in ('panels', 'subsets'):
        (parent / name).mkdir()
    for name in ('splits.json', 'signal_hashes.json'):
        run._write(parent / name, {})
    monkeypatch.setattr(run, '_source_snapshot', lambda: dict(revision='fixture', code_hashes={}))
    monkeypatch.setattr(run, '_freeze_inputs', lambda root: {'frozen': True})
    assert script.freeze(parent, destination, overwrite=True) == {'frozen': True}
    assert run._read(existing)['device'] == 'cuda'
