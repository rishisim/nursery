"""Step 6 inherits the frozen comparison and rejects mismatched preparation."""
import copy
import importlib.util
from pathlib import Path

import pytest

from nursery.learner import acquisition as run


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location('step6', Path(__file__).parents[1] / 'scripts/step6_acquisition_curves.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_only_changes_arms_and_fractions(script):
    parent = dict(subsets={'1.0': {'selected_clip_count': 630}},
                  gaze=dict(points='points', partitions='partitions', closeup_cache='closeups', shuffle_seed=0))
    saved = copy.deepcopy(parent)
    expected = script.comparison_protocol(parent)
    expected.update(conditions=['video', 'video_gaze', 'video_gaze_body_current'],
                    arms={k: v for k, v in expected['arms'].items() if k != 'video_gaze_body_rich'},
                    fractions=[.1, .25, .5, .75, 1.])
    protocol = script.curve_protocol(parent)
    assert parent == saved
    assert protocol == expected
    assert run.protocol_matrix(protocol) == (expected['fractions'], [0, 1, 2], expected['conditions'])


@pytest.fixture
def preparation(tmp_path, script):
    destination, comparison = tmp_path / 'curve', tmp_path / 'comparison'
    for root in (destination, comparison):
        (root / 'panels').mkdir(parents=True)
        (root / 'panels/trials.json').write_bytes(b'fixed trials')
        for count, fraction in enumerate(run.FRACTIONS, 1):
            subset = root / 'subsets' / str(fraction)
            subset.mkdir(parents=True)
            run._write(subset / 'stats.json', {'selected_clips': list('abcde'[:count])})
    run._write(destination / 'splits.json', {'source_keys': {'dev': ['dev'], 'test': ['test'], 's8': ['s8']}})
    return destination, comparison


def test_preparation_identity_and_nesting(script, preparation):
    result = script.check_preparation(*preparation)
    assert result['passed'] and result['strictly_nested'] and result['held_out_disjoint']
    assert len(result['sha256']) == 2


@pytest.mark.parametrize('failure', ['hash', 'missing', 'nesting', 'dev', 'test', 's8'])
def test_preparation_rejects_mismatch(script, preparation, failure):
    destination, comparison = preparation
    if failure == 'hash':
        (destination / 'panels/trials.json').write_bytes(b'changed')
    elif failure == 'missing':
        (destination / 'panels/trials.json').unlink()
    else:
        selected = ['z'] if failure == 'nesting' else ['a', failure]
        run._write(destination / 'subsets/0.25/stats.json', {'selected_clips': selected})
    with pytest.raises(AssertionError):
        script.check_preparation(destination, comparison)


def test_freeze_refuses_existing_root(script, tmp_path, monkeypatch):
    monkeypatch.setattr(script.subprocess, 'run', lambda *a, **k: None)
    with pytest.raises(FileExistsError):
        script.freeze(destination=tmp_path)


def test_smoke_uses_smallest_subset_and_all_three_arms(script, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    frozen, destination = tmp_path / 'frozen', tmp_path / 'smoke'
    subset = frozen / 'subsets/0.1'
    subset.mkdir(parents=True)
    (frozen / 'panels').mkdir()
    frames = pd.DataFrame(dict(video_id=['v']*12, frame_idx=range(12), clip=['a']*6 + ['b']*6))
    frames.to_parquet(tmp_path / 'frames.parquet')
    frames.iloc[[2, 3, 8, 9]].to_parquet(subset / 'manifest.parquet')
    np.savez(subset / 'fit_mask.npz', video_ids=np.array(['v']), include=np.ones((1, 12), dtype=bool))
    run._write(subset / 'stats.json', {'selected_clips': ['a', 'b']})
    run._write(frozen / 'protocol.json', dict(episodes=str(tmp_path), hashes={}))
    monkeypatch.setattr(script.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(run, '_source_snapshot', lambda: dict(revision='fixture', code_hashes={}))
    calls = []
    def train(root, fraction, seed, arm, **kwargs):
        calls.append((fraction, seed, arm, kwargs))
        return {'complete': True}
    monkeypatch.setattr(run, 'train_one', train)
    assert set(script.smoke(frozen, destination, clips=1)) == set(script.CURVE_ARMS)
    assert calls == [(.1, 0, arm, dict(smoke=True, epochs=1, device='cpu')) for arm in script.CURVE_ARMS]
    assert len(pd.read_parquet(destination / 'subsets/0.1/manifest.parquet')) == 2
    with np.load(destination / 'subsets/0.1/fit_mask.npz') as fit:
        assert fit['include'].tolist() == [[True]*6 + [False]*6]
