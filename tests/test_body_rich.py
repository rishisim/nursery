"""Step 3 lattice, isolation, masking and normalization contracts (CPU only)."""
import numpy as np
import pytest
import torch

from nursery.simulator.body_rich import bin_frames, contact_geometry
from nursery.learner.rich import standardize, enrich_store
from nursery.learner.windows import BodyData
from nursery.learner.body import BodyEncoder


@pytest.mark.parametrize('centre,length', [(0, 101), (1, 101), (59, 101), (100, 101), (120, 300)])
def test_bin_lattice_contains_snapshot_and_is_clip_isolated(centre, length):
    full, leading, trailing = [bin_frames(centre, length, v) for v in range(3)]
    assert centre in full and centre in leading and centre in trailing
    for frames in (full, leading, trailing):
        assert ((frames - centre) % 4 == 0).all()
        assert ((frames >= 0) & (frames < length)).all()
    np.testing.assert_array_equal(leading, full[full >= centre])
    np.testing.assert_array_equal(trailing, full[full <= centre])
    assert ((full >= centre - 60) & (full < centre + 60)).all()
    # Exact outer span is closed at +5 and begins at -5, with no +/-0.5s exposure.
    left = bin_frames(centre + 600, length + 1200, 1) - 600
    right = bin_frames(centre + 600, length + 1200, 2) + 600
    assert left.min() == centre and right.max() == centre + 1200


def test_contact_geometry_uses_only_contacted_hand_points():
    points = np.array([[-1., 0, 0], [1., 0, 0]])
    normals = np.array([[0., 1, 0], [0., -1, 0]])
    np.testing.assert_allclose(contact_geometry(points, normals), [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2])
    np.testing.assert_array_equal(contact_geometry(points[:0], normals[:0]), np.zeros(11))
    assert np.isfinite(contact_geometry(points[:1], normals[:1])).all()


def test_normalization_excludes_heldout_and_excluded_rows_and_zeroes_absence():
    values = np.array([[[[1.], [2.], [3.]], [[3.], [4.], [5.]], [[1e5], [1e5], [1e5]]],
                       [[[1e9], [1e9], [1e9]], [[1e9], [1e9], [1e9]], [[1e9], [1e9], [1e9]]]], dtype=np.float32)
    train = np.array([[True, True, False], [False, False, False]])
    valid = np.ones(values.shape[:-1], bool)
    valid[0, 0, 1] = False
    normalized, stats = standardize(values, train, valid)
    assert stats[0] == [2.]
    assert stats[1][0] == pytest.approx(1., abs=2e-6)
    assert normalized[0, 0, 1, 0] == 0
    changed = values.copy()
    changed[~train] *= 99
    assert standardize(changed, train, valid)[1] == stats


def test_loader_fit_mask_and_outer_summaries(tmp_path):
    root = tmp_path / 'body_rich'
    root.mkdir()
    steps = 12
    videos = ['train', 'heldout']
    for i, video in enumerate(videos):
        value = np.repeat((np.arange(steps) + i * 1e5)[:, None, None], 3, axis=1).astype(np.float32)
        np.savez(root / f'{video}.npz', frame_idx=np.arange(steps), regions22=np.full((steps, 3, 22), .25),
                 events=np.repeat(value, 4, axis=2), geometry=np.repeat(value, 22, axis=2),
                 fingers=np.repeat(value, 100, axis=2), arm=np.repeat(value, 31, axis=2),
                 touching=np.ones((steps, 3, 2), bool))
    store = {'present': np.ones((2, steps), bool), 'video_ids': videos, 'clip': np.zeros((2, steps), int),
             'groups': {'load': np.zeros((2, steps, 10)), 'skin': np.zeros((2, steps, 4))},
             'stats': {}, 'targets': {}, 'next_valid': np.ones((2, steps), bool)}
    exclude = np.zeros((2, steps), bool)
    exclude[0, 2:] = True
    rich = enrich_store(tmp_path, store, {'heldout'}, exclude, 22)
    assert rich['fit_mask'].sum() == 2
    assert rich['stats']['rich_fingers'][0] == [0.5] * 100
    assert {k: v.shape[-1] for k, v in rich['groups'].items()} == {'contact': 48, 'fingers': 100, 'arm': 31, 'load': 10, 'skin': 4}
    assert (rich['groups']['contact'][..., :22] == .25).all()
    # Distinct markers prove window-time selection, rather than merely a length check.
    for k, variants in rich['variants'].items():
        variants[:, :, 1] = 123
        variants[:, :, 2] = 456
    data = BodyData(rich, np.array([0]), np.array([5]), np.array([[5]]), None, None, None, None, 'cpu')
    groups, missing, seg = data.window(torch.tensor([0]), torch.tensor([5]))
    for values in groups.values():
        assert (values[:, 0] == 123).all() and (values[:, -1] == 456).all()
    assert not missing.any()
    _, missing, _ = data.window(torch.tensor([0]), torch.tensor([0]))
    assert missing[0, :5].all()
    encoder = BodyEncoder(data.dims)
    seg[:, :3] = 99
    state = encoder(groups, torch.zeros_like(missing), seg)[:, 5]
    changed = {k: v.clone() for k, v in groups.items()}
    for v in changed.values():
        v[:, :3] += 100
    torch.testing.assert_close(state, encoder(changed, torch.zeros_like(missing), seg)[:, 5])


def test_multinomial_probe_and_grouped_folds_use_training_fit_only():
    from scripts.step3_body_checks import fit_logistic, score_logistic, grouped_folds

    rng = np.random.default_rng(17)
    y = np.repeat(np.arange(3), 30)
    x = np.eye(3)[y] * 4 + rng.normal(0, .15, (90, 3))
    model = fit_logistic(x, y, 1., 3)
    score = score_logistic(model, x, y)
    assert score['accuracy'] == 1 and score['log_loss'] < .1
    np.testing.assert_allclose(model[0], x.mean(0))
    original = model[0].copy()
    score_logistic(model, x + 10000, y)
    np.testing.assert_array_equal(model[0], original)
    clips = np.repeat([f'clip{i}' for i in range(10)], 9)
    folds = grouped_folds(clips)
    assert set(folds) == set(range(5))
    for fold in range(5):
        assert not set(clips[folds == fold]) & set(clips[folds != fold])


def test_probe_geometry_absence_is_zero_after_fold_local_standardization():
    from scripts.step3_body_checks import fit_logistic, scaled_features

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 48))
    x[:, 22:24] = 1
    x[:20, 22] = 0  # left hand absent
    x[:20, 26:37] = 0
    model = fit_logistic(x, np.arange(40) % 2, .1, 2, granularity=22)
    transformed = scaled_features(x, model[0], model[1], 22)
    np.testing.assert_array_equal(transformed[:20, 26:37], 0)
    np.testing.assert_allclose(model[0][26:37], x[20:, 26:37].mean(0))
