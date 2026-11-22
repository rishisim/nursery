"""Step 3 CPU pre-comparison checks and diagnostic probes; never score test/s8.

Use PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 and the existing cue environment.
No new dependencies: multinomial L2 logistic regression uses scipy L-BFGS
with an analytic gradient (same sum-loss/1/C penalty convention as sklearn).
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / 'outputs/renders/clean_856'
RICH = EPISODES / 'body_rich'
OUT = ROOT / 'outputs/analysis/step3_body/build'
MATRIX = ROOT / 'outputs/runs/child_gaze/matrix'
CS = (.01, .1, 1., 10.)


def tables():
    frames = pd.read_parquet(EPISODES / 'frames.parquet').sort_values(['video_id', 'frame_idx']).reset_index(drop=True)
    partitions = json.loads((MATRIX / 'frame_partitions.json').read_text())
    frames['partition'] = [partitions[f'{v}|{f}'] for v, f in zip(frames.video_id, frames.frame_idx)]
    selected = set(json.loads((MATRIX / 'subsets/1.0/stats.json').read_text())['selected_clips'])
    return frames, selected


def record(key, value):
    path = OUT / 'checks.json'
    checks = json.loads(path.read_text()) if path.exists() else {}
    checks[key] = value
    path.write_text(json.dumps(checks, indent=2) + '\n')


def check_export():
    from nursery.simulator.body_rich import correspondences, bin_frames
    from nursery.simulator.swaps import assignment

    frames, _ = tables()
    config, meta = correspondences(EPISODES)
    swaps = json.loads(Path(config['swaps']).read_text())
    rows = sum(len(item['rows']) for item in meta.values())
    assert len(meta) == 1063 and rows == 102720
    mapping = {v: {} for v in frames.video_id.unique()}
    for clip, item in meta.items():
        for video, second, centre in item['rows']:
            mapping[video][second] = (clip, centre, item['length'])
    phases = defaultdict(set)
    for video, rows_by_second in mapping.items():
        for clip, centre, _ in rows_by_second.values():
            phases[(video, clip)].add(centre % 4)
    mismatch = {'native': 0, 'swapped': 0, 'hand_pose': 0, 'joints': 0}
    max_joint_error = 0.
    phase_changes = 0
    for video, group in frames.groupby('video_id'):
        with np.load(RICH / f'{video}.npz') as rich, np.load(EPISODES / 'body' / f'{video}.npz') as current:
            assert len(rich['frame_idx']) == 120
            for key in rich.files:
                assert np.isfinite(rich[key]).all(), (video, key)
            centre_bits = rich['contact_bits'][:, 15]
            for second in range(120):
                clip, centre, length = mapping[video][second]
                key = 'swapped' if assignment(swaps, clip) else 'native'
                mismatch[key] += int(not np.array_equal(centre_bits[second], current['hand_contact'][second]))
                expected = bin_frames(centre, length)
                actual = rich['source_frames'][second]
                np.testing.assert_array_equal(actual[actual >= 0], expected)
                assert actual[15] == centre
                if second and mapping[video][second - 1][0] == clip:
                    phase_changes += int((centre - mapping[video][second - 1][1]) % 4 != 0)
            expected_pose = np.c_[current['left_hand_pose'], current['right_hand_pose']]
            mismatch['hand_pose'] += int(np.any(rich['centre_hand_pose'] != expected_pose, axis=1).sum())
            error = np.abs(rich['centre_joints'] - current['joints']).max(axis=(1, 2))
            max_joint_error = max(max_joint_error, float(error.max()))
            mismatch['joints'] += int((error > 2e-6).sum())
    result = {'clips': len(meta), 'sessions': len(mapping), 'rows': rows, 'mismatched_rows': mismatch,
              'max_joint_error_m': max_joint_error, 'joint_tolerance_m': 2e-6, 'finite': True,
              'all_bin_indices_within_own_clip': True, 'centre_preserving_row_grids': True,
              'legacy_adjacent_centres_with_phase_change': phase_changes,
              'legacy_segments_with_phase_change': sum(len(x) > 1 for x in phases.values()),
              'alignment_passed': not any(mismatch.values())}
    record('export', result)
    print(json.dumps(result), flush=True)
    if not result['alignment_passed']:
        raise ValueError('Alignment failed; inspect checks.json')


def fit_logistic(x, y, c, classes, initial=None, granularity=None):
    """Train standardization and unpenalized-intercept multinomial logistic regression."""
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    if granularity is not None:
        for hand in range(2):
            positive = x[:, granularity + hand] > 0
            block = slice(granularity + 4 + 11*hand, granularity + 4 + 11*(hand+1))
            mean[block] = x[positive, block].mean(0) if positive.any() else 0
            scale[block] = x[positive, block].std(0) if positive.any() else 1
        scale = np.where(scale > 1e-8, scale, 1.)
    z = scaled_features(x, mean, scale, granularity)
    n, d = z.shape
    one_hot = np.eye(classes)[y]
    def objective(flat):
        weights = flat.reshape(d, classes)
        logits = z @ weights
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -logp[np.arange(n), y].mean() + np.square(weights[:-1]).sum() / (2 * c * n)
        residual = np.exp(logp) - one_hot
        grad = z.T @ residual / n
        grad[:-1] += weights[:-1] / (c * n)
        return loss, grad.ravel()
    start = np.zeros(d * classes) if initial is None else initial
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      options={'maxiter': 4000, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 40, 'maxcor': 30})
    if not result.success:
        raise RuntimeError(f'Logistic regression C={c} did not converge in {result.nit} iterations: {result.message}')
    return mean, scale, result.x.reshape(d, classes), int(result.nit), granularity


def scaled_features(x, mean, scale, granularity=None):
    z = (x - mean) / scale
    if granularity is not None:
        for hand in range(2):
            absent = x[:, granularity + hand] == 0
            z[absent, granularity+4+11*hand:granularity+4+11*(hand+1)] = 0
    return np.c_[z, np.ones(len(x))]


def feature_granularity(feature):
    return None if feature == 'current' else int(feature.replace('contact', '').replace('body', ''))


def score_logistic(model, x, y):
    mean, scale, weights, iterations, granularity = model
    logits = scaled_features(x, mean, scale, granularity) @ weights
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    valid = (y >= 0) & (y < weights.shape[1])
    likelihood = np.full(len(y), np.log(1e-15))
    likelihood[valid] = logp[np.arange(len(y))[valid], y[valid]]
    return {'accuracy': float((logp.argmax(1) == y).mean()), 'log_loss': float(-likelihood.mean()),
            'rows': len(y), 'unknown_labels': int((~valid).sum()), 'iterations': iterations}


def grouped_folds(groups):
    clips = np.array(sorted(set(groups)))
    if len(clips) < 5:
        raise ValueError('Need at least 5 distinct training source clips')
    rng = np.random.default_rng(0)
    fold = dict(zip(clips[rng.permutation(len(clips))], np.arange(len(clips)) % 5))
    return np.array([fold[c] for c in groups])


def cv_job(args):
    path, feature, c, fold, subject = args
    with np.load(path) as data:
        x, y, groups, subjects = data[feature], data['y'], data['clip'], data['subject']
        train = data['train'].copy()
    if subject is not None:
        train &= subjects != subject
    ix = np.flatnonzero(train)
    folds = grouped_folds(groups[ix])
    fit, val = ix[folds != fold], ix[folds == fold]
    assert not set(groups[fit]) & set(groups[val])
    model = fit_logistic(x[fit], y[fit], c, int(y[train].max()) + 1, granularity=feature_granularity(feature))
    result = score_logistic(model, x[val], y[val])
    return feature, c, fold, subject, result


def cv_select(pool, path, feature, subject=None):
    jobs = [(str(path), feature, c, fold, subject) for c in CS for fold in range(5)]
    values = list(pool.imap_unordered(cv_job, jobs))
    means = {c: float(np.mean([row[4]['accuracy'] for row in values if row[1] == c])) for c in CS}
    chosen = max(CS, key=lambda c: (means[c], -c))
    return chosen, {'fold_results': [{'C': c, 'fold': fold, **r} for _, c, fold, _, r in values],
                    'mean_accuracy': means, 'selected_C': chosen}


def diagnostic_data():
    """Only train/dev rows; labels never enter body export or learner features."""
    from nursery.learner.body import session_features
    from nursery.simulator.swaps import assignment

    frames, selected = tables()
    all_clips = {v: g.sort_values('frame_idx')['clip'].to_numpy() for v, g in frames.groupby('video_id')}
    frames = frames[frames.partition.isin(['train', 'dev'])].copy()
    # held_category is null when the 1Hz snapshot has no contact. Recover its
    # diagnostic label from the recorded world's constant source-clip assignment.
    swaps = json.loads((ROOT / 'outputs/assets/objaverse/swaps_v2.json').read_text())
    vocabulary = json.loads((ROOT / 'nursery/simulator/vocabulary.json').read_text())
    label = {}
    for clip, group in frames.groupby('clip'):
        observed = group.held_category.dropna().unique()
        assert len(observed) <= 1
        given = assignment(swaps, clip)
        if given:
            label[clip] = given['word']
        elif len(observed):
            label[clip] = observed[0]
        else:
            label[clip] = vocabulary['extras'][Path(clip).stem.split('_')[0]]
        if len(observed):
            assert label[clip] == observed[0]
    records = []
    features = defaultdict(list)
    current_vectors = []
    current_bits = []
    for video, group in frames.groupby('video_id'):
        with np.load(EPISODES / 'body' / f'{video}.npz') as data:
            current = session_features(data)
            bits = data['hand_contact'].copy()
        with np.load(EPISODES / 'physical_by_clip' / f'{video}.npz') as physical:
            phys = np.c_[physical['load'].reshape(120, 10), physical['skin_dT'], physical['contact_area']]
        # Current 1Hz position differences use full session clip-boundary rules.
        names = all_clips[video]
        first = np.r_[True, names[1:] != names[:-1]]
        delta = np.vstack([np.zeros((1, 12)), np.diff(current['arm'][:, :12], axis=0)])
        delta[first] = 0
        snapshot = np.c_[current['contact'], current['fingers'], current['arm'][:, :12], delta, current['arm'][:, 12:], phys]
        with np.load(RICH / f'{video}.npz') as archive:
            rich = {k: archive[k] for k in ('touching', 'regions22', 'regions32', 'regions64', 'events', 'geometry', 'fingers', 'arm')}
            for row in group.itertuples():
                if row.partition == 'train' and row.clip not in selected:
                    continue
                i = int(row.frame_idx)
                if not rich['touching'][i, 0].any():
                    continue
                records.append({'video_id': video, 'frame_idx': i, 'clip': row.clip, 'subject': row.subject,
                                'label': label[row.clip], 'partition': row.partition,
                                'label_filled': not isinstance(row.held_category, str)})
                for g in (22, 32, 64):
                    contact = np.r_[rich[f'regions{g}'][i, 0], rich['events'][i, 0], rich['geometry'][i, 0]]
                    features[f'contact{g}'].append(contact)
                    features[f'body{g}'].append(np.r_[contact, rich['fingers'][i, 0], rich['arm'][i, 0], phys[i]])
                features['current'].append(snapshot[i])
                current_vectors.append(current['contact'][i])
                current_bits.append(bits[i])
    rows = pd.DataFrame(records)
    rows.to_csv(OUT / 'probe_rows.csv', index=False)
    categories = sorted(rows[rows.partition == 'train'].label.unique())
    y = rows.label.map({name: i for i, name in enumerate(categories)}).fillna(-1).to_numpy(int)
    path = OUT / 'probe_features.npz'
    np.savez_compressed(path, **{k: np.array(v, np.float64) for k, v in features.items()}, y=y,
                        clip=rows['clip'].to_numpy(str), subject=rows.subject.to_numpy(str), train=rows.partition.eq('train'),
                        current_patches=np.array(current_vectors), current_bits=np.array(current_bits), classes=np.array(categories))
    record('probe_population', {'train_rows': int(rows.partition.eq('train').sum()), 'dev_rows': int(rows.partition.eq('dev').sum()),
                              'train_clips': int(rows[rows.partition == 'train']['clip'].nunique()),
                              'labels_filled_from_clip_assignment': int(rows.label_filled.sum()),
                              'no_test_or_s8': not rows.partition.isin(['test', 's8']).any()})
    return path


def probes(workers=8):
    started = time.monotonic()
    path = diagnostic_data()
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    train, dev, y = arrays['train'], ~arrays['train'], arrays['y']
    k = len(arrays['classes'])
    results = {'granularities': {}, 'LOSO': {}, 'all_body': {}, 'C_grid': CS,
               'standardization': 'Training-fold mean/std only; intercept unpenalized; C via mean grouped-fold accuracy',
               'unknown_dev_labels': 'Always wrong, log loss -log(1e-15)', 'train_classes': arrays['classes'].tolist()}
    with mp.get_context('spawn').Pool(workers) as pool:
        for g in (22, 32, 64):
            feature = f'contact{g}'
            c, cv = cv_select(pool, path, feature)
            model = fit_logistic(arrays[feature][train], y[train], c, k, granularity=feature_granularity(feature))
            results['granularities'][str(g)] = {**score_logistic(model, arrays[feature][dev], y[dev]), 'C': c, 'CV': cv}
            print('DEV', g, json.dumps(results['granularities'][str(g)]), flush=True)
        best = max(x['accuracy'] for x in results['granularities'].values())
        chosen = min(int(g) for g, x in results['granularities'].items() if x['accuracy'] >= best - .01 - 1e-12)
        results['chosen_G'] = chosen
        (OUT / 'probes.json').write_text(json.dumps(results, indent=2) + '\n')
        # Freeze G before any secondary diagnostic; no test or s8 scores exist.
        (RICH / 'selection.json').write_text(json.dumps({'granularity': chosen, 'rule': 'Smallest G within 0.01 dev accuracy of best',
                    'results': {g: {k: x[k] for k in ('accuracy', 'log_loss', 'C')} for g, x in results['granularities'].items()}}, indent=2) + '\n')
        for feature in ('current', f'body{chosen}'):
            c, cv = cv_select(pool, path, feature)
            model = fit_logistic(arrays[feature][train], y[train], c, k, granularity=feature_granularity(feature))
            results['all_body'][feature] = {**score_logistic(model, arrays[feature][dev], y[dev]), 'C': c, 'CV': cv}
        for g in (22, 32, 64):
            folds = []
            for subject in sorted(set(arrays['subject'][train])):
                feature = f'contact{g}'
                c, cv = cv_select(pool, path, feature, subject)
                fit = train & (arrays['subject'] != subject)
                val = train & (arrays['subject'] == subject)
                model = fit_logistic(arrays[feature][fit], y[fit], c, k, granularity=feature_granularity(feature))
                folds.append({'subject': subject, 'C': c, **score_logistic(model, arrays[feature][val], y[val]), 'CV': cv})
                print('LOSO', g, subject, folds[-1]['accuracy'], flush=True)
                results['LOSO'][str(g)] = folds
                (OUT / 'probes.json').write_text(json.dumps(results, indent=2) + '\n')
    results['LOSO_summary'] = {g: {'macro_accuracy': float(np.mean([x['accuracy'] for x in folds])),
                                       'row_weighted_accuracy': sum(x['accuracy'] * x['rows'] for x in folds) / sum(x['rows'] for x in folds),
                                       'row_weighted_log_loss': sum(x['log_loss'] * x['rows'] for x in folds) / sum(x['rows'] for x in folds)}
                               for g, folds in results['LOSO'].items()}
    results['seconds'] = time.monotonic() - started
    (OUT / 'probes.json').write_text(json.dumps(results, indent=2) + '\n')
    examples(arrays, chosen)
    event_visibility()
    record('granularity', {'chosen': chosen, 'results': {g: {k: x[k] for k in ('accuracy', 'log_loss', 'C')} for g, x in results['granularities'].items()}})


def examples(arrays, chosen):
    from nursery.learner.body import patch_matrix
    sizes = np.count_nonzero(patch_matrix(), axis=0)
    rows = pd.read_csv(OUT / 'probe_rows.csv')
    seen = {}
    out = []
    used = set()
    for i in np.flatnonzero(arrays['train']):
        patches = arrays['current_patches'][i]
        # Same integer patch counts give exactly equal unrounded patch averages.
        key = np.rint(patches * sizes).astype(np.uint16).tobytes()
        bits = arrays['current_bits'][i].tobytes()
        if key in seen:
            j, previous_bits = seen[key]
            if bits != previous_bits and key not in used:
                used.add(key)
                for ix in (j, i):
                    item = rows.iloc[ix].to_dict()
                    item.update(example=len(used), current_patch_values=json.dumps(arrays['current_patches'][ix].tolist()),
                                current_patch_max_difference=float(np.abs(arrays['current_patches'][i] - arrays['current_patches'][j]).max()),
                                rich_region_values=json.dumps(arrays[f'contact{chosen}'][ix, :chosen].tolist()),
                                rich_event_values=json.dumps(arrays[f'contact{chosen}'][ix, chosen:chosen+4].tolist()),
                                rich_geometry_values=json.dumps(arrays[f'contact{chosen}'][ix, chosen+4:].tolist()),
                                differing_snapshot_bits=int(np.unpackbits(arrays['current_bits'][i] ^ arrays['current_bits'][j])[:1556].sum()))
                    out.append(item)
                if len(used) == 5:
                    break
        else:
            seen[key] = (i, bits)
    pd.DataFrame(out).to_csv(OUT / 'contact_examples.csv', index=False)
    record('examples', {'pairs': len(used), 'file': 'contact_examples.csv', 'current_patch_max_difference': max((x['current_patch_max_difference'] for x in out), default=None)})


def event_visibility():
    frames, _ = tables()
    frames = frames[frames.partition == 'dev']
    chosen = json.loads((RICH / 'selection.json').read_text())['granularity']
    counts = {'either_hand': [0, 0, 0], 'per_hand': [0, 0, 0]}
    for video, group in frames.groupby('video_id'):
        with np.load(RICH / f'{video}.npz') as rich, np.load(EPISODES / 'body' / f'{video}.npz') as current:
            snapshot = np.unpackbits(current['hand_contact'], axis=1)[:, :1556].reshape(120, 2, 778).any(2)
            # Consecutive exported rows of one source clip; never join replay cuts.
            run = (group['clip'].ne(group['clip'].shift()) | group.frame_idx.diff().ne(1)).cumsum()
            for _, clip_rows in group.groupby(run):
                seconds = clip_rows.frame_idx.to_numpy()
                sf = rich['source_frames'][seconds]
                bits = np.unpackbits(rich['contact_bits'][seconds], axis=2)[..., :1556].reshape(len(seconds), 30, 2, 778).any(3)
                valid = sf >= 0
                indices, owner, contact = sf[valid], np.repeat(seconds, 30)[valid.ravel()], bits[valid]
                order = np.argsort(indices)
                indices, owner, contact = indices[order], owner[order], contact[order]
                centres = rich['source_frames'][seconds, 15]
                for name, signal in [('either_hand', contact.any(1)[:, None]), ('per_hand', contact)]:
                    for hand in range(signal.shape[1]):
                        x = signal[:, hand]
                        starts = np.flatnonzero(np.diff(x.astype(int)) == 1) + 1
                        ends = np.flatnonzero(np.diff(x.astype(int)) == -1) + 1
                        for start in starts:
                            later = ends[ends > start]
                            if not len(later):
                                continue
                            end = int(later[0])
                            if (indices[end] - indices[start]) / 120 >= 1:
                                continue
                            counts[name][0] += 1
                            in_episode = (centres >= indices[start]) & (centres < indices[end])
                            present = snapshot[seconds].any(1) if name == 'either_hand' else snapshot[seconds, hand]
                            counts[name][1] += int((in_episode & present).any())
                            # Every sampled contact frame contributes a nonzero region fraction to its owner's full bin.
                            trace_rows = np.unique(owner[start:end])
                            region_trace = rich[f'regions{chosen}'][trace_rows, 0].reshape(-1, 2, chosen//2).any(2)
                            channels = region_trace.any(1) if name == 'either_hand' else region_trace[:, hand]
                            counts[name][2] += int(channels.any())
    result = {k: {'short_episodes': n, 'current_visible': c, 'rich_visible': r,
                  'current_share': c/n if n else None, 'rich_share': r/n if n else None} for k, (n, c, r) in counts.items()}
    result['chosen_G'] = chosen
    result['definition'] = 'Dev only, per replay occurrence, fully observed episodes <1s; actual source-index durations; boundary-censored episodes excluded; legacy lattice phase changes preserved.'
    record('event_visibility', result)


def learner_checks():
    import torch
    from nursery.learner import body, _upstream_module
    from nursery.learner.windows import BodyData
    from nursery.learner.additive import AdditiveMIL

    torch.set_num_threads(1)
    frames, selected = tables()
    # No test/s8 features inform fitting, smoke data selection, compute comparisons or probes.
    ids = sorted(frames[frames.partition.isin(['train', 'dev'])].video_id.unique())
    include = np.zeros((len(ids), 120), bool)
    for s, video in enumerate(ids):
        group = frames[frames.video_id == video]
        include[s, group.frame_idx.to_numpy()] = group['clip'].isin(selected) & group.partition.eq('train')
    current = body.load_store(EPISODES, frames, ids, set(), exclude=~include, physical=EPISODES/'physical_by_clip')
    rich = body.load_store(EPISODES, frames, ids, set(), exclude=~include, representation='rich')
    np.testing.assert_array_equal(rich['fit_mask'], include)
    for key in ('load', 'skin'):
        np.testing.assert_array_equal(current['groups'][key], rich['groups'][key])
    for values in rich['variants'].values():
        assert np.isfinite(values).all()
    # Statistics independently reconstructed from selected full-bin raw features.
    raw_fingers = []
    raw_arm = []
    raw_events = []
    raw_geometry = [[], []]
    for s, video in enumerate(ids):
        with np.load(RICH/f'{video}.npz') as data:
            raw_fingers.append(data['fingers'][include[s], 0])
            raw_arm.append(data['arm'][include[s], 0])
            raw_events.append(data['events'][include[s], 0])
            for hand in range(2):
                positive = include[s] & data['touching'][:, 0, hand]
                raw_geometry[hand].append(data['geometry'][positive, 0, 11*hand:11*(hand+1)])
                block = rich['variants']['contact'][s, :, :, rich['granularity']+4+11*hand:rich['granularity']+4+11*(hand+1)]
                np.testing.assert_array_equal(block[~data['touching'][..., hand]], 0)
    for name, rows in [('fingers', raw_fingers), ('arm', raw_arm), ('events', raw_events),
                       ('geometry_0', raw_geometry[0]), ('geometry_1', raw_geometry[1])]:
        x = np.concatenate(rows)
        np.testing.assert_allclose(rich['stats'][f'rich_{name}'][0], x.mean(0), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(rich['stats'][f'rich_{name}'][1], x.std(0)+1e-6, rtol=2e-5, atol=2e-6)
    input_stats = {k: rich['stats'][v] for k, v in
                   {'events': 'rich_events', 'geometry_left': 'rich_geometry_0', 'geometry_right': 'rich_geometry_1',
                    'fingers': 'rich_fingers', 'arm': 'rich_arm', 'load': 'load', 'skin': 'skin'}.items()}
    (OUT / 'normalization.json').write_text(json.dumps({'granularity': rich['granularity'], 'stats': input_stats,
                   'selected_clips': sorted(selected), 'video_ids': ids, 'fit_rows': int(include.sum()),
                   'fit_mask_sha256': hashlib.sha256(include.tobytes()).hexdigest(),
                   'region_fractions': 'raw [0,1]; no standardization'}, indent=2) + '\n')
    record('normalization', {'fit_rows': int(include.sum()), 'fit_clips': len(selected), 'selected_train_only': True,
                             'load_skin_identical_to_current': True, 'all_rich_stats_independently_matched': True, 'absent_geometry_zero_after_normalization': True,
                             'geometry_fit': 'contact-positive selected full bins per hand; absence zeroed after normalization'})
    # Real cached visual/text inputs, train examples only, three additive CPU optimizer steps.
    man = pd.read_parquet(MATRIX/'subsets/1.0/manifest.parquet').iloc[:8].reset_index(drop=True)
    vocab = json.loads((MATRIX/'subsets/1.0/vocab.json').read_text())
    upstream = _upstream_module('train_frame_mil')
    source = ROOT/'outputs/features/child_gaze/world16'
    caches, lut = upstream.load_multi([source/'train_cache'])
    ds = upstream.FramePairs(caches, lut, man, vocab, 2, False)
    idx, v, text, n = upstream.collate([ds[i] for i in range(len(ds))])
    ps = np.array([rich['row'][r.video_id] for r in man.itertuples()])
    pt = man.frame_idx.to_numpy(copy=True)
    data = BodyData(rich, ps, pt, pt[:, None], None, None, None, None, 'cpu', dropout=0.)
    torch.manual_seed(0)
    model = AdditiveMIL(len(vocab), v.shape[-1], dims=data.dims, num_regions=257)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(3):
        ctx = data.batch(idx, torch.Generator().manual_seed(0), training=True)
        opt.zero_grad()
        loss = model.forward_loss(v, text, n, torch.ones(len(idx)), ctx)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None) > 0
        opt.step()
        losses.append(float(loss.detach()))
    record('smoke', {'device': 'cpu', 'path': 'existing AdditiveMIL, real cached train visual/text/body inputs, no gaze',
                     'steps': 3, 'losses': losses, 'encoder_gradients_finite_nonzero': True})
    compute = {}
    for name, store in [('current', current), ('rich', rich)]:
        dims = {k: x.shape[-1] for k, x in store['groups'].items()}
        encoder = body.BodyEncoder(dims)
        whole = AdditiveMIL(len(vocab), v.shape[-1], dims=dims, num_regions=257)
        input_width = 64*len(dims)+1
        macs = 11*(64*sum(dims.values()) + len(dims)*64*64 + 2*3*64*(input_width+64))
        example = {k: torch.from_numpy(x[ps[:, None], np.clip(pt[:, None]+np.arange(-5,6),0,119)]) for k,x in store['groups'].items()}
        missing = torch.zeros(len(ps),11)
        seg = torch.zeros(len(ps),11,dtype=torch.long)
        with torch.no_grad():
            for _ in range(3): encoder(example,missing,seg)
            t = time.monotonic()
            for _ in range(20): encoder(example,missing,seg)
        encoder_seconds = (time.monotonic()-t)/20/len(ps)
        measure_data = BodyData(store, ps, pt, pt[:, None], None, None, None, None, 'cpu', dropout=0.)
        ctx = measure_data.batch(idx, torch.Generator().manual_seed(0), training=False)
        whole.eval()
        with torch.no_grad():
            for _ in range(2): whole.forward_loss(v, text, n, torch.ones(len(idx)), ctx)
            tick = time.monotonic()
            for _ in range(5): whole.forward_loss(v, text, n, torch.ones(len(idx)), ctx)
        full_seconds = (time.monotonic()-tick)/5/len(ps)
        dim = whole.word.embedding_dim
        full_macs = macs + v.shape[1]*v.shape[-1]*dim + v.shape[1]*dim*len(ps) + 128*dim + dim*len(ps)
        compute[name] = {'dims': dims, 'total_input_dims': sum(dims.values()),
                         'encoder_parameters': sum(p.numel() for p in encoder.parameters()),
                         'additive_parameters': sum(p.numel() for p in whole.parameters()),
                         'encoder_MACs_per_11_step_example': macs,
                         'encoder_forward_seconds_per_example_batch8': encoder_seconds,
                         'additive_match_dense_MACs_per_example_batch8': full_macs,
                         'additive_match_forward_seconds_per_example_batch8': full_seconds}
    compute['convention'] = 'MACs count dense matrix multiplies only, exclude normalization/nonlinearities/embedding pooling; full match uses 5x257 visual regions and 8 text candidates. CPU timings use batch 8. Same width=64 hidden=64; full parameter counts include unchanged visual/text modules and auxiliary heads.'
    record('compute', compute)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export', 'probes', 'learner'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8')
    if args.action == 'export': check_export()
    elif args.action == 'probes': probes(args.workers)
    else: learner_checks()


if __name__ == '__main__':
    main()
