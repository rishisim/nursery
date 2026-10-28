import json

import pandas as pd
import pytest

from nursery.learner.cross_situational import CATEGORIES, eligible_categories, sample_rows


def test_visibility_proxy_requires_size_and_central_cell():
    objects = [dict(category='big', visible_px=100, cells=[5]),
               dict(category='small', visible_px=99, cells=[5]),
               dict(category='edge', visible_px=500, cells=[0])]
    assert eligible_categories(json.dumps(objects)) == {'big'}


def test_balanced_sampling_counts_replays_across_held_status():
    rows = []
    for part, clips in [('train', 20), ('dev', 5), ('test', 5)]:
        for category in CATEGORIES:
            for clip in range(clips):
                for held in (True, False):
                    for i in range(4):
                        rows.append(dict(partition=part, referent=category, held=held,
                                         clip=f'{part}-{clip}', frame_idx=i))
    selected = sample_rows(pd.DataFrame(rows))
    assert selected.groupby(['partition', 'referent', 'held']).size().tolist() == [8] * 16 + [32] * 8
    assert selected.groupby(['partition', 'referent', 'clip']).size().max() <= 4
    assert selected.equals(sample_rows(pd.DataFrame(rows)))


def test_insufficient_distinct_clips_fails():
    with pytest.raises(ValueError, match='Insufficient'):
        sample_rows(pd.DataFrame([dict(partition='train', referent='bell', held=True, clip='x')]))


def test_panel_rejects_duplicate_situation_and_foil_clip_excess():
    from nursery.learner.cross_situational import validate_panels
    rows = pd.DataFrame([
        dict(partition='dev', referent='bell', clip='a', video_id='v', frame_idx=1),
        dict(partition='dev', referent='tube', clip='b', video_id='v', frame_idx=1),
        dict(partition='dev', referent='tube', clip='c', video_id='v', frame_idx=2),
        dict(partition='dev', referent='tube', clip='d', video_id='v', frame_idx=3),
    ])
    with pytest.raises(ValueError, match='repeats a physical situation'):
        validate_panels(rows, {'dev': [{'candidates': [0, 1, 2, 3]}], 'test': []})
    repeated = pd.concat([rows.iloc[[0]]] * 5, ignore_index=True)
    with pytest.raises(ValueError, match='More than four'):
        validate_panels(repeated, {'dev': [], 'test': []})


def test_prepare_refuses_overwrite_before_reading_inputs(tmp_path):
    from nursery.learner.cross_situational import prepare
    (tmp_path / 'panels.json').write_text('frozen')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        prepare('/missing/episodes', '/missing/cache', tmp_path)
    assert (tmp_path / 'panels.json').read_text() == 'frozen'


def test_report_requires_complete_matching_runs_and_reports_sample_sd(tmp_path, monkeypatch):
    import hashlib
    from nursery.learner import cross_situational as cs

    monkeypatch.setattr(cs, '_output_directory', lambda p: (p.mkdir(parents=True, exist_ok=True) or p))
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps({'counts': [], 'foil_counts': {}, 'example_encounters': [
        {'video_id': 'a', 'frame_idx': 1, 'referent': 'bell'},
        {'video_id': 'b', 'frame_idx': 2, 'referent': 'bell'},
        {'video_id': 'c', 'frame_idx': 3, 'referent': 'bell'}]}))
    (tmp_path / 'manifest.parquet').write_bytes(b'manifest')
    (tmp_path / 'panels.json').write_text('{}')
    hashes = {f'study_{name}_sha256': hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
              for name, filename in [('protocol', 'protocol.json'), ('manifest', 'manifest.parquet'), ('panels', 'panels.json')]}
    paths = []
    for body, scores in [('none', [20, 30, 40]), ('real', [40, 50, 60])]:
        for seed, score in enumerate(scores):
            key = 'picture_soft' if body == 'none' else 'fused_soft'
            m = {**hashes, 'body': body, 'seed': seed, 'reported_test_acc': score,
                 'test': {'n_trials': 64, key: {'acc': score}},
                 'shuffled_test': {'n_trials': 64, 'fused_soft': {'acc': score - 10}} if body == 'real' else None}
            path = tmp_path / f'{body}-{seed}.json'
            path.write_text(json.dumps(m)); paths.append(path)
    result = cs.report(protocol_path, paths, tmp_path / 'report')
    assert result['video_only'] == {'mean': 30.0, 'sample_sd': 10.0}
    assert result['additive'] == {'mean': 50.0, 'sample_sd': 10.0}
    assert result['shuffled_additive'] == {'mean': 40.0, 'sample_sd': 10.0}
    assert len(result['example_encounters']) == 3
    with pytest.raises(ValueError, match='all three paired seeds'):
        cs.report(protocol_path, paths[:-1], tmp_path / 'incomplete')
    (tmp_path / 'panels.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='do not match'):
        cs.report(protocol_path, paths, tmp_path / 'mismatch')
    assert not (tmp_path / 'incomplete').exists()
    assert not (tmp_path / 'mismatch').exists()


def test_report_nine_matched_runs_checks_settings_panels_and_reports_sd(tmp_path, monkeypatch):
    import hashlib
    from nursery.learner import cross_situational as cs

    monkeypatch.setattr(cs, '_output_directory', lambda p: (p.mkdir(parents=True, exist_ok=True) or p))
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps({'counts': [], 'foil_counts': {}, 'example_encounters': []}))
    (tmp_path / 'manifest.parquet').write_bytes(b'manifest')
    frozen_trials = [{'candidates': [i, i + 64, i + 128, i + 192], 'correct': i % 4} for i in range(64)]
    (tmp_path / 'panels.json').write_text(json.dumps({'test': frozen_trials}))
    hashes = {f'study_{name}_sha256': hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
              for name, filename in [('protocol', 'protocol.json'), ('manifest', 'manifest.parquet'), ('panels', 'panels.json')]}
    settings = {'epochs': 20, 'batch': 32, 'lr': 0.0003, 'weight_decay': 0.1, 'window': 2, 'min_freq': 1}
    configs = [
        ('none', 'additive', 'picture_soft', [16, 32, 48]),
        ('real', 'additive', 'fused_soft', [20, 40, 50]),
        ('real', 'interaction', 'fused_soft', [24, 36, 52]),
    ]
    paths = []
    scores_by_mode = {}
    for body, arch, score_key, counts in configs:
        name = 'video_only' if body == 'none' else arch
        for seed, successes in enumerate(counts):
            score = round(100 * successes / 64, 3)
            flags = [int(i < successes) for i in range(64)]
            history = [{'ep': i, 'dev': 0.5} for i in range(20)]
            history[3]['dev'] = 0.75
            history[7]['dev'] = 0.75  # earliest tied maximum must be selected
            test = {'n_trials': 64, 'candidate_rows': [t['candidates'] for t in frozen_trials],
                    'correct_slots': [t['correct'] for t in frozen_trials], 'trials': {score_key: flags},
                    score_key: {'acc': score}}
            shuffled = None
            if body == 'real':
                sh_successes = successes - (2 + seed)
                sh_flags = [int(i < sh_successes) for i in range(64)]
                shuffled = {'n_trials': 64, 'candidate_rows': test['candidate_rows'],
                            'correct_slots': test['correct_slots'], 'trials': {'fused_soft': sh_flags},
                            'fused_soft': {'acc': round(100 * sh_successes / 64, 3)}}
            metrics = {**hashes, 'condition': 'none', 'body': body, 'arch': arch, 'seed': seed, 'device': 'cpu',
                       'settings': settings, 'n_pairs': 256, 'emb_dim': 512, 'pool': 'soft', 'tau_p': 0.1,
                       'body_dropout': 0.5,
                       'weights': {'pred': 0, 'episode': 0, 'consist': 0, 'mask': 0, 'bodymatch': 0},
                       'history': history, 'selected_epoch': 3, 'test': test, 'shuffled_test': shuffled,
                       'reported_test_acc': score}
            path = tmp_path / f'{name}-{seed}.json'
            path.write_text(json.dumps(metrics))
            paths.append(path)
            scores_by_mode[body, arch, seed] = score
    result = cs.report(protocol_path, paths, tmp_path / 'report')
    assert result['video_only'] == {'mean': 50.0, 'sample_sd': 25.0}
    assert result['additive'] == pytest.approx({'mean': 57.2916666667, 'sample_sd': 23.8675817446})
    assert result['interaction'] == pytest.approx({'mean': 58.3333333333, 'sample_sd': 21.9492786518})
    assert result['shuffled_additive'] == pytest.approx({'mean': 52.604, 'sample_sd': 22.3351309152})
    assert result['shuffled_interaction'] == pytest.approx({'mean': 53.6456666667, 'sample_sd': 20.3924963242})
    assert set(result) >= {'shuffled_additive', 'shuffled_interaction', 'per_seed'}
    assert (tmp_path / 'report' / 'results.md').read_text().count('Interaction video + body') == 1

    # Every failure below must happen before a report is created.
    bad_path = tmp_path / 'bad-device.json'
    bad = json.loads(paths[0].read_text()); bad['device'] = 'mps'; bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='requires CPU'):
        cs.report(protocol_path, [bad_path, *paths[1:]], tmp_path / 'bad-device-report')
    bad = json.loads(paths[0].read_text()); bad['test']['candidate_rows'][0][0] += 999
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='fixed test candidates'):
        cs.report(protocol_path, [bad_path, *paths[1:]], tmp_path / 'bad-panel-report')
    bad = json.loads(paths[-1].read_text()); bad['settings']['extra_setting'] = 1
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='identical settings'):
        cs.report(protocol_path, [*paths[:-1], bad_path], tmp_path / 'bad-settings-report')
    bad = json.loads(paths[0].read_text()); bad['study_panels_sha256'] = 'wrong'
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='frozen study inputs'):
        cs.report(protocol_path, [bad_path, *paths[1:]], tmp_path / 'bad-hash-report')
    assert not (tmp_path / 'bad-device-report').exists()
    assert not (tmp_path / 'bad-panel-report').exists()
    assert not (tmp_path / 'bad-settings-report').exists()
    assert not (tmp_path / 'bad-hash-report').exists()
