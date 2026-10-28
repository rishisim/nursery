"""Freeze source-clip acquisition subsets and evaluation panels before training."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nursery.common import _output_directory
from nursery.learner import _upstream_module, body, evaluate, train

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_keys(frames):
    if frames.duplicated(['video_id', 'frame_idx']).any():
        raise ValueError('Duplicate frame metadata')
    if frames[['clip', 'subject']].isna().any().any() or (frames['clip'] == '').any():
        raise ValueError('All frames require a source clip and subject')
    if (frames.groupby('clip').subject.nunique() != 1).any():
        raise ValueError('Source clip identity is shared across subjects')
    return frames['clip'].astype(str)


def _panel(rows, frames, lut):
    rows = rows[rows.named.astype(bool) & rows.referent_visible.astype(bool)].copy().reset_index(drop=True)
    rows['text'] = rows.referent
    if len(rows) < 8:
        raise ValueError('Too few body-present panel rows')
    context = evaluate.body_present_context(rows, frames, lut, 2, 5)
    trials = evaluate._body_trials(rows, context, 0, None)
    if not trials:
        raise ValueError('No feasible body-present trials')
    frame_key = frames.set_index(['video_id', 'frame_idx'])['clip']
    clips = [frame_key.loc[(r.video_id, int(r.frame_idx))] for r in rows.itertuples()]
    rng = np.random.default_rng(0)
    donors = []
    for i, subject in enumerate(context['subjects']):
        choices = [j for j, (s, c) in enumerate(zip(context['subjects'], clips)) if s == subject and c != clips[i]]
        if not choices:
            raise ValueError(f'No same-subject, different-source donor for row {i}')
        donors.append(int(rng.choice(choices)))
    return rows, {'trial_rows': [int(i) for i, _, _ in trials],
                  'candidate_rows': [c.tolist() for _, c, _ in trials],
                  'correct_slots': [a for _, _, a in trials], 'donor_rows': donors,
                  'donor_rule': 'seed 0, same panel and subject, different source clip, sampled with replacement',
                  'definition': context['definition']}



def _picture_panel(rows, index):
    lookup = {(r.video_id, int(r.frame_idx)): int(r.row) for r in index.itertuples()}
    pools = {}
    for cat, group in rows.groupby('category'):
        candidates = [lookup[(r.video_id, int(r.frame_idx))] for r in group.itertuples()
                      if (r.video_id, int(r.frame_idx)) in lookup]
        if len(candidates) >= 4:
            pools[cat] = candidates
    cats = sorted(pools)
    if len(cats) < 4:
        raise ValueError('Picture panel requires four categories with four cached images each')
    rng = np.random.default_rng(0)
    result = []
    for cat in cats:
        others = [c for c in cats if c != cat]
        for _ in range(100):
            candidates = [int(rng.choice(pools[cat]))] + [int(rng.choice(pools[c])) for c in rng.choice(others, 3, replace=False)]
            order = rng.permutation(4)
            result.append({'query': cat, 'candidate_rows': [candidates[i] for i in order],
                           'correct_slot': int(np.flatnonzero(order == 0)[0])})
    return {'seed': 0, 'cache_index_rows': True, 'trials': result,
            'sampling': '100 per category; three distinct foil categories; candidates frozen before vocabulary'}

def prepare(root, inputs, episodes):
    """Save fixed panels, clip-disjoint nested subsets, vocabularies and fit masks.

    fit_mask.npz contains video_ids and include; pass ~include as load_store's exclude.
    Subset manifest contains only vocabulary-bearing examples after fitting vocabulary.
    """
    root, inputs, episodes = map(lambda x: Path(x).resolve(), (root, inputs, episodes))
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty preparation root: {root}')
    frames = pd.read_parquet(episodes / 'frames.parquet')
    frames['clip'] = _source_keys(frames)
    sessions = pd.read_parquet(episodes / 'sessions.parquet')
    subject = dict(zip(sessions.video_id, sessions.subject))
    if frames.video_id.map(subject).isna().any() or not (frames.video_id.map(subject) == frames.subject).all():
        raise ValueError('Frame and session subjects disagree')
    manifest = pd.read_parquet(inputs / 'manifest.parquet').reset_index(drop=True)
    index = pd.read_parquet(inputs / 'train_cache/index.parquet')
    if index.duplicated(['video_id', 'frame_idx']).any():
        raise ValueError('Duplicate cache frame')
    lut = {f'{r.video_id}|{int(r.frame_idx)}': (0, int(r.row)) for r in index.itertuples()}
    if any(f'{r.video_id}|{int(r.frame_idx)}' not in lut for r in manifest.itertuples()):
        raise ValueError('Speech manifest is not cache-covered')
    part, counts = train.clip_partition(frames, 's8', {'seed': 0, 'dev': .15, 'test': .15})
    labels = train.split_pairs(manifest, part, 2)
    held = manifest.video_id.map(subject).eq('s8').to_numpy()
    # A global source key must never span partitions, including the held-out subject.
    frame_split = frames.apply(lambda r: 's8' if r.subject == 's8' else ('train', 'dev', 'test')[part[r.video_id][int(r.frame_idx)]], axis=1)
    membership = pd.DataFrame({'clip': frames['clip'], 'split': frame_split}).drop_duplicates()
    if membership.groupby('clip').split.nunique().max() != 1:
        raise ValueError('Source clip crosses partitions')
    source_map = frames.set_index(['video_id', 'frame_idx'])['clip']
    speech_clip = [source_map.loc[(r.video_id, int(r.frame_idx))] for r in manifest.itertuples()]
    # A selected example's entire cached visual window must belong to selected source clips.
    windows = []
    for r in manifest.itertuples():
        windows.append({source_map.loc[(r.video_id, f)] for f in range(int(r.frame_idx)-2, int(r.frame_idx)+3)
                        if f'{r.video_id}|{f}' in lut})
    base = (~held) & (labels == 0) & manifest.text.map(lambda x: isinstance(x, str) and bool(x.strip())).to_numpy()
    eligible = sorted({speech_clip[i] for i in np.flatnonzero(base)})
    if not eligible:
        raise ValueError('No eligible training clips')
    ordered = np.random.default_rng(0).permutation(eligible).tolist()
    upstream = _upstream_module('train_frame_mil')
    panels = {}
    for name, mask in [('clips_dev', (~held) & (labels == 1)), ('clips_test', (~held) & (labels == 2)), ('s8', held)]:
        rows, trials = _panel(manifest[mask], frames, lut)
        panels[name] = (rows, trials)
    provenance_files = [inputs / n for n in ('train_cache/emb.f16.npy', 'probes_cache/emb.f16.npy', 'konkle_cache/emb.f16.npy', 'manifest.parquet', 'generator.json', 'embedding.json', 'preserved_inputs.json', 'train_cache/index.parquet', 'probes_cache/index.parquet', 'konkle_cache/index.parquet', 'probes_frames.parquet', 'konkle_frames.parquet')]
    provenance_files += [episodes / 'frames.parquet', episodes / 'sessions.parquet']
    preserved = json.loads((inputs / 'preserved_inputs.json').read_text())
    for p, digest in preserved.get('sha256', {}).items():
        path = Path(p)
        if path.exists() and _sha(path) != digest:
            raise ValueError(f'Preserved input hash mismatch: {path}')
    for name, digest in preserved.get('run_input_sha256', {}).items():
        if _sha(inputs / name) != digest:
            raise ValueError(f'Preserved run input hash mismatch: {name}')
    for filename, expected in preserved.get('source_file_stats', {}).items():
        path = Path(filename)
        stat = path.stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Preserved source file metadata mismatch: {path}')
    signal_hashes = {}
    for video in sorted(frames.video_id.unique()):
        for directory in ('body', 'physical_by_clip'):
            path = episodes / directory / f'{video}.npz'
            signal_hashes[str(path)] = _sha(path)
    root = _output_directory(root)
    _write(root / 'signal_hashes.json', signal_hashes)
    _write(root / 'splits.json', {'counts': counts, 'source_keys': {s: sorted(g['clip'].tolist()) for s, g in membership.groupby('split')}, 'eligible_order_seed_0': ordered})
    for name, (rows, trials) in panels.items():
        directory = root / 'panels' / name
        directory.mkdir(parents=True)
        rows.to_parquet(directory / 'manifest.parquet', index=False)
        _write(directory / 'trials.json', trials)
    probe_rows = pd.read_parquet(inputs / 'probes_frames.parquet')
    picture_specs = [('picture_dev', probe_rows[probe_rows['split'] == 'dev'], 'probes_cache'),
                     ('picture_test', probe_rows[probe_rows['split'] == 'test'], 'probes_cache'),
                     ('konkle_test', pd.read_parquet(inputs / 'konkle_frames.parquet'), 'konkle_cache')]
    picture_panels = {}
    for name, rows, cache in picture_specs:
        directory = root / 'panels' / name
        directory.mkdir(parents=True)
        panel = _picture_panel(rows, pd.read_parquet(inputs / cache / 'index.parquet'))
        _write(directory / 'trials.json', panel)
        picture_panels[name] = (panel, cache)
    video_ids = sorted(frames.video_id.unique())
    steps = max(int(frames.frame_idx.max()) + 1, 120)
    subset_summary = {}
    for fraction in FRACTIONS:
        selected = set(ordered[:max(1, math.ceil(len(ordered) * fraction))])
        keep = np.array([base[i] and windows[i] <= selected for i in range(len(manifest))])
        raw = manifest[keep].copy().reset_index(drop=True)
        vocab = upstream.build_vocab(raw.text, 5)
        usable = raw.text.map(lambda x: bool(upstream.encode(x, vocab)))
        used = raw[usable].reset_index(drop=True)
        if len(used) < 2:
            raise ValueError(f'Insufficient usable speech at {fraction}')
        include = np.zeros((len(video_ids), steps), dtype=bool)
        video_rows = {v: i for i, v in enumerate(video_ids)}
        for r in frames.itertuples():
            if r.clip in selected:
                include[video_rows[r.video_id], int(r.frame_idx)] = True
        store = body.load_store(episodes, frames, video_ids, set(), steps=steps, exclude=~include, physical=episodes / 'physical_by_clip')
        centre = {source_map.loc[(r.video_id, int(r.frame_idx))] for r in used.itertuples()}
        used_indices = np.flatnonzero(keep)[usable.to_numpy()]
        exposed = set().union(*(windows[i] for i in used_indices))
        coverage = {name: {'known': int(sum(bool(upstream.encode(rows.iloc[i].referent, vocab)) for i in trials['trial_rows'])), 'total': len(trials['trial_rows'])} for name, (rows, trials) in panels.items()}
        coverage.update({name: {'known': sum(bool(upstream.encode(t['query'], vocab)) for t in panel['trials']), 'total': len(panel['trials'])} for name, (panel, _) in picture_panels.items()})
        if coverage['clips_dev']['known'] == 0:
            raise ValueError(f'No known development queries at {fraction}')
        summary = {'fraction': fraction, 'selected_clips': sorted(selected), 'selected_clip_count': len(selected),
                   'exposed_clip_count': len(exposed), 'exposed_clips': sorted(exposed), 'centre_clip_count': len(centre), 'utterances_before_vocab': len(raw),
                   'utterances': len(used), 'replay_exposure_seconds': int(include.sum()),
                   'unique_source_duration_seconds': None, 'vocabulary_coverage': coverage, 'normalization': store['stats']}
        directory = root / 'subsets' / str(fraction)
        directory.mkdir(parents=True)
        used.to_parquet(directory / 'manifest.parquet', index=False)
        np.savez_compressed(directory / 'fit_mask.npz', video_ids=np.asarray(video_ids), include=include)
        _write(directory / 'vocab.json', vocab)
        _write(directory / 'stats.json', summary)
        subset_summary[str(fraction)] = {k: summary[k] for k in ('selected_clip_count', 'exposed_clip_count', 'utterances', 'replay_exposure_seconds')}
    protocol = {'inputs': str(inputs), 'episodes': str(episodes), 'physical': str(episodes / 'physical_by_clip'),
                'hashes': {str(p): _sha(p) for p in provenance_files}, 'split_seed': 0, 'subset_seed': 0,
                'seeds': [0, 1, 2], 'fractions': list(FRACTIONS), 'window': 2, 'body_window': 5,
                'epochs': 60, 'batch': 256, 'optimizer': 'AdamW', 'lr': 0.0003, 'weight_decay': 0.1,
                'tau': 0.1, 'body_dropout': 0.5, 'streams': ['contact', 'fingers', 'arm', 'load', 'skin'],
                'auxiliary_losses': 0, 'device': 'mps', 'runtime_cap_hours': 24, 'unknown_query_score': 0.25,
                'picture_panels': {name: {'trials': str(root / 'panels' / name / 'trials.json'), 'cache': str(inputs / cache)} for name, (_, cache) in picture_panels.items()},
                'subsets': subset_summary}
    _write(root / 'protocol.json', protocol)
    return protocol


def audit(root):
    """Reconstruct and verify the frozen preparation, saving a concise audit record."""
    root = Path(root).resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['unknown_query_score'] != 0.25:
        raise ValueError('Unknown query credit must be 0.25')
    inputs, episodes = Path(protocol['inputs']), Path(protocol['episodes'])
    for filename, digest in protocol['hashes'].items():
        if _sha(Path(filename)) != digest:
            raise ValueError(f'Input changed: {filename}')
    for filename, digest in json.loads((root / 'signal_hashes.json').read_text()).items():
        if _sha(Path(filename)) != digest:
            raise ValueError(f'Signal changed: {filename}')
    frames = pd.read_parquet(episodes / 'frames.parquet')
    frames['clip'] = _source_keys(frames)
    manifest = pd.read_parquet(inputs / 'manifest.parquet').reset_index(drop=True)
    idx = pd.read_parquet(inputs / 'train_cache/index.parquet')
    lut = {f'{r.video_id}|{int(r.frame_idx)}': (0, int(r.row)) for r in idx.itertuples()}
    sessions = pd.read_parquet(episodes / 'sessions.parquet')
    subject = dict(zip(sessions.video_id, sessions.subject))
    held = manifest.video_id.map(subject).eq('s8').to_numpy()
    part, _ = train.clip_partition(frames, 's8', {'seed': 0, 'dev': .15, 'test': .15})
    labels = train.split_pairs(manifest, part, 2)
    source = frames.set_index(['video_id', 'frame_idx'])['clip']
    windows = [{source.loc[(r.video_id, f)] for f in range(int(r.frame_idx)-2, int(r.frame_idx)+3)
                if f'{r.video_id}|{f}' in lut} for r in manifest.itertuples()]
    base = (~held) & (labels == 0) & manifest.text.map(lambda x: isinstance(x, str) and bool(x.strip())).to_numpy()
    upstream = _upstream_module('train_frame_mil')
    splits = json.loads((root / 'splits.json').read_text())
    previous = set()
    summary = {}
    for fraction in FRACTIONS:
        directory = root / 'subsets' / str(fraction)
        stats = json.loads((directory / 'stats.json').read_text())
        selected = set(stats['selected_clips'])
        expected = set(splits['eligible_order_seed_0'][:math.ceil(len(splits['eligible_order_seed_0']) * fraction)])
        if selected != expected or not previous <= selected:
            raise ValueError('Subset selection or nesting mismatch')
        previous = selected
        keep = np.array([base[i] and windows[i] <= selected for i in range(len(manifest))])
        raw = manifest[keep].reset_index(drop=True)
        vocab = upstream.build_vocab(raw.text, 5)
        if vocab != json.loads((directory / 'vocab.json').read_text()):
            raise ValueError('Vocabulary does not match subset speech')
        expected_rows = raw[raw.text.map(lambda x: bool(upstream.encode(x, vocab)))].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_rows, pd.read_parquet(directory / 'manifest.parquet'))
        mask = np.load(directory / 'fit_mask.npz')
        ids, include = mask['video_ids'].tolist(), mask['include']
        positions = {v: i for i, v in enumerate(ids)}
        expected_mask = np.zeros_like(include)
        for row in frames.itertuples():
            if row.clip in selected:
                if row.subject == 's8' or part[row.video_id][int(row.frame_idx)] != 0:
                    raise ValueError('Fit mask includes held-out data')
                expected_mask[positions[row.video_id], int(row.frame_idx)] = True
        if not np.array_equal(include, expected_mask):
            raise ValueError('Fit mask mismatch')
        store = body.load_store(episodes, frames, ids, set(), steps=include.shape[1],
                                exclude=~include, physical=episodes / 'physical_by_clip')
        if store['stats'] != stats['normalization']:
            raise ValueError('Normalization does not match fit mask')
        summary[str(fraction)] = {'clips': len(selected), 'utterances': len(expected_rows),
                                  'fit_seconds': int(include.sum())}
    for name, selection in [('clips_dev', (~held) & (labels == 1)), ('clips_test', (~held) & (labels == 2)), ('s8', held)]:
        rows, trials = _panel(manifest[selection], frames, lut)
        directory = root / 'panels' / name
        pd.testing.assert_frame_equal(rows, pd.read_parquet(directory / 'manifest.parquet'))
        if trials != json.loads((directory / 'trials.json').read_text()):
            raise ValueError(f'Panel or donors changed: {name}')
        for fraction in FRACTIONS:
            directory_subset = root / 'subsets' / str(fraction)
            vocab = json.loads((directory_subset / 'vocab.json').read_text())
            stats = json.loads((directory_subset / 'stats.json').read_text())
            known = sum(bool(upstream.encode(rows.iloc[i].referent, vocab)) for i in trials['trial_rows'])
            if stats['vocabulary_coverage'][name] != {'known': known, 'total': len(trials['trial_rows'])}:
                raise ValueError(f'Coverage mismatch: {name}')
    probes = pd.read_parquet(inputs / 'probes_frames.parquet')
    specs = [('picture_dev', probes[probes['split'] == 'dev'], 'probes_cache'),
             ('picture_test', probes[probes['split'] == 'test'], 'probes_cache'),
             ('konkle_test', pd.read_parquet(inputs / 'konkle_frames.parquet'), 'konkle_cache')]
    for name, rows, cache in specs:
        expected = _picture_panel(rows, pd.read_parquet(inputs / cache / 'index.parquet'))
        if expected != json.loads((root / 'panels' / name / 'trials.json').read_text()):
            raise ValueError(f'Picture panel changed: {name}')
        for fraction in FRACTIONS:
            directory = root / 'subsets' / str(fraction)
            vocab = json.loads((directory / 'vocab.json').read_text())
            coverage = json.loads((directory / 'stats.json').read_text())['vocabulary_coverage'][name]
            known = sum(bool(upstream.encode(t['query'], vocab)) for t in expected['trials'])
            if coverage != {'known': known, 'total': len(expected['trials'])}:
                raise ValueError(f'Picture coverage mismatch: {name}')
    result = {'passed': True, 'subsets': summary, 'unknown_query_score': protocol['unknown_query_score']}
    if result['unknown_query_score'] != 0.25:
        raise ValueError('Unknown query credit must be 0.25')
    _output_directory(root)
    _write(root / 'audit.json', result)
    return result
