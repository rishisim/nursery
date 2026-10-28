"""Prepare a fixed ambiguous word dataset from existing rendered sessions.

Category labels in the output manifest are scoring metadata, never model features.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nursery.common import _output_directory

from .evaluate import body_present_context
from .train import clip_partition, split_pairs

CATEGORIES = ('bell', 'bottle', 'glass', 'tube')
WORDS = ('dax', 'blicket', 'wug', 'fep')
CENTRAL = {5, 6, 9, 10}


def eligible_categories(objects):
    """Conservative proxy for crop visibility; coarse cells do not guarantee recognizability."""
    return {o['category'] for o in json.loads(objects)
            if o['visible_px'] >= 100 and CENTRAL.intersection(o['cells'])}


def sample_rows(candidates, seed=0):
    rng = np.random.default_rng(seed)
    selected = []
    for partition, count in [('train', 64), ('dev', 16), ('test', 16)]:
        for category in CATEGORIES:
            clips = Counter()
            for held in (True, False):
                pool = candidates[(candidates.partition == partition) &
                                  (candidates.referent == category) & (candidates.held == held)]
                order = rng.permutation(len(pool))
                chosen = []
                for index in order:
                    row = pool.iloc[index]
                    if clips[row["clip"]] < 4:
                        chosen.append(row)
                        clips[row["clip"]] += 1
                        if len(chosen) == count // 2:
                            break
                if len(chosen) != count // 2:
                    raise ValueError(f'Insufficient {partition} {category} held={held}: {len(chosen)}')
                selected.extend(chosen)
    return pd.DataFrame(selected).reset_index(drop=True)


def validate_panels(selected, panels):
    if selected.groupby(['partition', 'referent', 'clip']).size().max() > 4:
        raise ValueError('More than four rows per source clip and word')
    for partition in ('dev', 'test'):
        rows = selected[selected.partition == partition].reset_index(drop=True)
        for trial in panels[partition]:
            situations = {(rows.iloc[i].video_id, int(rows.iloc[i].frame_idx))
                          for i in trial['candidates']}
            if len(trial['candidates']) != 4 or len(situations) != 4:
                raise ValueError(f'{partition}: trial repeats a physical situation')


def prepare(episodes, cache, output):
    episodes, cache, output = map(Path, (episodes, cache, output))
    if any((output / name).exists() for name in ('manifest.parquet', 'panels.json', 'protocol.json')):
        raise FileExistsError(f'Refusing to overwrite frozen dataset in {output}')
    output = _output_directory(output)
    frames = pd.read_parquet(episodes / 'frames.parquet')
    frames = frames[frames.subject != 's8'].copy().reset_index(drop=True)
    index = pd.read_parquet(cache / 'index.parquet')
    lut = {f'{r.video_id}|{r.frame_idx}': r.row for r in index.itertuples()}
    part, counts = clip_partition(frames, 's8', {'seed': 0, 'dev': .15, 'test': .15})
    frames['part'] = split_pairs(frames, part, 2)
    available = {}
    for video in frames.video_id.unique():
        with np.load(episodes / 'body' / f'{video}.npz') as body, np.load(episodes / 'physical_by_clip' / f'{video}.npz') as physical:
            valid = set(body['frame_idx'].tolist()) & set(physical['frame_idx'].tolist())
            keys = ('hand_contact', 'left_hand_pose', 'right_hand_pose', 'camera', 'joints')
            bmap = {int(t): i for i, t in enumerate(body['frame_idx'])}
            pmap = {int(t): i for i, t in enumerate(physical['frame_idx'])}
            b = {k: body[k] for k in keys}
            p = {k: physical[k] for k in ('load', 'skin_dT', 'contact_area')}
            available[video] = {t for t in valid if all(np.isfinite(a[bmap[t]]).all() for a in b.values())
                                and all(np.isfinite(a[pmap[t]]).all() for a in p.values())}
    rows = []
    for r in frames.itertuples():
        if r.part < 0 or f'{r.video_id}|{r.frame_idx}' not in lut or r.frame_idx not in available[r.video_id]:
            continue
        visible = eligible_categories(r.objects)
        if len(visible) < 2:
            continue
        for category, word in zip(CATEGORIES, WORDS):
            if category in visible:
                rows.append({'video_id': r.video_id, 'frame_idx': r.frame_idx, 'text': f'Look, {word}.',
                             'referent': category, 'partition': ('train', 'dev', 'test')[r.part],
                             'subject': r.subject, 'room': r.room, 'clip': r.clip,
                             'held': r.held_category == category, 'visible': sorted(visible)})
    candidates = pd.DataFrame(rows)
    # All valid clip-bounded body-window steps must exist and be finite, not just the centre.
    lookup = {(r.video_id, r.frame_idx): r.clip for r in frames.itertuples()}
    def body_ok(r):
        for direction in (-1, 1):
            for delta in range(1, 6):
                t = r.frame_idx + direction * delta
                if lookup.get((r.video_id, t)) != r.clip:
                    break
                if t not in available[r.video_id]:
                    return False
        return True
    candidates = candidates[[body_ok(r) for r in candidates.itertuples()]].reset_index(drop=True)
    # Precompute the evaluator's accessible sets once, before panel search.
    table = {(r.video_id, r.frame_idx): r for r in frames.itertuples()}
    seen = {key: {o['category'] for o in json.loads(r.objects) if o['visible_px'] > 0}
            for key, r in table.items()}
    context = {}
    for video, t in set(zip(candidates.video_id, candidates.frame_idx)):
        centre = table[video, t]
        visual = set().union(*(seen[video, f] for f in range(t - 2, t + 3)
                               if f'{video}|{f}' in lut and (video, f) in seen))
        body = {centre.held_category} if pd.notna(centre.held_category) else set()
        for direction in (-1, 1):
            for delta in range(1, 6):
                row = table.get((video, t + direction * delta))
                if row is None or row.clip != centre.clip:
                    break
                if pd.notna(row.held_category):
                    body.add(row.held_category)
        context[video, t] = visual | body
    # Screen target rows against the complete same-partition, same-subject pool.
    # Foils are explicit candidate-only rows and never extra word exposures.
    pool_by_key = {}
    for r in candidates.itertuples(index=False):
        key = (r.partition, r.subject)
        pool_by_key.setdefault(key, []).append(r)
    eligible = []
    for r in candidates.itertuples(index=False):
        if r.partition == 'train':
            eligible.append(True)
            continue
        eligible.append(sum(1 for q in pool_by_key[r.partition, r.subject]
                            if q.video_id != r.video_id and q.referent != r.referent
                            and r.referent not in context[q.video_id, q.frame_idx]) >= 3)
    eligible = np.asarray(eligible)
    selected = sample_rows(candidates[eligible].reset_index(drop=True), 0)
    # Deterministic local swaps remove subject/room values associated with one word.
    valid_candidates = candidates[eligible].reset_index(drop=True)
    for partition in ('train', 'dev', 'test'):
        for _ in range(100):
            group = selected[selected.partition == partition]
            bad_subject = set(group.groupby('subject').referent.nunique().loc[lambda x: x < 2].index)
            bad_room = set(group.groupby('room').referent.nunique().loc[lambda x: x < 2].index)
            bad = group[group.subject.isin(bad_subject) | group.room.isin(bad_room)]
            if bad.empty:
                break
            changed = False
            # First add another word to a single-word subject, preserving all quotas.
            for subject in sorted(bad_subject):
                existing = set(group[group.subject == subject].referent)
                for idx, old in group[~group.referent.isin(existing)].iterrows():
                    if group[(group.subject == old.subject) & (group.referent == old.referent)].shape[0] == 1:
                        continue
                    pool = valid_candidates[(valid_candidates.partition == partition) &
                        (valid_candidates.referent == old.referent) & (valid_candidates.held == old.held) &
                        (valid_candidates.subject == subject)]
                    for _, replacement in pool.iterrows():
                        rest = group.drop(index=idx)
                        if not ((rest.room == replacement.room) & (rest.referent != old.referent)).any():
                            continue
                        if ((rest.referent == old.referent) & (rest['clip'] == replacement['clip'])).sum() >= 4:
                            continue
                        if ((rest.video_id == replacement.video_id) & (rest.frame_idx == replacement.frame_idx)
                                & (rest.referent == old.referent)).any():
                            continue
                        selected.loc[idx] = replacement
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                continue
            for idx, old in bad.iterrows():
                candidates_for_row = valid_candidates[(valid_candidates.partition == partition) &
                    (valid_candidates.referent == old.referent) & (valid_candidates.held == old.held)]
                for _, replacement in candidates_for_row.iterrows():
                    rest = group.drop(index=idx)
                    if not ((rest.subject == replacement.subject) & (rest.referent != old.referent)).any():
                        continue
                    if not ((rest.room == replacement.room) & (rest.referent != old.referent)).any():
                        continue
                    if ((rest.referent == old.referent) & (rest['clip'] == replacement['clip'])).sum() >= 4:
                        continue
                    if ((rest.video_id == replacement.video_id) & (rest.frame_idx == replacement.frame_idx)
                            & (rest.referent == old.referent)).any():
                        continue
                    selected.loc[idx] = replacement
                    changed = True
                    break
                if changed:
                    break
            if not changed:
                raise ValueError(f'{partition}: cannot remove subject/room shortcut by local swaps; subjects={bad_subject}, rooms={bad_room}; rows={bad[["referent", "held", "subject", "room", "clip"]].to_dict("records")}')
        else:
            raise ValueError(f'{partition}: nuisance repair exceeded 100 swaps')
    selected['role'] = 'target' 
    panels, expanded = {}, []
    for partition in ('train', 'dev', 'test'):
        panel = selected[selected.partition == partition].reset_index(drop=True)
        if any(g.subject.nunique() < 2 or g.room.nunique() < 2 or
               set.intersection(*(seen[r.video_id, r.frame_idx] - {category} for r in g.itertuples()))
               for category, g in panel.groupby('referent')):
            raise ValueError(f'{partition}: deterministic nuisance association')
        if (panel.groupby('subject').referent.nunique() < 2).any() or (panel.groupby('room').referent.nunique() < 2).any():
            raise ValueError(f'{partition}: subject or room identifies a word')
        if partition == 'train':
            expanded.append(panel)
            continue
        records = panel.to_dict('records')
        target_count = len(records)
        identities = {(r['video_id'], r['frame_idx'], r['referent']) for r in records}
        for target in records[:target_count]:
            def valid(q):
                return (q['subject'] == target['subject'] and q['video_id'] != target['video_id']
                        and q['referent'] != target['referent']
                        and target['referent'] not in context[q['video_id'], q['frame_idx']])
            need = 3 - sum(valid(q) for q in records)
            if need > 0:
                for q in pool_by_key[partition, target['subject']]:
                    item = q._asdict()
                    identity = (item['video_id'], item['frame_idx'], item['referent'])
                    if identity not in identities and valid(item):
                        item['role'] = 'foil'
                        records.append(item)
                        identities.add(identity)
                        need -= 1
                        if need == 0:
                            break
            if need > 0:
                raise ValueError(f'{partition}: insufficient explicit candidate-only foils')
        panel = pd.DataFrame(records).reset_index(drop=True)
        checked = body_present_context(panel, frames, lut, 2, 5)
        rng = np.random.default_rng(0)
        trials = []
        for i in range(target_count):
            pool = checked['pools'][i]
            if len(pool) < 3:
                raise ValueError('Canonical evaluator rejected candidate foils')
            choices = rng.permutation(np.r_[i, rng.choice(pool, 3, replace=False)])
            trials.append({'target': i, 'candidates': choices.tolist(),
                           'correct': int(np.flatnonzero(choices == i)[0])})
        panels[partition] = trials
        if partition == 'test':
            rows = list(panel.itertuples())
            rng = np.random.default_rng(0)
            allowed = [rng.permutation([j for j, b in enumerate(rows)
                        if a.subject == b.subject and a.clip != b.clip]).tolist() for a in rows]
            owner = {}
            def assign(i, seen):
                for j in allowed[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in owner or assign(owner[j], seen):
                        owner[j] = i
                        return True
                return False
            if not all(assign(i, set()) for i in range(len(rows))):
                raise ValueError('No same-subject, different-source-clip body permutation')
            donor = [0] * len(rows)
            for j, i in owner.items():
                donor[i] = j
            panels['test_donors'] = donor
        expanded.append(panel)
    selected = pd.concat(expanded, ignore_index=True)
    validate_panels(selected, panels)
    # Check every cached visual token accessible to selected rows.
    embeddings = np.load(cache / 'emb.f16.npy', mmap_mode='r')
    for r in selected.itertuples():
        for t in range(r.frame_idx - 2, r.frame_idx + 3):
            key = f'{r.video_id}|{t}'
            if key in lut and not np.isfinite(embeddings[lut[key]]).all():
                raise ValueError(f'Non-finite cached visual features: {key}')
    selected.drop(columns='visible').to_parquet(output / 'manifest.parquet', index=False)
    (output / 'panels.json').write_text(json.dumps(panels, indent=2))
    summary = selected[selected.role == 'target'].groupby(['partition', 'referent', 'held']).agg(encounters=('clip', 'size'), clips=('clip', 'nunique')).reset_index().to_dict('records')
    protocol = {'categories': dict(zip(CATEGORIES, WORDS)), 'sampling_seed': 0, 'split_seed': 0,
                'source_clip_counts': counts, 'counts': summary, 'foil_counts': selected[selected.role == 'foil'].groupby('partition').agg(rows=('clip', 'size'), clips=('clip', 'nunique')).to_dict('index'), 'episodes': str(episodes.resolve()),
                'cache': str(cache.resolve()),
                'embedding_metadata': str((cache.parent / 'embedding.json').resolve()),
                'embedding_metadata_sha256': hashlib.sha256((cache.parent / 'embedding.json').read_bytes()).hexdigest(),
                'cache_index_sha256': hashlib.sha256((cache / 'index.parquet').read_bytes()).hexdigest(),
                'physical': str((episodes / 'physical_by_clip').resolve()), 'crop_proxy': 'visible_px >= 100 and intersects central grid cells',
                'visual_half_window': 2, 'body_half_window': 5, 'foil_seed': 0,
                'selection': 'Coverage-ranked candidates; turtle failed 32 held train encounters at max four per clip; glass substituted. Deterministic swaps remove subject/room shortcuts. No learner performance used.',
                'candidate_only_foils': 'Additional explicit partition-local rows; excluded from target counts.',
                'visual_qc': 'Two held and two nonheld selected targets per category and partition reviewed as five-frame crops. Some center frames are small or occluded; this is not a per-example recognizability guarantee.',
                'example_encounters': [{'video_id': 's7_0038', 'frame_idx': 2, 'referent': 'bottle'}, {'video_id': 's6_0102', 'frame_idx': 48, 'referent': 'bottle'}, {'video_id': 's5_0020', 'frame_idx': 118, 'referent': 'bottle'}],
                'donor_rule': 'Seed 0 one-to-one same-subject test permutation across different source clips'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.episodes, args.cache, args.output)
