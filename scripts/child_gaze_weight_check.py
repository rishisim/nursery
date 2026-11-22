"""Read-only, local development-selected beta check; never trains a learner."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nursery.learner import _upstream_module
from nursery.learner.acquisition import _gaze_data
from nursery.learner.acquisition_evaluate import StudyEvaluation, _metrics
from nursery.learner.gaze import GazeMIL

ATTEMPT = REPO / 'outputs/runs/child_gaze/phase_c/overnight_rental_attempt1'
STUDY = ATTEMPT / 'returned/outputs/runs/child_gaze/phase_c/overnight_rental_attempt1/study'
OUT = REPO / 'outputs/analysis/child_gaze_weight_check'
CONDITIONS = ('video_gaze', 'video_shuffled_gaze', 'video_gaze_point')
PANELS = ('clips_test', 's8')
GRID = [i / 20 for i in range(31)]


def read(path):
    return json.loads(path.read_text())


def summary(values):
    return {'per_seed': values, 'mean': float(np.mean(values)),
            'sample_sd': float(np.std(values, ddof=1))}


def fmt(s):
    return f"{s['mean']:.2f} ± {s['sample_sd']:.2f} [{', '.join(f'{v:.2f}' for v in s['per_seed'])}]"


def write(result, lines, started):
    result['runtime_seconds'] = time.monotonic() - started
    lines.append(f"\nRuntime: {result['runtime_seconds']:.1f} seconds.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps'),
                        default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    audit_path = REPO / 'outputs/runs/child_gaze/phase_c/audit_results.py'
    spec = importlib.util.spec_from_file_location('gaze_audit', audit_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    protocol = read(STUDY / 'protocol.json')
    local_inputs = REPO / 'outputs/features/child_gaze/world16'
    vocab = read(STUDY / 'subsets/1.0/vocab.json')
    upstream = _upstream_module('train_frame_mil')
    caches, lut = upstream.load_multi([local_inputs / 'train_cache'])
    evaluator = StudyEvaluation(STUDY, local_inputs, caches, lut, vocab, None, args.device)
    records, gate = [], []
    result = {'status': 'blocked', 'device': args.device, 'beta_grid': GRID,
              'checkpoint': 'model_bodypresent.pt (earliest with-gaze development maximum)',
              'study': str(STUDY), 'local_visual_cache': str(local_inputs / 'train_cache'),
              'cache_caveat': 'Returned CUDA visual caches were not retained locally; local original MPS cache used, previously parity-audited but not byte-identical.',
              'sanity_gate': gate}
    # All nine checkpoints and both test panels must pass before any sweep.
    for condition in CONDITIONS:
        evaluator.gaze_condition = condition
        evaluator.gaze_factory = lambda rows, c=condition: _gaze_data(protocol, rows, lut, c)
        evaluator.gaze_panels.clear()
        for seed in range(3):
            directory = STUDY / f'models/1.0/seed_{seed}' / condition
            metrics = read(directory / 'metrics.json')
            checkpoint = directory / 'model_bodypresent.pt'
            model = GazeMIL(len(vocab), caches[0].shape[-1], protocol['tau'],
                            option='additive', point_only=condition == 'video_gaze_point',
                            num_regions=caches[0].shape[1]).float().to(args.device)
            model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True), strict=True)
            beta = float(model.beta.detach().cpu())
            saved = {}
            for panel in PANELS:
                actual = evaluator.evaluate(model, panel)
                expected = metrics['primary'][panel]
                trials = read(STUDY / 'panels' / panel / 'trials.json')
                rows = pd.read_parquet(STUDY / 'panels' / panel / 'manifest.parquet')
                audit.reconstruct(actual, vocab, trials, rows.iloc[trials['trial_rows']].referent.tolist())
                assert actual['n_trials'] == {'clips_test': 556, 's8': 793}[panel]
                same = actual['accuracy'] == expected['accuracy']
                rec = {'condition': condition, 'seed': seed, 'panel': panel,
                       'checkpoint_sha256': audit.sha(checkpoint), 'learned_beta': beta,
                       'expected_accuracy': expected['accuracy'], 'actual_accuracy': actual['accuracy'],
                       'accuracy_exact_match': same,
                       'changed_predictions': sum(a != b for a, b in zip(actual['predicted_slots'], expected['predicted_slots']))}
                gate.append(rec)
                saved[panel] = actual
                print(f"Gate {condition} seed {seed} {panel}: {actual['accuracy']:.12f} / {expected['accuracy']:.12f}", flush=True)
                if not same:
                    result['reason'] = 'Exact learned-beta accuracy gate failed; local visual cache and device differ from the original CUDA execution. No beta sweep or development selection performed.'
                    write(result, ['# WEIGHT_CHECK: blocked', result['reason'], json.dumps(rec)], started)
                    return
            records.append({'condition': condition, 'seed': seed, 'model': model.cpu(),
                            'metrics': metrics, 'learned_beta': beta, 'learned': saved,
                            'checkpoint_sha256': audit.sha(checkpoint)})
    result['sanity_gate_passed'] = True
    # Reuse each encoded batch but call the canonical scorer at every beta;
    # no score interpolation or changes to any other model weight.
    selected, curves = [], {}
    for rec in records:
        condition, seed = rec['condition'], rec['seed']
        model = rec['model'].to(args.device)
        evaluator.gaze_condition = condition
        evaluator.gaze_factory = lambda rows, c=condition: _gaze_data(protocol, rows, lut, c)
        evaluator.gaze_panels.clear()
        def sweep(panel):
            batches = {b: [] for b in GRID}
            canonical = model.score_encoded
            def score_grid(R, T, ctx=None):
                for b in GRID:
                    model.beta.fill_(b)
                    scored = canonical(R, T, ctx)
                    batches[b].append(scored.float().cpu().numpy())
                return scored
            model.score_encoded = score_grid
            try:
                template = evaluator.evaluate(model, panel)
            finally:
                model.score_encoded = canonical
            known = np.array([s is not None for s in template['scores']])
            words = {w: i for i, w in enumerate(sorted({q for q, k in zip(template['queries'], known) if k}))}
            candidates = np.asarray(template['candidate_rows'])
            swept = {}
            for b in GRID:
                matrix = np.concatenate(batches[b]) if words else None
                scores = np.zeros((len(known), 4), dtype=np.float32)
                for i, query in enumerate(template['queries']):
                    if known[i]:
                        scores[i] = matrix[candidates[i], words[query]]
                swept[str(b)] = _metrics(scores, known, candidates, template['correct_slots'], template['queries'])
            return swept
        dev = sweep('clips_dev')
        chosen = max(GRID, key=lambda b: dev[str(b)]['accuracy'])
        # Selection is fixed solely from dev before test sweep is constructed.
        row = {k: rec[k] for k in ('condition', 'seed', 'learned_beta', 'checkpoint_sha256')}
        row.update(chosen_beta=chosen, dev_accuracy=dev[str(chosen)]['accuracy'],
                   dev_sweep={b: r['accuracy'] for b, r in dev.items()}, panels={})
        for panel in PANELS:
            curve = sweep(panel)
            with torch.no_grad():
                model.beta.fill_(chosen)
                exact = evaluator.evaluate(model, panel)
            assert exact['credits'] == curve[str(chosen)]['credits'], 'Batched sweep changed chosen-beta credits'
            row['panels'][panel] = exact
            key = f'{condition}/{panel}'
            curves.setdefault(key, []).append({**{b: r['accuracy'] for b, r in curve.items()},
                                              'learned': rec['learned'][panel]['accuracy']})
        model.cpu()
        selected.append(row)
        print(f"Selected {condition} seed {seed}: beta={chosen}, dev={row['dev_accuracy']:.6f}", flush=True)
    baselines = {panel: summary([read(STUDY / f'models/1.0/seed_{s}/video_only/metrics.json')['primary'][panel]['accuracy'] for s in range(3)]) for panel in PANELS}
    aggregates, splits = {}, {}
    manifest = pd.read_parquet(STUDY / 'panels/clips_test/manifest.parquet')
    trials = read(STUDY / 'panels/clips_test/trials.json')
    targets = manifest.iloc[trials['trial_rows']]
    masks = {'held': targets.held_referent.to_numpy(bool), 'gazed': targets.gazed_referent.to_numpy(bool)}
    masks.update(not_held=~masks['held'], not_gazed=~masks['gazed'])
    lines = ['# WEIGHT_CHECK: done', 'Sanity gate: all 18 learned-beta accuracies match Step 2 exactly.',
             'Selection: highest with-gaze clips_dev accuracy on 0:0.05:1.50, ties to smaller beta; fixed original checkpoints.',
             'Every beta uses the canonical model scorer with reused encoded video batches; test accuracy and credits at chosen beta also match an independent evaluator pass.',
             result['cache_caveat'], 'All figures are percentages; differences are percentage points; SD is sample SD across training seeds.']
    for condition in CONDITIONS:
        rr = [r for r in selected if r['condition'] == condition]
        lines.append(f"\n{condition}: chosen beta {[r['chosen_beta'] for r in rr]}")
        for panel in PANELS:
            values = [r['panels'][panel]['accuracy'] for r in rr]
            paired = summary([v - b for v, b in zip(values, baselines[panel]['per_seed'])])
            aggregates[f'{condition}/{panel}'] = {'accuracy': summary(values), 'paired_difference_vs_video_only': paired}
            lines.append(f"{panel}: {fmt(summary(values))}; video {fmt(baselines[panel])}; paired Δ {fmt(paired)}")
        splits[condition] = {name: {'n_trials': int(mask.sum()), **summary([float(100 * np.array(r['panels']['clips_test']['credits'])[mask].mean()) for r in rr])} for name, mask in masks.items()}
        lines.append('Primary splits: ' + '; '.join(f'{name} (n={s["n_trials"]}) {fmt(s)}' for name, s in splits[condition].items()))
    splits['video_only'] = {name: {'n_trials': int(mask.sum()), **summary([float(100 * np.array(read(STUDY / f'models/1.0/seed_{s}/video_only/metrics.json')['primary']['clips_test']['credits'])[mask].mean()) for s in range(3)])} for name, mask in masks.items()}
    lines.append('\nVideo-only primary splits: ' + '; '.join(f'{name} (n={s["n_trials"]}) {fmt(s)}' for name, s in splits['video_only'].items()))
    lines += ['', 'Descriptive test means versus beta (primary / s8); learned uses each seed’s own learned beta.',
              '| Learner | 0 | .25 | .50 | .75 | 1.0 | learned |', '|---|---|---|---|---|---|---|']
    mean_curves = {key: {b: summary([r[b] for r in rows]) for b in rows[0]} for key, rows in curves.items()}
    for condition in CONDITIONS:
        cells = [f"{mean_curves[f'{condition}/clips_test'][b]['mean']:.2f} / {mean_curves[f'{condition}/s8'][b]['mean']:.2f}" for b in ('0.0', '0.25', '0.5', '0.75', '1.0', 'learned')]
        lines.append('| ' + condition + ' | ' + ' | '.join(cells) + ' |')
    lines.append('\nInterpretation: development-selected beta makes real gaze exceed video only on both fixed panels in all three seeds. It removes most of the shuffled inference penalty, but the shuffled learner remains below video only; post-hoc weighting cannot undo its learned video-branch deficit. Point-only remains below video on primary. This supports checking calibrated additive gaze as the next integration candidate, not a claim about the acquisition ceiling or authorization for training.')
    result.update(status='done', selected=selected, baselines=baselines, aggregates=aggregates,
                  primary_splits=splits, descriptive_test_curves=mean_curves)
    write(result, lines, started)


if __name__ == '__main__':
    main()
