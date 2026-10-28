"""Fresh, paired acquisition runs with development-only checkpoint selection."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from nursery.common import _output_directory
from . import _upstream_module

CONDITIONS = ('video_only', 'additive', 'joint_interaction')


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot():
    repo = Path(__file__).resolve().parents[2]
    files = sorted((repo / 'nursery' / 'learner').glob('*.py'))
    files += sorted((repo / 'nursery' / 'learner').glob('*.json'))
    files += [repo / 'nursery' / 'common.py', repo / 'nursery' / 'speech' / 'config.json']
    files += sorted((repo / '.external' / 'vlm-headcam' / 'src').glob('*.py'))
    return {'revision': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'code_hashes': {str(p): _sha(p) for p in files if p.is_file()}}


def _freeze_inputs(root):
    """Audit first, then bind source, raw signals and prepared panels to timing."""
    from .acquisition_data import audit
    audit(root)
    protocol = _read(root / 'protocol.json')
    external = {**protocol['hashes'], **_read(root / 'signal_hashes.json')}
    for path, expected in external.items():
        if _sha(path) != expected:
            raise ValueError(f'Input provenance changed: {path}')
    prepared = [root / name for name in ('protocol.json', 'splits.json', 'signal_hashes.json')]
    prepared += sorted((root / 'subsets').rglob('*')) + sorted((root / 'panels').rglob('*'))
    snapshot = {**_source_snapshot(), 'prepared_hashes': {str(p): _sha(p) for p in prepared if p.is_file()},
                'external_hashes': external}
    _write(root / 'frozen_inputs.json', snapshot)
    return snapshot


def _verify_frozen(root, *, external=True):
    snapshot = _read(root / 'frozen_inputs.json')
    current = _source_snapshot()
    if any(current[k] != snapshot[k] for k in ('revision', 'code_hashes')):
        raise ValueError('Source changed after the timing freeze')
    hashes = dict(snapshot['prepared_hashes'])
    if external:
        hashes.update(snapshot['external_hashes'])
    for path, expected in hashes.items():
        if _sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    return snapshot


def _check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError('Acquisition execution reached its frozen 24-hour cap')


def _identity(root, fraction, seed, condition, epochs, device):
    return {'protocol_sha256': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest(),
            'fraction': float(fraction), 'seed': int(seed), 'condition': condition,
            'epochs': epochs, 'device': device}


def _output(root, fraction, seed, condition, smoke=False):
    return root / 'timing' / condition if smoke else root / 'models' / str(float(fraction)) / f'seed_{seed}' / condition


def _select(score, best, model, path):
    """Strict improvement preserves the earliest epoch on ties."""
    import torch
    if not np.isfinite(score):
        raise ValueError('Non-finite development score')
    if score > best:
        torch.save(model.state_dict(), path)
        return True
    return False


def train_one(root, fraction, seed, condition, epochs=None, smoke=False, deadline=None, *, device='mps'):
    """Train a fresh learner; CPU is an explicit tiny-test override only."""
    import torch
    from . import body
    from .additive import AdditiveMIL
    from .video_only import VideoOnlyMIL
    from .interaction import JointInteractionMIL, centre_clip_valid
    from .train import _pair_meta, _window_layout
    from .windows import BodyData
    from .acquisition_evaluate import StudyEvaluation

    root = Path(root).resolve()
    protocol = _read(root / 'protocol.json')
    epochs = (1 if smoke else protocol['epochs']) if epochs is None else epochs
    if condition not in CONDITIONS or epochs < 1:
        raise ValueError('Unknown condition or nonpositive epochs')
    if device not in ('cpu', 'mps') or (device == 'mps' and not torch.backends.mps.is_available()):
        raise RuntimeError('Scientific acquisition training requires available MPS; no automatic fallback')
    if smoke and epochs != 1:
        raise ValueError('Timing smoke is exactly one epoch')
    _check_deadline(deadline)
    if device == 'mps':
        _verify_frozen(root, external=False)
    identity = _identity(root, fraction, seed, condition, epochs, device)
    output = _output(root, fraction, seed, condition, smoke)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    output = _output_directory(output)
    started = time.time()
    _write(output / 'status.json', {'status': 'incomplete', **identity})
    try:
        subset = root / 'subsets' / str(float(fraction))
        manifest = pd.read_parquet(subset / 'manifest.parquet')
        vocab, stats = _read(subset / 'vocab.json'), _read(subset / 'stats.json')
        upstream = _upstream_module('train_frame_mil')
        inputs, episodes = Path(protocol['inputs']), Path(protocol['episodes'])
        caches, lut = upstream.load_multi([inputs / 'train_cache'])
        with np.load(subset / 'fit_mask.npz', allow_pickle=False) as fit:
            video_ids, include = fit['video_ids'].tolist(), fit['include']
        store = body.load_store(episodes, pd.read_parquet(episodes / 'frames.parquet'), video_ids, set(),
                                steps=include.shape[1], exclude=~include, physical=Path(protocol['physical']))
        if store['stats'] != stats['normalization']:
            raise ValueError('Frozen subset normalization differs from current inputs')
        if set(store['groups']) != {'contact', 'fingers', 'arm', 'load', 'skin'}:
            raise ValueError('Acquisition requires all five body streams')
        ds = upstream.FramePairs(caches, lut, manifest, vocab, protocol['window'], False)
        if len(ds) < 2 or ds.n_noframe or ds.n_novocab:
            raise ValueError('Frozen manifest must contain at least two cached vocabulary-bearing pairs')
        order_generator = torch.Generator().manual_seed(seed)
        body_generator = torch.Generator().manual_seed(seed + 1000)
        dl = torch.utils.data.DataLoader(ds, batch_size=protocol['batch'], shuffle=True, drop_last=False,
                                        collate_fn=upstream.collate, num_workers=0, generator=order_generator)
        torch.manual_seed(seed)
        np.random.seed(seed)
        dims = {k: v.shape[-1] for k, v in store['groups'].items()}
        emb_dim = int(caches[0].shape[-1])
        if caches[0].ndim != 3 or caches[0].shape[1] != 17:
            raise ValueError('Expected CLS plus 4x4 frozen region caches')
        constructors = {'video_only': VideoOnlyMIL, 'additive': AdditiveMIL, 'joint_interaction': JointInteractionMIL}
        kwargs = {} if condition == 'video_only' else {'dims': dims}
        model = constructors[condition](len(vocab), emb_dim, protocol['tau'], **kwargs).float().to(device)
        ps, pt, win = _pair_meta(manifest, store, lut, protocol['window'])
        data = BodyData(store, ps, pt, win, None, None, None, None, device,
                        protocol['body_window'], 0.0, protocol['body_dropout']) if condition != 'video_only' else None
        _, visual_valid = _window_layout(manifest, lut, protocol['window'])
        visual_valid = torch.as_tensor(visual_valid, device=device).repeat_interleave(17, dim=1)
        evaluator = StudyEvaluation(root, inputs, caches, lut, vocab, store, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['lr'], weight_decay=protocol['weight_decay'])
        setup_seconds = time.time() - started
        epoch_durations = []
        history, best_body, best_picture = [], -float('inf'), -float('inf')
        selection = {}
        for epoch in range(epochs):
            epoch_started = time.time()
            _check_deadline(deadline)
            model.train()
            loss_sum, count = 0.0, 0
            for idx, v, t, n in dl:
                _check_deadline(deadline)
                ctx = data.batch(idx.numpy(), body_generator) if data is not None else None
                if ctx is not None:
                    ctx['keep'] = ctx['keep'] * centre_clip_valid(ctx).any(1)
                    ctx['visual_valid'] = visual_valid[idx.to(device)]
                loss = model.forward_loss(v.to(device), t.to(device), n.to(device), torch.ones(len(idx), device=device), ctx)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(idx)
                count += len(idx)
            _check_deadline(deadline)
            model.eval()
            with torch.no_grad():
                body_dev = evaluator.evaluate(model, 'clips_dev')['accuracy']
                _check_deadline(deadline)
                picture_dev = evaluator.evaluate_picture(model, 'picture_dev')['accuracy']
            if _select(body_dev, best_body, model, output / 'model_bodypresent.pt'):
                best_body = body_dev
                selection['body_present'] = {'epoch': epoch + 1, 'dev_score': body_dev, 'panel': 'clips_dev'}
            if _select(picture_dev, best_picture, model, output / 'model.pt'):
                best_picture = picture_dev
                selection['picture'] = {'epoch': epoch + 1, 'dev_score': picture_dev, 'panel': 'picture_dev'}
            history.append({'epoch': epoch + 1, 'loss': loss_sum / count, 'clips_dev': body_dev, 'picture_dev': picture_dev})
            _write(output / 'history.json', history)
            epoch_durations.append(time.time() - epoch_started)
        final_started = time.time()
        model.load_state_dict(torch.load(output / 'model_bodypresent.pt', map_location=device, weights_only=True))
        primary, shuffled, picture, timing_dev = {}, {}, {}, {}
        with torch.no_grad():
            if smoke:
                # Timing never inspects held-out panels; charge four complete
                # development passes as a conservative final-evaluation estimate.
                _check_deadline(deadline)
                timing_dev['real'] = evaluator.evaluate(model, 'clips_dev')
                if data is not None:
                    _check_deadline(deadline)
                    timing_dev['shuffled'] = evaluator.evaluate(model, 'clips_dev', body_mode='shuffled')
                model.load_state_dict(torch.load(output / 'model.pt', map_location=device, weights_only=True))
                _check_deadline(deadline)
                timing_dev['picture'] = evaluator.evaluate_picture(model, 'picture_dev')
            else:
                for panel in ('clips_test', 's8'):
                    _check_deadline(deadline)
                    primary[panel] = evaluator.evaluate(model, panel)
                    if data is not None:
                        _check_deadline(deadline)
                        shuffled[panel] = evaluator.evaluate(model, panel, body_mode='shuffled')
                model.load_state_dict(torch.load(output / 'model.pt', map_location=device, weights_only=True))
                for panel in ('picture_test', 'konkle_test'):
                    _check_deadline(deadline)
                    picture[panel] = evaluator.evaluate_picture(model, panel)
        final_seconds = time.time() - final_started
        _check_deadline(deadline)
        metrics = {**identity, 'status': 'complete', 'smoke': smoke, 'selection': selection, 'history': history,
                   'primary': primary, 'shuffled': shuffled, 'picture': picture, 'data': stats,
                   'timing_dev': timing_dev,
                   'runtime_seconds': time.time() - started, 'setup_seconds': setup_seconds,
                   'epoch_seconds': float(np.mean(epoch_durations)),
                   'final_evaluation_seconds': final_seconds * (4 if smoke else 1),
                   'final_evaluation_estimate_rule': '4x development real/shuffled/picture pass' if smoke else 'measured held-out evaluation'}
        _write(output / 'metrics.json', metrics)
        _write(output / 'status.json', {'status': 'complete', **identity})
        return metrics
    except Exception as exc:
        _write(output / 'status.json', {'status': 'incomplete', **identity, 'error': str(exc), 'runtime_seconds': time.time() - started})
        raise


def timing(root):
    """Time one full-data epoch per condition; reject a projected over-cap matrix."""
    root = _output_directory(Path(root).resolve())
    protocol = _read(root / 'protocol.json')
    state_path = root / 'execution.json'
    if state_path.exists():
        raise FileExistsError('Timing has already started; inspect retained execution state')
    start = time.time()
    state = {'started_unix': start, 'deadline_unix': start + 24 * 3600, 'status': 'timing', 'completed': [],
             'protocol_sha256': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()}
    _write(state_path, state)
    try:
        _freeze_inputs(root)
        durations = {}
        for condition in CONDITIONS:
            metrics = train_one(root, 1.0, protocol['seeds'][0], condition, smoke=True, deadline=state['deadline_unix'])
            durations[condition] = {k: metrics[k] for k in ('runtime_seconds', 'setup_seconds', 'epoch_seconds', 'final_evaluation_seconds')}
        # Use full-data costs for every subset, with 25% headroom.
        projection = 1.25 * sum(d['setup_seconds'] + protocol['epochs'] * d['epoch_seconds'] +
                                d['final_evaluation_seconds'] for d in durations.values()) * len(protocol['fractions']) * len(protocol['seeds'])
        state.update(timing_seconds=durations, projected_matrix_seconds=projection)
        if projection + time.time() - start > 24 * 3600:
            raise RuntimeError('Conservative complete-matrix projection exceeds the 24-hour cap')
        state['status'] = 'ready'
        _write(state_path, state)
        return state
    except Exception as exc:
        state.update(status='stopped', error=str(exc))
        _write(state_path, state)
        raise
    finally:
        # Smoke checkpoints are task-created scratch; keep timing records only.
        for condition in CONDITIONS:
            directory = _output(root, 1.0, protocol['seeds'][0], condition, True)
            for name in ('model.pt', 'model_bodypresent.pt'):
                (directory / name).unlink(missing_ok=True)


def run(root):
    """Execute the frozen matrix sequentially, retaining partial runs on interruption."""
    root = _output_directory(Path(root).resolve())
    protocol = _read(root / 'protocol.json')
    state_path = root / 'execution.json'
    state = _read(state_path)
    digest = hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()
    if state.get('protocol_sha256') != digest:
        raise ValueError('Protocol changed after timing; timing gate no longer applies')
    if state['status'] not in ('ready', 'running', 'incomplete'):
        raise RuntimeError('A successful timing gate is required before scientific training')
    _verify_frozen(root)
    state['status'] = 'running'
    _write(state_path, state)
    try:
        for fraction in protocol['fractions']:
            for seed in protocol['seeds']:
                for condition in CONDITIONS:
                    _check_deadline(state['deadline_unix'])
                    output = _output(root, fraction, seed, condition)
                    expected = _identity(root, fraction, seed, condition, protocol['epochs'], 'mps')
                    if (output / 'metrics.json').exists():
                        metrics = _read(output / 'metrics.json')
                        if metrics.get('status') != 'complete' or any(metrics.get(k) != v for k, v in expected.items()):
                            raise ValueError(f'Existing run does not match frozen protocol: {output}')
                    else:
                        metrics = train_one(root, fraction, seed, condition, deadline=state['deadline_unix'])
                    key = f'{float(fraction)}/{seed}/{condition}'
                    if key not in state['completed']:
                        state['completed'].append(key)
                    state['consumed_run_seconds'] = state.get('consumed_run_seconds', {})
                    state['consumed_run_seconds'][key] = metrics['runtime_seconds']
                    _write(state_path, state)
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='incomplete', error=str(exc))
        _write(state_path, state)
        raise
    _write(state_path, state)
    return state
