"""Execute an immutable acquisition protocol with a bounded GPU job queue.

The caller transfers the original absolute paths and fetches before destroying.
This scheduler never freezes inputs or changes scientific source/protocols.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def jobs(protocol):
    # Body arms occupy the first six slots, as requested.
    body = [arm for arm in protocol['conditions'] if 'body' in arm]
    baseline = [arm for arm in protocol['conditions'] if 'body' not in arm]
    return ([(fraction, seed, arm) for arm in body for seed in protocol['seeds']
             for fraction in protocol['fractions']] +
            [(fraction, seed, arm) for seed in protocol['seeds'] for arm in baseline
             for fraction in protocol['fractions']])


def verify(frozen):
    from nursery.learner import acquisition as run
    snapshot = run._verify_frozen(frozen)
    protocol = run._read(frozen / 'protocol.json')
    for filename, digest in protocol['code_hashes'].items():
        assert run._sha(filename) == digest, filename
    return dict(complete=True, revision=snapshot['revision'],
                code_hashes=snapshot['code_hashes'],
                protocol_sha256=run._sha(frozen / 'protocol.json'),
                external_files=len(snapshot['external_hashes']), utc=time.time())


def worker(args):
    import torch
    from nursery.learner import acquisition as run
    torch.set_num_threads(args.threads)
    original = run._write
    def observed(path, value):
        original(path, value)
        if Path(path).name == 'history.json' and value:
            local = inspect.currentframe().f_back.f_locals
            write(Path(path).with_name('live_timing.json'), dict(
                epoch=value[-1]['epoch'], epoch_seconds=time.time()-local['epoch_started'],
                setup_seconds=local['setup_seconds'], utc=time.time()))
    run._write = observed
    run.train_one(args.frozen, args.fraction, args.seed, args.condition,
                  deadline=args.deadline, device='cuda')


def matrix(args):
    from nursery.learner import acquisition as run
    from scripts.child_gaze_rental import queued_epoch_projection
    protocol = run._read(args.frozen / 'protocol.json')
    receipt = verify(args.frozen)
    write(args.rental / 'host_input_verification.json', receipt)
    hard = min(args.launched + protocol['runtime_cap_hours']*3600,
               args.launched + (args.cost_cap-args.transfer_reserve)/args.hourly*3600)
    deadline = hard - args.return_reserve
    pending, active, completed = jobs(protocol), {}, []
    available = list(range(args.workers))
    gate_written = False
    try:
        while pending or active:
            if time.time() >= deadline or (args.rental / 'STOP').exists():
                raise TimeoutError('Rental return reserve reached or operator stop')
            while available and pending:
                gpu, job = available.pop(0), pending.pop(0)
                fraction, seed, arm = job
                log = args.rental / f'{arm}_seed_{seed}.log'
                handle = log.open('w')
                command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                           '--frozen', str(args.frozen), '--rental', str(args.rental),
                           '--fraction', str(fraction), '--seed', str(seed),
                           '--condition', arm, '--deadline', str(deadline), '--threads', str(args.threads)]
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                             OMP_NUM_THREADS=str(args.threads), PYTHONDONTWRITEBYTECODE='1'))
                active[gpu] = (process, handle, job)
            observations = []
            for gpu, (process, handle, job) in list(active.items()):
                directory = run._output(args.frozen, *job)
                timing = directory / 'live_timing.json'
                if timing.exists():
                    observations.append(dict(gpu=gpu, job=job, **run._read(timing)))
                if process.poll() is not None:
                    handle.close()
                    if process.returncode:
                        raise RuntimeError(f'Worker failed: {job}; inspect its log; no automatic rerun')
                    completed.append(job)
                    del active[gpu]
                    available.append(gpu)
            if not gate_written and len(observations) == args.workers:
                gate = queued_epoch_projection(
                    observations=[dict(condition=x['job'][2], epoch=x['epoch'],
                                       epoch_seconds=x['epoch_seconds'], setup_seconds=x['setup_seconds'])
                                  for x in observations],
                    pending_conditions=[job[2] for job in pending], epochs=protocol['epochs'],
                    elapsed_seconds=time.time()-args.launched, hourly=args.hourly,
                    hard_seconds=hard-args.launched, transfer_reserve=args.transfer_reserve,
                    return_reserve=args.return_reserve, cost_cap=args.cost_cap)
                write(args.rental / 'first_real_epoch_gate.json', gate)
                if not gate['passes']:
                    raise RuntimeError('Measured matrix projection exceeds caps')
                gate_written = True
            write(args.rental / 'progress.json', dict(utc=time.time(), completed=completed,
                pending=pending, active=observations, total=len(jobs(protocol))))
            time.sleep(2)
        write(args.rental / 'runs_complete.json', dict(complete=True, jobs=completed, utc=time.time()))
    except BaseException as error:
        write(args.rental / 'failure.json', dict(error=str(error), utc=time.time()))
        raise
    finally:
        for process, handle, _ in active.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['verify', 'worker', 'matrix'])
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--rental', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--hourly', type=float)
    p.add_argument('--launched', type=float)
    p.add_argument('--cost-cap', type=float, default=15)
    p.add_argument('--transfer-reserve', type=float, default=.75)
    p.add_argument('--return-reserve', type=float, default=600)
    p.add_argument('--fraction', type=float, default=1.)
    p.add_argument('--seed', type=int)
    p.add_argument('--condition')
    p.add_argument('--deadline', type=float)
    a = p.parse_args()
    if a.action == 'verify':
        write(a.rental / 'host_input_verification.json', verify(a.frozen))
    elif a.action == 'worker':
        worker(a)
    else:
        matrix(a)
