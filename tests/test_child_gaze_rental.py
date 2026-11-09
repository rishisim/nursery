"""Transport failures must not silently abandon or duplicate paid instances."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('child_gaze_rental', Path(__file__).parents[1] / 'scripts/child_gaze_rental.py')
rental = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rental)


def result(args, output='', code=0, error=''):
    return subprocess.CompletedProcess(args, code, output, error)


def test_dns_ssh_and_rsync_recover_without_single_failure_exit(tmp_path):
    for binary in ('vastai', 'ssh', 'rsync'):
        calls = []
        def invoke(args, **kwargs):
            calls.append(args)
            return result(args, 'ok') if len(calls) == 3 else result(args, code=1, error='Failed to resolve host')
        sleeps = []
        out = rental.run_transport([binary, 'operation'], state_file=tmp_path / f'{binary}.json',
                                   invoke=invoke, sleep=sleeps.append)
        assert out.stdout == 'ok' and len(calls) == 3 and sleeps == [5, 10]


def test_outage_age_persists_across_process_cycles_and_stops_on_recovery(tmp_path):
    state = tmp_path / 'state.json'
    failed = lambda args, **kw: result(args, code=1, error='Connection refused')
    with pytest.raises(rental.TransportUnavailable):
        rental.run_transport(['ssh', 'host'], state_file=state, attempts=1, clock=lambda: 100., invoke=failed)
    with pytest.raises(rental.ProlongedOutage):
        rental.run_transport(['ssh', 'host'], state_file=state, clock=lambda: 701., invoke=lambda args, **kw: result(args))
    assert json.loads(state.read_text())['stop_requested']
    assert rental.run_transport(['vastai', 'destroy', 'instance', '1', '-y', '--raw'],
                                state_file=state, teardown=True, invoke=lambda args, **kw: result(args)).returncode == 0


def test_ambiguous_create_reconciles_instead_of_creating_twice():
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        if args[1:3] == ['create', 'instance']:
            raise rental.AmbiguousCreate('timeout')
        return result(args, json.dumps([{'id': 42, 'label': 'unique-task'}]))
    out = rental.create_instance(['vastai', 'create', 'instance', '7', '--label', 'unique-task'],
                                 label='unique-task', runner=runner)
    assert out['new_contract'] == 42 and out['reconciled']
    assert sum(args[1:3] == ['create', 'instance'] for args in calls) == 1


def test_loading_and_transient_running_without_container_never_attempt_ssh():
    now = [0.]
    states = iter(['running', 'loading', 'running'])
    calls = []
    logs = iter(['No such container', 'sshd started'])
    def runner(args, **kw):
        calls.append(args)
        if args[1] == 'logs':
            return result(args, next(logs))
        return result(args, json.dumps({'id': 42, 'actual_status': next(states)}))
    value = rental.wait_running(42, started=0, runner=runner, clock=lambda: now[0],
                                sleep=lambda duration: now.__setitem__(0, now[0] + duration))
    assert value['actual_status'] == 'running' and now[0] == 30
    assert all(args[0] == 'vastai' for args in calls)


def test_loading_has_twenty_minute_bound():
    now = [0.]
    def runner(args, **kw):return result(args, json.dumps({'actual_status': 'loading'}))
    with pytest.raises(TimeoutError, match='twenty minutes'):
        rental.wait_running(42, started=0, runner=runner, clock=lambda: now[0],
                            sleep=lambda duration: now.__setitem__(0, now[0] + duration))
    assert now[0] == 1200
