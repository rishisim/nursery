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


def test_ambiguous_create_waits_through_inventory_dns_outage():
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise rental.AmbiguousCreate('create response lost')
        if len(calls) == 2:
            raise rental.TransportUnavailable('inventory DNS unavailable for retry cycle')
        return result(args, json.dumps([{'id': 43, 'label': 'unique-task'}]))
    out = rental.create_instance(['vastai', 'create', 'instance', '7', '--label', 'unique-task'],
                                 label='unique-task', runner=runner)
    assert out['new_contract'] == 43 and len(calls) == 3
    assert sum(args[1:3] == ['create', 'instance'] for args in calls) == 1


def test_create_response_is_retained_before_reconciliation(tmp_path):
    state = tmp_path / 'transport.json'
    def runner(args, **kwargs):
        if args[1:3] == ['create', 'instance']:
            return result(args, '{"success": true, "new_contract": 45, "instance_api_key": "private"}')
        pytest.fail('Successful create needs no reconciliation')
    value = rental.create_instance(['vastai', 'create', 'instance', '7', '--label', 'unique-task'],
                                    label='unique-task', state_file=state, runner=runner)
    receipt = json.loads((tmp_path / 'create_responses.jsonl').read_text())
    assert value['new_contract'] == 45 and receipt['returncode'] == 0
    assert json.loads(receipt['stdout']) == value and receipt['label'] == 'unique-task'
    assert 'instance_api_key' not in value and 'private' not in (tmp_path / 'create_responses.jsonl').read_text()


def test_persistent_wrapper_never_repeats_uncertain_create(monkeypatch):
    calls = []
    def fail(args, **kwargs):
        calls.append(args)
        raise rental.AmbiguousCreate('response lost')
    monkeypatch.setattr(rental, 'run_transport', fail)
    with pytest.raises(rental.AmbiguousCreate):
        rental.run_connected(['vastai', 'create', 'instance', '7'])
    assert len(calls) == 1


def test_early_auth_readiness_does_not_start_network_outage(tmp_path, monkeypatch):
    monkeypatch.setenv('CHILD_GAZE_SSH_STARTUP_DEADLINE', '1200')
    failed = lambda args, **kw: result(args, code=255, error='Permission denied (publickey)')
    state = tmp_path / 'transport.json'
    with pytest.raises(rental.TransportUnavailable):
        rental.run_transport(['ssh', 'host'], state_file=state, attempts=1,
                             clock=lambda: 100, invoke=failed)
    assert not state.exists()


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


def test_long_initial_upload_failure_still_gets_resumable_retry():
    now = [0.]
    timeouts = []
    def invoke(args, **kwargs):
        timeouts.append(kwargs['timeout'])
        if len(timeouts) == 1:
            now[0] += 900
            return result(args, code=255, error='Connection reset')
        return result(args, 'resumed')
    out = rental.run_transport(['rsync', 'source', 'host:dest'], timeout=1750,
                                invoke=invoke, clock=lambda: now[0],
                                sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    assert out.stdout == 'resumed' and timeouts == [1750, 295]


@pytest.mark.parametrize('recovered', [101., 1000.])
def test_fetch_first_recovery_preserves_contract_for_retrieval(tmp_path, monkeypatch, recovered):
    monkeypatch.setenv('CHILD_GAZE_RECOVERY_POLICY', 'fetch-first')
    state = tmp_path / 'state.json'
    rental._state(state, 'ssh', True, 100.)
    out = rental.run_transport(['ssh', 'host'], state_file=state, clock=lambda: recovered,
                               invoke=lambda args, **kw: result(args, 'available'))
    value = json.loads(state.read_text())
    assert out.stdout == 'available' and value['recovery_fetch_required']
    assert not value.get('stop_requested')


def test_non_teardown_never_starts_after_hard_cap(monkeypatch):
    monkeypatch.setenv('CHILD_GAZE_HARD_DEADLINE', '1')
    monkeypatch.setattr(rental, 'run_transport', lambda *args, **kw: pytest.fail('must not call transport'))
    with pytest.raises(rental.TransportUnavailable, match='hard deadline'):
        rental.run_connected(['ssh', 'host'])


def test_real_six_worker_first_epoch_projection():
    gate = rental.real_epoch_projection(epoch_seconds=[30.] * 6, setup_seconds=[20.] * 6,
                                        elapsed_seconds=1200, hourly=4.)
    assert gate['passes'] and gate['remaining_epochs'] == 59
    assert gate['projected_launch_to_verified_results_seconds'] == 4637.5
    assert not rental.real_epoch_projection(epoch_seconds=[80.] * 6, setup_seconds=[20.] * 6,
                                            elapsed_seconds=1200, hourly=4.)['passes']
    with pytest.raises(ValueError):
        rental.real_epoch_projection(epoch_seconds=[30.] * 5, setup_seconds=[20.] * 6,
                                      elapsed_seconds=1200, hourly=4.)
