"""Bounded transport retries and provider readiness for child-gaze rentals."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


class TransportUnavailable(RuntimeError):
    pass


class ProlongedOutage(TransportUnavailable):
    pass


class AmbiguousCreate(TransportUnavailable):
    pass


def _state(path, service, failed, now):
    if path is None:
        return False
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(path.read_text()) if path.exists() else {'outages': {}}
        outage = value['outages'].get(service)
        if failed:
            value['outages'].setdefault(service, {'started': now})['last_failed'] = now
        elif outage:
            duration = now - outage['started']
            value.setdefault('recoveries', []).append(dict(service=service, seconds=duration, utc=now))
            if os.environ.get('CHILD_GAZE_RECOVERY_POLICY') == 'fetch-first':
                value['recovery_fetch_required'] = True
            elif duration > 600:
                value['stop_requested'] = True
            del value['outages'][service]
        value['updated'] = now
        pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(path)
        return value.get('stop_requested', False)


def transient(text):
    text = text.lower()
    return any(word in text for word in (
        'nameresolutionerror', 'failed to resolve', 'temporary failure in name resolution',
        'nodename nor servname', 'could not resolve', 'connection refused',
        'connection reset', 'connection timed out', 'connectionerror', 'timed out',
        'network is unreachable', 'no route to host', 'connection closed',
        'connection unexpectedly closed', 'permission denied (publickey)',
        'remote host has closed', 'sslerror', 'remote end closed',
    ))


def run_transport(command, *, state_file=None, attempts=10, retry_seconds=300,
                  teardown=False, clock=time.time, sleep=time.sleep,
                  invoke=subprocess.run, **kwargs):
    """Retry only transport failures; create requests are always reconciled."""
    service = Path(str(command[0])).name
    if service not in ('vastai', 'ssh', 'rsync'):
        return invoke(command, **kwargs)
    state_file = state_file or os.environ.get('CHILD_GAZE_TRANSPORT_STATE')
    check = kwargs.pop('check', False)
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    kwargs.setdefault('timeout', 30)
    retry_started = None
    result = None
    create = service == 'vastai' and list(command[1:3]) == ['create', 'instance']
    for attempt in range(attempts):
        try:
            options = dict(kwargs)
            if retry_started is not None:
                options['timeout'] = min(options['timeout'], max(.1, retry_seconds - (clock()-retry_started)))
            result = invoke(command, **options)
        except subprocess.TimeoutExpired as error:
            result = subprocess.CompletedProcess(command, 124, '', str(error))
        output = str(result.stdout or '') + '\n' + str(result.stderr or '')
        if result.returncode == 0:
            stop = _state(state_file, service, False, clock())
            if stop and not teardown:
                raise ProlongedOutage('Connectivity recovered after an outage over ten minutes; teardown required')
            return result
        if not transient(output) and result.returncode != 124:
            if check:
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            return result
        if retry_started is None:
            retry_started = clock()
        readiness_failure = service == 'ssh' and clock() < float(os.environ.get('CHILD_GAZE_SSH_STARTUP_DEADLINE', '0')) and any(
            word in output.lower() for word in ('permission denied (publickey)', 'connection refused', 'connection closed'))
        if service == 'ssh' and os.environ.get('CHILD_GAZE_INSTANCE_ID'):
            # Provider logs must precede the first SSH recovery action.
            try:
                logs = run_transport(['vastai', 'logs', os.environ['CHILD_GAZE_INSTANCE_ID'],
                                      '--tail', '100', '--raw'], state_file=state_file,
                                     teardown=True)
                if state_file:
                    with Path(state_file).with_name('ssh_failure_logs.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(utc=time.time(), code=logs.returncode,
                                                     stdout=logs.stdout, stderr=logs.stderr)) + '\n')
            except TransportUnavailable:
                pass
            if state_file and Path(state_file).exists() and json.loads(Path(state_file).read_text()).get('stop_requested') and not teardown:
                raise ProlongedOutage('Provider recovered during SSH diagnostics after ten-minute outage; teardown required')
        if not readiness_failure:
            _state(state_file, service, True, clock())
        if create:
            raise AmbiguousCreate('Create response uncertain; reconcile task label before another create')
        elapsed = clock() - retry_started
        if attempt + 1 >= attempts or elapsed >= retry_seconds:
            break
        sleep(min(40, 5 * 2 ** attempt, max(0, retry_seconds - elapsed)))
    raise TransportUnavailable(f'{service} unavailable after {attempt + 1} tries: {output[-2000:]}')


def run_connected(command, *, teardown=False, **kwargs):
    """Continue bounded retry cycles; recovery after ten minutes stops science."""
    while True:
        deadline = float(os.environ.get('CHILD_GAZE_HARD_DEADLINE', 'inf'))
        if not teardown and time.time() >= deadline:
            raise TransportUnavailable('Approved hard deadline reached')
        state_file = kwargs.get('state_file') or os.environ.get('CHILD_GAZE_TRANSPORT_STATE')
        if not teardown and state_file and Path(state_file).exists() and json.loads(Path(state_file).read_text()).get('stop_requested'):
            raise ProlongedOutage('Prolonged-outage stop is pending; teardown required')
        try:
            return run_transport(command, teardown=teardown, **kwargs)
        except ProlongedOutage:
            raise
        except AmbiguousCreate:
            raise
        except TransportUnavailable:
            deadline = float(os.environ.get('CHILD_GAZE_HARD_DEADLINE', 'inf'))
            if not teardown and time.time() >= deadline:
                raise
            time.sleep(5)


def real_epoch_projection(*, epoch_seconds, setup_seconds, elapsed_seconds,
                          hourly, hard_seconds=7200, transfer_reserve=.25,
                          return_reserve=900, contingency=300):
    """Conservative cap gate from the first epochs of the actual six runs."""
    if len(epoch_seconds) != 6 or len(setup_seconds) != 6 or min(epoch_seconds) <= 0:
        raise ValueError('All six real workers must supply a positive epoch duration')
    remaining = 1.25 * (59 * max(epoch_seconds) + max(setup_seconds)) + return_reserve + contingency
    total = elapsed_seconds + remaining
    cost = total / 3600 * hourly + transfer_reserve
    return dict(passes=total <= hard_seconds and cost <= 15,
                projected_remaining_seconds=remaining,
                projected_launch_to_verified_results_seconds=total,
                projected_cost_usd=cost, max_epoch_seconds=max(epoch_seconds),
                max_setup_seconds=max(setup_seconds), headroom=1.25,
                remaining_epochs=59, return_reserve_seconds=return_reserve,
                contingency_seconds=contingency)


def create_instance(command, *, label, state_file=None, runner=run_transport,
                    clock=time.time, sleep=time.sleep):
    """Never repeat an uncertain create; first find its unique task label."""
    if '--label' not in command or command[command.index('--label') + 1] != label:
        raise ValueError('Unique task label is required for create reconciliation')
    try:
        result = runner(command, state_file=state_file, attempts=1, timeout=90, check=True)
        value = json.loads(result.stdout)
        if value.get('success') and value.get('new_contract'):
            return value
    except (TransportUnavailable, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    # The official inventory is the authority even if create stdout was lost.
    started = clock()
    attempt = 0
    while attempt < 10:
        try:
            inventory = runner(['vastai', 'show', 'instances', '--raw'], state_file=state_file,
                               teardown=True, check=True)
        except TransportUnavailable:
            # Never repeat POST merely because its reconciliation lookup is offline.
            continue
        instances = json.loads(inventory.stdout)
        matches = [item for item in instances if item.get('label') == label]
        if len(matches) == 1:
            return {'success': True, 'new_contract': matches[0]['id'], 'reconciled': True}
        if len(matches) > 1 or clock() - started >= 300:
            break
        sleep(min(40, 5 * 2 ** attempt))
        attempt += 1
    raise AmbiguousCreate(f'Create unresolved: {len(matches)} matching instances; do not create again')


def wait_running(instance, *, started, state_file=None, receipt=None, timeout=1200,
                 runner=run_transport, clock=time.monotonic, sleep=time.sleep):
    """Do not attempt SSH while the image/container is still loading."""
    while clock() - started < timeout:
        try:
            result = runner(['vastai', 'show', 'instance', str(instance), '--raw'],
                            state_file=state_file, check=True)
            value = json.loads(result.stdout)
            status = value.get('actual_status') or value.get('cur_state')
            if receipt:
                safe = {key: value.get(key) for key in ('id', 'actual_status', 'cur_state',
                        'intended_status', 'next_state', 'status_msg', 'public_ipaddr',
                        'ports', 'machine_dir_ssh_port')}
                Path(receipt).write_text(json.dumps(dict(utc=time.time(), elapsed_seconds=clock()-started,
                                                        provider=safe), indent=2) + '\n')
            if status in ('exited', 'unknown', 'offline', 'error') or value.get('intended_status') == 'stopped' or value.get('next_state') == 'stopped':
                raise RuntimeError('Provider reported a terminal startup error: ' + str(status))
            if status == 'running':
                logs = runner(['vastai', 'logs', str(instance), '--tail', '100', '--raw'],
                              state_file=state_file, check=True)
                if 'No such container' not in (logs.stdout or ''):
                    return value
        except ProlongedOutage:
            raise
        except TransportUnavailable:
            # Preserve the cross-phase outage clock; another bounded retry cycle
            # may recover without abandoning a healthy, still-loading rental.
            continue
        sleep(15)
    raise TimeoutError('Provider did not finish startup within twenty minutes')
