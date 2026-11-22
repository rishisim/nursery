"""Freeze the six-run steering variant against Step 2's unchanged inputs."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nursery.learner import acquisition as run

BASE_REVISION = 'bfa470c64e5ca2c531bd0ee520e378a1bd467d67'
CONDITIONS = ['video_gaze', 'video_shuffled_gaze']


def steering_protocol(parent):
    """Change only the requested readout and matrix conditions."""
    if parent['gaze']['option'] != 'additive':
        raise ValueError('Expected the frozen additive parent')
    child = copy.deepcopy(parent)
    child['gaze']['option'] = 'steering'
    child['conditions'] = list(CONDITIONS)
    return child


def verify_parent(parent, repo):
    """Permit reporting/orchestration commits, never changed scientific source."""
    frozen = run._read(parent / 'frozen_inputs.json')
    if frozen['revision'] != BASE_REVISION:
        raise ValueError('Unexpected Step 2 source revision')
    changed = subprocess.check_output(
        ['git', '-C', str(repo), 'diff', BASE_REVISION, '--', 'nursery'], text=True)
    if changed:
        raise ValueError('nursery/ changed since Step 2')
    if run._source_snapshot()['code_hashes'] != frozen['code_hashes']:
        raise ValueError('Frozen scientific source hashes differ')
    for group in ('prepared_hashes', 'external_hashes'):
        for filename, expected in frozen[group].items():
            if run._sha(filename) != expected:
                raise ValueError('Frozen input changed: ' + filename)
    return frozen


def freeze(parent, destination):
    repo = Path(__file__).resolve().parents[1]
    original = verify_parent(parent, repo)
    if not run._read(parent / 'protocol.json')['subsets']['1.0']['selected_clip_count'] == 630:
        raise ValueError('Expected 630 clips')
    subprocess.run(['git', '-C', str(repo), 'check-ignore', str(destination)], check=True,
                   stdout=subprocess.DEVNULL)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('panels', 'subsets'):
        shutil.copytree(parent / name, destination / name)
    for name in ('splits.json', 'signal_hashes.json'):
        shutil.copyfile(parent / name, destination / name)
    run._write(destination / 'protocol.json', steering_protocol(run._read(parent / 'protocol.json')))
    snapshot = run._freeze_inputs(destination)
    run._write(destination / 'parent_binding.json', {
        'parent_revision': original['revision'],
        'execution_revision': snapshot['revision'],
        'parent_protocol_sha256': run._sha(parent / 'protocol.json'),
        'parent_frozen_inputs_sha256': run._sha(parent / 'frozen_inputs.json'),
        'allowed_protocol_changes': ['conditions', 'gaze.option'],
    })
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    freeze(args.parent.resolve(), args.destination.resolve())
