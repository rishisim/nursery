"""Freeze Step 6 acquisition curves or run a tiny CPU smoke. Never rent hosts."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nursery.learner import acquisition as run, acquisition_data as data
from scripts.step4_body_comparison import ARMS, comparison_protocol, smoke as comparison_smoke

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = REPO / 'outputs/runs/child_gaze/steering/frozen'
DEFAULT_COMPARISON = REPO / 'outputs/runs/step4_body_comparison/frozen'
DEFAULT_ROOT = REPO / 'outputs/runs/step6_acquisition_curves'
CURVE_ARMS = {arm: spec for arm, spec in ARMS.items() if arm != 'video_gaze_body_rich'}


def curve_protocol(parent):
    protocol = comparison_protocol(parent)
    protocol.update(conditions=list(CURVE_ARMS), arms={arm: protocol['arms'][arm] for arm in CURVE_ARMS},
                    fractions=list(run.FRACTIONS))
    return protocol


def check_preparation(destination, comparison):
    """Require byte-identical full data/panels, strict nesting and no held-out clips."""
    hashes = {}
    for name in ('subsets/1.0', 'panels'):
        expected = {p.relative_to(comparison) for p in (comparison / name).rglob('*') if p.is_file()}
        actual = {p.relative_to(destination) for p in (destination / name).rglob('*') if p.is_file()}
        assert expected and actual == expected, f'File set differs: {name}'
        for relative in sorted(expected):
            digest = run._sha(destination / relative)
            assert digest == run._sha(comparison / relative), f'Hash mismatch: {relative}'
            hashes[str(relative)] = digest
    splits = run._read(destination / 'splits.json')['source_keys']
    held_out = set().union(*(set(splits[key]) for key in ('dev', 'test', 's8')))
    previous = set()
    for fraction in run.FRACTIONS:
        selected = set(run._read(destination / 'subsets' / str(fraction) / 'stats.json')['selected_clips'])
        assert previous < selected, f'Subsets not strictly nested: {fraction}'
        assert not selected & held_out, f'Held-out clips in subset: {fraction}'
        previous = selected
    return dict(passed=True, sha256=hashes, strictly_nested=True, held_out_disjoint=True)


def freeze(parent=DEFAULT_PARENT, destination=DEFAULT_ROOT / 'frozen', *, comparison=DEFAULT_COMPARISON):
    parent, destination, comparison = (Path(p).resolve() for p in (parent, destination, comparison))
    subprocess.run(['git', '-C', str(REPO), 'check-ignore', str(destination)], check=True,
                   stdout=subprocess.DEVNULL)
    if destination.exists():
        raise FileExistsError(f'Frozen destination already exists: {destination}')
    steering = run._read(parent / 'protocol.json')
    baseline = run._read(comparison / 'protocol.json')
    assert steering['subset_seed'] == steering['split_seed'] == 0, 'prepare uses seed 0'
    assert all(steering[key] == baseline[key] for key in ('inputs', 'episodes', 'subset_seed', 'split_seed', 'gaze'))
    prepared = data.prepare(destination, steering['inputs'], steering['episodes'],
                            fractions=list(run.FRACTIONS), conditions=['video_only'])
    checks = check_preparation(destination, comparison)
    protocol = curve_protocol(baseline)
    protocol['subsets'] = prepared['subsets']
    protocol.update(run._source_snapshot())
    for path in (Path(__file__).resolve(), REPO / 'scripts/step4_body_comparison.py'):
        protocol['code_hashes'][str(path)] = run._sha(path)
    run._write(destination / 'protocol.json', protocol)
    run._write(destination / 'preparation_checks.json', checks)
    return run._freeze_inputs(destination)


def smoke(frozen=DEFAULT_ROOT / 'frozen', destination=DEFAULT_ROOT / 'smoke', *, clips=8):
    return comparison_smoke(frozen, destination, device='cpu', clips=clips, epochs=1,
                            fraction=.1, arms=CURVE_ARMS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'smoke'])
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--comparison', type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--clips', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.parent, args.root / 'frozen', comparison=args.comparison)
    else:
        smoke(args.root / 'frozen', args.root / 'smoke', clips=args.clips)
