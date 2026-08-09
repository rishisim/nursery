#!/usr/bin/env python3
"""Compute and validate immutable ChildLens v1/v1.1 artifact-set baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


EXPECTED_V1_COUNT = 23
EXPECTED_V1_DIGEST = "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"
EXPECTED_V1_1_COUNT = 40
EXPECTED_V1_1_DIGEST = "3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48"

V1_PATHS = (
    "docs/childlens_feasibility_v1",
    "output/childlens_feasibility_v1",
    "scripts/validate_childlens_feasibility_v1.py",
    "tests/test_childlens_feasibility_v1.py",
)

V1_1_DIRECTORY_PATHS = (
    "docs/childlens_feasibility_v1_1",
    "output/childlens_feasibility_v1_1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand(root: Path, entries: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    return files


def v1_files(root: Path) -> list[Path]:
    return sorted(
        _expand(root, V1_PATHS),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def v1_1_files(root: Path) -> list[Path]:
    files = _expand(root, V1_1_DIRECTORY_PATHS)
    for parent_name in ("scripts", "tests"):
        parent = root / parent_name
        if parent.is_dir():
            files.extend(path for path in parent.glob("*v1_1*") if path.is_file())
    unique = {path.resolve(): path for path in files}
    return sorted(
        unique.values(),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def canonical_artifact_set(root: Path, files: list[Path]) -> tuple[str, str]:
    lines = sorted(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files
    )
    serialized = "".join(lines)
    return serialized, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_receipt(root: Path, observed_date: date) -> dict:
    root = root.resolve()
    v1 = v1_files(root)
    v1_1 = v1_1_files(root)
    _, v1_digest = canonical_artifact_set(root, v1)
    _, v1_1_digest = canonical_artifact_set(root, v1_1)
    return {
        "schema_version": "childlens-immutable-baseline-v1.2.0",
        "captured_date": observed_date.isoformat(),
        "canonical_serialization": (
            "UTF-8 lines lexicographically sorted as complete lines; each line is "
            "SHA256 two ASCII spaces repository-relative-PATH newline"
        ),
        "path_normalization": "POSIX_REPOSITORY_RELATIVE_NO_LEADING_DOT_SLASH",
        "v1": {
            "include_rules": list(V1_PATHS),
            "artifact_count": len(v1),
            "artifact_set_digest": v1_digest,
            "expected_artifact_count": EXPECTED_V1_COUNT,
            "expected_artifact_set_digest": EXPECTED_V1_DIGEST,
            "matches_prior_baseline": len(v1) == EXPECTED_V1_COUNT and v1_digest == EXPECTED_V1_DIGEST,
            "historical_decision_preserved": True,
        },
        "v1_1": {
            "include_rules": [
                *V1_1_DIRECTORY_PATHS,
                "scripts/*v1_1*",
                "tests/*v1_1*",
            ],
            "artifact_count": len(v1_1),
            "artifact_set_digest": v1_1_digest,
            "expected_artifact_count": EXPECTED_V1_1_COUNT,
            "expected_artifact_set_digest": EXPECTED_V1_1_DIGEST,
            "matches_v1_2_baseline": (
                len(v1_1) == EXPECTED_V1_1_COUNT
                and v1_1_digest == EXPECTED_V1_1_DIGEST
            ),
            "historical_decision_preserved": True,
        },
        "mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-v1-1-count", type=int, default=EXPECTED_V1_1_COUNT)
    parser.add_argument("--expect-v1-1-digest", default=EXPECTED_V1_1_DIGEST)
    args = parser.parse_args()

    receipt = build_receipt(args.repository_root, date.today())
    v1_pass = receipt["v1"]["matches_prior_baseline"]
    v1_1_pass = True
    if args.expect_v1_1_count is not None:
        v1_1_pass = v1_1_pass and receipt["v1_1"]["artifact_count"] == args.expect_v1_1_count
    if args.expect_v1_1_digest is not None:
        v1_1_pass = v1_1_pass and receipt["v1_1"]["artifact_set_digest"] == args.expect_v1_1_digest
    receipt["validation_status"] = "PASS" if v1_pass and v1_1_pass else "FAIL"

    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if v1_pass and v1_1_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
