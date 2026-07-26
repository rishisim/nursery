from pathlib import Path

import numpy as np
import pytest

from babyworld_lite.childlens_engine_bakeoff.bundle import (
    build_manifest,
    trace_equivalence,
    validate_shared_clock,
)


def test_shared_clock_requires_t_zero_and_equal_stream_lengths():
    clock = np.array([0.0, 0.1, 0.2])
    receipt = validate_shared_clock(clock, {"qpos": np.zeros((3, 2)), "rgb": np.zeros((3, 1))})
    assert receipt["passed"]
    with pytest.raises(ValueError, match="t=0"):
        validate_shared_clock(clock + 0.1, {"qpos": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="length mismatch"):
        validate_shared_clock(clock, {"qpos": np.zeros((2, 2))})


def test_trace_equivalence_is_numeric_and_bounded():
    first = {"qpos": np.array([[1.0, 2.0]]), "phase": np.array(["reach"])}
    same = {"qpos": np.array([[1.0, 2.0]]), "phase": np.array(["reach"])}
    assert trace_equivalence(first, same, absolute_tolerance=1e-9)["passed"]
    changed = {"qpos": np.array([[1.0, 2.1]]), "phase": np.array(["reach"])}
    assert not trace_equivalence(first, changed, absolute_tolerance=1e-9)["passed"]


def test_manifest_hashes_only_declared_files(tmp_path: Path):
    (tmp_path / "trace.npz").write_bytes(b"trace")
    manifest = build_manifest(
        tmp_path,
        ["trace.npz"],
        spec_sha256="a" * 64,
        provenance={"private_childlens_material": False},
        regeneration_command=["python3", "-m", "runner"],
    )
    assert manifest["files"][0]["bytes"] == 5
    assert manifest["provenance"]["private_childlens_material"] is False
