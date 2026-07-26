from pathlib import Path

import numpy as np

from babyworld_lite.childlens_engine_bakeoff.determinism import compare


def test_trace_comparison_detects_numeric_difference(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, time_s=np.asarray([0.0, 1.0]), phase=np.asarray(["a", "b"]))
    np.savez(second, time_s=np.asarray([0.0, 1.0 + 2e-8]), phase=np.asarray(["a", "b"]))
    assert not compare(first, second, atol=1e-9)["all_pass"]
    assert compare(first, second, atol=1e-7)["all_pass"]
