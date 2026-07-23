from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/acquire_childlens_alignment_bridge_expansion_v3.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "childlens_alignment_bridge_v3_acquisition_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_path_strips_only_bound_release_prefix() -> None:
    module = _module()
    assert module._repository_path("/ChildLens/videos/example") == "/videos/example"
    assert module._repository_path("/videos/example") == "/videos/example"
    with pytest.raises(module.AcquisitionError, match="E_REPOSITORY_PATH"):
        module._repository_path("/ChildLens/videos/../example")


def test_public_guard_rejects_restricted_paths() -> None:
    module = _module()
    with pytest.raises(module.AcquisitionError, match="E_PUBLIC_PRIVACY"):
        module._public_guard({"value": "/Users/restricted/example"})
