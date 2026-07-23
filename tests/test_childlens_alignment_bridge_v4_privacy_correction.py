from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/correct_childlens_alignment_bridge_v4_public_privacy.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "childlens_alignment_bridge_v4_privacy_correction_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suppressed_label_names_are_collapsed() -> None:
    result = _module().collapse_suppressed_labels(
        {
            "safe": {"participant_share": 0.5, "suppressed": False},
            "private-small-cell": {
                "participant_share": None,
                "suppressed": True,
            },
        }
    )
    assert result["reportable_categories"] == {
        "safe": {"participant_share": 0.5, "suppressed": False}
    }
    assert result["suppressed_category_count"] == 1
    assert result["suppressed_category_labels_exported"] is False
    assert "private-small-cell" not in str(result)
