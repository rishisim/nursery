from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/measure_childlens_alignment_bridge_expansion_v3.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "childlens_alignment_bridge_v3_measurement_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_summary() -> dict:
    return {
        "participant_count": 10,
        "accepted_segment_count": 100,
        "accepted_speech_seconds": 300.0,
        "nonempty_segment_participant_fraction": 1.0,
        "annotation_precision_median": 0.9,
        "annotation_recall_median": 0.8,
        "boundary_f1_median": 0.9,
        "boundary_f1_bootstrap_90pct": (0.7, 0.95),
        "duration_change_median": 0.05,
        "coverage_ratio_median": 1.0,
        "primary_nonempty_item_fraction": 1.0,
        "sensitivity_nonempty_item_fraction": 1.0,
        "primary_usable_segment_count": 80,
        "primary_usable_segment_fraction": 0.8,
        "primary_german_item_fraction": 0.9,
        "language_id_agreement_fraction": 0.9,
        "character_similarity_median": 0.8,
        "character_similarity_bootstrap_90pct": (0.6, 0.9),
        "embedding_cosine_median": 0.9,
        "embedding_cosine_bootstrap_90pct": (0.8, 0.95),
        "primary_character_self_median": 0.9,
        "sensitivity_character_self_median": 0.9,
        "primary_embedding_self_median": 0.95,
        "sensitivity_embedding_self_median": 0.95,
    }


def test_new_cohort_safeguard_omits_only_bootstrap_lower_requirements() -> None:
    module = _module()
    gates = json.loads(
        (
            ROOT / "configs/childlens_alignment_bridge_expansion_v3.json"
        ).read_text()
    )["gates"]
    summary = _passing_summary()
    summary["boundary_f1_bootstrap_90pct"] = (0.1, 0.95)
    summary["character_similarity_bootstrap_90pct"] = (0.1, 0.9)
    summary["embedding_cosine_bootstrap_90pct"] = (0.1, 0.95)
    expansion_checks = module._checks(summary, gates, require_bootstrap=False)
    combined_checks = module._checks(summary, gates, require_bootstrap=True)
    assert all(expansion_checks.values())
    assert not combined_checks["G1_boundary_lower"]
    assert not combined_checks["G2_character_lower"]
    assert not combined_checks["G2_embedding_lower"]


def test_locked_status_never_authorizes_automatic_evaluation() -> None:
    module = _module()
    result = module.locked_status()
    assert result["locked_evaluation_authorized"] is False
    assert result["locked_rows_loaded_or_evaluated"] == 0


def test_public_summary_suppresses_small_complementary_participant_cells() -> None:
    module = _module()
    summary = _passing_summary()
    summary["primary_nonempty_item_fraction"] = 0.9
    safe = module._rounded_summary(summary)
    assert safe["primary_nonempty_item_fraction"] is None
    assert safe["primary_nonempty_item_fraction_suppressed"] is True
    assert safe["nonempty_segment_participant_fraction"] == 1.0
    assert safe["nonempty_segment_participant_fraction_suppressed"] is False
