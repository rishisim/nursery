from __future__ import annotations

import json
from pathlib import Path

import pytest

from nursery_egobaby_preflight.contract import (
    canonical_json_sha256,
    lexical_macro_wiring,
    schedule_cycle,
)


def test_frozen_schedule_is_exact_4_1_1() -> None:
    config = json.loads(Path("configs/egobaby_cuda_preflight.json").read_text())
    assert schedule_cycle(config["learner"]["schedule"]) == [
        "contrastive",
        "contrastive",
        "contrastive",
        "contrastive",
        "mlm",
        "dinov2",
    ]


def test_config_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})


def test_lexical_wiring_requires_noun_then_adjective() -> None:
    result = lexical_macro_wiring({"noun": [1, 0], "adjective": [0, 1]})
    assert result == {"noun": 0.5, "adjective": 0.5, "lexical_macro": 0.5}
    with pytest.raises(ValueError, match="ordered exactly"):
        lexical_macro_wiring({"adjective": [0, 1], "noun": [1, 0]})


def test_phase3_preregistration_preserves_frozen_contract() -> None:
    config = json.loads(Path("configs/synthetic_video_preregistration.json").read_text())
    assert config["status"] == "INFRASTRUCTURE_PERMISSION_GATE"
    assert schedule_cycle(config["learner"]["schedule"]) == [
        "contrastive",
        "contrastive",
        "contrastive",
        "contrastive",
        "mlm",
        "dinov2",
    ]
    assert config["comparisons"]["primary"] == (
        "Synthetic-full_minus_Real-full_two_sided_equivalence"
    )
    assert config["endpoint"]["common_asset_across_arms"] is True
    assert config["analysis"]["equivalence_margin_bounds_absolute"] == [0.02, 0.05]
    assert config["analysis"]["minimum_common_seeds"] >= 3
    assert sum(config["split"]["target_proportions"].values()) == 1.0
    assert sum(config["split"]["minimum_children"].values()) == 8
    assert config["unblinding"]["synthetic_scores_sealed_until_real_only_gate_passes"] is True
    assert config["cost"]["existing_childlens_cost_is_zero"] is False
    assert config["gates"]["phase4_authorized"] is False
    assert config["gates"]["childlens_academic_noncommercial_access"] == "ESTABLISHED"
    assert config["gates"]["authorized_read_only_aggregate_inventory"] == (
        "AUTHORIZED_BUT_NOT_RUN"
    )
    assert config["gates"]["personnel_model"] == (
        "SINGLE_AUTHORIZED_APPLICANT_WITH_STAGED_PROCEDURAL_ROLES"
    )
    assert config["unblinding"]["personnel_independence"] is False
    assert all(value == "MISSING" for value in (
        config["gates"]["institutional_storage_and_derived_export_qualification"],
        config["gates"]["governed_cuda_qualification"],
    ))
