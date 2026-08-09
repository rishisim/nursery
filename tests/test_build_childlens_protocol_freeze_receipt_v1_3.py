from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_childlens_protocol_freeze_receipt_v1_3.py"
SPEC = importlib.util.spec_from_file_location("build_childlens_protocol_freeze_receipt_v1_3", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)
PROTOCOL = json.loads(builder.PROTOCOL.read_text(encoding="utf-8"))


def test_public_protocol_derives_complete_aggregate_freeze_receipt() -> None:
    receipt = builder.build_receipt(PROTOCOL, "a" * 64)
    assert receipt["status"] == "FROZEN"
    assert receipt["primary_audit_speech_seconds"] == 900
    assert receipt["maximum_additional_speech_seconds"] == 900
    assert receipt["second_human_required"] is False
    assert set(receipt["thresholds"]) == set(builder.GATE_MAP.values())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("author_audit", "author_blinded_to_predictions", False),
        ("scientific_boundaries", "hosted_model_content_access", True),
        ("bounded_escalation", "maximum_activations", 2),
        ("audit_sampling", "primary_target_total_ms", 899_000),
    ],
)
def test_safety_or_freeze_mutation_refuses_receipt(section: str, key: str, value: object) -> None:
    protocol = copy.deepcopy(PROTOCOL)
    protocol[section][key] = value
    with pytest.raises(builder.ReceiptError, match="E_PUBLIC_PROTOCOL_NOT_FROZEN"):
        builder.build_receipt(protocol, "a" * 64)


def test_threshold_mutation_cannot_generate_a_fresh_receipt() -> None:
    protocol = copy.deepcopy(PROTOCOL)
    protocol["agreement_gates"]["source_role"]["pass_thresholds"][
        "non_child_precision_min"
    ] = 0.5
    with pytest.raises(builder.ReceiptError, match="E_PUBLIC_PROTOCOL_NOT_FROZEN"):
        builder.build_receipt(protocol, "a" * 64)
