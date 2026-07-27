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
