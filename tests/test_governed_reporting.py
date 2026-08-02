from __future__ import annotations

import json

import pytest

from nursery_egobaby_preflight.contract import compact_aggregate_json


FIELDS = frozenset({"status", "count", "commitment_sha256"})


def test_compact_aggregate_allows_only_explicit_flat_aggregates() -> None:
    line = compact_aggregate_json(
        {"status": "PASS", "count": 94, "commitment_sha256": "a" * 64},
        allowed_fields=FIELDS,
        sha256_fields={"commitment_sha256"},
    )
    assert json.loads(line) == {
        "status": "PASS",
        "count": 94,
        "commitment_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    "unsafe",
    [
        {"status": "PASS", "count": [1, 2], "commitment_sha256": "a" * 64},
        {"status": "PASS", "count": {"row": 1}, "commitment_sha256": "a" * 64},
        {"status": "participant-name", "count": 1, "commitment_sha256": "a" * 64},
        {"status": "PASS", "count": 1, "commitment_sha256": "not-a-hash"},
    ],
)
def test_compact_aggregate_rejects_identifiers_containers_and_unapproved_hashes(
    unsafe: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="unsafe compact aggregate value"):
        compact_aggregate_json(
            unsafe, allowed_fields=FIELDS, sha256_fields={"commitment_sha256"}
        )


def test_compact_aggregate_rejects_unwhitelisted_fields() -> None:
    with pytest.raises(ValueError, match="fields do not match"):
        compact_aggregate_json(
            {
                "status": "PASS",
                "count": 1,
                "commitment_sha256": "a" * 64,
                "row_keys": ["b" * 64],
            },
            allowed_fields=FIELDS,
            sha256_fields={"commitment_sha256"},
        )
