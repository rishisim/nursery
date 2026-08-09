from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_childlens_author_audit_packet_v1_3.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_childlens_author_audit_packet_v1_3", SCRIPT
)
assert SPEC and SPEC.loader
sampler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampler
SPEC.loader.exec_module(sampler)


def _private_file(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.chmod(path, 0o600)


def _fixture(tmp_path: Path, *, seconds_by_item: list[int] | None = None):
    restricted = tmp_path / "restricted"
    restricted.mkdir(mode=0o700, parents=True)
    os.chmod(restricted, 0o700)
    (restricted / sampler.INDEX_SENTINEL).touch(mode=0o600)
    seconds_by_item = seconds_by_item or [180] * 15
    items = []
    for index, speech_seconds in enumerate(seconds_by_item):
        media_sha = __import__("hashlib").sha256(f"media-{index}".encode()).hexdigest()
        split = speech_seconds // 2
        windows = (
            [{"start_seconds": 0, "end_seconds": speech_seconds}]
            if speech_seconds < 2
            else [
                {"start_seconds": 0, "end_seconds": split},
                {
                    "start_seconds": split + 10,
                    "end_seconds": speech_seconds + 10,
                },
            ]
        )
        items.append(
            {
                "blinded_item_key": f"opaqueitemkey{index:04d}",
                "media_relpath": f"raw_v1_2/{media_sha}.bin",
                "expected_size_bytes": 1000 + index,
                "expected_media_sha256": media_sha,
                "reference_duration_seconds": speech_seconds + 20,
                "reference_duration_basis": "ANNOTATION_MAX_END",
                "speech_presence_expectation": "PRESENT",
                "speech_windows": windows,
            }
        )
    manifest = {
        "schema_version": sampler.INPUT_SCHEMA,
        "pilot_selection_sha256": sampler.FROZEN_V1_2_SELECTION_DIGEST,
        "items": items,
    }
    manifest_path = restricted / "official.json"
    _private_file(manifest_path, manifest)
    output_dir = restricted / "v1_3"
    output_dir.mkdir(mode=0o700)
    os.chmod(output_dir, 0o700)
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    return {
        "root": restricted,
        "manifest": manifest_path,
        "primary": output_dir / "primary.json",
        "reserve": output_dir / "reserve.json",
        "public": public_dir / "receipt.json",
        "manifest_value": manifest,
    }


def _run(paths):
    return sampler.prepare_author_audit_samples(
        restricted_root=paths["root"],
        official_windows_manifest_path=paths["manifest"],
        primary_packet_path=paths["primary"],
        reserve_packet_path=paths["reserve"],
        public_receipt_path=paths["public"],
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _duration(segments: list[dict]) -> int:
    return sum(row["end_ms"] - row["start_ms"] for row in segments)


def _overlap(first: list[dict], second: list[dict]) -> bool:
    return any(
        max(a["start_ms"], b["start_ms"]) < min(a["end_ms"], b["end_ms"])
        for a in first
        for b in second
    )


def test_preferred_primary_and_reserve_are_exact_disjoint_minutes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = _run(paths)
    primary = _load(paths["primary"])
    reserve = _load(paths["reserve"])
    assert receipt["status"] == "AUTHOR_AUDIT_SAMPLES_READY"
    assert receipt["primary_total_seconds"] == receipt["reserve_total_seconds"] == 900
    assert receipt["primary_one_minute_per_item"] is True
    assert receipt["reserve_one_minute_per_item"] is True
    assert receipt["model_prediction_independent"] is True
    assert receipt["audit_sample_frozen_before_predictions"] is True
    assert receipt["author_prediction_blinding_required"] is True
    assert receipt["deficit_redistribution_used"] is False
    assert len(primary["items"]) == len(reserve["items"]) == 15
    reserve_by_key = {row["audit_item_key"]: row for row in reserve["items"]}
    for row in primary["items"]:
        other = reserve_by_key[row["audit_item_key"]]
        assert row["duration_ms"] == other["duration_ms"] == 60_000
        assert _duration(row["segments"]) == _duration(other["segments"]) == 60_000
        assert not _overlap(row["segments"], other["segments"])
    assert stat.S_IMODE(paths["primary"].stat().st_mode) == 0o600
    assert stat.S_IMODE(paths["reserve"].stat().st_mode) == 0o600


def test_hash_selection_is_invariant_to_manifest_item_and_window_order(tmp_path: Path) -> None:
    first_paths = _fixture(tmp_path / "one")
    _run(first_paths)
    first_primary = _load(first_paths["primary"])["items"]
    first_reserve = _load(first_paths["reserve"])["items"]

    second_paths = _fixture(tmp_path / "two")
    manifest = second_paths["manifest_value"]
    manifest["items"].reverse()
    for item in manifest["items"]:
        item["speech_windows"].reverse()
    _private_file(second_paths["manifest"], manifest)
    _run(second_paths)
    assert _load(second_paths["primary"])["items"] == first_primary
    assert _load(second_paths["reserve"])["items"] == first_reserve


def test_prediction_or_lexical_field_is_rejected_not_silently_used(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = paths["manifest_value"]
    manifest["items"][0]["model_confidence"] = 0.99
    _private_file(paths["manifest"], manifest)
    with pytest.raises(sampler.SamplingError, match="E_ITEM_SCHEMA"):
        _run(paths)
    manifest["items"][0].pop("model_confidence")
    manifest["items"][0]["transcript"] = "synthetic lexical payload"
    _private_file(paths["manifest"], manifest)
    with pytest.raises(sampler.SamplingError, match="E_ITEM_SCHEMA"):
        _run(paths)


def test_deficit_is_prediction_blind_hash_redistributed_but_all_items_remain(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, seconds_by_item=[30] + [180] * 14)
    receipt = _run(paths)
    primary = _load(paths["primary"])
    reserve = _load(paths["reserve"])
    assert receipt["primary_one_minute_per_item"] is False
    assert receipt["reserve_one_minute_per_item"] is False
    assert receipt["deficit_redistribution_used"] is True
    assert sum(row["duration_ms"] for row in primary["items"]) == 900_000
    assert sum(row["duration_ms"] for row in reserve["items"]) == 900_000
    assert all(row["duration_ms"] >= 1000 for row in primary["items"])
    assert all(row["duration_ms"] >= 1000 for row in reserve["items"])
    assert {row["audit_item_key"] for row in primary["items"]} == {
        row["audit_item_key"] for row in reserve["items"]
    }


def test_total_or_cluster_coverage_shortfall_fails_closed(tmp_path: Path) -> None:
    total_short = _fixture(tmp_path / "total", seconds_by_item=[100] * 15)
    with pytest.raises(sampler.SamplingError, match="E_TOTAL_COVERAGE_INSUFFICIENT"):
        _run(total_short)
    cluster_short = _fixture(tmp_path / "cluster", seconds_by_item=[1] + [180] * 14)
    with pytest.raises(sampler.SamplingError, match="E_CLUSTER_COVERAGE_INSUFFICIENT"):
        _run(cluster_short)


def test_selection_digest_mismatch_and_media_binding_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "selection")
    manifest = paths["manifest_value"]
    manifest["pilot_selection_sha256"] = "f" * 64
    _private_file(paths["manifest"], manifest)
    with pytest.raises(sampler.SamplingError, match="E_SELECTION_BINDING"):
        _run(paths)

    paths = _fixture(tmp_path / "media")
    manifest = paths["manifest_value"]
    manifest["items"][0]["expected_media_sha256"] = "e" * 64
    _private_file(paths["manifest"], manifest)
    with pytest.raises(sampler.SamplingError, match="E_MEDIA_BINDING"):
        _run(paths)

    paths = _fixture(tmp_path / "traversal")
    manifest = paths["manifest_value"]
    manifest["items"][0]["media_relpath"] = (
        "../" + manifest["items"][0]["expected_media_sha256"] + ".bin"
    )
    _private_file(paths["manifest"], manifest)
    with pytest.raises(sampler.SamplingError, match="E_MEDIA_BINDING"):
        _run(paths)


def test_invalid_negative_official_window_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = paths["manifest_value"]
    manifest["items"][0]["speech_windows"][0]["start_seconds"] = -1
    _private_file(paths["manifest"], manifest)
    with pytest.raises(sampler.SamplingError, match="E_SPEECH_WINDOW"):
        _run(paths)


def test_public_receipt_has_no_row_level_or_interval_payload(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _run(paths)
    receipt_text = paths["public"].read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert "items" not in receipt
    assert "segments" not in receipt_text
    assert "start_ms" not in receipt_text
    assert "end_ms" not in receipt_text
    assert "media_relpath" not in receipt_text
    assert "audit_item_key" not in receipt_text
    assert "prediction_join_key" not in receipt_text
    assert "opaqueitemkey" not in receipt_text


def test_outputs_are_immutable_and_restricted_output_cannot_enter_repo(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "stable")
    first = _run(paths)
    assert _run(paths) == first
    paths["primary"].write_text("{}\n", encoding="utf-8")
    os.chmod(paths["primary"], 0o600)
    with pytest.raises(sampler.SamplingError, match="E_IMMUTABLE_OUTPUT_CONFLICT"):
        _run(paths)

    paths = _fixture(tmp_path / "boundary")
    with pytest.raises(sampler.SamplingError, match="E_RESTRICTED_PATH_BOUNDARY"):
        sampler.prepare_author_audit_samples(
            restricted_root=paths["root"],
            official_windows_manifest_path=paths["manifest"],
            primary_packet_path=ROOT / "forbidden-primary.json",
            reserve_packet_path=paths["reserve"],
        )
