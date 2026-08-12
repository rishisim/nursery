#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "pilot_p3_config.json"
AGGREGATE = ROOT / "pilots" / "juno_sample" / "p3_aggregate.json"
EXPECTED_LABELS = ["engineering_only", "non_comparable", "not_a_reproduction_result"]


def require(value, message):
    if not value:
        raise AssertionError(message)


def load_p3():
    spec = importlib.util.spec_from_file_location("pilot_p3", ROOT / "pilot_p3.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def expect_failure(call, message):
    try:
        call()
    except Exception:
        return
    raise AssertionError(message)


def synthetic_tests(config):
    p3 = load_p3()
    p3.validate_config(config)
    records = [{"record_key": "a", "sha256": "0" * 64, "size_bytes": 1, "status": "verified"},
               {"record_key": "b", "sha256": "1" * 64, "size_bytes": 1, "status": "completed"}]
    selected = p3.select_records({"queue": records})
    require(tuple(selected) == p3.SLOTS, "opaque two-input addressing failed")
    expect_failure(lambda: p3.select_records({"queue": records[:1]}), "one input accepted")
    expect_failure(lambda: p3.select_records({"queue": records + records[:1]}), "three inputs accepted")
    config_hash = "2" * 64; tool_hash = "3" * 64; source_hash = "4" * 64
    key = p3.cache_key(source_hash, config_hash, tool_hash)
    require(key == p3.cache_key(source_hash, config_hash, tool_hash), "cache key is nondeterministic")
    require(len({key, p3.cache_key("5" * 64, config_hash, tool_hash),
                 p3.cache_key(source_hash, "6" * 64, tool_hash),
                 p3.cache_key(source_hash, config_hash, "7" * 64)}) == 4, "cache invalidation failed")

    annotations = [{"start": 2.0, "end": 3.0, "label": "KCHI"}]
    require(p3.overlaps(1.5, 2.5, annotations) and not p3.overlaps(1.0, 2.0, annotations),
            "half-open KCHI overlap failed")
    raw = {"segments": [
        {"start": 0, "end": 1, "text": "raw", "words": [
            {"word": "keep", "score": 0.5}, {"word": "drop", "score": 0.4999},
            {"word": "missing"}]},
        {"start": 2.5, "end": 3.5, "text": "child", "words": [{"word": "x", "score": 1.0}]},
        {"start": 4, "end": 5, "text": "empty", "words": []},
    ]}
    normalized, counts = p3.normalize_segments(raw, annotations, 0.5)
    require(len(normalized) == 1 and normalized[0]["normalized_text"] == "keep", "confidence boundary failed")
    require(counts["confidence_filtered_words"] == 1 and counts["empty_score_words"] == 1,
            "empty-word accounting failed")
    require(counts["kchi_removed_segments"] == 1 and counts["empty_after_filter_utterances"] == 1,
            "filter reason accounting failed")
    selected_frames = p3.linear_selection(list(range(100)), 32)
    require(len(selected_frames) == 32 and selected_frames[0] == 0 and selected_frames[-1] == 99,
            "linear frame cap failed")
    require(p3.frame_candidates(1.0, 3.0, 5) == [1, 2, 3], "one-Hz interval alignment failed")
    visual, text, paired = p3.build_manifests("input_1", "opaque-a", 40, normalized)
    require(all(item["source"] == "opaque-a" for item in visual + text + paired), "cross-record pairing")

    with tempfile.TemporaryDirectory() as name:
        root = Path(name); stage = root / "sample"
        calls = []
        def builder(output):
            calls.append(1); (output / "value").write_text("fixture", encoding="utf-8")
            return {"count": 1}
        first, hit1 = p3.commit_stage(stage, key, {}, builder)
        second, hit2 = p3.commit_stage(stage, key, {}, builder)
        require(first == second and not hit1 and hit2 and len(calls) == 1, "atomic resume/skip failed")
        require(stat.S_IMODE((stage / "complete.json").stat().st_mode) == 0o600,
                "completion marker is not owner-only")


def main():
    config_text = CONFIG.read_text(encoding="utf-8")
    aggregate_text = AGGREGATE.read_text(encoding="utf-8")
    config = json.loads(config_text); aggregate = json.loads(aggregate_text)
    synthetic_tests(config)
    require(aggregate["status"] == "p3_complete" and aggregate["p3_gate"]["passed"], "P3 gate failed")
    require(aggregate["classification"] == EXPECTED_LABELS, "classification changed")
    require(aggregate["input_count"] == 2 and aggregate["inventory_status"] == "incomplete_inventory",
            "fixed input/inventory contract failed")
    require(aggregate["scientific_status_effect"] == "none" and aggregate["next_stage_started"] is False,
            "scientific or later-stage status changed")
    require(all(aggregate[key] for key in ("p0_status_preserved", "p1_status_preserved", "p2_status_preserved")),
            "prior pilot status changed")
    require(aggregate["verification"] == {"source_integrity_count": 2, "audio_contract_count": 2,
            "qa_passed_count": 2, "counts_reconciled": True}, "P3 reconciliation failed")
    require(aggregate["counts"]["retained_utterances"] > 0 and aggregate["counts"]["paired_records"] > 0,
            "no retained paired examples")
    require(aggregate["resume"]["controlled_interruption_tested"] and
            aggregate["resume"]["idempotent_rerun_passed"], "resume/idempotence gate failed")
    require(aggregate["provenance"]["p3_config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "config provenance mismatch")
    forbidden = ["/work/", "/scratch/", "participant", "session", "record_key", "media_path",
                 "filename", "asset_id", "transcript", "utterance_text", "source_sha256", "stderr"]
    require(not any(token in aggregate_text.lower() for token in forbidden), "aggregate violates privacy boundary")
    print("Pilot P3 verification passed")


if __name__ == "__main__":
    main()
