import hashlib
import json

import pytest

from nursery_egobaby_preflight.language_gate import (
    ARMS, chrf, corpus_wer, select_candidate, validate_common_assets,
    validate_timestamps,
)


def config():
    return json.loads(open("configs/synthetic_video_language_gate.json").read())


def passing(candidate_id, wer):
    return {
        "candidate_id": candidate_id,
        "public_corpus_wer": wer,
        "translation_chrf": 0.7,
        "mean_word_confidence": 0.8,
        "abstention_rate": 0.1,
        "timestamp_valid_fraction": 1.0,
        "round_trip_fraction": 1.0,
        "manifest_completeness_fraction": 1.0,
        "crashes": 0,
        "silent_truncations": 0,
        "offline_reload_pass": True,
        "telemetry_disabled": True,
        "artifact_manifest_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
        "runtime": {"wall_seconds": 1, "peak_rss_bytes": 1, "stored_bytes": 1},
    }


def test_metrics_and_timestamps():
    assert corpus_wer([{"reference": "Der rote Ball", "hypothesis": "der Ball"}]) == pytest.approx(1 / 3)
    assert chrf("the red cup", "the red cup") == 1
    assert validate_timestamps(
        [{"word": "der", "start": 0.0, "end": 0.2}, {"word": "Ball", "start": 0.2, "end": 0.5}], 0.5,
    )
    assert not validate_timestamps([{"word": "Ball", "start": 0.5, "end": 0.4}], 1.0)


def test_selection_is_closed_and_lexicographic():
    cfg = config()
    base = passing("whisper-base-opus-de-en", 0.2)
    small = passing("whisper-small-opus-de-en", 0.1)
    decision = select_candidate(cfg, [base, small])
    assert decision["status"] == "PASS"
    assert decision["selected_candidate"] == "whisper-small-opus-de-en"
    assert len(decision["decision_sha256"]) == 64
    with pytest.raises(ValueError):
        select_candidate(cfg, [base])


def test_no_go_does_not_relax_threshold():
    cfg = config()
    decision = select_candidate(cfg, [
        passing("whisper-base-opus-de-en", 0.9),
        passing("whisper-small-opus-de-en", 0.8),
    ])
    assert decision["status"] == "NO-GO"
    assert decision["selected_candidate"] is None


def test_all_arms_must_consume_identical_sealed_assets():
    assets = {
        "machine_devbench_lexical": {"commitment_sha256": "a" * 64, "sealed": True},
        "held_out_real_temporal_retrieval": {"commitment_sha256": "b" * 64, "sealed": True},
    }
    expected = {name: item["commitment_sha256"] for name, item in assets.items()}
    record = {"assets": assets, "arm_consumers": {arm: expected for arm in ARMS}}
    validate_common_assets(record)
    record["arm_consumers"]["Mixed"] = {**expected, "machine_devbench_lexical": "c" * 64}
    with pytest.raises(ValueError):
        validate_common_assets(record)
