from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from babyworld_lite.aea.config import AEA_SCIENTIFIC_ROLE, load_aea_config, validate_aea_config
from babyworld_lite.aea.experiment import locked_training_config_matches
from babyworld_lite.aea.manifest import AEASequenceId, load_safe_manifest
from babyworld_lite.aea.preprocess import SpeechWord, build_windows, resample_imu
from babyworld_lite.aea.smoke import build_aea_smoke_fixture
from babyworld_lite.aea.splits import held_out_location_splits
from babyworld_lite.grounding.pilot_data import (
    AEAWindowAdapter,
    ArmDataset,
    GroundingCorpus,
    WordTokenizer,
)
from babyworld_lite.grounding.pilot_experiment import PilotConfig, metadata_only_shortcut_check
from babyworld_lite.grounding.pilot_experiment import motor_only_manipulation_check


ROOT = Path(__file__).parents[1]


def test_safe_manifest_boundary_discards_signed_urls(tmp_path: Path) -> None:
    path = tmp_path / "links.json"
    path.write_text(json.dumps({
        "sequence_config": {
            "dataset_name": "AriaEverydayActivities",
            "release": "1.0",
            "private_url": "https://secret.invalid/CONFIG_SECRET_SENTINEL",
        },
        "sequences": {
            "loc1_script1_seq1_rec1": {
                "main_vrs": {
                    "filename": "private.vrs",
                    "sha1sum": "abc",
                    "file_size_bytes": 12,
                    "download_url": "https://secret.invalid/SECRET_SENTINEL",
                }
            }
        },
    }))
    safe = load_safe_manifest(path)
    representation = repr(safe)
    assert "SECRET_SENTINEL" not in representation
    assert safe.sequences["loc1_script1_seq1_rec1"].assets["main_vrs"].file_size_bytes == 12


def test_sequence_groups_are_explicit() -> None:
    value = AEASequenceId.parse("loc3_script2_seq7_rec2")
    assert value.event_group == "loc3_script2_seq7"
    assert value.wearer_session_group == "loc3_script2_rec2"


def test_config_refuses_developmental_claim() -> None:
    config = load_aea_config(ROOT / "configs" / "aea_real.yaml")
    assert config["profile"]["scientific_role"] == AEA_SCIENTIFIC_ROLE
    config["profile"]["scientific_role"] = "developmental_evidence"
    with pytest.raises(ValueError, match="developmental-evidence claims are not allowed"):
        validate_aea_config(config)


def test_claim_schedule_requires_the_locked_training_configuration() -> None:
    protocol = load_aea_config(ROOT / "configs" / "aea_real.yaml")["experiment"]
    locked = PilotConfig(**protocol["locked_training"])
    assert locked_training_config_matches(locked, protocol)
    assert not locked_training_config_matches(
        PilotConfig(**{**protocol["locked_training"], "epochs": 1}), protocol
    )


def test_speech_windows_and_six_axis_resampling() -> None:
    config = load_aea_config(ROOT / "configs" / "aea_real.yaml")
    words = [
        SpeechWord(0, 100_000_000, "please", "please", 1.0),
        SpeechWord(2_900_000_000, 3_100_000_000, "Grab", "grab", 0.9),
        SpeechWord(3_200_000_000, 3_400_000_000, "cup", "cup", 0.8),
    ]
    windows = build_windows(words, config)
    assert len(windows) == 1
    assert (windows[0].action, windows[0].object_noun) == ("grab", "cup")
    timestamps = np.arange(0, 6_000_000_001, 10_000_000, dtype=np.int64)
    values = np.stack([np.sin(timestamps / 1e9 + i) for i in range(6)], axis=1)
    result, quality = resample_imu(timestamps, values, 0, 6_000_000_000, 50, 50)
    assert result.shape == (300, 6)
    assert quality["coverage_fraction"] > 0.99


def test_fixture_adapter_episode_local_derangement_and_location_splits(tmp_path: Path) -> None:
    examples = build_aea_smoke_fixture(tmp_path)
    records = [json.loads(line) for line in examples.read_text().splitlines()]
    adapter = AEAWindowAdapter(tmp_path)
    corpus = GroundingCorpus(records, adapter, frame_count=2, image_size=20, motor_sample_count=10)
    train = [i for i, row in enumerate(records) if row["location"] != 3]
    tokenizer = WordTokenizer.fit([corpus.text(i) for i in train])
    shuffled = ArmDataset(corpus, train, tokenizer, "shuffled", 10, 7)
    index = train[0]
    assert corpus.motor(index).shape == (10, 6)
    assert corpus.video(index).shape == (2, 3, 20, 20)
    assert corpus.metadata(index).episode_group != corpus.metadata(shuffled.donors[index]).episode_group
    assert torch.equal(shuffled._motor(index), corpus.motor(shuffled.donors[index]))
    splits = held_out_location_splits(records, (1, 2, 3), 2)
    assert len(splits) == 3
    assert all(split.audit["valid"] for split in splits)
    assert all(not split.audit["sequence_overlap"] for split in splits)


def test_concurrent_recordings_share_a_shuffled_donor_group(tmp_path: Path) -> None:
    examples = build_aea_smoke_fixture(tmp_path)
    records = [json.loads(line) for line in examples.read_text().splitlines()]
    first = records[0]
    partner = json.loads(json.dumps(first))
    partner["sequence_id"] = first["sequence_id"].replace("_rec1", "_rec2")
    partner["recording"] = 2
    adapter = AEAWindowAdapter(tmp_path)
    assert adapter.metadata(first).episode_group == adapter.metadata(partner).episode_group


def test_metadata_control_ignores_train_only_actions(tmp_path: Path) -> None:
    examples = build_aea_smoke_fixture(tmp_path)
    records = [json.loads(line) for line in examples.read_text().splitlines()]
    corpus = GroundingCorpus(
        records,
        AEAWindowAdapter(tmp_path),
        frame_count=2,
        image_size=20,
        motor_sample_count=10,
    )
    retained_actions = ("get", "put")
    train_indices = list(range(len(records)))
    test_indices = [
        index for index, row in enumerate(records)
        if row["evaluation_targets"]["action_verb"] in retained_actions
    ][:4]
    result = metadata_only_shortcut_check(
        corpus,
        train_indices,
        test_indices,
        retained_actions,
    )
    assert 0.0 <= result["action_2afc_macro_accuracy"] <= 1.0


def test_shuffled_training_manipulation_is_reported_when_test_derangement_is_impossible(
    tmp_path: Path,
) -> None:
    examples = build_aea_smoke_fixture(tmp_path)
    records = [json.loads(line) for line in examples.read_text().splitlines()]
    corpus = GroundingCorpus(
        records,
        AEAWindowAdapter(tmp_path),
        frame_count=2,
        image_size=20,
        motor_sample_count=10,
    )
    train = list(range(16, len(records)))
    test = list(range(8))
    tokenizer = WordTokenizer.fit([corpus.text(index) for index in train])
    result = motor_only_manipulation_check(
        corpus,
        train,
        test,
        tokenizer,
        "shuffled",
        ("get", "put", "cook", "grab"),
        PilotConfig(frame_count=2, image_size=20, motor_sample_count=10),
        11,
    )
    assert result["test_episode_derangement_feasible"] is False
    assert result["shuffled_training_self_match_rate"] == 0.0
    assert result["shuffled_training_episode_group_match_rate"] == 0.0
