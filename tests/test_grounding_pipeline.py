from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from babyworld_lite.grounding.config import load_grounding_config, validate_grounding_config
from babyworld_lite.grounding.pilot_data import GroundingRecordAdapter
from babyworld_lite.grounding.pipeline import (
    BaseEpisode,
    MODEL_INPUT_ALLOWLIST,
    generate_grounding_dataset,
    observable_leakage_paths,
    render_raw_frame,
)
from babyworld_lite.sim import simulate_episode


ROOT = Path(__file__).parents[1]


def _small_config() -> dict:
    config = load_grounding_config(ROOT / "configs" / "grounding_provisional.yaml")
    config["generation"].update({"base_episodes": 18, "frame_stride": 7, "render_frames": True})
    config["splits"]["minimum_base_episodes_per_holdout"] = 2
    return config


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, list[dict], list[dict]]:
    out = tmp_path_factory.mktemp("grounding")
    config = _small_config()
    summary = generate_grounding_dataset(config, out)
    examples = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
    oracle = [json.loads(line) for line in (out / "oracle.jsonl").read_text().splitlines()]
    return out, summary, examples, oracle


def test_config_is_explicitly_provisional_and_rejects_matched_claim() -> None:
    config = _small_config()
    assert config["profile"]["calibration_status"] == "provisional_not_babyview_matched"
    invalid = copy.deepcopy(config)
    invalid["profile"]["calibration_status"] = "babyview_matched"
    with pytest.raises(ValueError, match="BabyView-matched claims are not allowed"):
        validate_grounding_config(invalid)


def test_factorial_conditions_model_allowlist_and_separate_oracle(generated) -> None:
    out, summary, examples, oracle = generated
    assert summary["base_episodes"] == 18
    assert summary["examples"] == 18 * 3 * 4
    assert summary["audit"]["valid"] is True
    assert {row["alignment_condition"] for row in examples} == {"strong", "weak", "shuffled"}
    assert {row["motor_condition"] for row in examples} == {"null", "synchronized", "shuffled", "time_shifted"}
    assert all(set(row["model_inputs"]) == MODEL_INPUT_ALLOWLIST["top"] for row in examples)
    assert all(not observable_leakage_paths(row) for row in examples)
    assert all("oracle" not in row and "event_label" not in row for row in examples)
    assert all("causal_graph" in row for row in oracle)
    assert (out / "oracle.jsonl").is_file()


def test_composition_and_hash_disjoint_splits_and_equal_alignment_inventory(generated) -> None:
    _out, summary, examples, _oracle = generated
    fairness = summary["audit"]["fairness"]
    assert fairness["split_hash_overlap"] == {}
    assert fairness["split_composition_overlap"] == {}
    assert fairness["episode_multisets_identical_across_alignment"] is True
    assert fairness["utterance_multisets_identical_across_alignment"] is True
    assert fairness["shuffled_self_matches"] == []
    assert fairness["cross_split_donors"] == []
    test_compositions = {
        (row["evaluation_targets"]["object_noun"], row["evaluation_targets"]["action_verb"])
        for row in examples if row["split"] == "test"
    }
    assert test_compositions == {("cup", "push"), ("plush", "grasp")}


def test_motor_controls_are_whole_sequence_derangements_shifts_and_nulls(generated) -> None:
    _out, _summary, examples, oracle = generated
    metadata = {row["example_id"]: row for row in oracle}
    strong = [row for row in examples if row["alignment_condition"] == "strong"]
    synchronized = {row["base_episode_id"]: row["model_inputs"]["motor"] for row in strong if row["motor_condition"] == "synchronized"}
    for row in strong:
        motor = row["model_inputs"]["motor"]
        if row["motor_condition"] == "null":
            assert len(motor) == len(synchronized[row["base_episode_id"]])
            assert all(sample["available"] == 0 and sample["x"] == 0 for sample in motor)
        elif row["motor_condition"] == "shuffled":
            donor = metadata[row["example_id"]]["motor_source_episode_id"]
            assert donor != row["base_episode_id"]
            donor_motor = synchronized[donor]
            overlap = min(len(motor), len(donor_motor))
            assert motor[:overlap] == donor_motor[:overlap]
            assert all(sample["available"] == 0 for sample in motor[overlap:])
        elif row["motor_condition"] == "time_shifted":
            shift = 5
            own = synchronized[row["base_episode_id"]]
            assert all(sample["available"] == 0 for sample in motor[:shift])
            assert [sample["x"] for sample in motor[shift:]] == [sample["x"] for sample in own[:-shift]]


def test_alignment_changes_in_window_relevance_without_changing_text_inventory(generated) -> None:
    _out, _summary, examples, _oracle = generated
    rows = {
        (row["base_episode_id"], row["alignment_condition"]): row
        for row in examples
        if row["motor_condition"] == "synchronized"
    }
    for base_episode_id in {key[0] for key in rows}:
        strong = rows[(base_episode_id, "strong")]
        weak = rows[(base_episode_id, "weak")]
        shuffled = rows[(base_episode_id, "shuffled")]
        strong_texts = sorted(item["text"] for item in strong["model_inputs"]["utterances"])
        weak_texts = sorted(item["text"] for item in weak["model_inputs"]["utterances"])
        in_window = lambda row: [
            item for item in row["model_inputs"]["utterances"]
            if 0 <= item["onset_frame"] < len(row["model_inputs"]["motor"])
        ]
        assert len(in_window(strong)) == 1
        assert len(in_window(shuffled)) == 1
        assert len(in_window(weak)) <= 1
        assert strong_texts == weak_texts


def test_raw_renderer_is_invariant_to_labels_and_force_and_has_no_overlay() -> None:
    original = simulate_episode(0, 123)
    mutated = copy.deepcopy(original)
    mutated.action = "DO_NOT_RENDER"
    mutated.event_label = "DO_NOT_RENDER"
    for frame in mutated.frames:
        frame["touch"]["force"] = 999999.0
        frame["touch"]["contact"] = 1.0
    before = render_raw_frame(original, 0)
    after = render_raw_frame(mutated, 0)
    assert before.tobytes() == after.tobytes()
    assert before.getpixel((1, 1)) == (245, 239, 225)


def test_raw_renderer_consumes_visibility_occlusion_and_camera_controls() -> None:
    episode = simulate_episode(0, 123)
    base = BaseEpisode(
        episode=episode,
        split="train",
        distractors=[],
        episode_hash="test",
        activity_frames=28,
        target_visible=True,
        target_occlusion_fraction=0.0,
        camera_motion_amplitude_px=0.0,
    )
    visible = render_raw_frame(base, 0).tobytes()
    invisible = render_raw_frame(replace(base, target_visible=False), 0).tobytes()
    occluded = render_raw_frame(replace(base, target_occlusion_fraction=0.5), 0).tobytes()
    moving_camera = render_raw_frame(replace(base, camera_motion_amplitude_px=5.0), 0).tobytes()
    assert visible != invisible
    assert visible != occluded
    assert visible != moving_camera


def test_rendered_paths_exist_and_audit_is_persisted(generated) -> None:
    out, summary, examples, _oracle = generated
    assert all((out / path).is_file() for path in examples[0]["model_inputs"]["frame_paths"])
    persisted = json.loads((out / "audit_summary.json").read_text())
    assert persisted == summary["audit"]
    snapshot = yaml.safe_load((out / "config_snapshot.yaml").read_text())
    assert snapshot["profile"]["calibration_status"] == "provisional_not_babyview_matched"


def test_calibration_targets_are_consumed_and_realized_distributions_are_audited(generated) -> None:
    _out, summary, _examples, oracle = generated
    configured = summary["audit"]["configured_calibration_targets"]
    realized = summary["audit"]["realized_calibration"]
    assert configured["utterance_rate_per_minute"] == 52.0
    assert sum(realized["utterances_per_activity_window"].values()) == 18
    assert sum(realized["activity_window_frames"].values()) == 18
    assert sum(realized["target_visibility"].values()) == 18
    assert sum(realized["target_occlusion_fraction"].values()) == 18
    assert sum(realized["camera_motion_amplitude_px"].values()) == 18
    assert realized["utterance_rate_per_minute"] > 0
    assert 0.0 <= realized["silent_activity_frame_fraction"] <= 1.0
    assert realized["utterance_length_words"]
    assert realized["inter_utterance_interval_frames"]
    assert realized["target_word_frequency"]
    calibration_rows = {row["base_episode_id"]: row["calibration_realization"] for row in oracle}
    assert {row["activity_window_frames"] for row in calibration_rows.values()} <= {20, 24, 28}
    assert {row["camera_motion_amplitude_px"] for row in calibration_rows.values()} <= {0.0, 2.0, 5.0}


def test_grounding_record_consumer_honors_window_timing_and_allowlist(generated) -> None:
    out, _summary, examples, _oracle = generated
    adapter = GroundingRecordAdapter(out)
    record = next(
        row for row in examples
        if row["alignment_condition"] == "strong" and row["motor_condition"] == "synchronized"
    )
    window_frames = len(record["model_inputs"]["motor"])
    expected_text = " ".join(
        item["text"] for item in sorted(
            record["model_inputs"]["utterances"], key=lambda item: item["onset_frame"]
        )
        if 0 <= item["onset_frame"] < window_frames
    )
    assert adapter.text(record) == expected_text
    assert adapter.video(record, frame_count=2, image_size=32).shape == (2, 3, 32, 32)
    assert adapter.motor(record, frame_count=3).shape == (3, 4)

    silent = copy.deepcopy(record)
    for item in silent["model_inputs"]["utterances"]:
        item["onset_frame"] = window_frames + 5
    assert adapter.text(silent) == ""

    invalid = copy.deepcopy(record)
    invalid["model_inputs"]["event_label"] = "leak"
    with pytest.raises(ValueError, match="model input schema violations"):
        adapter.text(invalid)
