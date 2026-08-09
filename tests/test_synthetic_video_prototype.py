import json
from pathlib import Path

import pytest

from scripts.run_synthetic_video_prototype import PrototypeError, compare, compile_prompt, validate_scene


def scene():
    return {
        "schema_version": 1,
        "setting": {"public_room_category": "kitchen", "lighting": "daylight"},
        "camera": {"first_person": True, "height": "child-height", "framing": "centered", "motion": "gentle", "blur": "low", "continuity": "continuous shot"},
        "entities": [{"category": "cup", "role": "object", "attributes": ["red"], "recurs": True}],
        "activity": "reaching for a cup",
        "temporal_beats": [{"start": 0, "end": 4, "action": "approach"}, {"start": 4, "end": 10, "action": "grasp"}],
        "hands": {"visibility": "right hand", "contact": "cup contact", "manipulation": "grasp", "intervals": [{"start": 4, "end": 10}]},
        "speech": {"observable": None, "semantic_summary": "unsupported", "tts_de_paraphrase": "", "intervals": []},
        "learning_opportunity": {"visible_nouns": ["cup"], "attribute_contrasts": [], "referent_candidates": ["cup"], "ambiguity": "low", "no_referent_intervals": []},
        "scene_dynamics": {"clutter": "low", "distractors": "few", "occlusion": "low", "recurrence": "cup persists", "idle_intervals": [], "transitions": []},
        "uncertainty": ["speech unavailable from sampled frames"],
    }


def test_schema_and_deterministic_compiler():
    schema = json.loads(Path("configs/synthetic_video_scene_schema.json").read_text())
    assert schema["$schema"].endswith("2020-12/schema")
    value = scene()
    validate_scene(value)
    assert compile_prompt(value) == compile_prompt(value)
    plan = compile_prompt(value)
    assert plan["model"] == "minimax/hailuo-3"
    assert plan["duration"] == 10
    assert plan["generate_audio"] is False
    assert "identifiable faces" in plan["negative_constraints"]


def test_privacy_guard_rejects_reconstructive_values():
    value = scene()
    value["uncertainty"] = ["source filename secret.mp4"]
    with pytest.raises(PrototypeError, match="E_RECONSTRUCTIVE_DESCRIPTION"):
        validate_scene(value)


def test_privacy_guard_allows_ordinary_category_slash():
    value = scene()
    value["uncertainty"] = ["motion is low/medium"]
    validate_scene(value)


def test_prototype_declares_exactly_two_engineering_attempts():
    source = Path("scripts/run_synthetic_video_prototype.py").read_text()
    assert 'choices=(1, 2)' in source
    assert 'except URLError:' in source
    assert 'video_job.json' in source


def test_feature_comparison_reports_items_without_omnibus_score():
    source = scene()
    generated = scene()
    generated["hands"]["manipulation"] = "lift and grasp the cup"
    result = compare(source, generated, {"duration": 10.0, "audio": True})
    assert result["checks"]["activity_broadly_retained"] is False
    assert result["omnibus_score"] is None
