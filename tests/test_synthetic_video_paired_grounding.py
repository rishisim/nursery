from scripts.run_synthetic_video_paired_grounding import aggregate, qualitative


PROBES = {
    "noun": {"target": "garment", "distractors": ["cup", "book", "ball"]},
    "adjective": {"target": "white", "distractors": ["black", "red", "blue"]},
}


def test_aggregate_reports_arm_metrics_and_deltas_without_omnibus():
    rows = []
    for arm, scores in (("real", [0.7, 0.1, 0.1, 0.1]), ("synthetic", [0.4, 0.5, 0.05, 0.05])):
        for frame in range(10):
            for probe in PROBES:
                rows.append({"arm": arm, "frame_ordinal": frame, "probe": probe, "scores": scores})
    result = aggregate(rows, PROBES, ("real", "synthetic"))
    assert result["noun"]["real"]["top1_accuracy"] == 1.0
    assert result["noun"]["synthetic"]["top1_accuracy"] == 0.0
    assert result["noun"]["synthetic_minus_real"]["top1_accuracy"] == -1.0
    assert set(result) == {"noun", "adjective"}


def test_qualitative_mapping_preserves_direct_mismatch():
    existing = {
        "checks": {
            "setting_broadly_retained": False,
            "activity_broadly_retained": True,
            "primary_public_category_objects_retained": True,
            "intended_hand_action_beat_retained_when_present": True,
            "first_person_camera_retained": True,
            "no_cuts_or_severe_object_identity_drift": True,
            "no_caption_logo_watermark_physics_or_anatomy_defect": True,
            "speech_and_referent_timing_within_0_75_seconds": None,
        },
        "null_check_meaning": "not_applicable",
    }
    result = qualitative(existing)
    assert result["setting"] is False
    assert result["attribute"] == "covered_by_grounded_adjective_probe"
    assert result["speech_and_referent_timing"] is None
    assert "score" not in result


def test_node_scorer_handles_commonjs_default_export():
    source = open("scripts/synthetic_video_clip_scores.mjs").read()
    assert "imported.default ?? imported" in source
    assert "classifier.dispose()" in source
