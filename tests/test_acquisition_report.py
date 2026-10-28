import hashlib
import json

import pytest

from nursery.learner.acquisition_report import CONDITIONS, _PLOT_CODE, _write_tables, aggregate


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _eval_result(accuracy, known=5, total=8):
    return {"accuracy": accuracy, "known_accuracy": accuracy - 1,
            "coverage": known / total, "n_trials": total, "n_known": known,
            "unknown_rule": "25% expected accuracy; retained in fixed denominator"}


def _matrix(root):
    protocol = {"fractions": [0.1, 0.25, 0.5, 0.75, 1.0], "seeds": [0, 1, 2], "epochs": 3,
                "unknown_query_score": 0.25}
    _write(root / "protocol.json", protocol)
    digest = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    for fraction in protocol["fractions"]:
        selected = round(fraction * 100)
        stats = {"fraction": fraction, "selected_clip_count": selected, "exposed_clip_count": selected + 1,
                 "utterances": selected * 2, "replay_exposure_seconds": selected * 4,
                 "unique_source_duration_seconds": None,
                 "vocabulary_coverage": {name: {"known": 5, "total": 8}
                                         for name in ("clips_test", "s8", "picture_test", "konkle_test")}}
        _write(root / "subsets" / str(float(fraction)) / "stats.json", stats)
        for seed in protocol["seeds"]:
            for condition_index, condition in enumerate(CONDITIONS):
                # Paired condition differences are [3, 5, 7] at each point.
                base = 30 + seed
                offsets = {"video_only": 0, "additive": 3 + 2 * seed, "joint_interaction": 8 + 3 * seed}
                primary = {"clips_test": base + offsets[condition],
                           "s8": base + condition_index, "chance": 25}
                shuffled = {} if condition == "video_only" else {
                    "clips_test": _eval_result(26 + seed), "s8": _eval_result(27 + seed)}
                data = {key: stats[key] for key in ("selected_clip_count", "exposed_clip_count", "utterances", "replay_exposure_seconds")}
                data["vocabulary_coverage"] = stats["vocabulary_coverage"]
                metrics = {
                    "status": "complete", "smoke": False, "fraction": fraction, "seed": seed,
                    "condition": condition, "protocol_sha256": digest, "epochs": 3,
                    "selection": {"body_present": {"epoch": 1}, "picture": {"epoch": 1}},
                    "primary": {key: _eval_result(value) for key, value in primary.items() if key in ("clips_test", "s8")},
                    "shuffled": shuffled,
                    "picture": {"picture_test": _eval_result(35 + seed), "konkle_test": _eval_result(40 + seed)},
                    "data": data,
                }
                _write(root / "models" / str(float(fraction)) / f"seed_{seed}" / condition / "metrics.json", metrics)
    return protocol


def test_aggregate_full_matrix_keeps_seed_paired_differences(tmp_path):
    _matrix(tmp_path)
    result = aggregate(tmp_path)
    assert result["n_runs"] == 45
    point = result["points"][0]
    assert point["conditions"]["video_only"]["clips_test"]["mean"] == pytest.approx(31)
    difference = point["paired_differences"]["additive_minus_video_only"]["clips_test"]
    assert difference["mean"] == pytest.approx(5)
    assert difference["sample_std"] == pytest.approx(2)
    assert difference["seed_values"] == {"0": 3, "1": 5, "2": 7}
    assert point["conditions"]["additive"]["shuffled_clips_test"]["n_seeds"] == 3
    assert point["conditions"]["additive"]["known_accuracy"]["clips_test"]["mean"] == pytest.approx(35)


def test_aggregate_refuses_missing_run_instead_of_returning_partial_curve(tmp_path):
    _matrix(tmp_path)
    (tmp_path / "models" / "0.5" / "seed_1" / "joint_interaction" / "metrics.json").unlink()
    with pytest.raises(ValueError, match="partial acquisition report"):
        aggregate(tmp_path)


def test_aggregate_rejects_smoke_result(tmp_path):
    _matrix(tmp_path)
    path = tmp_path / "models" / "0.1" / "seed_0" / "video_only" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["smoke"] = True
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Run identity/status mismatch"):
        aggregate(tmp_path)


def test_report_tables_are_written_after_full_aggregation(tmp_path):
    _matrix(tmp_path)
    summary = aggregate(tmp_path)
    _write_tables(tmp_path, summary)
    assert json.loads((tmp_path / "summary.json").read_text())["n_runs"] == 45
    csv_text = (tmp_path / "summary.csv").read_text()
    assert "paired_difference" in csv_text and "vocabulary_coverage" in csv_text
    assert "Konkle transfer" in (tmp_path / "report.md").read_text()
    assert "Known-word accuracy (secondary)" in (tmp_path / "report.md").read_text()
    assert "5/8" in (tmp_path / "report.md").read_text()
    assert "not recorded" in (tmp_path / "report.md").read_text()
    compile(_PLOT_CODE, "acquisition_report_plot", "exec")
