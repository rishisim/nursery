from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from babyworld_lite.interactmove_intermimic import cli
from babyworld_lite.interactmove_intermimic.activity import activity_spec_sha256
from babyworld_lite.interactmove_intermimic.common import content_sha256


REQUIRED_METRICS = (
    "target_instance_correct",
    "phase_order_fraction",
    "intended_contact_fraction",
    "stable_grasp_fraction",
    "max_penetration_m",
    "balance_success",
    "final_relation_satisfied",
)


def valid_activity_spec() -> dict:
    boolean_metrics = {
        "target_instance_correct",
        "balance_success",
        "final_relation_satisfied",
    }
    return {
        "schema": "InteractMoveInterMimicActivitySpec",
        "schema_version": 1,
        "protocol_id": "interactmove_intermimic",
        "activity_id": "lift-blue-bowl",
        "prompt": "Lift the blue bowl from the table with both hands.",
        "actor": {
            "body_model": "SMPL-X",
            "morphology": "adult",
            "gender": "neutral",
            "betas": [0.0] * 10,
            "initial_stance": "standing_neutral",
            "initial_placement": {
                "method": "target_relative",
                "target_relative_position_m": [-1.0, 0.0, 0.0],
                "face_target": True,
                "selected_by": "nursery",
            },
        },
        "target_request": {
            "description": "the blue bowl on the table",
            "category": "bowl",
            "prompt_span": "blue bowl",
            "resolution_rule": {
                "method": "sole_manipulated_instance",
                "expected_source_node_key": None,
            },
        },
        "relations": {
            "start": [
                {
                    "subject": "target",
                    "predicate": "on",
                    "object_description": "the table",
                    "description": "the target starts on the table",
                }
            ],
            "final": [
                {
                    "subject": "target",
                    "predicate": "held_by",
                    "object_description": "the adult actor",
                    "description": "the target finishes held by the actor",
                }
            ],
        },
        "phases": [
            {
                "id": "approach",
                "description": "approach the target",
                "start_frame": 0,
                "end_frame_exclusive": 75,
                "expected_target_contact_hands": [],
            },
            {
                "id": "grasp-and-lift",
                "description": "grasp and lift the target",
                "start_frame": 75,
                "end_frame_exclusive": 300,
                "expected_target_contact_hands": ["left", "right"],
            },
        ],
        "timing": {
            "duration_s": 10.0,
            "fps": 30,
            "frame_count": 300,
            "sampling": "half_open_[0,duration)",
        },
        "hand_use": {"mode": "two_hand", "hands": ["left", "right"]},
        "seeds": {
            "master": 10,
            "embodiedgen_image": 11,
            "embodiedgen_asset": 12,
            "embodiedgen_layout": 13,
            "interactmove": 14,
            "intermimic": 15,
            "render": 16,
        },
        "success_criteria": [
            {
                "id": f"criterion-{index}",
                "metric": metric,
                "operator": (
                    "eq"
                    if metric in boolean_metrics
                    else "lte" if metric == "max_penetration_m" else "gte"
                ),
                "threshold": (
                    True
                    if metric in boolean_metrics
                    else 0.01 if metric == "max_penetration_m" else 0.9
                ),
                "unit": "m" if metric == "max_penetration_m" else None,
                "evaluation_window": "full_episode",
            }
            for index, metric in enumerate(REQUIRED_METRICS)
        ],
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cli_runner(capsys):
    """Call the real emitter with pytest's active stdout as its default."""

    def run(arguments):
        defaults = cli._emit.__kwdefaults__
        original = defaults["stream"]
        defaults["stream"] = sys.stdout
        try:
            return cli.main(arguments)
        finally:
            defaults["stream"] = original

    return run, capsys


def test_validate_activity_positional_spec_prints_digest(tmp_path, cli_runner) -> None:
    run_cli, captured_io = cli_runner
    spec = valid_activity_spec()
    spec_path = tmp_path / "activity.json"
    write_json(spec_path, spec)

    assert run_cli(["validate-activity", str(spec_path), "--print-digest"]) == 0

    status = json.loads(captured_io.readouterr().out)
    assert status == {
        "activity_id": "lift-blue-bowl",
        "activity_spec_sha256": activity_spec_sha256(spec),
        "command": "validate-activity",
        "status": "ok",
    }


def test_prepare_request_writes_tmp_output_and_refuses_overwrite(
    tmp_path, cli_runner
) -> None:
    run_cli, captured_io = cli_runner
    spec_path = tmp_path / "activity.json"
    output_path = tmp_path / "embodiedgen_request.json"
    write_json(spec_path, valid_activity_spec())
    arguments = [
        "prepare-embodiedgen-request",
        str(spec_path),
        "--background-list",
        '["kitchen", "studio"]',
        "--output",
        str(output_path),
    ]

    assert run_cli(arguments) == 0
    success = json.loads(captured_io.readouterr().out)
    assert success["status"] == "ok"
    assert success["artifact_sha256"] == content_sha256(
        json.loads(output_path.read_text(encoding="utf-8"))
    )

    assert run_cli(arguments) == 2
    error = json.loads(captured_io.readouterr().err)
    assert error["status"] == "error"
    assert "refusing to overwrite" in error["message"]


def test_compile_scene_bundle_requires_and_forwards_request(
    tmp_path, monkeypatch, cli_runner
) -> None:
    run_cli, captured_io = cli_runner
    spec = valid_activity_spec()
    request = {"schema": "test-request", "binding": "parsed-json"}
    spec_path = tmp_path / "activity.json"
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "scene_bundle.json"
    write_json(spec_path, spec)
    write_json(request_path, request)

    with pytest.raises(SystemExit) as missing:
        run_cli(
            [
                "compile-scene-bundle",
                str(spec_path),
                "--scene-root",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )
    assert missing.value.code == 2
    assert json.loads(captured_io.readouterr().err)["status"] == "error"

    calls: dict[str, object] = {}
    bundle = {"schema": "test-scene-bundle", "value": 1}

    def fake_compile(scene_root, layout, activity_spec, **kwargs):
        calls.update(
            scene_root=scene_root,
            layout=layout,
            activity_spec=activity_spec,
            kwargs=kwargs,
        )
        return bundle

    monkeypatch.setattr(cli, "compile_scene_bundle", fake_compile)
    monkeypatch.setattr(cli, "validate_scene_bundle", lambda value: None)
    assert (
        run_cli(
            [
                "compile-scene-bundle",
                str(spec_path),
                "--request",
                str(request_path),
                "--scene-root",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    assert calls["scene_root"] == tmp_path
    assert calls["layout"] == "layout.json"
    assert calls["activity_spec"] == spec
    assert calls["kwargs"]["embodiedgen_request"] == request
    assert json.loads(output_path.read_text(encoding="utf-8")) == bundle
    assert json.loads(captured_io.readouterr().out)["status"] == "ok"


def test_validate_scene_bundle_prints_digest(tmp_path, monkeypatch, cli_runner) -> None:
    run_cli, captured_io = cli_runner
    bundle = {"schema": "test-scene-bundle", "value": 2}
    bundle_path = tmp_path / "bundle.json"
    write_json(bundle_path, bundle)
    seen = []
    monkeypatch.setattr(cli, "validate_scene_bundle", seen.append)

    assert (
        run_cli(
            ["validate-scene-bundle", str(bundle_path), "--print-digest"]
        )
        == 0
    )

    assert seen == [bundle]
    assert json.loads(captured_io.readouterr().out) == {
        "command": "validate-scene-bundle",
        "scene_bundle_sha256": content_sha256(bundle),
        "status": "ok",
    }


def test_contract_error_is_json_on_stderr_with_exit_two(tmp_path, cli_runner) -> None:
    run_cli, captured_io = cli_runner
    invalid = tmp_path / "invalid.json"
    write_json(invalid, {})

    assert run_cli(["validate-activity", str(invalid)]) == 2

    captured = captured_io.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["status"] == "error"
    assert error["error"] == "ContractError"


def test_in_repository_output_outside_run_root_is_rejected_without_write(
    tmp_path, monkeypatch, cli_runner
) -> None:
    run_cli, captured_io = cli_runner
    fake_repository = tmp_path / "repository"
    fake_repository.mkdir()
    disallowed = fake_repository / "result.json"
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", fake_repository)
    monkeypatch.setattr(
        cli,
        "IN_REPOSITORY_RUN_ROOT",
        fake_repository / "runs/interactmove_intermimic",
    )

    assert (
        run_cli(
            [
                "prepare-embodiedgen-request",
                str(tmp_path / "unused-spec.json"),
                "--background-list",
                "kitchen",
                "--output",
                str(disallowed),
            ]
        )
        == 2
    )

    assert not disallowed.exists()
    error = json.loads(captured_io.readouterr().err)
    assert error["status"] == "error"
    assert "runs/interactmove_intermimic" in error["message"]
