from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

import nursery_egobaby_preflight.synthetic_video_model_bakeoff as bakeoff_module
from nursery_egobaby_preflight.contract import canonical_json_sha256
from nursery_egobaby_preflight.synthetic_video_model_bakeoff import (
    GeminiInteractionsClient,
    MiniMaxVideoClient,
    compile_bakeoff_work_order,
    execute_bakeoff,
    finalize_blinded_review,
    inspect_blinded_review_status,
    load_bakeoff_config,
    planned_cost_usd,
    validate_bakeoff_config,
    validate_work_order,
    write_bakeoff_work_order,
)
from nursery_egobaby_preflight.synthetic_video_pilot import (
    compile_prompt,
    load_config as load_public_pilot_config,
)
from nursery_egobaby_preflight.synthetic_video_quality_ceiling import (
    load_quality_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BAKEOFF_CONFIG_PATH = Path("configs/synthetic_video_model_bakeoff.json")
PUBLIC_CONFIG_PATH = Path("configs/synthetic_video_public_pilot.json")
COMPLETED_QUALITY_PATH = Path("configs/synthetic_video_quality_ceiling.json")


@pytest.fixture
def bakeoff_config() -> dict:
    return load_bakeoff_config(BAKEOFF_CONFIG_PATH)


@pytest.fixture
def public_config() -> dict:
    return load_public_pilot_config(PUBLIC_CONFIG_PATH)


@pytest.fixture
def completed_quality() -> dict:
    return load_quality_config(COMPLETED_QUALITY_PATH)


def _order(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
) -> dict:
    return compile_bakeoff_work_order(
        bakeoff_config,
        public_config,
        completed_quality,
        REPOSITORY_ROOT,
        "model-bakeoff-test",
    )


def test_bakeoff_contract_is_frozen_public_only_and_cost_bounded(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
) -> None:
    validate_bakeoff_config(
        bakeoff_config,
        public_config,
        completed_quality,
        REPOSITORY_ROOT,
    )
    assert planned_cost_usd(bakeoff_config) == Decimal("4.66")
    assert bakeoff_config["authorization"]["paid_execution_authorized"] is True
    assert bakeoff_config["provider_cost"]["user_authorized_new_spend_usd"] == 6
    assert bakeoff_config["authorization"]["scientific_training_use_authorized"] is False
    assert bakeoff_config["comparison"]["provider_specific_prompt_changes"] == "none"
    assert bakeoff_config["new_families"]["gemini_omni_flash"]["model"] == (
        "gemini-omni-flash-preview"
    )
    assert bakeoff_config["new_families"]["minimax_h3"]["model"] == "MiniMax-H3"
    assert bakeoff_config["provider_cost"]["gemini"][
        "maximum_expected_charge_usd"
    ] == 2.06
    assert bakeoff_config["provider_cost"]["minimax"][
        "maximum_expected_charge_usd"
    ] == 2.6


def test_work_order_compiles_eight_exact_prompt_requests_and_sixteen_cards(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
) -> None:
    order = _order(bakeoff_config, public_config, completed_quality)
    assert len(order["attempts"]) == 8
    assert len(order["scenes"]) == 4
    assert sum(len(scene["presentation"]) for scene in order["scenes"]) == 16
    assert order["planned_cost_usd"] == 4.66
    assert set(order["blinding_key"]) == {
        card["blinded_display_id"]
        for scene in order["scenes"]
        for card in scene["presentation"]
    }
    public_order = {
        key: value for key, value in order.items() if key != "blinding_key"
    }
    expected_hash = public_order.pop("work_order_sha256")
    assert canonical_json_sha256(public_order) == expected_hash

    for attempt in order["attempts"]:
        prompt = compile_prompt(public_config, "ltx", attempt["scene_id"])
        assert attempt["prompt"] == prompt
        assert attempt["provider_seed_requested"] is None
        assert "seed" not in json.dumps(attempt["request"]).lower()
        if attempt["family"] == "gemini_omni_flash":
            assert attempt["request"]["input"] == prompt
            assert attempt["request"]["store"] is False
            assert attempt["request"]["response_format"]["duration"] == "5s"
            assert attempt["request"]["response_format"]["delivery"] == "inline"
            assert attempt["planned_charge_usd"] == pytest.approx(0.515)
        else:
            assert attempt["request"]["content"] == [
                {"type": "text", "text": prompt}
            ]
            assert attempt["request"]["resolution"] == "2K"
            assert attempt["request"]["duration"] == 5
            assert attempt["planned_charge_usd"] == pytest.approx(0.65)


def test_work_order_rejects_rehashed_request_tampering(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
) -> None:
    order = _order(bakeoff_config, public_config, completed_quality)
    order["attempts"][0]["request"]["input"] = "tampered"
    unhashed = {
        key: value
        for key, value in order.items()
        if key not in {"work_order_sha256", "blinding_key"}
    }
    order["work_order_sha256"] = canonical_json_sha256(unhashed)
    with pytest.raises(ValueError, match="attempts changed"):
        validate_work_order(
            bakeoff_config,
            public_config,
            completed_quality,
            REPOSITORY_ROOT,
            order,
        )


def test_work_order_separates_key_and_writes_family_free_qa(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
    tmp_path: Path,
) -> None:
    order = _order(bakeoff_config, public_config, completed_quality)
    work_order_path = write_bakeoff_work_order(
        tmp_path,
        order,
        bakeoff_config,
        public_config,
    )
    public_order = json.loads(work_order_path.read_text())
    key_path = tmp_path / "blinding_key.json"
    key = json.loads(key_path.read_text())
    assert "blinding_key" not in public_order
    assert canonical_json_sha256(key) == public_order["blinding_key_sha256"]
    assert key_path.stat().st_mode & 0o777 == 0o600
    qa_paths = sorted((tmp_path / "qa").glob("*.json"))
    assert len(qa_paths) == 16
    for path in qa_paths:
        qa = json.loads(path.read_text())
        assert "family" not in qa
        assert "attempt_id" not in qa
        assert len(qa["items"]) == 10
        assert [item["scope"] for item in qa["items"][-2:]] == [
            "secondary_presentation",
            "secondary_presentation",
        ]


def test_review_status_remains_blinded_while_records_are_pending(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
    tmp_path: Path,
) -> None:
    order = _order(bakeoff_config, public_config, completed_quality)
    write_bakeoff_work_order(tmp_path, order, bakeoff_config, public_config)
    public_order = json.loads((tmp_path / "work_order.json").read_text())
    status = inspect_blinded_review_status(
        bakeoff_config,
        public_config,
        completed_quality,
        public_order,
        repository_root=REPOSITORY_ROOT,
        run_root=tmp_path,
    )
    assert status["status"] == "BLINDED_QA_INCOMPLETE"
    assert status["record_counts"] == {
        "complete": 0,
        "pending": 16,
        "missing": 0,
    }
    serialized = json.dumps(status)
    assert '"family"' not in serialized
    assert '"attempt_id"' not in serialized


def test_paid_runner_refuses_preapproval_protocol_before_contacting_providers(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
    tmp_path: Path,
) -> None:
    awaiting = copy.deepcopy(bakeoff_config)
    awaiting["status"] = "FROZEN_AWAITING_CREDENTIALS_AND_SPEND_APPROVAL"
    awaiting["authorization"]["paid_execution_authorized"] = False
    awaiting["provider_cost"]["user_authorized_new_spend_usd"] = None
    order = compile_bakeoff_work_order(
        awaiting,
        public_config,
        completed_quality,
        REPOSITORY_ROOT,
        "model-bakeoff-test",
    )
    with pytest.raises(ValueError, match="not authorized"):
        execute_bakeoff(
            awaiting,
            public_config,
            completed_quality,
            order,
            repository_root=REPOSITORY_ROOT,
            run_root=tmp_path,
            approved_spend_usd=Decimal("4.66"),
            gemini_api_key="unused",
            minimax_api_key="unused",
            implementation_commit="a" * 40,
        )


def test_authorized_protocol_still_requires_exact_spend(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
    tmp_path: Path,
) -> None:
    authorized = copy.deepcopy(bakeoff_config)
    authorized["status"] = "FROZEN_EXECUTION_AUTHORIZED"
    authorized["authorization"]["paid_execution_authorized"] = True
    authorized["provider_cost"]["user_authorized_new_spend_usd"] = 4.66
    validate_bakeoff_config(
        authorized,
        public_config,
        completed_quality,
        REPOSITORY_ROOT,
    )
    order = compile_bakeoff_work_order(
        authorized,
        public_config,
        completed_quality,
        REPOSITORY_ROOT,
        "model-bakeoff-test",
    )
    with pytest.raises(ValueError, match="exactly match"):
        execute_bakeoff(
            authorized,
            public_config,
            completed_quality,
            order,
            repository_root=REPOSITORY_ROOT,
            run_root=tmp_path,
            approved_spend_usd=Decimal("5"),
            gemini_api_key="unused",
            minimax_api_key="unused",
            implementation_commit="a" * 40,
        )


def test_provider_clients_reject_untrusted_urls_and_parse_documented_outputs(
) -> None:
    gemini = GeminiInteractionsClient("secret")
    with pytest.raises(ValueError, match="untrusted"):
        gemini.create("https://example.com/v1beta/interactions", {"input": "x"})
    with pytest.raises(ValueError, match="untrusted"):
        gemini.download_uri(
            "https://example.com/v1beta/files/x:download",
            Path("unused"),
        )
    assert GeminiInteractionsClient.video_output(
        {
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "video",
                            "mime_type": "video/mp4",
                            "uri": (
                                "https://generativelanguage.googleapis.com/"
                                "v1beta/files/file_123:download?alt=media"
                            ),
                        }
                    ],
                }
            ]
        }
    )["mime_type"] == "video/mp4"
    assert GeminiInteractionsClient.file_name_from_uri(
        "https://generativelanguage.googleapis.com/"
        "v1beta/files/file_123:download?alt=media"
    ) == "file_123"

    minimax = MiniMaxVideoClient("secret")
    with pytest.raises(ValueError, match="untrusted"):
        minimax.submit("https://example.com/v2/video_generation", {"model": "x"})
    with pytest.raises(ValueError, match="untrusted"):
        minimax.download_public_file(
            "https://example.com/output.mp4",
            Path("unused"),
        )


def test_finalizer_freezes_sixteen_records_before_four_family_decisions(
    bakeoff_config: dict,
    public_config: dict,
    completed_quality: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = _order(bakeoff_config, public_config, completed_quality)
    write_bakeoff_work_order(tmp_path, order, bakeoff_config, public_config)
    for path in (tmp_path / "qa").glob("*.json"):
        qa = json.loads(path.read_text())
        family = order["blinding_key"][qa["blinded_display_id"]]["family"]
        qa["rater_id"] = "blinded-rater-1"
        qa["rated_at"] = "2026-07-31T12:00:00Z"
        for item in qa["items"]:
            if item["scope"] != "primary_visual":
                response = "pass"
            elif family in {"seedance", "gemini_omni_flash"}:
                response = "pass"
            elif family == "ltx":
                response = "fail" if item["id"] == "contact_action" else "pass"
            else:
                response = (
                    "fail"
                    if item["id"] in {"contact_action", "identity"}
                    else "pass"
                )
            item["response"] = response
            item["confidence"] = "high"
        path.write_text(json.dumps(qa))

    references = {
        family: {
            scene_id: {
                "attempt_id": (
                    f"ltx__{scene_id}__s314159__a1__modular"
                    if family == "ltx"
                    else f"seedance__{scene_id}__sprovider__a1__modular"
                ),
                "record": {
                    "final_media": {"sha256": f"{family}-{scene_id}-sha"}
                },
            }
            for scene_id in bakeoff_module.SCENE_IDS
        }
        for family in bakeoff_module.REFERENCE_FAMILIES
    }
    execution_records = {
        attempt["attempt_id"]: {
            "attempt": attempt,
            "status": "media_valid",
            "provider": {"nursery_adapter_commit": "a" * 40},
            "final_media": {"sha256": f"{attempt['attempt_id']}-sha"},
        }
        for attempt in order["attempts"]
    }
    monkeypatch.setattr(
        bakeoff_module,
        "verify_reference_media",
        lambda *args, **kwargs: references,
    )
    monkeypatch.setattr(
        bakeoff_module,
        "_verify_new_execution",
        lambda *args, **kwargs: {
            "planned_attempt_count": 8,
            "media_valid_count": 8,
            "generated_attempt_failure_count": 0,
            "records": execution_records,
        },
    )

    public_order = json.loads((tmp_path / "work_order.json").read_text())
    result = finalize_blinded_review(
        bakeoff_config,
        public_config,
        completed_quality,
        public_order,
        repository_root=REPOSITORY_ROOT,
        run_root=tmp_path,
    )
    assert result["status"] == "COMPLETE_QUALITATIVE_SCREEN"
    assert result["decisions"]["gemini_omni_flash"]["label"] == (
        "CLEAR_TASK_ADVANTAGE_OVER_LTX"
    )
    assert result["decisions"]["gemini_omni_flash"][
        "competitive_with_seedance"
    ] is True
    assert result["decisions"]["minimax_h3"]["label"] == (
        "BELOW_SEEDANCE_ON_THIS_SCREEN"
    )
    summary = json.loads((tmp_path / "review" / "review_summary.json").read_text())
    assert summary["qualitative"]["families"]["gemini_omni_flash"][
        "counts_by_scope"
    ]["primary_visual"]["pass"] == 28
    assert summary["provenance"]["qa_bundle_sha256"]
    assert (tmp_path / "review" / "qa_freeze.json").is_file()
