from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/compare_childlens_model_human_audit_v1_3.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_childlens_model_human_audit_v1_3", MODULE_PATH
)
assert SPEC and SPEC.loader
comparator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparator
SPEC.loader.exec_module(comparator)

TERMINAL_PATH = REPO_ROOT / "scripts/synthesize_childlens_terminal_v1_3.py"
TERMINAL_SPEC = importlib.util.spec_from_file_location(
    "synthesize_childlens_terminal_v1_3_for_comparator_test", TERMINAL_PATH
)
assert TERMINAL_SPEC and TERMINAL_SPEC.loader
terminal = importlib.util.module_from_spec(TERMINAL_SPEC)
sys.modules[TERMINAL_SPEC.name] = terminal
TERMINAL_SPEC.loader.exec_module(terminal)


def _protocol() -> dict:
    return json.loads(comparator.PROTOCOL_PATH.read_text(encoding="utf-8"))


def _passing_items() -> list[comparator.ItemEvidence]:
    items: list[comparator.ItemEvidence] = []
    for item_index in range(15):
        humans: list[comparator.HumanUtterance] = []
        machines: list[comparator.MachineUtterance] = []
        windows: list[comparator.MachineWindow] = []
        intervals = ((0, 20_000), (20_000, 40_000), (40_000, 60_000))
        second_status = (
            "VISIBLE_CANDIDATE"
            if item_index < 5
            else "NULL_NOT_VISIBLE"
            if item_index < 9
            else "IRRELEVANT"
            if item_index < 12
            else "UNDECIDABLE"
        )
        statuses = ("VISIBLE_CANDIDATE", second_status, "IRRELEVANT")
        roles = ("NON_CHILD", "NON_CHILD", "CHILD")
        for utterance_index, (interval, role, status) in enumerate(
            zip(intervals, roles, statuses)
        ):
            text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
            mentions: list[comparator.HumanMention] = []
            noun: tuple[str, ...] = ()
            verb: tuple[str, ...] = ()
            if status == "VISIBLE_CANDIDATE" and role == "NON_CHILD":
                mentions.append(
                    comparator.HumanMention(
                        family="NOUN_OBJECT", text="alpha", visible_interval=interval
                    )
                )
                noun = ("alpha",)
                if utterance_index == 0:
                    mentions.append(
                        comparator.HumanMention(
                            family="VERB_ACTION", text="beta", visible_interval=interval
                        )
                    )
                    verb = ("beta",)
            humans.append(
                comparator.HumanUtterance(
                    interval=interval,
                    text=text,
                    role=role,
                    referential_status=status,
                    mentions=tuple(mentions),
                )
            )
            machines.append(
                comparator.MachineUtterance(interval=interval, text=text, role=role)
            )
            windows.append(
                comparator.MachineWindow(
                    interval=interval,
                    referential_status=status,
                    role=role,
                    noun_candidates=noun,
                    verb_candidates=verb,
                )
            )
        items.append(
            comparator.ItemEvidence(
                masks=((0, 60_000),),
                human_utterances=tuple(humans),
                machine_utterances=tuple(machines),
                machine_vad=((0, 60_000),),
                machine_windows=tuple(windows),
                wer_applicable=True,
            )
        )
    return items


def test_frozen_normalization_and_edit_distance() -> None:
    assert comparator.normalize_text("  HéLLO—world! ") == "héllo world"
    assert comparator.edit_distance("kitten", "sitting") == 3
    assert comparator.edit_distance("", "abc") == 3


def test_boundary_match_maximizes_cardinality_then_iou() -> None:
    human = [(0, 1000), (900, 1900)]
    machine = [(0, 1000), (1000, 2000)]
    assert comparator.maximum_cardinality_temporal_matching(
        human, machine, tolerance_ms=1000
    ) == [(0, 0), (1, 1)]


def test_all_six_frozen_gates_pass_on_perfect_supported_synthetic_fixture() -> None:
    results, reserve = comparator.evaluate_items(
        _passing_items(),
        _protocol(),
        protocol_digest="a" * 64,
        sample_digest="b" * 64,
    )
    assert set(results) == set(comparator.GATE_NAMES)
    assert all(result["status"] == "PASS" for result in results.values())
    assert all(result["threshold_frozen"] is True for result in results.values())
    assert all(result["leave_one_item_out"]["reported"] is True for result in results.values())
    assert reserve is False


def test_supported_role_failure_is_hard_fail_and_cannot_activate_reserve() -> None:
    items = []
    for item in _passing_items():
        items.append(
            comparator.ItemEvidence(
                masks=item.masks,
                human_utterances=item.human_utterances,
                machine_utterances=tuple(
                    comparator.MachineUtterance(
                        interval=row.interval, text=row.text, role="UNCERTAIN"
                    )
                    for row in item.machine_utterances
                ),
                machine_vad=item.machine_vad,
                machine_windows=tuple(
                    comparator.MachineWindow(
                        interval=row.interval,
                        referential_status=row.referential_status,
                        role="UNCERTAIN",
                        noun_candidates=row.noun_candidates,
                        verb_candidates=row.verb_candidates,
                    )
                    for row in item.machine_windows
                ),
                wer_applicable=item.wer_applicable,
            )
        )
    results, reserve = comparator.evaluate_items(
        items,
        _protocol(),
        protocol_digest="a" * 64,
        sample_digest="b" * 64,
    )
    assert results["source_role_non_child_priority"]["status"] == "HARD_FAIL"
    assert reserve is False


def test_primary_support_shortfall_is_borderline_and_reserve_eligible() -> None:
    items = []
    for item in _passing_items():
        only = item.human_utterances[:1]
        items.append(
            comparator.ItemEvidence(
                masks=item.masks,
                human_utterances=only,
                machine_utterances=item.machine_utterances[:1],
                machine_vad=(only[0].interval,),
                machine_windows=item.machine_windows[:1],
                wer_applicable=True,
            )
        )
    results, reserve = comparator.evaluate_items(
        items,
        _protocol(),
        protocol_digest="a" * 64,
        sample_digest="b" * 64,
    )
    assert results["speech_utterance_boundary"]["status"] == "BORDERLINE"
    assert reserve is True


def _private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.chmod(path, 0o600)


def test_reserve_activation_is_one_time_sealed_and_only_for_borderline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".childlens-test"
    root.mkdir(mode=0o700)
    selection = "c" * 64
    packet = {
        "schema_version": "childlens-author-audit-reserve-sample-v1.3.0",
        "sample_kind": "BORDERLINE_ESCALATION_RESERVE",
        "route": "AUTHOR_AUDIT_A_BORDERLINE_RESERVE",
        "activation": "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK",
        "maximum_activation_count": 1,
        "frozen_selection_digest": selection,
        "sampler_policy_sha256": "d" * 64,
        "official_windows_manifest_sha256": "e" * 64,
        "item_count": 15,
        "opaque_key_attestation": True,
        "selection_independent_of_model_outputs": True,
        "sample_frozen_before_predictions": True,
        "model_predictions_present": False,
        "model_predictions_used_for_selection": False,
        "exact_intervals_restricted": True,
        "total_duration_ms": 900_000,
        "items": [],
    }
    packet_path = root / "reserve.json"
    _private_json(packet_path, packet)
    reserve_digest = hashlib.sha256(comparator._canonical(packet)).hexdigest()
    receipt_path = tmp_path / "sampling.json"
    receipt_path.write_text(
        json.dumps(
            {
                "reserve_packet_sha256": reserve_digest,
                "frozen_v1_2_selection_digest": selection,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(comparator, "SAMPLING_RECEIPT_PATH", receipt_path)
    meta = {
        "author_record_hmac": "f" * 64,
        "sample_digest": "a" * 64,
        "protocol_digest": "b" * 64,
        "author_pass_locked_at_utc": "2026-07-22T00:00:00+00:00",
        "workflow_state": "AUTHOR_LOCKED",
    }
    thresholds = {
        name: {"status": "BORDERLINE" if index == 0 else "PASS"}
        for index, name in enumerate(comparator.GATE_NAMES)
    }
    assert comparator._activate_reserve_once(root, meta, thresholds) is True
    seal = root / comparator.COMPARISON_DIRECTORY / comparator.RESERVE_ACTIVATION_FILE
    first = seal.read_bytes()
    assert comparator._activate_reserve_once(root, meta, thresholds) is True
    assert seal.read_bytes() == first
    assert json.loads(first)["activation_count"] == 1

    hard = dict(thresholds)
    hard[comparator.GATE_NAMES[0]] = {"status": "HARD_FAIL"}
    with pytest.raises(comparator.ComparatorError, match="E_RESERVE_ACTIVATION_NOT_ALLOWED"):
        comparator._activate_reserve_once(root, meta, hard)

    unlocked = dict(meta)
    unlocked["workflow_state"] = "AUTHOR_BLIND_OPEN"
    with pytest.raises(comparator.ComparatorError, match="E_RESERVE_REQUIRES_AUTHOR_LOCK"):
        comparator._activate_reserve_once(root, unlocked, thresholds)


def test_comparator_refuses_unlocked_synthetic_author_database(tmp_path: Path) -> None:
    root = tmp_path / ".childlens-unlocked"
    root.mkdir(mode=0o700)
    directory = root / comparator.workflow.WORKFLOW_DIR
    directory.mkdir(mode=0o700)
    database = directory / comparator.workflow.DATABASE_FILE
    secret = directory / comparator.workflow.SECRET_FILE
    connection = comparator.sqlite3.connect(database)
    connection.executescript(comparator.workflow.SCHEMA)
    connection.execute(
        """INSERT INTO workflow_meta(
           singleton,schema_version,workflow_version,route,sample_digest,
           packet_digest,protocol_digest,workflow_state,estimated_author_minutes,
           created_at_utc,inter_human_reliability_available,
           pseudo_labels_are_ground_truth,primary_evaluation_truth)
           VALUES(1,1,?,'AUTHOR_AUDIT_A',?,?,?,'AUTHOR_BLIND_OPEN',75,?,0,0,
                  'SIMULATOR_ORACLE_ONLY')""",
        (
            comparator.workflow.VERSION,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "2026-07-22T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()
    os.chmod(database, 0o600)
    secret.write_bytes(b"x" * 32)
    os.chmod(secret, 0o600)
    with pytest.raises(comparator.ComparatorError, match="E_AUTHOR_AUDIT_NOT_LOCKED"):
        comparator._locked_context(root)


def test_source_contains_no_network_or_hosted_model_client() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "httpx", "openai", "socket", "subprocess"):
        assert f"import {forbidden}" not in source


def test_public_receipt_validator_rejects_restricted_shape_and_small_cells() -> None:
    receipt = {
        "schema_version": comparator.RECEIPT_SCHEMA,
        "audit_complete": True,
        "threshold_results": {
            name: {"status": "PASS", "threshold_frozen": True}
            for name in comparator.GATE_NAMES
        },
    }
    comparator.validate_public_receipt(receipt)
    unsafe = dict(receipt)
    unsafe["transcript_text"] = "synthetic"
    with pytest.raises(comparator.ComparatorError, match="E_PUBLIC_RECEIPT_PRIVACY"):
        comparator.validate_public_receipt(unsafe)
    small = dict(receipt)
    small["support"] = {"observed": 3}
    with pytest.raises(comparator.ComparatorError, match="E_PUBLIC_RECEIPT_SMALL_CELL"):
        comparator.validate_public_receipt(small)


def test_terminal_accepts_pending_one_time_reserve_shape_without_reactivation() -> None:
    receipt = {
        "schema_version": comparator.RECEIPT_SCHEMA,
        "audit_complete": True,
        "qualified_author_confirmed": True,
        "author_used_raw_audio_video": True,
        "author_blinded_until_irreversible_lock": True,
        "author_record_irreversibly_locked": True,
        "model_predictions_revealed_only_after_lock": True,
        "sample_selection_digest_match": True,
        "thresholds_digest_match": True,
        "cluster_aware_uncertainty_reported": True,
        "small_sample_limitation_reported": True,
        "author_record_changed_after_lock": False,
        "thresholds_changed_after_author_labels": False,
        "sample_changed_after_author_labels": False,
        "human_evidence_fabricated": False,
        "inter_human_reliability_claimed": False,
        "additional_audit_speech_seconds_used": 0,
        "reserve_activation": {
            "eligible": True,
            "activated_once": True,
            "pending_author_lock": True,
        },
        "threshold_results": {
            name: {
                "status": "BORDERLINE" if index == 0 else "PASS",
                "threshold_frozen": True,
            }
            for index, name in enumerate(comparator.GATE_NAMES)
        },
    }
    assert terminal._audit_valid_shape(receipt) is True
