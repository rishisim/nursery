from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_calibration_conditioned_symbolic_grounding_v2.py"
PROTOCOL = ROOT / "docs/nursery_program_convergence_v1/frozen_calibration_conditioned_sensitivity_protocol.json"
CONTRACT = ROOT / "docs/nursery_program_convergence_v1/calibration_conditioned_symbolic_execution_contract_v2.json"
FIXTURE = ROOT / "tests/fixtures/calibration_conditioned_symbolic_ranges_v1.json"
HISTORICAL_CALIBRATION = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json"
SUBSTITUTION_CALIBRATION = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
SPEC = importlib.util.spec_from_file_location("calibration_conditioned_symbolic_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


@pytest.fixture(scope="module")
def contract():
    return module.load_execution_contract(CONTRACT, PROTOCOL)


@pytest.fixture(scope="module")
def fixture_binding():
    return module.v1.bind_calibration_ranges(
        json.loads(FIXTURE.read_text(encoding="utf-8")), fixture_only=True
    )


def test_contract_freezes_existing_protocol_seeds_ranges_arms_and_thresholds(contract) -> None:
    assert contract.corpus_seeds == tuple(range(740001, 740080, 2))
    assert contract.model_seeds == (750001, 750003, 750007)
    assert contract.range_draws == 7
    assert contract.arms == module.v1.ARMS
    assert contract.range_families == module.REPLACEMENT_RANGE_FAMILIES
    assert len(contract.erratum_digest) == 64
    assert str(SUBSTITUTION_CALIBRATION.relative_to(ROOT)) in contract.planned_command
    assert str(HISTORICAL_CALIBRATION.relative_to(ROOT)) not in contract.planned_command
    assert "--authorization-seal" in contract.planned_command
    assert "--preoutcome-receipt" in contract.planned_command


def test_maximin_lhs_is_deterministic_unique_and_stratified(contract, fixture_binding) -> None:
    family = fixture_binding.families[0]
    first = module.deterministic_range_points(
        family, draws=7, inference_seed=contract.inference_seed
    )
    second = module.deterministic_range_points(
        family, draws=7, inference_seed=contract.inference_seed
    )
    assert module._digest(first) == module._digest(second)
    assert len({module._digest(value) for value in first}) == 7
    assert all(value["range_point_policy"] == "DETERMINISTIC_MAXIMIN_LATIN_HYPERCUBE" for value in first)


def test_fixture_preoutcome_gates_cover_leakage_correction_and_cue_withholding(
    contract, fixture_binding
) -> None:
    receipt = module.build_preoutcome_receipt(
        fixture_binding, contract, fixture_only=True
    )
    assert receipt["status"] == "FIXTURE_PASS"
    assert tuple(receipt["gates"]) == module.PREOUTCOME_GATES
    assert all(receipt["gates"].values())
    assert receipt["gates"]["confidence_only_noun_wrong_or_tied_mapping_corrected"] is True
    assert receipt["gates"]["confidence_only_verb_wrong_or_tied_mapping_corrected"] is True
    assert receipt["scientific_endpoints_opened"] is False


def test_live_calibration_cannot_produce_passed_preoutcome(contract) -> None:
    document = json.loads(HISTORICAL_CALIBRATION.read_text(encoding="utf-8"))
    binding = module.v1.bind_calibration_ranges(
        document, fixture_only=False
    )
    receipt = module.build_preoutcome_receipt(
        binding,
        contract,
        fixture_only=False,
        calibration_document=document,
    )
    assert receipt["status"] == "FAIL"
    assert receipt["gates"]["calibration_ranges_complete"] is False
    assert receipt["gates"]["calibration_receipt_public_privacy_policy"] is False
    assert receipt["scientific_endpoints_opened"] is False


def _synthetic_shaped_caf_success_receipt() -> dict:
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"synthetic_fixture_nominal"}:
                return {
                    family: copy.deepcopy(value["synthetic_fixture_nominal"])
                    for family in module.REPLACEMENT_RANGE_FAMILIES
                }
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value

    return {
        "schema_version": module.PROTOTYPE_CALIBRATION_SCHEMA,
        "status": "CALIBRATION_PASS",
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE",
        "caf_transport_amendment_sha256": module.CAF_TRANSPORT_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": module.NO_FALLBACK_OVERRIDE_SHA256,
        "automatic_fallback_allowed": False,
        "scientific_endpoint_if_gemma_fails": False,
        "instrument_paths": list(module.REPLACEMENT_RANGE_FAMILIES[:2]),
        "calibration_ranges": expand(source["calibration_ranges"]),
        "public_export": {
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            **{field: False for field in module.PUBLIC_FALSE_FIELDS},
        },
        "security": {
            "restricted_inference_network_disabled": True,
            "external_api_used": False,
            "hosted_or_cloud_content_path": False,
            "quarantine_only_pseudo_payloads": True,
        },
        "ancestry": {
            "AEA_empirical_ancestry": False,
            "BabyView_empirical_ancestry": False,
            "cross_corpus_pooling": False,
            "learner_ancestry_from_instruments": False,
        },
        "pseudo_labels_are_ground_truth": False,
        "human_evidence_available": False,
        "simulator_oracle_only_evaluation_truth": True,
        "scientific_outcome_run": False,
    }


def test_v2_preflight_accepts_synthetic_shaped_caf_success_receipt(contract) -> None:
    document = _synthetic_shaped_caf_success_receipt()
    binding = module.bind_calibration_ranges_v2(document, fixture_only=False)
    assert binding.digest == module._digest(document)
    assert tuple(family.name for family in binding.families) == tuple(sorted(module.REPLACEMENT_RANGE_FAMILIES))
    assert module._public_calibration_receipt_safe(document, binding)
    receipt = module.build_preoutcome_receipt(
        binding,
        contract,
        fixture_only=False,
        calibration_document=document,
    )
    assert receipt["gates"]["calibration_ranges_complete"] is True
    assert receipt["gates"]["calibration_receipt_public_privacy_policy"] is True
    assert receipt["scientific_endpoints_opened"] is False


def test_public_payload_sentinel_rejects_restricted_extra_field() -> None:
    document = json.loads(HISTORICAL_CALIBRATION.read_text(encoding="utf-8"))
    assert module._public_payload_safe(document)
    document["transcript_text"] = "forbidden"
    assert not module._public_payload_safe(document)


def test_complete_bundle_has_every_family_draw_model_and_arm(contract, fixture_binding) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    value = module.execute_complete_bundle(
        corpus_seed=plan.corpus_seeds[0],
        binding=fixture_binding,
        contract=contract,
        plan=plan,
    )
    assert value["complete_paired_bundle"] is True
    assert len(value["result_rows"]) == 1 * 2 * 1 * 5
    assert {row["arm"] for row in value["result_rows"]} == set(module.v1.ARMS)
    assert all(set(module.ENDPOINTS).issubset(row) for row in value["result_rows"])


def test_self_rehashed_incomplete_bundle_is_rejected(contract, fixture_binding) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    value = dict(
        module.execute_complete_bundle(
            corpus_seed=plan.corpus_seeds[0],
            binding=fixture_binding,
            contract=contract,
            plan=plan,
        )
    )
    value["result_rows"] = list(value["result_rows"][:-1])
    payload = dict(value)
    payload.pop("bundle_sha256")
    value["bundle_sha256"] = module._digest(payload)
    with pytest.raises(module.OneShotError, match="E_BUNDLE_CHECKPOINT"):
        module._validate_bundle(
            value,
            corpus_seed=plan.corpus_seeds[0],
            request_sha256=module._bundle_request_digest(
                corpus_seed=plan.corpus_seeds[0],
                binding=fixture_binding,
                contract=contract,
                plan=plan,
            ),
            plan=plan,
        )


def test_merge_is_byte_identical_under_completion_order(contract, fixture_binding) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    bundles = [
        module.execute_complete_bundle(
            corpus_seed=seed,
            binding=fixture_binding,
            contract=contract,
            plan=plan,
        )
        for seed in plan.corpus_seeds
    ]
    shuffled = list(bundles)
    random.Random(5).shuffle(shuffled)
    assert module._canonical(module.merge_bundles(bundles, plan=plan)) == module._canonical(
        module.merge_bundles(shuffled, plan=plan)
    )
    merged = module.merge_bundles(bundles, plan=plan)
    assert merged["confidence_only_falsification"]["confidence_endpoint_present"] is False
    assert merged["side_only_leakage_falsification"]["required_pre_outcome"] is True


def test_checkpoint_resume_reuses_only_valid_complete_bundle(
    tmp_path: Path, contract, fixture_binding, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    output = tmp_path / "first"
    receipt = module.execute_checkpointed_plan(
        binding=fixture_binding,
        contract=contract,
        plan=plan,
        authorization_id="fixture-checkpoint-0001",
        output_root=output,
        jobs=2,
    )
    assert receipt["status"] == "FIXTURE_COMPLETE"
    assert output.is_dir()
    assert len(list((output / "bundles").glob("*.json"))) == 2
    with pytest.raises(module.OneShotError, match="E_ONE_SHOT_COMPLETE"):
        module.execute_checkpointed_plan(
            binding=fixture_binding,
            contract=contract,
            plan=plan,
            authorization_id="fixture-checkpoint-0001",
            output_root=output,
            jobs=2,
        )


def test_completed_authorization_cannot_replay_if_published_tree_is_moved(
    tmp_path: Path, contract, fixture_binding
) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    output = tmp_path / "published"
    module.execute_checkpointed_plan(
        binding=fixture_binding,
        contract=contract,
        plan=plan,
        authorization_id="fixture-registry-0001",
        output_root=output,
        jobs=2,
    )
    output.rename(tmp_path / "published-preserved")
    with pytest.raises(module.OneShotError, match="E_ONE_SHOT_COMPLETE"):
        module.execute_checkpointed_plan(
            binding=fixture_binding,
            contract=contract,
            plan=plan,
            authorization_id="fixture-registry-0001",
            output_root=output,
            jobs=2,
        )
    with pytest.raises(module.OneShotError, match="E_ONE_SHOT_REGISTRY"):
        module.execute_checkpointed_plan(
            binding=fixture_binding,
            contract=contract,
            plan=plan,
            authorization_id="fixture-registry-0002",
            output_root=output,
            jobs=2,
        )


def test_interrupted_checkpoint_resumes_without_recomputing_complete_bundle(
    tmp_path: Path, contract, fixture_binding, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    output = tmp_path / "resumed"
    original = module.execute_complete_bundle
    calls: list[int] = []

    def interrupted(**kwargs):
        seed = int(kwargs["corpus_seed"])
        calls.append(seed)
        if seed == plan.corpus_seeds[1]:
            raise module.OneShotError("E_SYNTHETIC_INTERRUPTION")
        return original(**kwargs)

    monkeypatch.setattr(module, "execute_complete_bundle", interrupted)
    with pytest.raises(module.OneShotError, match="E_SYNTHETIC_INTERRUPTION"):
        module.execute_checkpointed_plan(
            binding=fixture_binding,
            contract=contract,
            plan=plan,
            authorization_id="fixture-resume-000001",
            output_root=output,
            jobs=1,
        )
    staging = tmp_path / ".resumed.fixture-resume-000001.staging"
    assert len(list((staging / "bundles").glob("*.json"))) == 1
    monkeypatch.setattr(module, "execute_complete_bundle", original)
    calls.clear()

    def counted(**kwargs):
        calls.append(int(kwargs["corpus_seed"]))
        return original(**kwargs)

    monkeypatch.setattr(module, "execute_complete_bundle", counted)
    receipt = module.execute_checkpointed_plan(
        binding=fixture_binding,
        contract=contract,
        plan=plan,
        authorization_id="fixture-resume-000001",
        output_root=output,
        jobs=1,
    )
    assert receipt["status"] == "FIXTURE_COMPLETE"
    assert calls == [plan.corpus_seeds[1]]


def test_publication_failure_consumes_authorization_fail_closed(
    tmp_path: Path,
    contract,
    fixture_binding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = module.fixture_plan(contract, fixture_binding.families[0].name)
    output = tmp_path / "publication-failure"
    original_replace = module.os.replace

    def fail_publication(source, destination):
        if Path(source).name.endswith(".staging") and Path(destination) == output:
            raise OSError("synthetic publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_publication)
    with pytest.raises(module.OneShotError, match="E_PUBLICATION_AFTER_CONSUMPTION"):
        module.execute_checkpointed_plan(
            binding=fixture_binding,
            contract=contract,
            plan=plan,
            authorization_id="fixture-publish-00001",
            output_root=output,
            jobs=2,
        )
    monkeypatch.setattr(module.os, "replace", original_replace)
    with pytest.raises(module.OneShotError, match="E_ONE_SHOT_COMPLETE"):
        module.execute_checkpointed_plan(
            binding=fixture_binding,
            contract=contract,
            plan=plan,
            authorization_id="fixture-publish-00001",
            output_root=output,
            jobs=2,
        )


def test_authorization_seal_is_required_private_and_exactly_bound(
    tmp_path: Path, contract, fixture_binding
) -> None:
    preoutcome = module.build_preoutcome_receipt(
        fixture_binding, contract, fixture_only=True
    )
    preoutcome_path = tmp_path / "preoutcome.json"
    preoutcome_path.write_text(json.dumps(preoutcome), encoding="utf-8")
    os.chmod(preoutcome_path, 0o600)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("{}", encoding="utf-8")
    os.chmod(seal_path, 0o644)
    with pytest.raises(module.OneShotError, match="E_AUTHORIZATION_PRIVATE"):
        module.validate_authorization_seal(
            seal_path,
            protocol_path=PROTOCOL,
            contract_path=CONTRACT,
            calibration_path=FIXTURE,
            calibration_document=None,
            binding=fixture_binding,
            preoutcome_path=preoutcome_path,
            contract=contract,
        )


def test_authorization_revalidates_calibration_public_policy(
    tmp_path: Path, contract, fixture_binding
) -> None:
    preoutcome = module.build_preoutcome_receipt(
        fixture_binding, contract, fixture_only=True
    )
    preoutcome_path = tmp_path / "preoutcome.json"
    preoutcome_path.write_text(json.dumps(preoutcome), encoding="utf-8")
    os.chmod(preoutcome_path, 0o600)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("{}", encoding="utf-8")
    os.chmod(seal_path, 0o600)
    fixture_document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(module.OneShotError, match="E_CALIBRATION_PUBLIC_POLICY"):
        module.validate_authorization_seal(
            seal_path,
            protocol_path=PROTOCOL,
            contract_path=CONTRACT,
            calibration_path=FIXTURE,
            calibration_document=fixture_document,
            binding=fixture_binding,
            preoutcome_path=preoutcome_path,
            contract=contract,
        )


def test_current_exact_run_is_blocked_before_output(tmp_path: Path) -> None:
    preoutcome = tmp_path / "failed-preoutcome.json"
    preoutcome.write_text("{}", encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text("{}", encoding="utf-8")
    os.chmod(seal, 0o600)
    output = tmp_path / "must-not-exist"
    code = module.main(
        [
            "run",
            "--protocol",
            str(PROTOCOL),
            "--execution-contract",
            str(CONTRACT),
            "--calibration-receipt",
            str(HISTORICAL_CALIBRATION),
            "--preoutcome-receipt",
            str(preoutcome),
            "--authorization-seal",
            str(seal),
            "--output-root",
            str(output),
            "--jobs",
            "4",
        ]
    )
    assert code == 2
    assert not output.exists()


def test_scientific_plan_is_complete_but_not_executed(contract) -> None:
    plan = module.scientific_plan(contract)
    assert len(plan.corpus_seeds) == 40
    assert len(plan.model_seeds) == 3
    assert len(plan.family_names) == 5
    assert plan.range_draws == 7
    assert plan.fixture_only is False
