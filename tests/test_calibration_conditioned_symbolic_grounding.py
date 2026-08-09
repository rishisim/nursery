from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_calibration_conditioned_symbolic_grounding.py"
FIXTURE = ROOT / "tests/fixtures/calibration_conditioned_symbolic_ranges_v1.json"
PROTOCOL = ROOT / "docs/nursery_program_convergence_v1/frozen_calibration_conditioned_sensitivity_protocol.json"
CALIBRATION = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json"
SPEC = importlib.util.spec_from_file_location("calibration_conditioned_symbolic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _fixture_document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_range_binding_is_nonempirical_and_complete() -> None:
    binding = module.bind_calibration_ranges(_fixture_document(), fixture_only=True)
    assert binding.fixture_only is True
    assert binding.gate_passed is True
    assert binding.suppressed_dimensions == ()
    assert [family.name for family in binding.families] == ["synthetic_fixture_nominal"]


def test_fixture_binding_rejects_restricted_or_outcome_fields() -> None:
    value = _fixture_document()
    value["transcript"] = "not allowed"
    with pytest.raises(module.SymbolicRunnerError, match="E_FIXTURE_RESTRICTED_FIELD"):
        module.bind_calibration_ranges(value, fixture_only=True)
    value = _fixture_document()
    value["scientific_outcome_allowed"] = True
    with pytest.raises(module.SymbolicRunnerError, match="E_FIXTURE_BOUNDARY"):
        module.bind_calibration_ranges(value, fixture_only=True)


def test_live_aggregate_ranges_are_accepted_but_gate_remains_closed() -> None:
    binding = module.bind_calibration_ranges(
        json.loads(CALIBRATION.read_text(encoding="utf-8")), fixture_only=False
    )
    assert binding.status == "CALIBRATION_REVISE"
    assert binding.gate_passed is False
    assert binding.suppressed_dimensions


def test_five_arms_share_primary_inventory_and_side_marginals() -> None:
    binding = module.bind_calibration_ranges(_fixture_document(), fixture_only=True)
    episodes = module.generate_fixture_episodes(
        module.parameter_point(binding.families[0]), 910001
    )
    views = module.condition_views(episodes)
    assert tuple(views) == module.ARMS
    assert len({module.primary_digest(rows) for rows in views.values()}) == 1
    assert len(
        {
            module.side_multiset_digest(rows)
            for arm, rows in views.items()
            if arm != "weak_vl_absent_side"
        }
    ) == 1
    assert all(
        not ({"noun_truth", "verb_truth"} & set(row))
        for rows in views.values()
        for row in rows
    )


def test_dual_corrective_microcase_changes_wrong_or_tied_mappings() -> None:
    result = module.corrective_microcase()
    assert result["noun"]["wrong_or_tied_to_correct"] is True
    assert result["verb"]["wrong_or_tied_to_correct"] is True
    assert result["noun"]["disconnected_equals_baseline_top"] is True
    assert result["verb"]["disconnected_equals_baseline_top"] is True


def test_evaluator_rejects_side_fields_and_model_serializes_none() -> None:
    binding = module.bind_calibration_ranges(_fixture_document(), fixture_only=True)
    episodes = module.generate_fixture_episodes(
        module.parameter_point(binding.families[0]), 910001
    )
    rows = module.condition_views(episodes)["weak_vl_synchronized_side"]
    model = module.fit_dual_corrective(rows, model_seed=920001)
    assert model["side_state_serialized"] is False
    items = [dict(item) for item in module.evaluation_items()]
    items[0]["side_cue"] = [1.0]
    with pytest.raises(module.SymbolicRunnerError, match="E_EVALUATION_SIDE_OR_EXTRA_FIELD"):
        module.evaluate_without_side(model, items)


def test_fixture_smoke_has_paired_bundles_and_all_falsifications() -> None:
    binding = module.bind_calibration_ranges(_fixture_document(), fixture_only=True)
    receipt = module.execute_fixture_smoke(
        binding, corpus_seeds=(910001,), model_seeds=(920001,)
    )
    assert receipt["status"] == "FIXTURE_SMOKE_PASS"
    assert receipt["scientific_outcome_run"] is False
    assert receipt["planned_outcome_command_executed"] is False
    assert receipt["five_matched_arms"] == list(module.ARMS)
    bundle = receipt["bundles"][0]
    assert all(bundle["falsification_gates"].values())
    assert all(value == 1.0 for value in bundle["strong_alignment_ceiling"].values())


def test_planned_command_is_exact_and_run_interface_refuses_outcome(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["execution"]["planned_launch_command"] == module.PLANNED_COMMAND
    output = tmp_path / "must-not-exist"
    code = module.main(
        [
            "run",
            "--protocol",
            str(PROTOCOL),
            "--calibration-receipt",
            str(CALIBRATION),
            "--output-root",
            str(output),
            "--jobs",
            "4",
        ]
    )
    assert code == 2
    assert not output.exists()

def test_even_forged_authorization_cannot_execute_scientific_path(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    forged = copy.deepcopy(protocol)
    forged["execution"]["outcome_authorized"] = True
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(forged), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    code = module.main(
        [
            "run",
            "--protocol",
            str(protocol_path),
            "--calibration-receipt",
            str(CALIBRATION),
            "--output-root",
            str(output),
            "--jobs",
            "4",
        ]
    )
    assert code == 2
    assert not output.exists()
