from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


runner = _load("test_gemma_delegation_runner_v1_5", "scripts/run_gemma4_prototype_common_schema_canary_v1_5.py")
coordinator = _load("test_gemma_delegation_coordinator_v1_5", "scripts/nursery_gemma_substitution_calibration_v1_5.py")


def test_amendment_and_historical_lineage_are_exact() -> None:
    amendment = runner._validate_delegation_amendment()
    assert amendment["status"] == "FROZEN_PRE_CANARY_PREFLIGHT_DELEGATION_ONLY"
    assert runner._sha256_file(runner.V14_RUNNER) == runner.V14_RUNNER_SHA256
    assert runner._sha256_file(runner.V14_COORDINATOR) == runner.V14_COORDINATOR_SHA256
    assert runner._sha256_file(runner.V14_FAILURE) == runner.V14_FAILURE_SHA256
    assert runner._sha256_file(runner.WORKER) == runner.WORKER_SHA256


def test_seal_is_precomputed_once_projected_purely_and_callables_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    seal = {"schema_version": "synthetic", "parser_mode": "PRIMARY", "nested": {"a": 1}}
    calls = {"preflight": 0, "nested": 0}
    original_v13 = runner.v14.v13._public_preflight
    original_v12 = runner.v14.v13.v12._public_preflight

    def preflight(mode: str):
        calls["preflight"] += 1
        assert mode == "PRIMARY"
        return seal

    def nested(supplied):
        calls["nested"] += 1
        first = runner.v14.v13._public_preflight("PRIMARY")
        second = runner.v14.v13._public_preflight("PRIMARY")
        assert first == seal and second == seal and first is not second
        assert supplied == seal
        return {"runtime_lock": {}, "receipt": {"status": "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS"}}

    monkeypatch.setattr(runner, "_public_preflight", preflight)
    monkeypatch.setattr(runner.v14.v13, "sandboxed_execute", nested)
    result = runner.sandboxed_execute(seal)
    assert calls == {"preflight": 1, "nested": 1}
    assert result["receipt"]["preflight_seal_precompute_count"] == 1
    assert result["receipt"]["nested_amendment_preflight_recomputed"] is False
    assert runner.v14.v13._public_preflight is original_v13
    assert runner.v14.v13.v12._public_preflight is original_v12


def test_projection_fails_closed_on_parser_mismatch_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    seal = {"schema_version": "synthetic", "parser_mode": "PRIMARY"}
    original_v13 = runner.v14.v13._public_preflight

    def nested(_seal):
        runner.v14.v13._public_preflight("PRIMARY")
        runner.v14.v13._public_preflight("ONE_OUTER_FENCE")
        raise AssertionError("mismatch accepted")

    monkeypatch.setattr(runner, "_public_preflight", lambda _mode: seal)
    monkeypatch.setattr(runner.v14.v13, "sandboxed_execute", nested)
    with pytest.raises(runner.DelegationCanaryError, match="E_PARSER_MODE"):
        runner.sandboxed_execute(seal)
    assert runner.v14.v13._public_preflight is original_v13


def test_coordinator_fails_before_restricted_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coordinator, "ACTIVATION", tmp_path / "missing-activation.json")
    monkeypatch.setattr(coordinator, "PUBLIC_CANARY", tmp_path / "missing-canary.json")
    monkeypatch.setattr(coordinator.legacy.base, "_restricted_inputs", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("restricted discovery")))
    with pytest.raises(coordinator.GemmaSubstitutionError, match="E_DELEGATION_ACTIVATION_MISSING"):
        coordinator._public_gemma_seal()


def test_no_fallback_and_original_success_path_are_preserved() -> None:
    assert coordinator.PUBLIC_RECEIPT == ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
    for common in (runner._delegation_common_receipt(),):
        assert common["nested_amendment_preflight_recomputed"] is False
    source = (ROOT / "scripts/nursery_gemma_substitution_calibration_v1_5.py").read_text(encoding="utf-8")
    assert '"automatic_fallback_allowed": False' in source
    assert '"scientific_endpoint_if_gemma_fails": False' in source
