"""Non-outcome construction gate and engineering-only microbenchmark."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import torch

from .calibration import load_measurement_spec
from .fixtures import build_construction_fixture
from .model import (
    FreshCorpusTokenizer,
    TemporalCLIPConfig,
    TemporalCLIPPlusTrainingModel,
    VisionTextEvalBatch,
)
from .parallel import build_parallel_plan, validate_plan
from .policy import (
    CONSTRUCTION_PROFILE,
    canonical_digest,
    canonical_json_bytes,
    load_policy,
    validate_provenance,
)
from .protocol import load_and_validate_evaluation_spec, load_and_validate_protocol
from .schema import validate_bundle


FORBIDDEN_ENGINEERING_REPORT_KEYS = {
    "accuracy",
    "lift",
    "effect",
    "acquisition_score",
    "noun_score",
    "verb_score",
    "composition_score",
}


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _host_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_intraop_threads_observed": torch.get_num_threads(),
        "construction_note": "HOST_METADATA_ONLY_NO_SCIENTIFIC_DATA",
    }


def engineering_microbenchmark(
    model: TemporalCLIPPlusTrainingModel,
    *,
    device: torch.device,
    warmup_steps: int = 2,
    timed_steps: int = 5,
) -> dict[str, Any]:
    """Time one fixture-shaped forward path; do not train, score, or compare arms."""

    if warmup_steps < 1 or timed_steps < 1:
        raise ValueError("microbenchmark step counts must be positive")
    torch.manual_seed(710031)
    config = model.config
    batch = 2
    frames = 4
    token_count = 8
    candidates = 3
    inputs = {
        "video": torch.randint(0, 256, (batch, frames, 3, config.image_size, config.image_size), dtype=torch.uint8, device=device),
        "tokens": torch.randint(0, config.vocabulary_size, (batch, token_count), dtype=torch.long, device=device),
        "attention_mask": torch.ones((batch, token_count), dtype=torch.bool, device=device),
        "raw_side": torch.randn((batch, frames, config.side_input_dim), device=device),
        "candidate_event_mask": torch.tensor(
            [[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]] * batch,
            dtype=torch.float32,
            device=device,
        ),
        "candidate_valid": torch.ones((batch, candidates), dtype=torch.bool, device=device),
    }
    model = model.to(device).eval()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model.training_forward(**inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(timed_steps):
            model.training_forward(**inputs)
        _synchronize(device)
        elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "profile_label": CONSTRUCTION_PROFILE,
        "purpose": "ENGINEERING_FORWARD_THROUGHPUT_AND_RESOURCE_ESTIMATE_ONLY",
        "outcome_execution": False,
        "scientific_interpretation": False,
        "fixture_shape": {
            "batch": batch,
            "frames": frames,
            "image_size": config.image_size,
            "tokens": token_count,
            "raw_side_channels": config.side_input_dim,
            "candidate_events": candidates,
        },
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_forward": elapsed / timed_steps,
        "fixture_examples_per_second": batch * timed_steps / elapsed,
        "parameter_count": parameters,
        "fp32_parameter_bytes": parameters * 4,
        "rough_adam_training_state_bytes": parameters * 16,
        "architecture_digest": config.architecture_digest,
        "host": _host_record(device),
    }
    if FORBIDDEN_ENGINEERING_REPORT_KEYS.intersection(result):
        raise AssertionError("engineering report accidentally contains an acquisition endpoint")
    return result


def run_construction_validation(repo_root: str | Path, *, benchmark_device: str = "cpu") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "docs" / "child_only_prototype_v1"
    provenance = json.loads((docs / "construction_provenance_v1.json").read_text(encoding="utf-8"))
    validate_provenance(provenance)
    load_policy(docs / "provenance_policy_v1.json")
    protocol = load_and_validate_protocol(docs / "causal_protocol_v1.json")
    protocol_digest = canonical_digest(protocol)
    load_and_validate_evaluation_spec(docs / "evaluation_spec_v1.json")
    load_measurement_spec(docs / "calibration" / "babyview_measurement_spec_v1.json", "BABYVIEW")
    load_measurement_spec(docs / "calibration" / "childlens_measurement_spec_v1.json", "CHILDLENS")

    with tempfile.TemporaryDirectory(prefix="child-only-construction-") as temp:
        fixture_root = Path(temp) / "fixture"
        first_manifest = build_construction_fixture(fixture_root)
        bundle_result = validate_bundle(fixture_root)
        second_root = Path(temp) / "fixture_repeat"
        second_manifest = build_construction_fixture(second_root)
        if first_manifest["canonical_digest"] != second_manifest["canonical_digest"]:
            raise AssertionError("construction fixture manifest is not deterministic")

    tokenizer = FreshCorpusTokenizer.fit(
        ["look at it move", "see it move"],
        source_family="SYNTHETIC_FIXTURE",
        corpus_instance_id="construction-fixture-instance-v1",
        tokenizer_artifact_id="fixture-tokenizer-v1",
        max_vocabulary_size=64,
        profile_label=CONSTRUCTION_PROFILE,
    )
    config = TemporalCLIPConfig(vocabulary_size=64)
    initialization_receipt = {
        "model_initialization": "SCRATCH",
        "parent_checkpoint": None,
        "model_artifact_id": "fixture-model-v1",
        "corpus_instance_id": "construction-fixture-instance-v1",
        "construction_receipt_id": "fixture-model-construction-receipt-v1",
    }
    model = TemporalCLIPPlusTrainingModel(config, initialization_receipt=initialization_receipt)
    device = torch.device(benchmark_device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS benchmark requested but unavailable")
    benchmark = engineering_microbenchmark(model, device=device)

    # Structural primary-evaluation smoke: an exported module has no side state
    # and consumes only the exact typed V/T batch.
    eval_model = model.export_primary_evaluation_model().to(device).eval()
    tokens, mask = tokenizer.encode(["look at it move", "see it move"], config.max_tokens)
    eval_batch = VisionTextEvalBatch.from_mapping(
        {
            "video": torch.zeros((2, 4, 3, config.image_size, config.image_size), dtype=torch.uint8, device=device),
            "tokens": tokens.to(device),
            "attention_mask": mask.to(device),
        }
    )
    with torch.inference_mode():
        eval_outputs = eval_model(eval_batch)
    if eval_outputs["similarity"].shape != (2, 2):
        raise AssertionError("primary evaluation interface shape is invalid")

    plan = build_parallel_plan(
        protocol_digest=protocol_digest,
        corpus_instance_id="construction-fixture-instance-v1",
        corpus_seeds=[101, 202],
        model_seeds=[11, 22],
        backend="mps",
        max_trainers=1,
        cpu_workers=2,
        profile_label=CONSTRUCTION_PROFILE,
        outcome_task_authorized=False,
    )
    validate_plan(plan)
    report_core = {
        "validation_version": "child-only-construction-validation-v1",
        "terminal_candidate": "CHILD_ONLY_PROTOTYPE_CONSTRUCTION_READY",
        "profile_label": CONSTRUCTION_PROFILE,
        "scientific_outcome_executed": False,
        "restricted_child_data_accessed": False,
        "selected_corpus": None,
        "gates": {
            "provenance_policy": "PASS",
            "schema_and_oracle_separation": "PASS",
            "separate_calibration_specs": "PASS",
            "temporal_clip_plus_interface": "PASS",
            "vision_text_only_evaluation_export": "PASS",
            "causal_condition_contract": "PASS",
            "corpus_grounded_evaluation_spec": "PASS",
            "deterministic_fixture_manifest": "PASS",
            "device_aware_parallel_plan": "PASS",
        },
        "fixture_bundle": bundle_result,
        "model": {
            "architecture_digest": config.architecture_digest,
            "tokenizer_receipt_digest": tokenizer.receipt()["digest"],
            "evaluation_state_keys": sorted(eval_model.state_dict()),
        },
        "engineering_microbenchmark": benchmark,
        "parallel_plan_digest": plan["plan_digest"],
    }
    return {**report_core, "validation_digest": canonical_digest(report_core)}


def write_validation_report(report: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    if path.exists():
        raise FileExistsError("validation report writer refuses to overwrite evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(report) + b"\n")
