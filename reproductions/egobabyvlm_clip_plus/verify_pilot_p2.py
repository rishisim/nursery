#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "pilot_p2_config.json"
AGGREGATE = ROOT / "pilots" / "juno_sample" / "p2_aggregate.json"


def require(value, message):
    if not value:
        raise AssertionError(message)


def load_preflight():
    spec = importlib.util.spec_from_file_location("pilot_p2_preflight", ROOT / "pilot_p2_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_tests(config):
    preflight = load_preflight()
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        scratch = root / "scratch"; durable = root / "durable"
        source = scratch / "caches" / "source"; output = durable / "run_records" / "detail.json"
        source.mkdir(parents=True); output.parent.mkdir(parents=True)
        preflight.validate_contract(config, source, scratch, durable, output)
        for field in ("from_pretrained_allowed", "torch_hub_allowed", "checkpoint_paths_allowed",
                      "network_allowed_during_construction", "hugging_face_model_names_allowed"):
            bad = json.loads(json.dumps(config)); bad["learned_initialization"][field] = True
            try:
                preflight.validate_contract(bad, source, scratch, durable, output)
            except preflight.PreflightError:
                pass
            else:
                raise AssertionError(f"forbidden learned initialization accepted: {field}")
        try:
            preflight.validate_contract(config, root / "worktree", scratch, durable, output)
        except preflight.PreflightError:
            pass
        else:
            raise AssertionError("non-Juno source path accepted")
        first = {"schema_version": 1, "values": [1, 2, 3]}
        preflight.atomic_json(output, first)
        require(json.loads(output.read_text()) == first, "deterministic JSON fixture failed")


def main():
    config_text = CONFIG.read_text(encoding="utf-8")
    aggregate_text = AGGREGATE.read_text(encoding="utf-8")
    config = json.loads(config_text); aggregate = json.loads(aggregate_text)
    synthetic_tests(config)
    require(aggregate["status"] == "p2_complete" and aggregate["p2_gate"]["passed"], "P2 gate failed")
    require(aggregate["inventory_status"] == "incomplete_inventory", "inventory overclaimed")
    require(aggregate["scientific_status_effect"] == "none", "scientific status changed")
    require(aggregate["p0_status_preserved"] and aggregate["p1_status_preserved"], "prior pilot gate changed")
    require(aggregate["next_stage_started"] is False, "P3 or later started")
    require(aggregate["full_corpus_discrepancy"] == "unresolved_and_out_of_scope", "corpus blocker changed")
    vision = aggregate["construction"]["vision"]; text = aggregate["construction"]["text"]
    require((vision["hidden_size"], vision["layers"], vision["attention_heads"]) == (768, 12, 12),
            "DINOv2 shape changed")
    require((text["hidden_size"], text["layers"], text["attention_heads"]) == (768, 12, 12),
            "BERT shape changed")
    require(vision["constructed"] and text["constructed"], "real architecture did not construct")
    boundary = aggregate["weight_boundary"]
    require(boundary["learned_initialization"] == "explicit_config_random_only", "random init policy changed")
    require(not any(boundary[key] for key in ("external_pretrained_allowed", "from_pretrained_allowed",
                "torch_hub_allowed", "learned_checkpoint_paths_allowed", "network_allowed_during_construction",
                "checkpoint_loaded", "network_fallback_used")), "forbidden initialization enabled")
    require(boundary["preprocessing_namespace_isolated"], "preprocessing namespace not isolated")
    require(aggregate["preprocessing_only_provenance"]["preprocessing_executed"] is False,
            "preprocessing was executed")
    require(aggregate["construction"]["training_step_executed"] is False, "training step executed")
    require(aggregate["construction"]["checkpoint_created"] is False, "checkpoint created")
    require(aggregate["provenance"]["p2_config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "P2 config digest mismatch")
    forbidden = ["/work/", "/scratch/", "participant", "session", "record_key", "media_path",
                 "filename", "asset_id", "credential", "token="]
    require(not any(token in aggregate_text.lower() for token in forbidden), "aggregate violates privacy boundary")
    print("Pilot P2 verification passed")


if __name__ == "__main__":
    main()
