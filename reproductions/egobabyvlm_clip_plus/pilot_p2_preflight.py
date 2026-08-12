#!/usr/bin/env python3
"""Construct the real Pilot P2 shapes under a fail-closed weight boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
from pathlib import Path


LABELS = ["engineering_only", "non_comparable", "not_a_reproduction_result"]


class PreflightError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise PreflightError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_contract(config, source_root, scratch_root, durable_root, output):
    require(config["stage"] == "P2", "wrong pilot stage")
    require(config["classification"] == LABELS, "classification changed")
    require(inside(source_root, Path(scratch_root) / "caches"), "source is outside governed scratch caches")
    require(inside(output, Path(durable_root) / "run_records"), "detail output is outside durable run records")
    policy = config["learned_initialization"]
    require(policy == {
        "policy": "explicit_config_random_only",
        "external_pretrained_allowed": False,
        "network_allowed_during_construction": False,
        "checkpoint_paths_allowed": False,
        "hugging_face_model_names_allowed": False,
        "from_pretrained_allowed": False,
        "torch_hub_allowed": False,
    }, "learned initialization boundary is not fail closed")
    preprocessing = config["preprocessing_only"]
    require(preprocessing["referenceable_by_learned_initialization"] is False,
            "preprocessing namespace is referenceable by learned initialization")
    learned_text = json.dumps(policy, sort_keys=True)
    require(preprocessing["namespace"] not in learned_text, "namespace separation failed")


def verify_source(config, source_root):
    source_root = Path(source_root)
    head = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True,
                          text=True, stdout=subprocess.PIPE).stdout.strip()
    require(head == config["source"]["commit"], "upstream commit mismatch")
    status = subprocess.run(["git", "-C", str(source_root), "status", "--porcelain"], check=True,
                            text=True, stdout=subprocess.PIPE).stdout
    require(not status.strip(), "upstream checkout is modified")
    observed = {}
    for relative, expected in config["source"]["required_files"].items():
        digest = sha256(source_root / relative)
        require(digest == expected, "upstream source checksum mismatch")
        observed[relative] = digest
    return head, observed


def install_runtime_guards(torch, transformers):
    def blocked(*_args, **_kwargs):
        raise PreflightError("external initialization or network access blocked")

    torch.hub.load = blocked
    for owner, name in (
        (transformers.PreTrainedModel, "from_pretrained"),
        (transformers.AutoModel, "from_pretrained"),
        (transformers.AutoConfig, "from_pretrained"),
        (transformers.AutoTokenizer, "from_pretrained"),
    ):
        setattr(owner, name, classmethod(blocked))
    socket.create_connection = blocked
    socket.socket.connect = blocked
    return blocked


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def construct(config, source_root):
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TORCH_HOME": os.devnull, "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
    })
    import torch
    import torchvision
    import transformers
    from omegaconf import OmegaConf
    from transformers import BertConfig, BertModel

    install_runtime_guards(torch, transformers)
    source_root = Path(source_root)
    dinov2_root = source_root / "apps" / "baselines" / "dinov2" / "third_party"
    import sys
    sys.path.insert(0, str(dinov2_root))
    from dinov2.models import build_model_from_cfg

    default = OmegaConf.load(dinov2_root / "dinov2" / "configs" / "ssl_default_config.yaml")
    babyview = OmegaConf.load(dinov2_root / "dinov2" / "configs" / "train" / "vitb14_babyview.yaml")
    vision_cfg = OmegaConf.merge(default, babyview)
    require(not vision_cfg.student.pretrained_weights, "DINO pretrained weight configured")
    vision, vision_dim = build_model_from_cfg(vision_cfg, only_teacher=True)

    text_spec = config["architectures"]["text"]
    bert_config = BertConfig(
        vocab_size=text_spec["vocab_size"], hidden_size=text_spec["hidden_size"],
        num_hidden_layers=text_spec["layers"], num_attention_heads=text_spec["attention_heads"],
        intermediate_size=text_spec["intermediate_size"],
        max_position_embeddings=text_spec["max_position_embeddings"],
        type_vocab_size=text_spec["type_vocab_size"],
    )
    text = BertModel(bert_config, add_pooling_layer=False)

    require(vision_dim == 768 and vision.embed_dim == 768, "DINO hidden size mismatch")
    require(vision.n_blocks == 12 and vision.patch_size == 14, "DINO architecture mismatch")
    require(text.config.hidden_size == 768 and len(text.encoder.layer) == 12,
            "BERT architecture mismatch")
    require(text.config.num_attention_heads == 12 and text.config.intermediate_size == 3072,
            "BERT attention/MLP mismatch")
    require(text.pooler is None, "BERT pooling layer unexpectedly present")

    device = torch.device("cuda:0")
    require(torch.cuda.is_available(), "GPU allocation is unavailable")
    vision.to(device)
    text.to(device)
    torch.cuda.synchronize()
    gpu_name = torch.cuda.get_device_name(0)
    results = {
        "vision": {"constructed": True, "arch": "vit_base_patch14", "hidden_size": 768,
                   "layers": 12, "attention_heads": 12, "parameter_count": parameter_count(vision)},
        "text": {"constructed": True, "arch": "bert_base", "vocab_size": text_spec["vocab_size"],
                 "hidden_size": 768, "layers": 12, "attention_heads": 12,
                 "parameter_count": parameter_count(text)},
        "checkpoint_loaded": False, "training_step_executed": False,
        "checkpoint_created": False, "network_fallback_used": False,
    }
    del vision, text
    torch.cuda.empty_cache()
    return {
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "transformers": transformers.__version__,
                     "cuda_runtime": torch.version.cuda},
        "hardware": {"gpu_count_used": 1, "gpu_model": gpu_name},
        "construction": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_contract(config, args.source_root, args.scratch_root, args.durable_root, args.output)
    head, hashes = verify_source(config, args.source_root)
    runtime = construct(config, args.source_root)
    detail = {
        "schema_version": 1, "record_type": "pilot_p2_governed_detailed_record",
        "pilot_id": config["pilot_id"], "engineering_run_id": config["engineering_run_id"],
        "stage": "P2", "classification": LABELS, "scientific_status_effect": "none",
        "source": {"commit": head, "verified_file_sha256": hashes},
        "environment_contract": config["environment"],
        "preprocessing_only_provenance": config["preprocessing_only"],
        **runtime,
        "boundary": {**config["learned_initialization"], "runtime_guards_installed": True,
                     "preprocessing_namespace_isolated": True},
        "actions": {"media_accessed": False, "preprocessing_run": False,
                    "evaluation_data_accessed": False, "training_or_optimization_run": False,
                    "p3_or_later_started": False},
        "gate": {"passed": True, "stop_reason": None},
    }
    atomic_json(args.output, detail)
    print(json.dumps({"status": "p2_complete", "detail_sha256": sha256(args.output),
                      "construction": detail["construction"], "versions": detail["versions"],
                      "hardware": detail["hardware"]}, sort_keys=True))


if __name__ == "__main__":
    main()
