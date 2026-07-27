"""Real-CUDA, public/dummy preflight against the pinned EgoBabyVLM checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .bridge import convert_hf_dinov2_base, strict_state_equality
from .contract import canonical_json_sha256, file_sha256, lexical_macro_wiring, schedule_cycle


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _require_upstream(upstream_root: Path, expected_commit: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"upstream commit mismatch: expected {expected_commit}, got {actual}")


def _finite_metrics(metrics: dict[str, float], required: tuple[str, ...]) -> dict[str, float]:
    missing = [key for key in required if key not in metrics]
    if missing:
        raise AssertionError(f"required loss metrics missing: {missing}; available={sorted(metrics)}")
    for key, value in metrics.items():
        if not math.isfinite(float(value)):
            raise AssertionError(f"non-finite metric {key}={value}")
    return {key: float(metrics[key]) for key in required}


def _finite_gradients(module: Any) -> dict[str, int]:
    import torch

    checked = 0
    nonzero = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        checked += 1
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError("non-finite gradient")
        if torch.count_nonzero(parameter.grad).item():
            nonzero += 1
    if checked == 0 or nonzero == 0:
        raise AssertionError("objective produced no finite nonzero gradients")
    return {"gradient_tensors": checked, "nonzero_gradient_tensors": nonzero}


def _parameter_digest(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _assert_update(before: str, module: Any, objective: str) -> str:
    after = _parameter_digest(module)
    if before == after:
        raise AssertionError(f"{objective} optimizer did not update its objective module")
    return after


def _configure_ssl(base_config: Path, prior_checkpoint: Path, output_config: Path) -> Any:
    from apps.baselines.clip.modeling.dinov2_ssl import load_dinov2_config
    from omegaconf import OmegaConf

    cfg = load_dinov2_config(base_config)
    cfg = OmegaConf.merge(
        cfg,
        {
            "student": {
                "arch": "vit_base",
                "patch_size": 14,
                "ffn_layer": "mlp",
                "block_chunks": 0,
                "pretrained_weights": str(prior_checkpoint),
                "num_register_tokens": 0,
            },
            "crops": {
                "global_crops_size": 224,
                "local_crops_size": 98,
                "local_crops_number": 1,
            },
            "dino": {"head_n_prototypes": 128, "head_hidden_dim": 256, "head_bottleneck_dim": 64},
            "ibot": {
                "head_n_prototypes": 128,
                "head_hidden_dim": 256,
                "head_bottleneck_dim": 64,
                "separate_head": True,
            },
            "train": {"OFFICIAL_EPOCH_LENGTH": 8},
            "optim": {"epochs": 2, "warmup_epochs": 0, "freeze_last_layer_epochs": 0},
        },
    )
    OmegaConf.save(cfg, output_config)
    return cfg


def _load_shared_prior(config: dict[str, Any], work_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    vision = config["priors"]["vision"]
    weight_path = Path(
        hf_hub_download(
            repo_id=vision["repository"],
            revision=vision["revision"],
            filename=vision["filename"],
        )
    )
    actual_hash = file_sha256(str(weight_path))
    if actual_hash != vision["sha256"]:
        raise AssertionError(f"DINO prior sha256 mismatch: expected {vision['sha256']}, got {actual_hash}")
    source_state = load_file(str(weight_path), device="cpu")
    text = config["priors"]["text"]
    text_weight_path = Path(
        hf_hub_download(
            repo_id=text["repository"],
            revision=text["revision"],
            filename=text["filename"],
        )
    )
    text_hash = file_sha256(str(text_weight_path))
    if text_hash != text["sha256"]:
        raise AssertionError(f"BERT prior sha256 mismatch: expected {text['sha256']}, got {text_hash}")
    converted, record = convert_hf_dinov2_base(source_state, target_position_tokens=257)
    checkpoint_path = work_dir / "shared_dinov2_vitb14_224.pth"
    import torch

    torch.save({"model": converted, "teacher": converted}, checkpoint_path)
    return converted, {
        "source_file_sha256": actual_hash,
        "text_source_file_sha256": text_hash,
        "converted_checkpoint": str(checkpoint_path),
        "conversion": _jsonable(record),
    }


def _dummy_lineage(seed: int) -> tuple[list[dict[str, Any]], str]:
    rows = [
        {"id": index, "caption": caption, "pixel_seed": seed + index}
        for index, caption in enumerate(
            (
                "a red cup",
                "a blue block",
                "a small car",
                "a wooden spoon",
                "the cup is red",
                "the block is blue",
                "the car is small",
                "the spoon is wooden",
            )
        )
    ]
    return rows, canonical_json_sha256(rows)


def _build_runtime(
    upstream_root: Path,
    frozen: dict[str, Any],
    work_dir: Path,
    converted: dict[str, Any],
    lineage_hash: str,
) -> dict[str, Any]:
    import torch
    from apps.baselines.clip.modeling import DINOv2SSL, MLMHead, MultiModalModel
    from apps.baselines.clip.modeling.text_encoder import TextEncoder
    from apps.baselines.clip.modeling.vision_encoder import CustomDINOv2VisionEncoder
    from apps.baselines.clip.training.interleave import InterleaveScheduler
    from apps.baselines.clip.training.loop import ContrastiveTrainer
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Dataset

    learner = frozen["learner"]
    seed = learner["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")

    prior_checkpoint = work_dir / "shared_dinov2_vitb14_224.pth"
    ssl_config_path = work_dir / "ssl_config.yaml"
    base_ssl_config = (
        upstream_root
        / "apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_coco.yaml"
    )
    ssl_config = _configure_ssl(base_ssl_config, prior_checkpoint, ssl_config_path)

    vision = CustomDINOv2VisionEncoder(
        config_path=ssl_config_path,
        checkpoint_path=prior_checkpoint,
        embedding_dim=64,
    ).to(device)
    vision_init = {key: value.detach().cpu() for key, value in vision.backbone.state_dict().items()}
    strict_state_equality(converted, vision_init)

    text_prior = frozen["priors"]["text"]
    text = TextEncoder(
        text_prior["repository"],
        hf_revision=text_prior["revision"],
        embedding_dim=64,
        dropout=0.0,
    ).to(device)
    model = MultiModalModel(vision, text, normalize_features=True).to(device)
    mlm_head = MLMHead(text).to(device)
    ssl = DINOv2SSL(ssl_config_path, device=device).to(device) if hasattr(DINOv2SSL, "to") else DINOv2SSL(
        ssl_config_path,
        device=device,
    )

    teacher_init = {key: value.detach().cpu() for key, value in ssl.teacher_backbone_state_dict().items()}
    pre_step_teacher_equality = strict_state_equality(converted, teacher_init)
    pre_step_clip_equality = strict_state_equality(vision_init, teacher_init)

    rows, _ = _dummy_lineage(seed)

    class DummyContrastiveDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            generator = torch.Generator().manual_seed(rows[index]["pixel_seed"])
            return {
                "images": torch.randn(3, 224, 224, generator=generator),
                "captions": rows[index]["caption"],
            }

    def collate_contrastive(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "images": torch.stack([item["images"] for item in batch]),
            "captions": [item["captions"] for item in batch],
        }

    tokens = text.tokenizer(
        [row["caption"] for row in rows],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.full_like(tokens["input_ids"], -100)
    for row_index in range(labels.shape[0]):
        labels[row_index, 1] = tokens["input_ids"][row_index, 1]
        tokens["input_ids"][row_index, 1] = text.tokenizer.mask_token_id

    class DummyMLMDataset(Dataset):
        def __len__(self) -> int:
            return labels.shape[0]

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "input_ids": tokens["input_ids"][index],
                "attention_mask": tokens["attention_mask"][index],
                "labels": labels[index],
            }

    train_loader = DataLoader(
        DummyContrastiveDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_contrastive,
    )
    mlm_loader = DataLoader(DummyMLMDataset(), batch_size=2, shuffle=False)
    schedule = InterleaveScheduler(dict(learner["schedule"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    seen: set[int] = set()
    mlm_parameters = []
    for parameter in list(text.parameters()) + list(mlm_head.parameters()):
        if id(parameter) not in seen:
            seen.add(id(parameter))
            mlm_parameters.append(parameter)
    mlm_optimizer = torch.optim.AdamW(mlm_parameters, lr=1e-6)
    resolved_config = {
        "seed": seed,
        "log_interval": 1000,
        "optim": {"grad_clip": 1.0},
        "checkpoint": {"save_dir": str(work_dir), "keep_last": 1, "save_every": 1},
        "model": {
            "embedding_dim": 64,
            "normalize_features": True,
            "temperature": 0.07,
            "fix_temperature": False,
            "text_encoder": {
                "_target_": "apps.baselines.clip.modeling.TextEncoder",
                "hf_model_name": text_prior["repository"],
                "hf_revision": text_prior["revision"],
                "embedding_dim": 64,
                "dropout": 0.0,
                "freeze": False,
                "pooling": "cls",
            },
            "vision_encoder": {
                "_target_": "apps.baselines.clip.modeling.CustomDINOv2VisionEncoder",
                "config_path": str(ssl_config_path),
                "checkpoint_path": str(prior_checkpoint),
                "embedding_dim": 64,
                "freeze": False,
            },
        },
        "preflight": {
            "frozen_config_sha256": canonical_json_sha256(frozen),
            "data_lineage_sha256": lineage_hash,
        },
    }
    trainer = ContrastiveTrainer(
        model=model,
        contrastive_optimizer=optimizer,
        scheduler=schedule,
        train_loader=train_loader,
        config=OmegaConf.create(resolved_config),
        device=device,
        mlm_head=mlm_head,
        mlm_optimizer=mlm_optimizer,
        mlm_loader=mlm_loader,
        ssl=ssl,
        sync_vision_from_ssl=True,
    )
    return {
        "trainer": trainer,
        "model": model,
        "mlm_head": mlm_head,
        "ssl": ssl,
        "train_loader": train_loader,
        "pre_step_teacher_equality": pre_step_teacher_equality,
        "pre_step_clip_equality": pre_step_clip_equality,
        "resolved_config": resolved_config,
        "ssl_config": OmegaConf.to_container(ssl_config, resolve=True),
    }


def _run_cycle(runtime: dict[str, Any]) -> dict[str, Any]:
    import torch

    trainer = runtime["trainer"]
    batches = iter(runtime["train_loader"])
    observed: list[str] = []
    objective_evidence: dict[str, Any] = {}
    before = {
        "contrastive": _parameter_digest(runtime["model"]),
        "mlm": _parameter_digest(runtime["mlm_head"]),
        "dinov2": _parameter_digest(runtime["ssl"].model.student),
    }
    for _ in range(6):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(runtime["train_loader"])
            batch = next(batches)
        mode, advanced = trainer.scheduler.step()
        observed.append(mode)
        metrics = trainer._dispatch_step(mode, batch)
        if mode == "contrastive":
            losses = _finite_metrics(metrics, ("loss",))
            grads = _finite_gradients(runtime["model"])
        elif mode == "mlm":
            losses = _finite_metrics(metrics, ("mlm_loss",))
            grads = _finite_gradients(runtime["mlm_head"])
        else:
            losses = _finite_metrics(metrics, ("total_loss", "dino_global_crops_loss", "ibot_loss"))
            grads = _finite_gradients(runtime["ssl"].model.student)
        if mode not in objective_evidence:
            target = runtime["model"] if mode == "contrastive" else runtime["mlm_head"]
            if mode == "dinov2":
                target = runtime["ssl"].model.student
            objective_evidence[mode] = {
                "losses": losses,
                "gradients": grads,
                "updated_parameter_sha256": _assert_update(before[mode], target, mode),
            }
        if advanced and mode == "dinov2":
            teacher = runtime["ssl"].teacher_backbone_state_dict()
            clip_before = runtime["model"].image_embed.backbone.state_dict()
            changed_before_sync = any(
                not torch.equal(clip_before[key].detach().cpu(), teacher[key].detach().cpu())
                for key in teacher
            )
            if not changed_before_sync:
                raise AssertionError("SSL teacher did not diverge before expected post-block synchronization")
            trainer._on_dinov2_block_exit()
            post_sync = strict_state_equality(
                {key: value.detach().cpu() for key, value in runtime["model"].image_embed.backbone.state_dict().items()},
                {key: value.detach().cpu() for key, value in teacher.items()},
            )
            objective_evidence["dinov2"]["post_ssl_sync"] = post_sync
        trainer.global_step += 1
    expected = ["contrastive"] * 4 + ["mlm", "dinov2"]
    if observed != expected:
        raise AssertionError(f"scheduler ordering mismatch: {observed} != {expected}")
    return {"ordering": observed, "objectives": objective_evidence}


def _next_step(runtime: dict[str, Any]) -> tuple[str, float]:
    batch = next(iter(runtime["train_loader"]))
    mode, _ = runtime["trainer"].scheduler.step()
    metrics = runtime["trainer"]._dispatch_step(mode, batch)
    if mode != "contrastive":
        raise AssertionError(f"expected contrastive after 4:1:1 cycle, got {mode}")
    return mode, float(metrics["loss"])


def _extractor_wiring(checkpoint: Path, work_dir: Path) -> dict[str, Any]:
    import torch
    from apps.baselines.clip.trained_extractor import ContrastiveFeatureExtractor

    extractor = ContrastiveFeatureExtractor(checkpoint, device="cuda")
    generator = torch.Generator().manual_seed(9001)
    images = torch.randn(2, 3, 224, 224, generator=generator, device="cuda")
    features = extractor.extract_features({"image": images, "text": ["a cup", "a red object"]})
    similarities = extractor.compute_similarity(
        features["image_features"],
        features["text_features"],
        normalize=True,
    )
    if similarities.shape != (2, 2) or not torch.isfinite(similarities).all():
        raise AssertionError("official feature extractor returned invalid similarity wiring")
    aggregate = lexical_macro_wiring({"noun": [1.0, 0.0], "adjective": [0.0, 1.0]})
    fixture_path = work_dir / "fabricated_lexical_fixture.json"
    fixture_path.write_text(json.dumps({"noun": [1, 0], "adjective": [0, 1]}, sort_keys=True))
    return {
        "official_interface": "apps.baselines.clip.trained_extractor.ContrastiveFeatureExtractor",
        "feature_shape": list(features["image_features"].shape),
        "similarity_shape": list(similarities.shape),
        "noun_wiring": "complete",
        "adjective_wiring": "complete",
        "lexical_macro_wiring": "complete",
        "fixture_sha256": file_sha256(str(fixture_path)),
        "scientific_result_retained": False,
        "aggregate_field_names": list(aggregate),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA is mandatory; CPU/MPS/mock execution is not a pass")
    upstream_root = Path(args.upstream_root).resolve()
    frozen_path = Path(args.config).resolve()
    frozen = json.loads(frozen_path.read_text())
    _require_upstream(upstream_root, frozen["upstream"]["commit"])

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="nursery-egobaby-cuda-") as temporary:
        work_dir = Path(temporary)
        converted, prior_evidence = _load_shared_prior(frozen, work_dir)
        rows, lineage_hash = _dummy_lineage(frozen["learner"]["seed"])
        runtime = _build_runtime(upstream_root, frozen, work_dir, converted, lineage_hash)
        cycle = _run_cycle(runtime)
        checkpoint = runtime["trainer"]._save("preflight")
        if checkpoint is None:
            raise AssertionError("rank zero did not save checkpoint")
        checkpoint_hash = file_sha256(str(checkpoint))
        reference_mode, reference_loss = _next_step(runtime)

        fresh = _build_runtime(upstream_root, frozen, work_dir, converted, lineage_hash)
        fresh["trainer"].resume(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_contract = payload["config"]["preflight"]
        expected_contract = runtime["resolved_config"]["preflight"]
        if saved_contract != expected_contract:
            raise AssertionError(f"checkpoint contract mismatch: {saved_contract} != {expected_contract}")
        if fresh["trainer"].ssl_iteration != 1:
            raise AssertionError(f"SSL scheduler position not restored: {fresh['trainer'].ssl_iteration}")
        resumed_mode, resumed_loss = _next_step(fresh)
        tolerance = frozen["learner"]["reproducibility"]["fresh_process_next_loss"]
        if resumed_mode != reference_mode or not math.isclose(
            resumed_loss,
            reference_loss,
            rel_tol=tolerance["rtol"],
            abs_tol=tolerance["atol"],
        ):
            raise AssertionError(
                f"fresh-process continuation mismatch: {resumed_mode}/{resumed_loss} "
                f"!= {reference_mode}/{reference_loss} under {tolerance}"
            )
        evaluator = _extractor_wiring(checkpoint, work_dir)
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        env_hash = hashlib.sha256(pip_freeze.encode()).hexdigest()
        pixi_lock = upstream_root / "pixi.lock"
        result = {
            "schema_version": 1,
            "status": "PASS",
            "proof": "public/dummy engineering compatibility only",
            "upstream": {
                **frozen["upstream"],
                "commit_verified": True,
                "patch_sha256": file_sha256(
                    str(Path(__file__).resolve().parents[2] / "patches/egobabyvlm_shared_prior.patch")
                ),
            },
            "pins": frozen["priors"],
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0),
                "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
                "single_gpu": torch.cuda.device_count() == 1,
                "distributed_smoke": "not_run_budget_optional",
                "pixi_lock_sha256": file_sha256(str(pixi_lock)),
                "pip_freeze_sha256": env_hash,
            },
            "network_destinations": [
                "github.com/facebookresearch/egobabyvlm",
                "huggingface.co/facebook/dinov2-base",
                "huggingface.co/google-bert/bert-base-uncased",
            ],
            "initialization": {
                **prior_evidence,
                "pre_step_teacher_equality": runtime["pre_step_teacher_equality"],
                "pre_step_clip_equality": runtime["pre_step_clip_equality"],
                "architecture": "vit_base14",
                "image_size": 224,
            },
            "data": {
                "kind": "self-authored/dummy",
                "rows": len(rows),
                "lineage_sha256": lineage_hash,
                "restricted_data_used": False,
            },
            "schedule": cycle,
            "checkpoint_resume": {
                "checkpoint_sha256": checkpoint_hash,
                "stored_outside_repository": True,
                "config_and_lineage_equal": True,
                "scheduler_state_restored": True,
                "ssl_iteration_restored": fresh["trainer"].ssl_iteration,
                "next_objective": resumed_mode,
                "next_loss_within_predeclared_tolerance": True,
                "criterion": tolerance,
            },
            "evaluator": evaluator,
            "runtime": {
                "wall_time_seconds": time.monotonic() - started,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "provider": args.provider,
                "maximum_wall_time": args.maximum_wall_time,
                "upper_bound_cost_usd": args.upper_bound_cost_usd,
                "actual_cost_usd": args.actual_cost_usd,
            },
            "proof_limits": frozen["proof_limits"],
        }
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--maximum-wall-time", required=True)
    parser.add_argument("--upper-bound-cost-usd", required=True, type=float)
    parser.add_argument("--actual-cost-usd", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
