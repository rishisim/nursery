#!/usr/bin/env python3
"""Pinned matched-arm learner for the coverage-based one-hour pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nursery_egobaby_preflight.bridge import strict_state_equality
from nursery_egobaby_preflight.cuda_preflight import _configure_ssl, _load_shared_prior


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_private(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(canonical(value) + b"\n")
    os.chmod(path, 0o600)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(canonical(list(tensor.shape)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_runtime(upstream: Path, data_root: Path, output_root: Path, public: Path, frozen: dict, seed: int):
    from apps.baselines.clip.data import MLMCollator, TextOnlyDataset, build_eval_transform, build_train_transform, contrastive_collate
    from apps.baselines.clip.data.captions import Ego4DCaptionsDataset
    from apps.baselines.clip.modeling import DINOv2SSL, MLMHead, MultiModalModel
    from apps.baselines.clip.modeling.text_encoder import TextEncoder
    from apps.baselines.clip.modeling.vision_encoder import CustomDINOv2VisionEncoder
    from apps.baselines.clip.training.interleave import InterleaveScheduler
    from apps.baselines.clip.training.loop import ContrastiveTrainer
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(False)
    device = torch.device("cuda:0")
    runtime_root = output_root / "learner_runtime"; runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    converted, prior_evidence = _load_shared_prior(json.loads((public / "source/egobaby_cuda_preflight.json").read_text()), runtime_root)
    prior_checkpoint = Path(prior_evidence["converted_checkpoint"])
    ssl_config_path = runtime_root / "ssl_config.yaml"
    ssl_cfg = _configure_ssl(upstream / "apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_coco.yaml", prior_checkpoint, ssl_config_path)
    ssl_cfg.train.OFFICIAL_EPOCH_LENGTH = 95
    ssl_cfg.optim.epochs = 1
    ssl_cfg.optim.warmup_epochs = 0
    ssl_cfg.optim.lr = 3.3145630368e-6
    ssl_cfg.optim.min_lr = 1e-6
    OmegaConf.save(ssl_cfg, ssl_config_path)
    dim = frozen["learner"]["embedding_dim"]
    vision = CustomDINOv2VisionEncoder(config_path=ssl_config_path, checkpoint_path=prior_checkpoint, embedding_dim=dim).to(device)
    strict_state_equality(converted, {k:v.detach().cpu() for k,v in vision.backbone.state_dict().items()})
    text_cfg = json.loads((public / "source/egobaby_cuda_preflight.json").read_text())["priors"]["text"]
    text = TextEncoder(text_cfg["repository"], hf_revision=text_cfg["revision"], embedding_dim=dim, dropout=0.1).to(device)
    model = MultiModalModel(vision, text, normalize_features=True, temperature=0.07).to(device)
    mlm_head = MLMHead(text).to(device)
    ssl = DINOv2SSL(ssl_config_path, device=device)
    strict_state_equality(converted, {k:v.detach().cpu() for k,v in ssl.teacher_backbone_state_dict().items()})
    train_ds = Ego4DCaptionsDataset(data_root / "restricted_train_1h_manifest.json", data_root / "frames/training", transform=build_train_transform(augment=False), multiple_frames=True)
    val_ds = Ego4DCaptionsDataset(data_root / "restricted_validation_manifest.json", data_root / "frames/validation", transform=build_eval_transform(), multiple_frames=False)
    batch_size = frozen["learner"]["batch_size"]
    expected_records = frozen["learner"]["reference_records_per_arm"]
    if len(train_ds) != expected_records:
        raise RuntimeError(f"E_RECORD_COUNT:{len(train_ds)}:{expected_records}")
    contrastive_generator = torch.Generator().manual_seed(seed + 1009)
    ssl_generator = torch.Generator().manual_seed(seed + 2003)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=contrastive_generator, num_workers=4, pin_memory=True, collate_fn=contrastive_collate, drop_last=True)
    ssl_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=ssl_generator, num_workers=4, pin_memory=True, collate_fn=contrastive_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=contrastive_collate)
    text_ds = TextOnlyDataset(data_root / "restricted_train_1h_text.txt", text.tokenizer, max_seq_len=512)
    mlm_loader = DataLoader(text_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 3001), num_workers=2, collate_fn=MLMCollator(text.tokenizer, mlm_probability=0.15), drop_last=True)
    model_optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    seen=set(); mlm_params=[]
    for p in list(text.model.embeddings.parameters())+list(text.model.encoder.parameters())+list(mlm_head.parameters()):
        if id(p) not in seen: seen.add(id(p)); mlm_params.append(p)
    mlm_optim = torch.optim.AdamW(mlm_params, lr=3e-4, weight_decay=0.01)
    config = OmegaConf.create({
        "seed":seed,"epochs":1,"log_interval":100,"optim":{"grad_clip":None},
        "checkpoint":{"save_dir":str(output_root/"checkpoints"),"keep_last":1,"save_every":1},
        "model":{"embedding_dim":dim,"normalize_features":True,"temperature":0.07,"fix_temperature":False,
          "text_encoder":{"_target_":"apps.baselines.clip.modeling.TextEncoder","hf_model_name":text_cfg["repository"],"hf_revision":text_cfg["revision"],"embedding_dim":dim,"dropout":0.1,"freeze":False,"pooling":"cls"},
          "vision_encoder":{"_target_":"apps.baselines.clip.modeling.CustomDINOv2VisionEncoder","config_path":str(ssl_config_path),"checkpoint_path":str(prior_checkpoint),"embedding_dim":dim,"freeze":False}},
    })
    trainer = ContrastiveTrainer(model=model, contrastive_optimizer=model_optim, scheduler=InterleaveScheduler(dict(frozen["learner"]["schedule"])), train_loader=train_loader, val_loader=val_loader, config=config, device=device, mlm_head=mlm_head, mlm_optimizer=mlm_optim, mlm_loader=mlm_loader, ssl=ssl, sync_vision_from_ssl=True)
    return trainer, model, train_loader, ssl_loader, config


def evaluate(checkpoint: Path, restricted: Path) -> dict:
    from apps.baselines.clip.trained_extractor import ContrastiveFeatureExtractor
    from evaluation.data.machine_devbench import MachineDevBenchLexicalDataset
    from PIL import Image
    extractor = ContrastiveFeatureExtractor(checkpoint, device="cuda")
    lexical = {}
    for style in ("realistic", "cartoon"):
        for pos in ("nouns", "adjectives"):
            ds=MachineDevBenchLexicalDataset(restricted/"lexical",style,pos); correct=0
            for index in range(len(ds)):
                item=ds[index]; features=extractor.extract_features({"image":item["image"],"text":item["text"]})
                sim=extractor.compute_similarity(features["image_features"],features["text_features"],normalize=True)
                correct += int(float(sim[0,0]) > float(sim[1,0]))
            lexical[f"{style}_{pos}"]=correct/len(ds)
    temporal=json.loads((restricted/"temporal/restricted_manifest.json").read_text()); reciprocal=[]; r1=[]
    for row in temporal["items"]:
        images=[Image.open(restricted/"temporal"/name).convert("RGB") for name in row["frames"]]
        features=extractor.extract_features({"image":images,"text":row["candidate_texts"]})
        image=features["image_features"].mean(0,keepdim=True); scores=extractor.compute_similarity(image,features["text_features"],normalize=True)[0]
        order=torch.argsort(scores,descending=True).tolist(); rank=order.index(row["positive_index"])+1
        r1.append(int(rank==1)); reciprocal.append(1.0/rank)
    noun=lexical["realistic_nouns"]; adjective=lexical["realistic_adjectives"]
    return {**lexical,"realistic_macro":(noun+adjective)/2,"temporal_recall_at_1":sum(r1)/len(r1),"temporal_mrr":sum(reciprocal)/len(reciprocal)}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--upstream",type=Path,required=True); parser.add_argument("--data-root",type=Path,required=True); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--public",type=Path,required=True); parser.add_argument("--restricted",type=Path,required=True); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--arm",choices=("real","synthetic"),required=True); parser.add_argument("--seed",type=int,required=True); args=parser.parse_args()
    frozen=json.loads(args.config.read_text())
    if args.seed not in frozen["learner"]["seeds"]:
        raise RuntimeError("E_UNREGISTERED_SEED")
    trainer,model,loader,ssl_loader,config=build_runtime(args.upstream,args.data_root,args.output_root,args.public,frozen,args.seed)
    checkpoints=args.output_root/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True,mode=0o700)
    from omegaconf import OmegaConf
    initial=checkpoints/"initialized.pt"; torch.save({"model_state_dict":model.state_dict(),"config":OmegaConf.to_container(config,resolve=True)},initial); os.chmod(initial,0o600)
    initialization_state_hash=state_hash(model.state_dict())
    initialized=evaluate(initial,args.restricted)
    iterator=iter(loader); ssl_iterator=iter(ssl_loader); mode_counts={"contrastive":0,"mlm":0,"dinov2":0}
    for _ in range(frozen["learner"]["objective_steps"]):
        mode,advanced=trainer.scheduler.step()
        if mode == "contrastive":
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader); batch=next(iterator)
        elif mode == "dinov2":
            try: batch=next(ssl_iterator)
            except StopIteration: ssl_iterator=iter(ssl_loader); batch=next(ssl_iterator)
        else:
            batch={}
        trainer._dispatch_step(mode,batch); mode_counts[mode]+=1
        if advanced and mode=="dinov2": trainer._on_dinov2_block_exit()
        trainer.global_step+=1
    if mode_counts != frozen["learner"]["objective_counts"]:
        raise RuntimeError(f"E_MODE_COUNTS:{mode_counts}")
    arm=f"{args.arm}_1h"; final=trainer._save(f"{arm}_seed_{args.seed}_step_{trainer.global_step}")
    health=trainer.validate(); trained=evaluate(final,args.restricted)
    result={"status":"PASS","arm":arm,"seed":args.seed,"credited_hours":1,"objective_steps":trainer.global_step,"objective_counts":mode_counts,"initialization_state_hash":initialization_state_hash,"initialized":initialized,"trained":trained,
      "delta_trained_minus_initialized":trained["realistic_macro"]-initialized["realistic_macro"],
      "temporal_noncatastrophic":trained["temporal_recall_at_1"]>=initialized["temporal_recall_at_1"]-0.05 and trained["temporal_mrr"]>=initialized["temporal_mrr"]-0.05,
      "validation_runtime_health_finite":health is not None and all(__import__('math').isfinite(v) for v in health.values())}
    result["commitment"]=hashlib.sha256(canonical(result)).hexdigest(); write_private(args.output_root/"compact_learner_result.json",result); print(json.dumps(result,sort_keys=True))


if __name__ == "__main__": main()
