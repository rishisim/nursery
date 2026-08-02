#!/usr/bin/env python3
"""Pinned initialized and Real-1h learner run for the lean pilot."""

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


def build_runtime(upstream: Path, root: Path, public: Path, frozen: dict):
    from apps.baselines.clip.data import MLMCollator, TextOnlyDataset, build_eval_transform, build_train_transform, contrastive_collate
    from apps.baselines.clip.data.captions import Ego4DCaptionsDataset
    from apps.baselines.clip.modeling import DINOv2SSL, MLMHead, MultiModalModel
    from apps.baselines.clip.modeling.text_encoder import TextEncoder
    from apps.baselines.clip.modeling.vision_encoder import CustomDINOv2VisionEncoder
    from apps.baselines.clip.training.interleave import InterleaveScheduler
    from apps.baselines.clip.training.loop import ContrastiveTrainer
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    seed = frozen["learner"]["seed"]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(False)
    device = torch.device("cuda:0")
    runtime_root = root / "learner_runtime"; runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
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
    train_ds = Ego4DCaptionsDataset(root / "restricted_train_1h_manifest.json", root / "frames/training", transform=build_train_transform(augment=False), multiple_frames=True)
    val_ds = Ego4DCaptionsDataset(root / "restricted_validation_manifest.json", root / "frames/validation", transform=build_eval_transform(), multiple_frames=False)
    batch_size = frozen["learner"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=contrastive_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=contrastive_collate)
    text_ds = TextOnlyDataset(root / "restricted_train_1h_text.txt", text.tokenizer, max_seq_len=512)
    mlm_loader = DataLoader(text_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=MLMCollator(text.tokenizer, mlm_probability=0.15), drop_last=True)
    model_optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    seen=set(); mlm_params=[]
    for p in list(text.model.embeddings.parameters())+list(text.model.encoder.parameters())+list(mlm_head.parameters()):
        if id(p) not in seen: seen.add(id(p)); mlm_params.append(p)
    mlm_optim = torch.optim.AdamW(mlm_params, lr=3e-4, weight_decay=0.01)
    config = OmegaConf.create({
        "seed":seed,"epochs":1,"log_interval":100,"optim":{"grad_clip":None},
        "checkpoint":{"save_dir":str(root/"checkpoints"),"keep_last":1,"save_every":1},
        "model":{"embedding_dim":dim,"normalize_features":True,"temperature":0.07,"fix_temperature":False,
          "text_encoder":{"_target_":"apps.baselines.clip.modeling.TextEncoder","hf_model_name":text_cfg["repository"],"hf_revision":text_cfg["revision"],"embedding_dim":dim,"dropout":0.1,"freeze":False,"pooling":"cls"},
          "vision_encoder":{"_target_":"apps.baselines.clip.modeling.CustomDINOv2VisionEncoder","config_path":str(ssl_config_path),"checkpoint_path":str(prior_checkpoint),"embedding_dim":dim,"freeze":False}},
    })
    trainer = ContrastiveTrainer(model=model, contrastive_optimizer=model_optim, scheduler=InterleaveScheduler(dict(frozen["learner"]["schedule"])), train_loader=train_loader, val_loader=val_loader, config=config, device=device, mlm_head=mlm_head, mlm_optimizer=mlm_optim, mlm_loader=mlm_loader, ssl=ssl, sync_vision_from_ssl=True)
    return trainer, model, train_loader, config


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
    parser=argparse.ArgumentParser(); parser.add_argument("--upstream",type=Path,required=True); parser.add_argument("--root",type=Path,required=True); parser.add_argument("--public",type=Path,required=True); parser.add_argument("--restricted",type=Path,required=True); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args()
    frozen=json.loads(args.config.read_text()); trainer,model,loader,config=build_runtime(args.upstream,args.root,args.public,frozen)
    checkpoints=args.root/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True,mode=0o700)
    from omegaconf import OmegaConf
    initial=checkpoints/"initialized.pt"; torch.save({"model_state_dict":model.state_dict(),"config":OmegaConf.to_container(config,resolve=True)},initial); os.chmod(initial,0o600)
    initialized=evaluate(initial,args.restricted)
    iterator=iter(loader)
    for _ in range(frozen["learner"]["objective_steps"]):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        mode,advanced=trainer.scheduler.step(); trainer._dispatch_step(mode,batch)
        if advanced and mode=="dinov2": trainer._on_dinov2_block_exit()
        trainer.global_step+=1
    final=trainer._save("real_1h_step_570")
    health=trainer.validate(); real=evaluate(final,args.restricted)
    result={"status":"PASS","objective_steps":trainer.global_step,"initialized":initialized,"real_1h":real,
      "delta_real_minus_initialized":real["realistic_macro"]-initialized["realistic_macro"],
      "temporal_noncatastrophic":real["temporal_recall_at_1"]>=initialized["temporal_recall_at_1"]-0.05 and real["temporal_mrr"]>=initialized["temporal_mrr"]-0.05,
      "validation_runtime_health_finite":health is not None and all(__import__('math').isfinite(v) for v in health.values())}
    result["real_1h_gate_pass"] = real["realistic_macro"]>=0.52 and result["delta_real_minus_initialized"]>=0.02 and result["temporal_noncatastrophic"]
    result["commitment"]=hashlib.sha256(canonical(result)).hexdigest(); write_private(args.root/"compact_learner_result.json",result); print(json.dumps(result,sort_keys=True))


if __name__ == "__main__": main()
