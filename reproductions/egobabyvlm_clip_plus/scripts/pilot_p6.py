#!/usr/bin/env python3
"""Governed Pilot P6 tokenizer gate and BERT rehearsal entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


RUN_ID = "p6-6d31a4e7"
P3_RUN_ID = "p3-3d96f71c"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
EXPECTED_SCRATCH_SUFFIX = Path("runs/pilot_p6") / RUN_ID
EXPECTED_DURABLE_SUFFIX = Path("run_records/pilot_p0/p0-4cc3af23/pilot_p6")


class GateError(RuntimeError):
    """A frozen P6 gate failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_owner_only(path: Path, directory: bool = True) -> None:
    if not path.is_absolute() or not path.exists():
        raise GateError("governed root is absent or not absolute")
    if directory and not path.is_dir():
        raise GateError("governed root is not a directory")
    mode = stat.S_IMODE(path.stat().st_mode)
    expected = 0o700 if directory else 0o600
    if mode != expected:
        raise GateError(f"governed path mode must be {expected:o}")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["engineering_run_id"] != RUN_ID or value["stage"] != "P6":
        raise GateError("wrong frozen P6 config")
    if value["tokenizer"]["vocab_size"] != 380 or value["model"]["vocab_size"] != 380:
        raise GateError("active vocabulary must be the authorized natural size 380")
    active = json.dumps({key: value[key] for key in ("source", "environment", "tokenizer", "model", "training", "storage", "forbidden")}).lower()
    for needle in ("model_name_or_path", "from_pretrained", "torch.hub", "pilot_p4_calibration_only", "pilot_p5/"):
        if needle in active:
            raise GateError("forbidden initialization/runtime reference")
    return value


def governed_paths(cfg: dict[str, Any], scratch_root: Path, durable_root: Path) -> tuple[Path, Path]:
    run_root = scratch_root.resolve() / "runs" / cfg["storage"]["scratch_namespace"]
    record_root = durable_root.resolve() / "run_records" / cfg["storage"]["durable_namespace"]
    if run_root.relative_to(scratch_root.resolve()) != EXPECTED_SCRATCH_SUFFIX:
        raise GateError("wrong scratch namespace")
    if record_root.relative_to(durable_root.resolve()) != EXPECTED_DURABLE_SUFFIX:
        raise GateError("wrong durable namespace")
    require_owner_only(scratch_root.resolve()); require_owner_only(durable_root.resolve())
    for parent in (scratch_root.resolve() / "runs", durable_root.resolve() / "run_records"):
        if parent.exists(): require_owner_only(parent)
    return run_root, record_root


def reject_completed(record_root: Path) -> None:
    if (record_root / "completion.json").exists():
        raise GateError("P6 already completed; mutation refused")


def validate_checkpoint_state(state: dict[str, Any], cfg: dict[str, Any], expected_step: int, tokenizer_hash: str, config_hash: str) -> None:
    required = {"model", "optimizer", "scheduler", "optimizer_step", "iteration", "tokenizer_sha256", "config_sha256", "torch_cpu_rng", "torch_cuda_rng", "sampler_state", "slurm_job_id"}
    if not required <= state.keys(): raise GateError("checkpoint required state missing")
    if state["optimizer_step"] != expected_step or state["iteration"] != expected_step: raise GateError("checkpoint step/iteration mismatch")
    if state["config_sha256"] != config_hash or state["tokenizer_sha256"] != tokenizer_hash: raise GateError("checkpoint config/tokenizer digest mismatch")
    if not isinstance(state["model"], dict) or not state["model"] or not isinstance(state["optimizer"], dict) or not isinstance(state["scheduler"], dict): raise GateError("checkpoint payload incompatible")
    if state["scheduler"].get("last_epoch") != expected_step: raise GateError("checkpoint scheduler step mismatch")
    if state["torch_cpu_rng"] is None or state["torch_cuda_rng"] is None or state["sampler_state"] is None: raise GateError("checkpoint RNG/sampler state missing")
    if state.get("python_rng") is not None: raise GateError("Python RNG is unused and must not be claimed")
    if not str(state["slurm_job_id"]).isdigit(): raise GateError("checkpoint Slurm job identity missing")


def verify_source_environment(cfg: dict[str, Any], source_root: Path, python_bin: Path, p2_config: Path | None = None, durable_root: Path | None = None) -> None:
    source_root=source_root.resolve(); python_bin=python_bin.resolve()
    if python_bin != (source_root / ".pixi/envs/default/bin/python").resolve(): raise GateError("not the verified P2 Python")
    head=subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    if head != cfg["source"]["commit"]: raise GateError("pinned source commit mismatch")
    if sha256(source_root/"pixi.lock") != cfg["environment"]["pixi_lock_sha256"]: raise GateError("P2 pixi lock mismatch")
    for relative, expected in cfg["source"]["required_files"].items():
        if sha256(source_root/relative) != expected: raise GateError("pinned source file hash mismatch")
    version=subprocess.run([str(python_bin),"-c","import transformers; print(transformers.__version__)"],check=True,text=True,capture_output=True).stdout.strip()
    if version != cfg["environment"]["transformers"]: raise GateError("P2 transformers version mismatch")
    if p2_config is not None and sha256(p2_config.resolve()) != cfg["environment"]["p2_config_sha256"]: raise GateError("P2 environment config mismatch")
    if durable_root is not None:
        p2_detail=durable_root.resolve()/"run_records/pilot_p2/p2-7e49c8a1.json"
        require_owner_only(p2_detail,directory=False)
        if sha256(p2_detail) != cfg["environment"]["p2_detail_sha256"]: raise GateError("P2 detail provenance mismatch")


def find_manifest(p3_root: Path, slot: str) -> Path:
    matches = list((p3_root / slot).glob("*/manifests/text.json"))
    if len(matches) != 1:
        raise GateError("expected exactly one governed retained-text manifest per slot")
    return matches[0]


def load_text(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise GateError("retained-text manifest is not a list")
    texts: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise GateError("retained-text row is malformed")
        text = row["text"].strip()
        if text:
            texts.append(text)
    return texts


def write_corpus(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text("".join(f"{text}\n" for text in texts), encoding="utf-8")
    os.chmod(path, 0o600)


def train_tokenizer(texts: list[str], output: Path, cfg: dict[str, Any]) -> tuple[int, str]:
    from tokenizers import BertWordPieceTokenizer

    tok_cfg = cfg["tokenizer"]
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=False,
    )
    tokenizer.train_from_iterator(
        texts,
        vocab_size=tok_cfg["vocab_size"],
        min_frequency=tok_cfg["min_frequency"],
        limit_alphabet=tok_cfg["limit_alphabet"],
        special_tokens=list(SPECIAL_TOKENS),
        wordpieces_prefix=tok_cfg["continuing_subword_prefix"],
        show_progress=False,
    )
    vocab = tokenizer.get_vocab()
    if len(vocab) != len(set(vocab)):
        raise GateError("tokenizer vocabulary is not unique")
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if vocab.get(token) != expected_id:
            raise GateError("special-token ID contract failed")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    tokenizer.save_model(str(output))
    tokenizer.save(str(output / "tokenizer.json"))
    for path in output.iterdir():
        os.chmod(path, 0o600)
    return len(vocab), sha256(output / "tokenizer.json")


def runtime_imports():
    import torch
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
    return torch, BertConfig, BertForMaskedLM, BertTokenizerFast


def model_config(cfg: dict[str, Any], BertConfig: Any) -> Any:
    m = cfg["model"]
    return BertConfig(**{key: m[key] for key in (
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "intermediate_size", "hidden_act", "hidden_dropout_prob",
        "attention_probs_dropout_prob", "max_position_embeddings", "type_vocab_size",
        "initializer_range", "layer_norm_eps", "position_embedding_type", "pad_token_id",
    )})


def make_tokenizer(run_root: Path, BertTokenizerFast: Any) -> Any:
    return BertTokenizerFast(
        vocab_file=str(run_root / "tokenizer" / "vocab.txt"), do_lower_case=False,
        pad_token=SPECIAL_TOKENS[0], unk_token=SPECIAL_TOKENS[1],
        cls_token=SPECIAL_TOKENS[2], sep_token=SPECIAL_TOKENS[3], mask_token=SPECIAL_TOKENS[4],
    )


def masked_batch(torch: Any, tokenizer: Any, texts: list[str], index: int, cfg: dict[str, Any], device: Any, generator: Any):
    encoded = tokenizer(texts[index], max_length=cfg["training"]["sequence_length"], truncation=True, return_tensors="pt")
    ids = encoded["input_ids"].clone(); labels = ids.clone()
    special = torch.tensor([tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True) for row in labels], dtype=torch.bool)
    probability = torch.full(labels.shape, cfg["training"]["mlm_probability"])
    masked = torch.bernoulli(probability.masked_fill(special, 0.0), generator=generator).bool()
    if not masked.any():
        candidates=(~special).nonzero(as_tuple=False)
        if not len(candidates): raise GateError("sequence has no maskable tokens")
        masked[tuple(candidates[0].tolist())]=True
    labels[~masked]=-100
    replace = torch.bernoulli(torch.full(labels.shape, 0.8), generator=generator).bool() & masked
    ids[replace]=tokenizer.mask_token_id
    random_replace = torch.bernoulli(torch.full(labels.shape, 0.5), generator=generator).bool() & masked & ~replace
    random_words=torch.randint(len(tokenizer), labels.shape, generator=generator)
    ids[random_replace]=random_words[random_replace]
    return {"input_ids":ids.to(device), "attention_mask":encoded["attention_mask"].to(device), "labels":labels.to(device)}


def checkpoint_path(run_root: Path, step: int) -> Path:
    return run_root / "checkpoints" / f"step_{step}.pt"


def save_checkpoint(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary=path.with_suffix(".tmp")
    torch.save(payload, temporary); os.chmod(temporary,0o600); os.replace(temporary,path)


def build_durable_inventory(durable_root: Path, record: Path, promoted: Path, jobs: dict[str, str]) -> dict[str, Any]:
    log_root=durable_root/"logs/pilot_p6"/RUN_ID
    invalid=durable_root/"run_records/pilot_p0/p0-4cc3af23/pilot_p6_invalid_config_attempt"
    roots={"canonical_checkpoint":promoted,"canonical_records":record,"important_logs":log_root}
    if invalid.exists(): roots["invalid_attempt"]=invalid
    entries=[]
    for namespace,root in roots.items():
        require_owner_only(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name=="durable_inventory.json": continue
            require_owner_only(path,directory=False)
            entries.append({"namespace":namespace,"opaque_name":path.name,"sha256":sha256(path),"bytes":path.stat().st_size})
    for role,job in jobs.items():
        expected=log_root/f"valid-{role}-{job}.log"
        if not expected.is_file(): raise GateError("important Slurm log missing")
    return {"schema_version":1,"record_type":"pilot_p6_durable_inventory","owner_only_verified":True,"entries":entries}


def train_phase(args: argparse.Namespace) -> int:
    torch, BertConfig, BertForMaskedLM, BertTokenizerFast = runtime_imports()
    cfg=load_config(args.config.resolve()); run_root,record_root=governed_paths(cfg,args.scratch_root,args.durable_root); reject_completed(record_root)
    gate=json.loads((run_root/"tokenizer_gate.json").read_text()); target=cfg["tokenizer"]["vocab_size"]
    if gate["tokenizer"]["observed_unique_vocab_size"] != target: raise GateError("tokenizer gate does not match authorized vocabulary")
    tokenizer=make_tokenizer(run_root,BertTokenizerFast)
    if len(tokenizer)!=target: raise GateError("tokenizer/model vocabulary mismatch")
    train_text=(run_root/"corpus/train.txt").read_text().splitlines()
    device=torch.device("cuda"); generator=torch.Generator(device="cpu")
    resume=args.resume_step is not None
    if resume:
        state=torch.load(checkpoint_path(run_root,args.resume_step),map_location="cpu",weights_only=False)
        validate_checkpoint_state(state,cfg,args.resume_step,sha256(run_root/"tokenizer/tokenizer.json"),sha256(args.config))
        model=BertForMaskedLM(model_config(cfg,BertConfig)); model.load_state_dict(state["model"])
    else:
        torch.manual_seed(cfg["training"]["seed"]); torch.cuda.manual_seed_all(cfg["training"]["seed"])
        model=BertForMaskedLM(model_config(cfg,BertConfig)); state=None
    model.to(device); model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s: min((s+1)/cfg["training"]["warmup_steps"],max(0.0,(cfg["training"]["final_optimizer_step"]-s)/(cfg["training"]["final_optimizer_step"]-cfg["training"]["warmup_steps"]))))
    if state:
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_cpu_rng"]); torch.cuda.set_rng_state_all(state["torch_cuda_rng"]); generator.set_state(state["sampler_state"]); start=state["optimizer_step"]
    else:
        generator.manual_seed(cfg["training"]["seed"]); start=0
    initial_encoder=next(model.bert.encoder.parameters()).detach().float().cpu().clone()
    initial_head=model.cls.predictions.bias.detach().float().cpu().clone()
    losses=[]; started=time.time(); peak=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(start+1,args.target_step+1):
        batch=masked_batch(torch,tokenizer,train_text,(step-1)%len(train_text),cfg,device,generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): output=model(**batch); loss=output.loss
        if not torch.isfinite(loss): raise GateError("nonfinite MLM loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise GateError("nonfinite gradient")
        optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
    peak=torch.cuda.max_memory_allocated(); elapsed=time.time()-started
    encoder_updated=not torch.equal(initial_encoder,next(model.bert.encoder.parameters()).detach().float().cpu())
    head_updated=not torch.equal(initial_head,model.cls.predictions.bias.detach().float().cpu())
    job_id=os.environ.get("SLURM_JOB_ID","")
    if not job_id.isdigit(): raise GateError("training requires Slurm job identity")
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":None,"optimizer_step":args.target_step,"iteration":args.target_step,"tokenizer_sha256":sha256(run_root/"tokenizer/tokenizer.json"),"config_sha256":sha256(args.config),"python_rng":None,"torch_cpu_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all(),"sampler_state":generator.get_state(),"process_id":os.getpid(),"slurm_job_id":job_id}
    save_checkpoint(torch,checkpoint_path(run_root,args.target_step),payload)
    atomic_json(run_root/f"segment_{args.target_step}.json",{"start_step":start,"end_step":args.target_step,"process_id":os.getpid(),"slurm_job_id":job_id,"losses":losses,"all_losses_finite":True,"encoder_updated":encoder_updated,"mlm_head_updated":head_updated,"scheduler_last_epoch":scheduler.last_epoch,"peak_allocated_gpu_memory_bytes":peak,"wall_seconds":elapsed})
    print(json.dumps({"status":"segment_complete","start":start,"end":args.target_step,"pid":os.getpid()})); return 0


def finalize(args: argparse.Namespace) -> int:
    torch,BertConfig,BertForMaskedLM,BertTokenizerFast=runtime_imports(); cfg=load_config(args.config.resolve())
    run_root,durable=governed_paths(cfg,args.scratch_root,args.durable_root); reject_completed(durable)
    tokenizer=make_tokenizer(run_root,BertTokenizerFast); state=torch.load(checkpoint_path(run_root,100),map_location="cpu",weights_only=False)
    validate_checkpoint_state(state,cfg,100,sha256(run_root/"tokenizer/tokenizer.json"),sha256(args.config))
    model=BertForMaskedLM(model_config(cfg,BertConfig)); model.load_state_dict(state["model"]); model.cuda().eval()
    validation=(run_root/"corpus/validation.txt").read_text().splitlines(); generator=torch.Generator().manual_seed(cfg["training"]["seed"]+1)
    losses=[]
    with torch.no_grad():
        for i in range(len(validation)):
            batch=masked_batch(torch,tokenizer,validation,i,cfg,torch.device("cuda"),generator)
            losses.append(float(model(**batch).loss))
    if not losses or not all(__import__('math').isfinite(x) for x in losses): raise GateError("validation did not produce finite loss")
    smoke=masked_batch(torch,tokenizer,validation,0,cfg,torch.device("cuda"),generator)
    with torch.no_grad(): logits=model(input_ids=smoke["input_ids"],attention_mask=smoke["attention_mask"]).logits
    if logits.shape[-1]!=len(tokenizer): raise GateError("fresh-load smoke incompatibility")
    s50=json.loads((run_root/"segment_50.json").read_text()); s100=json.loads((run_root/"segment_100.json").read_text())
    finalize_job=os.environ.get("SLURM_JOB_ID","")
    jobs={str(s50.get("slurm_job_id","")),str(s100.get("slurm_job_id","")),finalize_job}
    if len(jobs)!=3 or not all(x.isdigit() for x in jobs) or s100["start_step"]!=50: raise GateError("three distinct Slurm jobs not proven")
    promoted=args.durable_root.resolve()/"checkpoints/pilot_p6"/RUN_ID; promoted.mkdir(parents=True,exist_ok=True,mode=0o700)
    for source in (checkpoint_path(run_root,100),run_root/"tokenizer/vocab.txt",run_root/"tokenizer/tokenizer.json",args.config.resolve()):
        target=promoted/source.name; shutil.copy2(source,target); os.chmod(target,0o600)
    inventory={p.name:sha256(p) for p in promoted.iterdir() if p.is_file()}
    detail={"schema_version":1,"status":"p6_complete","optimizer_step":100,"iteration":100,"slurm_job_ids":{"health":s50["slurm_job_id"],"resume":s100["slurm_job_id"],"finalize":finalize_job},"losses":s50["losses"]+s100["losses"],"encoder_updated":s50["encoder_updated"] and s100["encoder_updated"],"mlm_head_updated":s50["mlm_head_updated"] and s100["mlm_head_updated"],"validation_runs":1,"validation_used_for_selection":False,"validation_loss":sum(losses)/len(losses),"validation_non_generalizing":True,"fresh_load_smoke":True,"python_rng_restored":False,"torch_rng_and_sampler_restored":True,"resources":{"peak_allocated_gpu_memory_bytes":max(s50["peak_allocated_gpu_memory_bytes"],s100["peak_allocated_gpu_memory_bytes"]),"training_wall_seconds":s50["wall_seconds"]+s100["wall_seconds"],"checkpoint_bytes":checkpoint_path(run_root,100).stat().st_size,"driver_memory_bytes":None},"inventory":inventory}
    atomic_json(durable/"detail.json",detail); atomic_json(durable/"checksum_inventory.json",inventory); atomic_json(durable/"completion.json",{"status":"p6_complete","detail_sha256":sha256(durable/"detail.json"),"inventory_sha256":sha256(durable/"checksum_inventory.json")})
    jobs={"health":str(s50["slurm_job_id"]),"resume":str(s100["slurm_job_id"]),"finalize":finalize_job}
    atomic_json(durable/"audit_attestation.json",{"schema_version":1,"record_type":"pilot_p6_completion_attestation","slurm_job_ids":jobs,"three_distinct_jobs":len(set(jobs.values()))==3,"fresh_final_load":{"optimizer_step":100,"iteration":100,"config_and_tokenizer_hashes_verified":True,"exact_model_shape_and_vocab_verified":True,"smoke_forward_passed":True},"python_rng_used":False,"python_rng_restored":False,"torch_rng_and_sampler_restored":True,"validation_runs":1,"validation_used_for_selection":False,"driver_memory_bytes":None})
    atomic_json(durable/"durable_inventory.json",build_durable_inventory(args.durable_root.resolve(),durable,promoted,jobs))
    print(json.dumps({"status":"p6_complete","validation_loss":detail["validation_loss"]})); return 0


def prepare(args: argparse.Namespace) -> int:
    scratch_root = args.scratch_root.resolve()
    durable_root = args.durable_root.resolve()
    require_owner_only(scratch_root)
    require_owner_only(durable_root)
    config = load_config(args.config.resolve())
    config_hash = sha256(args.config.resolve())
    run_root,durable=governed_paths(config,scratch_root,durable_root)
    reject_completed(durable)
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_root, 0o700)
    durable.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(durable, 0o700)
    frozen_copy = run_root / "resolved_config.json"
    if frozen_copy.exists() and sha256(frozen_copy) != config_hash:
        raise GateError("existing frozen config hash mismatch")
    if not frozen_copy.exists():
        shutil.copy2(args.config, frozen_copy)
        os.chmod(frozen_copy, 0o600)

    p3_root = scratch_root / "runs" / "pilot_p3" / P3_RUN_ID
    split = config["input"]["recording_level_split"]
    train_manifest = find_manifest(p3_root, split["train_slot"])
    validation_manifest = find_manifest(p3_root, split["validation_slot"])
    if train_manifest.parent.parent.parent == validation_manifest.parent.parent.parent:
        raise GateError("recording-level split is not disjoint")
    if sha256(train_manifest) != split["train"]["manifest_sha256"]:
        raise GateError("training retained-text checksum mismatch")
    if sha256(validation_manifest) != split["validation"]["manifest_sha256"]:
        raise GateError("validation retained-text checksum mismatch")
    train_text = load_text(train_manifest)
    validation_text = load_text(validation_manifest)
    counts = {
        "train_utterances": len(train_text),
        "train_words": sum(len(text.split()) for text in train_text),
        "validation_utterances": len(validation_text),
        "validation_words": sum(len(text.split()) for text in validation_text),
    }
    expected_counts = {
        "train_utterances": split["train"]["utterances"],
        "train_words": split["train"]["words"],
        "validation_utterances": split["validation"]["utterances"],
        "validation_words": split["validation"]["words"],
    }
    if counts != expected_counts:
        raise GateError("privacy-safe split counts mismatch")
    corpus_dir = run_root / "corpus"
    write_corpus(corpus_dir / "train.txt", train_text)
    write_corpus(corpus_dir / "validation.txt", validation_text)
    tokenizer_dir = run_root / "tokenizer"
    if tokenizer_dir.exists():
        raise GateError("tokenizer gate is single-execution and output already exists")
    vocab_size, tokenizer_hash = train_tokenizer(train_text, tokenizer_dir, config)
    result = {
        "schema_version": 1,
        "record_type": "pilot_p6_tokenizer_gate_detail",
        "engineering_run_id": RUN_ID,
        "classification": config["classification"],
        "status": "tokenizer_gate_passed" if vocab_size == config["tokenizer"]["vocab_size"] else "stopped_natural_vocabulary_below_target",
        "split": {"recording_level_disjoint": True, "fallback_used": False, **counts},
        "tokenizer": {
            "algorithm": "WordPiece",
            "training_split_only": True,
            "external_tokens": False,
            "fabricated_tokens": False,
            "observed_unique_vocab_size": vocab_size,
            "target_vocab_size": config["tokenizer"]["vocab_size"],
            "required_special_token_count": len(SPECIAL_TOKENS),
            "special_token_ids_match": True,
            "tokenizer_json_sha256": tokenizer_hash,
        },
        "config_sha256": config_hash,
        "bert_training_started": False,
        "p5_or_dino_loaded": False,
        "p4_loaded": False,
        "p7_started": False,
        "scientific_status_effect": "none",
        "inventory_status": "incomplete_inventory",
    }
    atomic_json(run_root / "tokenizer_gate.json", result)
    atomic_json(durable / "tokenizer_gate.json", result)
    if vocab_size != config["tokenizer"]["vocab_size"]:
        print(json.dumps({"status": result["status"], "observed_unique_vocab_size": vocab_size, "target_vocab_size": config["tokenizer"]["vocab_size"]}))
        return 3
    print(json.dumps({"status": result["status"], "vocab_size": vocab_size}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight","prepare-tokenizer","train","finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--python-bin", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "preflight":
            if args.source_root is None or args.python_bin is None: raise GateError("source root and Python are required")
            verify_source_environment(load_config(args.config.resolve()),args.source_root,args.python_bin,args.config.resolve().parent/"pilot_p2.json",args.durable_root); print("P6 source/environment preflight passed"); return 0
        if args.command == "prepare-tokenizer":
            return prepare(args)
        if args.command == "train":
            if args.target_step not in (50,100): raise GateError("target step must be 50 or 100")
            return train_phase(args)
        if args.command == "finalize": return finalize(args)
        raise GateError("unsupported command")
    except (GateError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"P6 gate error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
