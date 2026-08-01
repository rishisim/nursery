#!/usr/bin/env python3
"""Build and seal both governed Phase 4 common evaluation assets."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import torch
import torch.distributed as dist


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(canonical(value) + b"\n")
    os.chmod(path, 0o600)


def translate_segments(tokenizer, translator, segments, language, audio_duration, confidence_min):
    from synthetic_video_language_adapter import translate_accepted, validate_asr_prediction
    out = []
    for segment in segments:
        prediction = {"text": segment.get("text", ""), "language": language, "words": segment.get("words", [])}
        adjudication = validate_asr_prediction(prediction, audio_duration, confidence_min)
        with torch.inference_mode():
            adjudication = translate_accepted(adjudication, tokenizer, translator)
        if adjudication["status"] == "ACCEPT":
            out.append({"start": adjudication["words"][0]["start"], "end": adjudication["words"][-1]["end"], "de": adjudication["text_de"], "en": adjudication["text_en"], "words": adjudication["words"], "status": "ACCEPT", "reason": None})
        else:
            out.append({"start": float(segment.get("start",0.0)), "end": float(segment.get("end",0.0)), "de": adjudication.get("text_de",""), "en": "", "words": adjudication.get("words",[]), "status": "ABSTAIN", "reason": adjudication["reason"]})
    return out


def transcribe_inventory(root: Path, public: Path, rows, checkpoint_name: str):
    import whisper
    from transformers import MarianMTModel, MarianTokenizer
    checkpoint_path = root / checkpoint_name
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"items": {}}
    if checkpoint.get("adapter_contract") != "frozen_language_adapter_v1":
        checkpoint = {"adapter_contract": "frozen_language_adapter_v1", "items": {}}
    asr = whisper.load_model(str(public / "models/whisper/small.pt"), device="cuda")
    translator_root = public / "models/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(translator_root, local_files_only=True)
    translator = MarianMTModel.from_pretrained(translator_root, local_files_only=True).to("cuda")
    for row in rows:
        key = row["asset_key"]
        if key in checkpoint["items"]:
            continue
        result = asr.transcribe(
            str(root / row["file"]), language="de", task="transcribe", temperature=0,
            beam_size=5, word_timestamps=True, condition_on_previous_text=False,
            fp16=True, verbose=False,
        )
        audio_duration = len(whisper.load_audio(str(root / row["file"]))) / whisper.audio.SAMPLE_RATE
        segments = translate_segments(tokenizer, translator, result["segments"], result.get("language"), audio_duration, 0.35)
        checkpoint["items"][key] = {"segments": segments, "language": result.get("language"), "child_key": row["child_key"], "session_key": row["session_key"], "file": row["file"], "accepted_segments":sum(segment["status"]=="ACCEPT" for segment in segments), "abstained_segments":sum(segment["status"]=="ABSTAIN" for segment in segments), "total_segments":len(segments)}
        write_json(checkpoint_path, checkpoint)
    del asr, translator
    torch.cuda.empty_cache()
    return checkpoint


def frequency_bins(counts, bins=3):
    ordered = sorted(counts, key=lambda w: (counts[w], w))
    return {word: min(bins - 1, index * bins // max(1, len(ordered))) for index, word in enumerate(ordered)}


def freeze_vocabulary(root: Path, cfg, checkpoint):
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.data.path[:] = [str(Path(os.environ["PHASE4_PUBLIC_ROOT"]) / "models/nltk_data")]
    counts = Counter()
    for item in checkpoint["items"].values():
        for segment in item["segments"]:
            if segment.get("status") != "ACCEPT": continue
            counts.update(re.findall(r"[a-z]+", segment["en"].lower()))
    tagged = nltk.pos_tag(sorted(counts))
    nouns = {w: counts[w] for w, pos in tagged if pos.startswith("NN") and counts[w] >= cfg["lexical"]["noun_minimum_count"] and wn.synsets(w, pos=wn.NOUN)}
    adjectives = {w: counts[w] for w, pos in tagged if pos.startswith("JJ") and counts[w] >= cfg["lexical"]["adjective_minimum_count"] and wn.synsets(w, pos=wn.ADJ)}
    sys.path.insert(0, os.environ["PHASE4_EGOBABY_ROOT"])
    from apps.benchmark_creation.pipeline.lexical.constants import ADJ_ANTONYMS
    adjectives = {w: c for w, c in adjectives.items() if w in ADJ_ANTONYMS}
    noun_groups = defaultdict(list)
    for word in nouns:
        noun_groups[wn.synsets(word, pos=wn.NOUN)[0].lexname()].append(word)
    nouns = {w: c for w, c in nouns.items() if len(noun_groups[wn.synsets(w, pos=wn.NOUN)[0].lexname()]) >= 2}
    selected = {}
    for pos, values, cap in (("nouns", nouns, cfg["lexical"]["nouns_per_frequency_bin"]), ("adjectives", adjectives, cfg["lexical"]["adjectives_per_frequency_bin"])):
        bins = frequency_bins(values)
        words = []
        for bin_id in range(cfg["lexical"]["frequency_bins"]):
            candidates = sorted((w for w in values if bins[w] == bin_id), key=lambda w: (-values[w], w))[:cap]
            words.extend({"word": w, "count": values[w], "frequency_bin": bin_id} for w in candidates)
        selected[pos] = words
    if len(selected["nouns"]) < 2 or len(selected["adjectives"]) < 1:
        raise RuntimeError("E_LEXICAL_SUPPORT")
    value = {"schema_version": 1, "source": "calibration_C_only", "nouns": selected["nouns"], "adjectives": selected["adjectives"], "antonyms": {x["word"]: ADJ_ANTONYMS[x["word"]] for x in selected["adjectives"]}}
    write_json(root / "lexical/restricted_vocabulary.json", value)
    return value


def image_work(vocab, cfg):
    work = []
    for style in cfg["lexical"]["styles"]:
        for pos in ("nouns", "adjectives"):
            concepts = vocab[pos]
            if pos == "adjectives":
                concepts = concepts + [{"word": vocab["antonyms"][x["word"]], "count": 0, "frequency_bin": x["frequency_bin"]} for x in concepts]
            for concept in concepts:
                for seed in cfg["lexical"]["seeds"]:
                    work.append((style, pos, concept["word"], seed))
    return sorted(set(work))


def generate_images(root: Path, public: Path, vocab, cfg, rank, world):
    from diffusers import Flux2KleinPipeline
    work = image_work(vocab, cfg)
    pipe = Flux2KleinPipeline.from_pretrained(public / "models/FLUX.2-klein-4B", torch_dtype=torch.bfloat16, local_files_only=True).to(f"cuda:{rank}")
    for index, (style, pos, word, seed) in enumerate(work):
        if index % world != rank:
            continue
        target = root / "lexical/images" / style / pos / f"{digest([word, seed])}.png"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        provisional = root / "superseded/initial_stage_b/lexical/images" / style / pos / target.name
        if provisional.is_file():
            import shutil
            shutil.copyfile(provisional, target); os.chmod(target, 0o600)
            if file_digest(provisional) != file_digest(target): raise RuntimeError("E_PROVISIONAL_IMAGE_REUSE_HASH")
            continue
        subject = f"one clearly visible {word}" if pos == "nouns" else f"one clearly {word} everyday object"
        prefix = "a realistic child-eye-level photograph of" if style == "realistic" else "a clean colorful children's-book cartoon of"
        prompt = f"{prefix} {subject}, centered, uncluttered background, no text, no watermark"
        image = pipe(prompt=prompt, height=cfg["lexical"]["image_size"], width=cfg["lexical"]["image_size"], num_inference_steps=4, guidance_scale=1.0, generator=torch.Generator(device=f"cuda:{rank}").manual_seed(seed)).images[0]
        image.save(target)
        os.chmod(target, 0o600)


def build_lexical_manifests(root: Path, vocab, cfg):
    from nltk.corpus import wordnet as wn
    from apps.benchmark_creation.utils.vision_scoring import build_caption
    manifests = []
    for style in cfg["lexical"]["styles"]:
        noun_rows = []
        nouns = vocab["nouns"]
        groups = defaultdict(list)
        for item in nouns: groups[wn.synsets(item["word"], pos=wn.NOUN)[0].lexname()].append(item)
        for item in nouns:
            peers = [x for x in groups[wn.synsets(item["word"], pos=wn.NOUN)[0].lexname()] if x["word"] != item["word"]]
            if not peers: continue
            neg = sorted(peers, key=lambda x: digest([item["word"], x["word"]]))[0]
            for seed in cfg["lexical"]["seeds"]:
                category=wn.synsets(item["word"], pos=wn.NOUN)[0].lexname()
                noun_rows.append({"word": item["word"], "category": category, "caption_positive": build_caption(item["word"],category=category,task="nouns"), "caption_negative": build_caption(neg["word"],category=category,task="nouns"), "negative_word": neg["word"], "image_positive": f"../../images/{style}/nouns/{digest([item['word'], seed])}.png", "image_negative": f"../../images/{style}/nouns/{digest([neg['word'], seed])}.png", "frequency_bin": item["frequency_bin"]})
        adjective_rows = []
        for item in vocab["adjectives"]:
            neg = vocab["antonyms"][item["word"]]
            for seed in cfg["lexical"]["seeds"]:
                adjective_rows.append({"word": item["word"], "caption_positive": f"a {item['word']} object", "caption_negative": f"a {neg} object", "negative_word": neg, "image_positive": f"../../images/{style}/adjectives/{digest([item['word'], seed])}.png", "image_negative": f"../../images/{style}/adjectives/{digest([neg, seed])}.png", "frequency_bin": item["frequency_bin"]})
        for pos, rows in (("nouns", noun_rows), ("adjectives", adjective_rows)):
            path = root / "lexical/Lexical" / pos.capitalize() / f"manifest_{pos}_{style}.json"
            write_json(path, {"schema_version": 1, "style": style, "items": rows})
            manifests.append(path)
    return manifests


def load_frozen_pe_engine(public: Path, cfg, device="cuda"):
    from apps.benchmark_creation.utils.vision_scoring import ScoringEngine
    from apps.alignment_scoring.third_party.perception_models.core.vision_encoder import pe, transforms
    expected_revision = cfg["lexical"]["upstream_filter"]["model_revision"]
    candidates = list((public / "models/pe-hf-home/hub").glob(f"models--facebook--PE-Core-L14-336/snapshots/{expected_revision}/PE-Core-L14-336.pt"))
    if len(candidates) != 1 or file_digest(candidates[0]) != cfg["lexical"]["upstream_filter"]["weights_sha256"]:
        raise RuntimeError("E_PE_CHECKPOINT")
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
    model.load_ckpt(str(candidates[0])); model = model.to(device).eval()
    return ScoringEngine(model=model, processor=(transforms.get_image_transform(model.image_size), transforms.get_text_tokenizer(model.context_length)), backend="pe")


def filter_lexical(root: Path, public: Path, cfg, manifests):
    from PIL import Image
    from apps.benchmark_creation.utils.vision_scoring import score_image_text_matrix
    engine = load_frozen_pe_engine(public, cfg)
    threshold = cfg["lexical"]["upstream_filter"]["min_score"]
    audits = {}
    for manifest_path in manifests:
        value = json.loads(manifest_path.read_text()); retained = []
        for row in value["items"]:
            with Image.open(manifest_path.parent / row["image_positive"]) as image:
                positive = image.convert("RGB").copy()
            with Image.open(manifest_path.parent / row["image_negative"]) as image:
                negative = image.convert("RGB").copy()
            matrix = score_image_text_matrix(engine, [positive, negative], [row["caption_positive"], row["caption_negative"]], "cuda")
            passes = matrix[0,0].item() >= threshold and matrix[1,1].item() >= threshold
            if cfg["lexical"]["upstream_filter"]["require_contrastive"]:
                passes = passes and matrix[0,0].item() > matrix[1,0].item() and matrix[1,1].item() > matrix[0,1].item()
            if passes: retained.append(row)
        key=f"{value['style']}_{manifest_path.parent.name.lower()}"
        audits[key]={"before":len(value["items"]),"retained":len(retained),"removed":len(value["items"])-len(retained)}
        value["items"]=retained; value["upstream_filter"]={"model":cfg["lexical"]["upstream_filter"]["model"],"revision":cfg["lexical"]["upstream_filter"]["model_revision"],"min_score":threshold,"require_contrastive":True}
        write_json(manifest_path,value)
    del engine
    torch.cuda.empty_cache()
    return audits


def official_lexical_loader_smoke(root: Path, cfg):
    import numpy as np
    from evaluation.data.machine_devbench import MachineDevBenchLexicalDataset
    from evaluation.multimodal.machine_devbench.base import MachineDevBenchEvalModule
    class StructuralSmoke(MachineDevBenchEvalModule):
        def _compute_similarity(self, model, batch):
            return np.asarray([[1.0],[0.0]])
    totals={}
    for style in cfg["lexical"]["styles"]:
        for pos in ("nouns","adjectives"):
            dataset=MachineDevBenchLexicalDataset(root/"lexical",style,pos)
            for index in range(len(dataset)):
                sample=dataset[index]
                batch=dataset.collate_fn([sample])
                if len(batch["image"])!=2 or len(batch["text"])!=1: raise RuntimeError("E_LEXICAL_LOADER_SHAPE")
                smoke=object.__new__(StructuralSmoke)
                if smoke._predict_batch(None,f"lex_{pos}",batch)!=[0]: raise RuntimeError("E_LEXICAL_EVALUATOR_SMOKE")
            totals[f"{style}_{pos}"]=len(dataset)
    return totals


def perceptual_hash(image):
    import numpy as np
    pixels=np.asarray(image.convert("L").resize((32,32)),dtype=np.float64)
    n=32; indices=np.arange(n); frequencies=np.arange(n)[:,None]
    basis=np.cos(np.pi*(2*indices+1)*frequencies/(2*n)); basis[0]*=1/np.sqrt(2); basis*=np.sqrt(2/n)
    low=(basis@pixels@basis.T)[:8,:8].reshape(-1)[1:]
    median=float(np.median(low))
    return sum((int(value>median)<<index) for index,value in enumerate(low))


def overlap_near_duplicate_diagnostics(root: Path, public: Path, cfg, stage):
    from PIL import Image
    ffmpeg=__import__("imageio_ffmpeg").get_ffmpeg_exe()
    sample_root=root/"overlap_diagnostics/frames"; rows=[]
    for role,items in (("calibration",stage["calibration"]),("evaluation",stage["evaluation"])):
        for item in items:
            target=sample_root/f"{role}-{item['asset_key']}.png"
            if not target.exists():
                extract_frame(ffmpeg,root/item["file"],cfg["overlap_audit"]["perceptual_sample_seconds"],target,120)
            if target.is_file(): rows.append({"role":role,"asset_key":item["asset_key"],"path":target})
    engine=load_frozen_pe_engine(public,cfg)
    transform=engine.processor[0]; features=[]
    for row in rows:
        with Image.open(row["path"]) as image: rgb=image.convert("RGB").copy()
        row["phash"]=perceptual_hash(rgb)
        tensor=transform(rgb).unsqueeze(0).to("cuda")
        with torch.inference_mode(): feature=engine.model.encode_image(tensor,normalize=True).cpu()[0]
        features.append(feature)
    calibration_indices=[i for i,row in enumerate(rows) if row["role"]=="calibration"]
    evaluation_indices=[i for i,row in enumerate(rows) if row["role"]=="evaluation"]
    phash_pairs=embedding_pairs=0
    if calibration_indices and evaluation_indices:
        c=torch.stack([features[i] for i in calibration_indices]); e=torch.stack([features[i] for i in evaluation_indices])
        embedding_pairs=int((c@e.T>=cfg["overlap_audit"]["embedding_cosine_min_diagnostic"]).sum().item())
        for i in calibration_indices:
            for j in evaluation_indices:
                phash_pairs += int((rows[i]["phash"]^rows[j]["phash"]).bit_count()<=cfg["overlap_audit"]["perceptual_hash_hamming_max_diagnostic"])
    restricted={"schema_version":1,"rows":[{"role":row["role"],"asset_key":row["asset_key"],"phash":row["phash"],"embedding":features[i].tolist()} for i,row in enumerate(rows)]}
    write_json(root/"overlap_diagnostics/restricted_near_duplicate_diagnostics.json",restricted)
    for row in rows: row["path"].unlink(missing_ok=True)
    del engine
    torch.cuda.empty_cache()
    return {"sampled_calibration":len(calibration_indices),"sampled_evaluation":len(evaluation_indices),"cross_role_phash_pairs_at_threshold":phash_pairs,"cross_role_embedding_pairs_at_threshold":embedding_pairs,"blocking":False}


def extract_frame(ffmpeg, source, timestamp, target, timeout_seconds):
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_suffix(".partial.png")
    try:
        subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}", "-i", str(source), "-frames:v", "1", "-vf", "scale=512:-2", "-y", str(temporary)], check=True, timeout=timeout_seconds)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            return False
        os.chmod(temporary, 0o600)
        temporary.replace(target)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    finally:
        temporary.unlink(missing_ok=True)


def render_temporal_row(plan):
    row, jobs = plan
    completed = []
    for job in jobs:
        target = job[-2]
        if target.is_file() and target.stat().st_size > 0:
            completed.append(target)
            continue
        if not extract_frame(*job):
            for path in completed:
                path.unlink(missing_ok=True)
            return None
        completed.append(target)
    return row


def build_temporal(root: Path, cfg, checkpoint):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    utterances = []
    for asset_key, item in checkpoint["items"].items():
        for idx, segment in enumerate(item["segments"]):
            if segment.get("status") != "ACCEPT": continue
            if segment["end"] - segment["start"] < 0.4: continue
            utterances.append({"utterance_key": digest([asset_key, idx]), "asset_key": asset_key, "child_key": item["child_key"], "session_key": item["session_key"], "file": item["file"], "start": segment["start"], "end": segment["end"], "text": segment["en"]})
    plans=[]
    for u in utterances:
        same_session=[x for x in utterances if x["session_key"]==u["session_key"] and abs((x["start"]+x["end"])/2-(u["start"]+u["end"])/2)>=cfg["temporal"]["temporal_exclusion_buffer_seconds"]]
        same_child=[x for x in utterances if x["child_key"]==u["child_key"] and x["session_key"]!=u["session_key"]]
        other=[x for x in utterances if x["child_key"]!=u["child_key"]]
        selected=[]; strata=[]
        for label,pool,n in (("same_session",same_session,3),("same_child_other_session",same_child,2),("other_child",other,2)):
            chosen=sorted(pool,key=lambda x:digest([u["utterance_key"],x["utterance_key"]]))[:n]
            selected += chosen; strata += [label] * len(chosen)
        if len({x["utterance_key"] for x in selected})<7: continue
        center=max(2.0,(u["start"]+u["end"])/2); window_start=max(0.0,center-2.0)
        frames=[]; jobs=[]
        for off in cfg["temporal"]["frame_offsets_seconds"]:
            target=root/"temporal/frames"/f"{u['utterance_key']}-{int(off*10):02d}.png"
            jobs.append((ffmpeg,root/u["file"],window_start+off,target,cfg["temporal"]["frame_decode_timeout_seconds"]))
            frames.append(str(target.relative_to(root/"temporal")))
        candidates=[u]+selected
        order=sorted(range(8),key=lambda i:digest([u["utterance_key"],candidates[i]["utterance_key"]]))
        labels=["positive"]+strata
        plans.append(({"query_key":u["utterance_key"],"frames":frames,"candidate_texts":[candidates[i]["text"] for i in order],"candidate_strata":[labels[i] for i in order],"positive_index":order.index(0),"child_key":u["child_key"],"session_key":u["session_key"]},jobs))
    if not plans: raise RuntimeError("E_TEMPORAL_SUPPORT")
    with ThreadPoolExecutor(max_workers=8) as executor:
        rendered=list(executor.map(render_temporal_row,plans))
    rows=[row for row in rendered if row is not None]
    if not rows: raise RuntimeError("E_TEMPORAL_FRAME_SUPPORT")
    frame_paths=[root/"temporal"/path for row in rows for path in row["frames"]]
    frame_hashes=[file_digest(path) for path in frame_paths]
    manifest={"schema_version":1,"candidate_count":8,"metrics":["Recall@1","MRR"],"aggregation":"macro_child_then_overall_mean","referent_ground_truth":False,"human_german_validation":False,"items":rows}
    write_json(root/"temporal/restricted_manifest.json",manifest)
    return root/"temporal/restricted_manifest.json", {"frame_duplicate_count":len(frame_hashes)-len(set(frame_hashes)),"row_count":len(rows),"frame_count":len(frame_paths),"decode_excluded_row_count":len(plans)-len(rows),"all_candidate_counts_exact":all(len(row["candidate_texts"])==cfg["temporal"]["candidate_count"] for row in rows)}


def seal(root: Path, cfg, vocab, lexical_paths, temporal_path, audits):
    lexical_rows=[{"path":str(p.relative_to(root/"lexical")),"sha256":file_digest(p)} for p in sorted(lexical_paths)]
    for p in sorted((root/"lexical/images").rglob("*.png")): lexical_rows.append({"path":str(p.relative_to(root/"lexical")),"sha256":file_digest(p)})
    lexical_commitment=digest(lexical_rows)
    temporal_rows=[{"path":str(temporal_path.relative_to(root/"temporal")),"sha256":file_digest(temporal_path)}]
    for p in sorted((root/"temporal/frames").rglob("*.png")): temporal_rows.append({"path":str(p.relative_to(root/"temporal")),"sha256":file_digest(p)})
    temporal_commitment=digest(temporal_rows)
    common={arm:{"lexical":lexical_commitment,"temporal":temporal_commitment} for arm in cfg["sealing"]["all_later_arms"]}
    provisional_images=root/"superseded/initial_stage_b/lexical/images"
    reused=sum((provisional_images/p.relative_to(root/"lexical/images")).is_file() and file_digest(provisional_images/p.relative_to(root/"lexical/images"))==file_digest(p) for p in (root/"lexical/images").rglob("*.png"))
    audits.update({"lexical_noun_count":len(vocab["nouns"]),"lexical_adjective_count":len(vocab["adjectives"]),"lexical_style_count":len(cfg["lexical"]["styles"]),"lexical_image_reused_byte_identical_count":reused,"lexical_image_generated_or_changed_count":len(list((root/"lexical/images").rglob("*.png")))-reused})
    record={"schema_version":1,"status":"PASS","lexical_commitment":lexical_commitment,"temporal_commitment":temporal_commitment,"common_asset_references":common,"test_assets_may_steer_later_work":False,"generated_images_not_held_out_real":True,"temporal_model_derived_no_referent_ground_truth":True,"public_provenance":{"language_pipeline":cfg["language_pipeline"],"image_generator":{"repository":cfg["lexical"]["generator"],"revision":cfg["lexical"]["generator_revision"],"license":cfg["lexical"].get("generator_license","Apache-2.0")}},"filters":{"lexical_source":"calibration_C_only","temporal_source":"evaluation_children_sessions_only","frame_decode_failure":cfg["temporal"]["frame_decode_failure"]},"hash_inventory":{"lexical_files":len(lexical_rows),"temporal_files":len(temporal_rows)},"audits":audits}
    write_json(root/"sealed/restricted_phase4_seal.json",record)
    compact=dict(record)
    compact["contract_identical_all_arms"]=len({canonical(x) for x in common.values()})==1
    write_json(root/"sealed/compact_phase4_decision.json",compact)


def main():
    rank=int(os.environ.get("SLURM_PROCID","0")); world=int(os.environ.get("SLURM_NTASKS","1")); torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(hours=12), device_id=torch.device(f"cuda:{rank}"))
    root=Path(os.environ["PHASE4_RESTRICTED_ROOT"]); public=Path(os.environ["PHASE4_PUBLIC_ROOT"])
    cfg=json.loads(Path(os.environ["PHASE4_CONFIG"]).read_text()); stage=json.loads((root/"restricted_stage_manifest.json").read_text())
    overlap=json.loads((root/"restricted_overlap_audit.json").read_text())
    if overlap["status"]!="PASS" or overlap["post_resolution_unresolved_exact_or_temporal_count"]!=0:
        raise RuntimeError("E_BLOCKING_OVERLAP_AUDIT")
    calibration_children={row["child_key"] for row in stage["calibration"]}
    evaluation_children={row["child_key"] for row in stage["evaluation"]}
    child_overlap=len(calibration_children & evaluation_children)
    if child_overlap: raise RuntimeError("E_CALIBRATION_EVALUATION_CHILD_OVERLAP")
    if rank==0:
        c=transcribe_inventory(root,public,stage["calibration"],"asr/restricted_calibration.json")
        vocab=freeze_vocabulary(root,cfg,c)
    dist.barrier(); vocab=json.loads((root/"lexical/restricted_vocabulary.json").read_text())
    generate_images(root,public,vocab,cfg,rank,world); dist.barrier()
    if rank==0:
        lexical=build_lexical_manifests(root,vocab,cfg)
        lexical_filter_audits=filter_lexical(root,public,cfg,lexical)
        lexical_loader_counts=official_lexical_loader_smoke(root,cfg)
        evaluation=transcribe_inventory(root,public,stage["evaluation"],"asr/restricted_evaluation.json")
        near_duplicate_audits=overlap_near_duplicate_diagnostics(root,public,cfg,stage)
        temporal,audits=build_temporal(root,cfg,evaluation)
        audits.update({"calibration_evaluation_child_overlap_count":child_overlap,"evaluation_child_count":len(evaluation_children),"evaluation_session_count":len({row["session_key"] for row in stage["evaluation"]}),"blocking_overlap_audit_commitment":file_digest(root/"restricted_overlap_audit.json"),"blocking_overlap_post_resolution_unresolved":0,"lexical_filter":lexical_filter_audits,"official_loader_counts":lexical_loader_counts,"official_evaluator_structural_smoke":True,"near_duplicate_diagnostics":near_duplicate_audits})
        seal(root,cfg,vocab,lexical,temporal,audits)
        print(json.dumps({"status":"PASS","phase":4},sort_keys=True))
    dist.barrier();dist.destroy_process_group()


if __name__=="__main__": main()
