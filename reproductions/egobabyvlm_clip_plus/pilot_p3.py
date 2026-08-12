#!/usr/bin/env python3
"""Canonical, fail-closed Pilot P3 preprocessing orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


LABELS = ["engineering_only", "non_comparable", "not_a_reproduction_result"]
SLOTS = ("input_1", "input_2")
STAGES = ("audio", "vtc", "transcript", "frames", "manifests", "qa")
PRIVATE_LOG = None


class P3Error(RuntimeError):
    """A redacted P3 contract or execution failure."""


class ControlledStop(P3Error):
    """Intentional interruption after a completed stage."""


def require(value, message):
    if not value:
        raise P3Error(message)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
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


def owner_only(path, directory=False):
    path = Path(path)
    required = 0o700 if directory else 0o600
    require(stat.S_IMODE(path.stat().st_mode) == required, "owner_only_mode_check_failed")


def select_records(ledger):
    queue = ledger.get("queue")
    require(isinstance(queue, list), "governed_ledger_queue_missing")
    records = [item for item in queue if isinstance(item, dict) and
               item.get("status") in {"downloaded", "completed", "verified"}]
    require(len(records) == 2, "exact_two_input_contract_failed")
    required = {"record_key", "sha256", "size_bytes"}
    require(all(required.issubset(item) for item in records), "governed_verification_fields_missing")
    require(len({str(item["record_key"]) for item in records}) == 2, "governed_keys_not_unique")
    return dict(zip(SLOTS, records))


def resolve_media(media_root, governed_key):
    matches = [item for item in Path(media_root).rglob(str(governed_key) + ".*") if item.is_file()]
    require(len(matches) == 1, "governed_key_resolution_failed")
    return matches[0]


def opaque_key(run_id, slot, governed_key):
    return hashlib.sha256(f"{run_id}:{slot}:{governed_key}".encode()).hexdigest()[:24]


def tool_revision_hash(config):
    selected = {name: config[name] for name in ("audio", "speaker", "transcription", "frames", "pairing")}
    return canonical_hash(selected)


def cache_key(source_hash, config_hash, revision_hash):
    return hashlib.sha256(f"{source_hash}:{config_hash}:{revision_hash}".encode()).hexdigest()


def run_redacted(command, *, env=None, cwd=None):
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False, text=True)
    if completed.returncode:
        if PRIVATE_LOG:
            with Path(PRIVATE_LOG).open("a", encoding="utf-8") as stream:
                stream.write("\n[governed-private-subprocess-output]\n")
                stream.write(completed.stdout)
                stream.write(completed.stderr)
                stream.write("\n[/governed-private-subprocess-output]\n")
            os.chmod(PRIVATE_LOG, 0o600)
        raise P3Error("external_stage_failed")
    return time.monotonic() - started


def directory_inventory(root):
    items = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            items.append({"relative": str(path.relative_to(root)), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    return items


def valid_completion(stage_dir, expected_key):
    marker = Path(stage_dir) / "complete.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("cache_key") != expected_key or data.get("stage") != Path(stage_dir).name:
            return False
        for item in data.get("inventory", []):
            path = Path(stage_dir) / item["relative"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                return False
        return True
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def commit_stage(stage_dir, key, metrics, builder):
    stage_dir = Path(stage_dir)
    if valid_completion(stage_dir, key):
        marker = json.loads((stage_dir / "complete.json").read_text(encoding="utf-8"))
        return marker["metrics"], True
    require(not stage_dir.exists(), "invalid_completed_stage_requires_review")
    stage_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(stage_dir.parent, 0o700)
    temporary = Path(tempfile.mkdtemp(prefix="." + stage_dir.name + ".", dir=stage_dir.parent))
    os.chmod(temporary, 0o700)
    try:
        stage_metrics = builder(temporary)
        merged = {**metrics, **stage_metrics}
        inventory = directory_inventory(temporary)
        marker = {"schema_version": 1, "stage": stage_dir.name, "cache_key": key,
                  "metrics": merged, "inventory": inventory}
        atomic_json(temporary / "complete.json", marker)
        os.replace(temporary, stage_dir)
        return merged, False
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_rttm(path):
    annotations = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                start, duration = float(parts[3]), float(parts[4])
            except ValueError:
                continue
            if start >= 0 and duration >= 0:
                annotations.append({"start": start, "end": start + duration, "label": parts[7]})
    return annotations


def overlaps(start, end, annotations, label="KCHI"):
    return any(item["label"] == label and start < item["end"] and end > item["start"] for item in annotations)


def normalize_segments(transcription, annotations, threshold):
    output, counts = [], {"raw_segments": 0, "raw_words": 0, "kchi_removed_segments": 0,
                          "kchi_removed_duration_seconds": 0.0, "confidence_filtered_words": 0,
                          "empty_score_words": 0, "empty_after_filter_utterances": 0,
                          "retained_utterances": 0, "retained_words": 0}
    for segment in transcription.get("segments", []):
        if not isinstance(segment, dict):
            continue
        counts["raw_segments"] += 1
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            continue
        words = segment.get("words") if isinstance(segment.get("words"), list) else []
        counts["raw_words"] += len(words)
        if overlaps(start, end, annotations):
            counts["kchi_removed_segments"] += 1
            counts["kchi_removed_duration_seconds"] += end - start
            continue
        retained = []
        for word in words:
            if not isinstance(word, dict):
                counts["empty_score_words"] += 1
                continue
            score = word.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                counts["empty_score_words"] += 1
                continue
            if score < threshold:
                counts["confidence_filtered_words"] += 1
                continue
            text = str(word.get("word") or "").strip()
            if not text:
                counts["empty_score_words"] += 1
                continue
            retained.append({"word": text, "score": float(score),
                             "start": word.get("start"), "end": word.get("end")})
        if not retained:
            counts["empty_after_filter_utterances"] += 1
            continue
        normalized = " ".join(item["word"] for item in retained)
        output.append({"start": start, "end": end, "raw_text": segment.get("text", ""),
                       "normalized_text": normalized, "words": retained})
        counts["retained_utterances"] += 1
        counts["retained_words"] += len(retained)
    counts["kchi_removed_duration_seconds"] = round(counts["kchi_removed_duration_seconds"], 6)
    return output, counts


def linear_selection(items, maximum):
    if len(items) <= maximum:
        return list(items)
    require(maximum >= 2, "invalid_pairing_limit")
    indices = []
    for index in range(maximum):
        value = index * (len(items) - 1) / (maximum - 1)
        selected = int(math.floor(value + 0.5))
        if not indices or selected != indices[-1]:
            indices.append(selected)
    return [items[index] for index in indices]


def frame_candidates(start, end, frame_count):
    return [index for index in range(frame_count) if start <= float(index) <= end]


def build_manifests(slot, opaque, frame_count, normalized, maximum=32):
    visual = [{"source": opaque, "frame_index": index, "timestamp_seconds": float(index)}
              for index in range(frame_count)]
    text, paired = [], []
    for utterance_index, segment in enumerate(normalized):
        text.append({"source": opaque, "utterance_index": utterance_index,
                     "start": segment["start"], "end": segment["end"],
                     "text": segment["normalized_text"], "word_count": len(segment["words"])})
        selected = linear_selection(frame_candidates(segment["start"], segment["end"], frame_count), maximum)
        if selected:
            paired.append({"source": opaque, "utterance_index": utterance_index,
                           "start": segment["start"], "end": segment["end"],
                           "frame_indices": selected, "text": segment["normalized_text"]})
    require(all(record["source"] == opaque for record in visual + text + paired), "cross_record_mixing")
    return visual, text, paired


def ffprobe_audio(path, ffprobe):
    command = [str(ffprobe), "-v", "error", "-select_streams", "a:0", "-show_entries",
               "stream=sample_rate,channels,duration", "-of", "json", str(path)]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(completed.returncode == 0, "audio_probe_failed")
    streams = json.loads(completed.stdout).get("streams", [])
    require(len(streams) == 1, "audio_stream_count_invalid")
    return streams[0]


def process_unit(slot, record, args, config, roots, tools, stop_after=None):
    source = resolve_media(args.media_root, record["record_key"])
    source_hash = sha256_file(source)
    require(source_hash == record["sha256"] and source.stat().st_size == int(record["size_bytes"]),
            "source_integrity_failed")
    config_hash = sha256_file(args.config)
    revision_hash = tool_revision_hash(config)
    key = cache_key(source_hash, config_hash, revision_hash)
    opaque = opaque_key(config["engineering_run_id"], slot, record["record_key"])
    unit = roots["run"] / slot / key
    unit.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(unit, 0o700)
    stage_hits, stage_runs, metrics = [], [], {"slot": slot}

    def stage(name, builder):
        nonlocal metrics
        metrics, hit = commit_stage(unit / name, key, metrics, builder)
        (stage_hits if hit else stage_runs).append(name)
        if stop_after == name:
            atomic_json(unit / "controlled_stop.json", {"stage": name, "cache_key": key})
            raise ControlledStop("controlled_stop")

    def audio_builder(root):
        output = root / "audio.wav"
        wall = run_redacted([str(tools["ffmpeg"]), "-v", "error", "-i", str(source), "-vn",
                             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)])
        probe = ffprobe_audio(output, tools["ffprobe"])
        require(int(probe["sample_rate"]) == 16000 and int(probe["channels"]) == 1,
                "audio_contract_failed")
        return {"source_duration_seconds": float(probe.get("duration") or 0),
                "audio_wall_seconds": wall, "audio_bytes": output.stat().st_size}
    stage("audio", audio_builder)

    def vtc_builder(root):
        audio_dir = root / "input"; audio_dir.mkdir(mode=0o700)
        os.symlink(unit / "audio" / "audio.wav", audio_dir / "source.wav")
        output = root / "output"
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        vtc_env = os.environ.copy()
        vtc_env["PATH"] = str(tools["ffmpeg"].parent) + os.pathsep + vtc_env.get("PATH", "")
        vtc_env["LD_LIBRARY_PATH"] = str(tools["ffmpeg"].parent.parent / "lib") + os.pathsep + vtc_env.get("LD_LIBRARY_PATH", "")
        wall = run_redacted([str(tools["vtc_python"]), str(tools["vtc_runner"]),
                             "--upstream-script", str(tools["vtc_script"]), "--wavs",
                             str(audio_dir), "--output", str(output), "--device", "cuda",
                             "--config", str(tools["vtc_config"]),
                             "--checkpoint", str(tools["vtc_checkpoint"]),
                             "--thresholds", str(tools["vtc_thresholds"])], env=vtc_env, cwd=tools["vtc_root"])
        rttms = list((output / "rttm").glob("*.rttm"))
        require(len(rttms) == 1, "vtc_output_invalid")
        shutil.copy2(rttms[0], root / "speaker.rttm")
        annotations = parse_rttm(root / "speaker.rttm")
        return {"vtc_wall_seconds": wall, "speech_segments": len(annotations),
                "peak_cpu_ram_kb": max(before, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)}
    stage("vtc", vtc_builder)

    def transcript_builder(root):
        sys.path.insert(0, str(tools["upstream_root"]))
        import torch
        import whisperx
        device = "cuda"
        model = whisperx.load_model(str(tools["whisper_model"]), device,
                                    compute_type=config["transcription"]["compute_type"], language="en")
        align, metadata = whisperx.load_align_model(language_code="en", device=device,
                                                    model_name=str(tools["alignment_model"]),
                                                    model_dir=str(tools["alignment_dir"]),
                                                    model_cache_only=True)
        audio = whisperx.load_audio(str(unit / "audio" / "audio.wav"))
        chosen, raw, retries = None, None, 0
        for batch in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
            try:
                started = time.monotonic(); raw = model.transcribe(audio, batch_size=batch)
                wall = time.monotonic() - started; chosen = batch; break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise P3Error("transcription_failed") from None
                retries += 1; torch.cuda.empty_cache()
        require(raw is not None, "no_implementation_valid_transcription_batch")
        aligned = whisperx.align(raw["segments"], align, metadata, audio, device,
                                 return_char_alignments=False)
        aligned["language"] = raw.get("language", "en")
        atomic_json(root / "raw.json", aligned)
        annotations = parse_rttm(unit / "vtc" / "speaker.rttm")
        normalized, counts = normalize_segments(aligned, annotations,
                                                config["transcription"]["confidence_threshold"])
        atomic_json(root / "normalized.json", {"segments": normalized, "policy": {
            "threshold": 0.5, "comparison": "greater_than_or_equal", "empty_word": "drop",
            "empty_utterance": "drop", "aggregation": "space_join_source_order"}})
        peak = int(torch.cuda.max_memory_allocated())
        return {**counts, "whisperx_wall_seconds": wall, "whisperx_batch_size": chosen,
                "whisperx_batch_retries": retries, "peak_gpu_memory_bytes": peak}
    stage("transcript", transcript_builder)

    def frames_builder(root):
        pattern = root / "frame_%08d.jpg"
        wall = run_redacted([str(tools["ffmpeg"]), "-v", "error", "-i", str(source),
                             "-vf", "fps=1", "-start_number", "0", str(pattern)])
        frames = sorted(root.glob("frame_*.jpg"))
        require(frames, "frame_extraction_empty")
        atomic_json(root / "frames.json", [{"index": index, "timestamp_seconds": float(index),
                    "source_sha256": source_hash, "sha256": sha256_file(path)}
                    for index, path in enumerate(frames)])
        return {"frame_wall_seconds": wall, "frames": len(frames),
                "frame_bytes": sum(path.stat().st_size for path in frames)}
    stage("frames", frames_builder)

    def manifest_builder(root):
        normalized = json.loads((unit / "transcript" / "normalized.json").read_text())["segments"]
        visual, text, paired = build_manifests(slot, opaque, metrics["frames"], normalized,
                                                config["pairing"]["maximum_candidate_frames"])
        for name, value in (("visual.json", visual), ("text.json", text), ("paired.json", paired)):
            atomic_json(root / name, value)
        return {"visual_records": len(visual), "text_records": len(text), "paired_records": len(paired),
                "paired_frame_references": sum(len(item["frame_indices"]) for item in paired)}
    stage("manifests", manifest_builder)

    def qa_builder(root):
        visual = json.loads((unit / "manifests" / "visual.json").read_text())
        text = json.loads((unit / "manifests" / "text.json").read_text())
        paired = json.loads((unit / "manifests" / "paired.json").read_text())
        annotations = parse_rttm(unit / "vtc" / "speaker.rttm")
        require(len(visual) == metrics["frames"], "visual_reconciliation_failed")
        require(len(text) == metrics["retained_utterances"], "text_reconciliation_failed")
        require(all(0 < len(item["frame_indices"]) <= 32 for item in paired), "pair_limit_failed")
        require(all(all(item["start"] <= index <= item["end"] for index in item["frame_indices"])
                    for item in paired), "pair_alignment_failed")
        require(all(not overlaps(item["start"], item["end"], annotations) for item in text),
                "kchi_exclusion_failed")
        require(all(item["source"] == opaque for item in visual + text + paired), "cross_record_mixing")
        require(all(item.get("text", "").strip() for item in text + paired), "empty_record_failed")
        sample_count = min(3, len(paired))
        atomic_json(root / "spot_check.json", {"sample_count": sample_count, "content_printed": False,
                    "timestamp_alignment_passed": True, "kchi_exclusion_passed": True})
        return {"qa_spot_checks": sample_count, "qa_passed": True}
    stage("qa", qa_builder)
    metrics["controlled_stop_observed"] = (unit / "controlled_stop.json").is_file()
    return {"slot": slot, "cache_key": key, "opaque_source": opaque, "metrics": metrics,
            "cache_hits": stage_hits, "stages_executed": stage_runs, "unit": str(unit)}


def validate_config(config):
    require(config["stage"] == "P3" and config["classification"] == LABELS, "config_identity_failed")
    require(config["input_contract"]["count"] == 2 and config["input_contract"]["slots"] == list(SLOTS),
            "config_input_contract_failed")
    require(config["transcription"]["confidence_threshold"] == 0.5, "confidence_contract_failed")
    require(config["frames"]["fps"] == 1 and config["pairing"]["maximum_candidate_frames"] == 32,
            "visual_contract_failed")
    require(config["weight_boundary"]["referenceable_by_learned_initialization"] is False,
            "weight_boundary_failed")


def aggregate_results(results, config, detail_path, log_path=None):
    metrics = [item["metrics"] for item in results]
    total_duration = sum(item["source_duration_seconds"] for item in metrics)
    def total(name): return sum(item.get(name, 0) for item in metrics)
    def per_hour(name): return round(total(name) / (total_duration / 3600), 6) if total_duration else None
    failures = total("whisperx_batch_retries")
    passed = (len(results) == 2 and all(item.get("qa_passed") for item in metrics) and
              total("paired_records") > 0 and total("retained_utterances") > 0)
    return {
        "schema_version": 1, "record_type": "pilot_p3_privacy_safe_aggregate",
        "pilot_id": config["pilot_id"], "engineering_run_id": config["engineering_run_id"],
        "stage": "P3", "status": "p3_complete" if passed else "p3_incomplete",
        "classification": LABELS, "scientific_status_effect": "none",
        "inventory_status": "incomplete_inventory", "input_count": len(results),
        "provenance": {"p3_config_sha256": sha256_file(Path(__file__).with_name("pilot_p3_config.json")),
                       "upstream_commit": config["source"]["commit"],
                       "tool_model_revision_sha256": tool_revision_hash(config)},
        "aggregate_total_source_hours": round(total_duration / 3600, 6),
        "counts": {name: total(name) for name in ("frames", "speech_segments", "raw_segments",
                   "retained_utterances", "retained_words", "confidence_filtered_words",
                   "empty_score_words", "empty_after_filter_utterances", "kchi_removed_segments",
                   "paired_records", "paired_frame_references")},
        "kchi_removed_duration_seconds": round(total("kchi_removed_duration_seconds"), 6),
        "wall_seconds_per_video_hour": {name: per_hour(name) for name in
             ("audio_wall_seconds", "vtc_wall_seconds", "whisperx_wall_seconds", "frame_wall_seconds")},
        "storage_bytes": {"audio": total("audio_bytes"), "frames": total("frame_bytes")},
        "resources": {"peak_cpu_ram_kb": max((item.get("peak_cpu_ram_kb", 0) for item in metrics), default=0),
                      "peak_gpu_memory_bytes": max((item.get("peak_gpu_memory_bytes", 0) for item in metrics), default=0)},
        "resume": {"controlled_interruption_tested": any(item.get("controlled_stop_observed") for item in metrics),
                   "stage_cache_hits": sum(len(item["cache_hits"]) for item in results),
                   "stages_executed": sum(len(item["stages_executed"]) for item in results),
                   "idempotent_rerun_passed": all(len(item["stages_executed"]) == 0 for item in results)},
        "failures": {"retry_count": failures, "reason_categories":
                     (["gpu_batch_resource_deviation"] if failures else [])},
        "batch": {"largest_implementation_safe": min(item["whisperx_batch_size"] for item in metrics),
                  "paper_target": 1024, "lower_value_is_engineering_only_deviation": True},
        "verification": {"source_integrity_count": 2, "audio_contract_count": 2,
                         "qa_passed_count": sum(bool(item.get("qa_passed")) for item in metrics),
                         "counts_reconciled": all(item["visual_records"] == item["frames"] and
                                                  item["text_records"] == item["retained_utterances"] for item in metrics)},
        "governed_detailed_record": {"opaque_run_id": config["engineering_run_id"],
                                     "sha256": sha256_file(detail_path), "owner_only": True,
                                     **({"log_sha256": sha256_file(log_path)} if log_path and Path(log_path).is_file() else {})},
        "p3_gate": {"passed": passed, "stop_reason": None if passed else "fixed_inputs_no_retained_pairs"},
        "p0_status_preserved": True, "p1_status_preserved": True, "p2_status_preserved": True,
        "full_corpus_discrepancy": "unresolved_and_out_of_scope", "next_stage_started": False,
    }


def main():
    global PRIVATE_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--slot", choices=(*SLOTS, "all"), default="all")
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--vtc-python", type=Path, required=True)
    parser.add_argument("--vtc-script", type=Path, required=True)
    parser.add_argument("--vtc-runner", type=Path, required=True)
    parser.add_argument("--vtc-root", type=Path, required=True)
    parser.add_argument("--vtc-config", type=Path, required=True)
    parser.add_argument("--vtc-checkpoint", type=Path, required=True)
    parser.add_argument("--vtc-thresholds", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--alignment-model", required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    PRIVATE_LOG = args.log
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")); validate_config(config)
        owner_only(args.ledger)
        require(inside(args.media_root, args.scratch_root), "media_root_outside_governed_scratch")
        roots = {"run": args.scratch_root / "runs" / config["storage"]["scratch_run_subdirectory"],
                 "detail": args.durable_root / "run_records" / config["storage"]["durable_record_subdirectory"]}
        require(inside(roots["run"], args.scratch_root / "runs") and
                inside(roots["detail"], args.durable_root / "run_records"), "storage_boundary_failed")
        tools = {"ffmpeg": args.ffmpeg, "ffprobe": args.ffprobe, "vtc_python": args.vtc_python,
                 "vtc_script": args.vtc_script, "vtc_runner": args.vtc_runner, "vtc_root": args.vtc_root,
                 "vtc_config": args.vtc_config, "vtc_checkpoint": args.vtc_checkpoint,
                 "vtc_thresholds": args.vtc_thresholds, "upstream_root": args.upstream_root,
                 "whisper_model": args.whisper_model, "alignment_model": args.alignment_model,
                 "alignment_dir": args.alignment_dir}
        require(all(Path(value).exists() for name, value in tools.items() if name != "alignment_model"),
                "pinned_tool_missing")
        ledger = json.loads(args.ledger.read_text(encoding="utf-8")); records = select_records(ledger)
        selected = SLOTS if args.slot == "all" else (args.slot,)
        results = [process_unit(slot, records[slot], args, config, roots, tools, args.stop_after)
                   for slot in selected]
        if args.slot == "all":
            detail = {"schema_version": 1, "record_type": "pilot_p3_governed_completion",
                      "classification": LABELS, "stage": "P3", "results": results,
                      "config_sha256": sha256_file(args.config), "tool_revision_sha256": tool_revision_hash(config)}
            detail_path = roots["detail"] / "completion.json"; atomic_json(detail_path, detail)
            aggregate = aggregate_results(results, config, detail_path, args.log)
            atomic_json(roots["detail"] / "aggregate.json", aggregate)
            inventory_path = roots["detail"] / "checksum_inventory.json"
            records = [item for item in directory_inventory(roots["detail"])
                       if item["relative"] != "checksum_inventory.json"]
            inventory = {"records": records, "important_logs": []}
            if args.log and args.log.is_file():
                inventory["important_logs"].append({"opaque_role": "idempotence_log",
                    "bytes": args.log.stat().st_size, "sha256": sha256_file(args.log)})
            atomic_json(inventory_path, inventory)
            print(json.dumps(aggregate, sort_keys=True))
        else:
            print(json.dumps({"status": "slot_complete", "slot": args.slot,
                              "cache_hits": results[0]["cache_hits"],
                              "stages_executed": results[0]["stages_executed"]}, sort_keys=True))
        return 0
    except ControlledStop:
        print(json.dumps({"status": "controlled_stop", "error": "ControlledStop"}, sort_keys=True))
        return 75
    except Exception as error:
        print(json.dumps({"status": "p3_incomplete", "error": type(error).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
