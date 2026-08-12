#!/usr/bin/env python3
"""Read-only Pilot P1 audit for exactly two governed media records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

LABELS = ["engineering_only", "non_comparable", "not_a_reproduction_result"]
SLOTS = ("input_1", "input_2")


class AuditError(RuntimeError):
    pass


def stream_hash(path, chunk_size=1024 * 1024):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def rate(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+/\d+", value):
        return None
    numerator, denominator = map(int, value.split("/"))
    return str(Fraction(numerator, denominator)) if denominator else None


def probe_media(path, executable):
    version_run = subprocess.run([executable, "-version"], check=False, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    version = version_run.stdout.splitlines()[0] if version_run.returncode == 0 else None
    run = subprocess.run(
        [executable, "-v", "error", "-show_error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        payload = json.loads(run.stdout) if run.stdout else {}
    except json.JSONDecodeError:
        payload = {}
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    format_info = payload.get("format", {}) if isinstance(payload, dict) else {}
    try:
        duration = float(format_info.get("duration"))
    except (TypeError, ValueError):
        duration = None
    usable_audio = bool(audio.get("codec_name") and str(audio.get("sample_rate", "")).isdigit()
                        and int(audio["sample_rate"]) > 0 and int(audio.get("channels", 0)) > 0)
    decodable = run.returncode == 0 and bool(video.get("codec_name"))
    return {
        "tool_version": version,
        "returncode": run.returncode,
        "error_status": None if decodable else "media_probe_failed",
        "video_decodable": decodable,
        "duration_seconds": duration,
        "container": format_info.get("format_name"),
        "video_codec": video.get("codec_name"),
        "frame_rate": rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "width": video.get("width"), "height": video.get("height"),
        "audio_present": bool(audio), "usable_audio": usable_audio,
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio["sample_rate"]) if str(audio.get("sample_rate", "")).isdigit() else None,
        "audio_channels": audio.get("channels"),
    }


def select_records(ledger):
    queue = ledger.get("queue")
    if not isinstance(queue, list):
        raise AuditError("governed ledger queue missing")
    records = [item for item in queue if isinstance(item, dict) and
               item.get("status") in {"downloaded", "completed", "verified"}]
    if len(records) != 2:
        raise AuditError("Pilot P1 requires exactly two completed governed records")
    required = {"record_key", "sha256", "size_bytes"}
    if any(not required.issubset(item) for item in records):
        raise AuditError("governed verification fields missing")
    return records


def resolve_media(root, key):
    matches = [item for item in Path(root).rglob(str(key) + ".*") if item.is_file()]
    if len(matches) != 1:
        raise AuditError("opaque governed key did not resolve exactly one media file")
    return matches[0]


def audit(ledger_path, media_root, output_path, run_id, executable, tool_provenance):
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    records, details = select_records(ledger), []
    for slot, record in zip(SLOTS, records):
        media = resolve_media(media_root, record["record_key"])
        size, digest = stream_hash(media)
        details.append({
            "slot": slot,
            "pilot_record_key": hashlib.sha256(
                (run_id + ":" + str(record["record_key"])).encode()
            ).hexdigest()[:24],
            "governed_record_key": record["record_key"],
            "expected_size_bytes": int(record["size_bytes"]), "observed_size_bytes": size,
            "expected_sha256": record["sha256"], "observed_sha256": digest,
            "size_consistent": size == int(record["size_bytes"]),
            "sha256_consistent": digest == record["sha256"],
            "probe": probe_media(media, executable),
        })
    checksums = sum(x["size_consistent"] and x["sha256_consistent"] for x in details)
    videos = sum(x["probe"]["video_decodable"] for x in details)
    audios = sum(x["probe"]["usable_audio"] for x in details)
    passed = checksums == 2 and videos == 2 and audios >= 1
    governed = {
        "schema_version": 1, "record_type": "pilot_p1_governed_detailed_audit",
        "pilot_id": "juno_sample", "engineering_run_id": run_id, "stage": "P1",
        "classification": LABELS, "scientific_status_effect": "none",
        "inventory_status": "incomplete_inventory", "input_count": 2,
        "tool_provenance": tool_provenance, "records": details,
        "gate": {"checksum_consistent_count": checksums, "video_decodable_count": videos,
                 "usable_audio_count": audios, "passed": passed,
                 "stop_reason": None if passed else "p1_media_gate_failed"},
        "p0_status_preserved": True, "later_stage_started": False,
    }
    output_path = Path(output_path)
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("engineering_run_id") != run_id or prior.get("stage") != "P1":
            raise AuditError("refusing to overwrite unrelated governed record")
    atomic_json(output_path, governed)
    record_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    def categories(field):
        return sorted({x["probe"][field] for x in details if x["probe"].get(field) is not None}, key=str)
    versions = categories("tool_version")
    safe_tool_provenance = {key: tool_provenance[key] for key in (
        "source_repository", "release_tag", "asset_name", "asset_sha256",
        "asset_size_bytes", "platform", "license_variant", "ffprobe_sha256",
        "ffprobe_version",
    ) if key in tool_provenance}
    return {
        "schema_version": 1, "record_type": "pilot_p1_privacy_safe_aggregate",
        "pilot_id": "juno_sample", "engineering_run_id": run_id, "stage": "P1",
        "status": "p1_complete" if passed else "p1_incomplete", "classification": LABELS,
        "scientific_status_effect": "none", "inventory_status": "incomplete_inventory",
        "sample_scope": "two_video_engineering_sample_not_full_release_ledger", "input_count": 2,
        "verification": {"juno_owner_mode_check_passed": True,
                         "byte_count_consistent_count": sum(x["size_consistent"] for x in details),
                         "sha256_consistent_count": sum(x["sha256_consistent"] for x in details),
                         "video_decodable_count": videos, "usable_audio_count": audios},
        "aggregate_total_duration_seconds": round(sum(x["probe"]["duration_seconds"] or 0 for x in details), 6),
        "aggregate_categories": {
            "containers": categories("container"), "video_codecs": categories("video_codec"),
            "frame_rates": categories("frame_rate"),
            "dimensions": sorted({"%sx%s" % (x["probe"]["width"], x["probe"]["height"])
                                  for x in details if x["probe"].get("width") and x["probe"].get("height")}),
            "audio_codecs": categories("audio_codec"),
            "audio_sample_rates": categories("audio_sample_rate"),
            "audio_channel_counts": categories("audio_channels"),
        },
        "ffprobe_version": versions[0] if len(versions) == 1 else versions,
        "tool_provenance": safe_tool_provenance,
        "governed_detailed_record": {"opaque_run_id": run_id, "sha256": record_hash,
                                     "owner_only": True},
        "p1_gate": {"passed": passed, "stop_reason": None if passed else "p1_media_gate_failed"},
        "p0_status_preserved": True, "full_corpus_discrepancy": "unresolved_and_out_of_scope",
        "later_stage_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering-run-id", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--tool-provenance", type=Path, required=True)
    args = parser.parse_args()
    try:
        provenance = json.loads(args.tool_provenance.read_text(encoding="utf-8"))
        print(json.dumps(audit(args.ledger, args.media_root, args.output,
                               args.engineering_run_id, args.ffprobe, provenance), sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "p1_incomplete", "error": type(error).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
