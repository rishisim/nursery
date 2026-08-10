#!/usr/bin/env python3
"""Read-only, resumable inventory of authorized BabyView media.

The JSONL output is sensitive. The program refuses repository-local output.
Identifiers are never printed; a secret-keyed HMAC supplies stable private keys.
"""
from __future__ import annotations

import argparse, hashlib, hmac, json, os, stat, subprocess, tempfile
from pathlib import Path
import storage

VERSION = "audit_phase1/1"
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

def private_key(secret: bytes, value: str) -> str:
    return hmac.new(secret, value.encode(), hashlib.sha256).hexdigest()

def validate_governed_output(path: Path) -> None:
    for tier in ("durable", "scratch"):
        root = storage.storage_root(tier).resolve()
        if root not in path.parents: continue
        marker = root / storage.MARKER_NAME
        if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o700:
            raise ValueError("governed output root is absent or not owner-only")
        try: observed = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc: raise ValueError("governed output marker is unavailable") from exc
        if observed != storage.expected_marker(tier): raise ValueError("governed output marker does not match policy")
        return
    raise ValueError("ledger must be beneath a configured governed storage root")

def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def probe(path: Path, executable: str) -> tuple[float | None, bool | None, bool | None, str | None]:
    command = [executable, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        audio = any(item.get("codec_type") == "audio" for item in data.get("streams", []))
        return duration, audio, audio, None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return None, None, None, type(exc).__name__

def discover(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.relative_to(root).as_posix())

def atomic_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def load_existing(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists(): return {}
    rows: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"invalid ledger JSON at line {line_number}") from exc
            key = row.get("record_key")
            if not isinstance(key, str) or len(key) != 64: raise ValueError(f"invalid ledger key at line {line_number}")
            rows[key] = row
    return rows

def merged_rows(existing: dict[str, dict[str, object]], current: list[dict[str, object]]) -> list[dict[str, object]]:
    combined = dict(existing); combined.update((str(row["record_key"]), row) for row in current)
    return [combined[key] for key in sorted(combined)]

def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    reasons: dict[str, dict[str, float | int]] = {}
    def seconds(row: dict[str, object]) -> float: return float(row["duration_seconds"] or 0)
    for row in rows:
        reason = str(row["exclusion_reason"])
        slot = reasons.setdefault(reason, {"count": 0, "duration_seconds": 0.0})
        slot["count"] = int(slot["count"]) + 1; slot["duration_seconds"] = float(slot["duration_seconds"]) + seconds(row)
    return {"schema_version": 1, "recordings": len(rows), "duration_seconds": sum(map(seconds, rows)),
            "included_seconds": sum(seconds(r) for r in rows if r["decision"] == "include"),
            "excluded_seconds": sum(seconds(r) for r in rows if r["decision"] == "exclude"),
            "by_exclusion_reason": dict(sorted(reasons.items()))}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("ledger", type=Path)
    parser.add_argument("--secret-file", required=True, type=Path); parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--limit", type=int); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args(); source = args.source.resolve(); ledger = args.ledger.resolve()
    repo = Path(__file__).resolve().parents[2]
    if ledger == repo or repo in ledger.parents: parser.error("ledger output must not be inside the Git repository")
    try: validate_governed_output(ledger)
    except ValueError as exc: parser.error(str(exc))
    secret = args.secret_file.read_bytes()
    if len(secret) < 32: parser.error("secret file must contain at least 32 bytes")
    paths = discover(source); paths = paths[:args.limit] if args.limit is not None else paths
    if args.checkpoint_every < 1: parser.error("--checkpoint-every must be positive")
    existing = {} if args.dry_run else load_existing(ledger)
    rows = []
    for path in paths:
        relative = path.relative_to(source).as_posix(); key = private_key(secret, "record:" + relative)
        if key in existing:
            rows.append(existing[key]); continue
        duration, present, usable, error = probe(path, args.ffprobe)
        rows.append({"record_key": key, "databrary_reference": None,
          "source_filename": relative, "sha256": None if args.dry_run else stream_sha256(path), "duration_seconds": duration,
          "audio_present": present, "audio_usable": usable, "decision": "undetermined", "exclusion_reason": "probe_error" if error else "undetermined",
          "group_key": None, "release_membership": "unverified", "probe_version": VERSION, "probe_error": error})
        if not args.dry_run and len(rows) % args.checkpoint_every == 0: atomic_write(ledger, merged_rows(existing, rows))
    if not args.dry_run: atomic_write(ledger, merged_rows(existing, rows))
    print(json.dumps({"status": "dry-run" if args.dry_run else "written", "recordings": len(rows)}, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
