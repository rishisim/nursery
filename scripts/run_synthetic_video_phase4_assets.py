#!/usr/bin/env python3
"""Governed Phase 4 allocation, selective acquisition, sealing, and verification.

Restricted paths, identifiers, text, and row-level manifests are written only
below the caller-provided private root. Stdout contains compact aggregates.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/synthetic_video_phase4_assets.json"
TOKEN_SERVICE = "ChildLens-v1.2-Keeper-Repo-Token"
TOKEN_ACCOUNT = "childlens-v1.2-read-only"
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~+/=-]{16,1024}$")


class Phase4Error(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise Phase4Error("E_PRIVATE_MODE")


def write_private(path: Path, value: object) -> None:
    private_dir(path.parent)
    payload = canonical(value) + b"\n"
    pending = path.parent / f".pending-{secrets.token_hex(8)}"
    pending.write_bytes(payload)
    os.chmod(pending, 0o600)
    os.replace(pending, path)


def locate_catalog(mount: Path, expected_hash: str) -> Path:
    matches = [p for p in mount.rglob("*.json") if p.is_file() and file_digest(p) == expected_hash]
    if not matches:
        raise Phase4Error("E_CATALOG")
    return matches[0]


def calibration_children(mount: Path) -> set[str]:
    children: set[str] = set()
    for name in ("restricted_development_manifest.json", "restricted_measurement_manifest.json"):
        for path in mount.rglob(name):
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for row in value.get("items", []) if isinstance(value, dict) else []:
                child = row.get("participant_key") if isinstance(row, dict) else None
                if isinstance(child, str):
                    children.add(child)
    return children


def deal_children(eligible: list[str], key: bytes, allocation: dict[str, object]) -> dict[str, list[str]]:
    study_id = str(allocation["study_id_utf8"]).encode("utf-8")
    ranked = sorted(eligible, key=lambda child: hmac.digest(key, study_id + child.encode("utf-8"), "sha256"))
    counts = allocation["counts"]
    assignments = {role: [] for role in counts}
    deal_order = allocation["deal_order"]
    role_index = 0
    for child in ranked:
        while len(assignments[deal_order[role_index]]) >= counts[deal_order[role_index]]:
            role_index = (role_index + 1) % len(deal_order)
        assignments[deal_order[role_index]].append(child)
        role_index = (role_index + 1) % len(deal_order)
    if {role: len(items) for role, items in assignments.items()} != counts:
        raise Phase4Error("E_ALLOCATION_COUNTS")
    return assignments


def allocate(mount: Path, private_root: Path) -> dict[str, object]:
    cfg = json.loads(CONFIG.read_text())
    catalog_path = locate_catalog(mount, cfg["catalog_sha256"])
    catalog = json.loads(catalog_path.read_text())
    all_children = sorted({row["participant_key"] for row in catalog["media"]})
    calibration = calibration_children(mount)
    if len(calibration) != 18 or len(all_children) != 58 or not calibration <= set(all_children):
        raise Phase4Error("E_CHILD_INVENTORY")
    eligible = sorted(set(all_children) - calibration)
    if len(eligible) != 40:
        raise Phase4Error("E_ELIGIBLE_INVENTORY")
    private_dir(private_root)
    key_path = private_root / "allocation.key"
    if not key_path.exists():
        key_path.write_bytes(secrets.token_bytes(cfg["allocation"]["secret_bytes"]))
        os.chmod(key_path, 0o600)
    key = key_path.read_bytes()
    counts = cfg["allocation"]["counts"]
    assignments = deal_children(eligible, key, cfg["allocation"])
    media = catalog["media"]
    objects = {row["object_key"]: row for row in catalog["objects"]}
    role_sets = {role:set(children) for role,children in assignments.items()}
    role_media = {role:[] for role in assignments}
    for row in media:
        for role,children in role_sets.items():
            if row["participant_key"] in children:
                obj = objects[row["object_key"]]
                role_media[role].append({**row,"source_locator":obj["source_locator"]})
                break
    evaluation_media = role_media["evaluation"]
    previous_path = private_root / "restricted_allocation.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    ledger = {
        "schema_version": 1,
        "catalog_sha256": cfg["catalog_sha256"],
        "config_sha256": file_digest(CONFIG),
        "calibration_children": sorted(calibration),
        "assignments": assignments,
        "role_media": role_media,
        "evaluation_media": evaluation_media,
    }
    overlap = None
    if previous is not None:
        prior_assignments = previous["assignments"]
        overlap = {
            new_role: {old_role: len(set(assignments[new_role]) & set(prior_assignments[old_role])) for old_role in counts}
            for new_role in counts
        }
        superseded = private_root / "superseded_allocation.json"
        if not superseded.exists():
            write_private(superseded, previous)
    write_private(previous_path, ledger)
    return {
        "status": "PASS",
        "calibration_children": len(calibration),
        "eligible_children": len(eligible),
        "training_children": len(assignments["training"]),
        "evaluation_children": len(assignments["evaluation"]),
        "validation_children": len(assignments["validation"]),
        "evaluation_recordings": len(evaluation_media),
        "role_recording_counts": {role:len(rows) for role,rows in role_media.items()},
        "old_new_role_overlap_counts": overlap,
        "allocation_commitment": digest(ledger),
    }


def read_token() -> str:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", TOKEN_SERVICE, "-a", TOKEN_ACCOUNT, "-w"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    token = result.stdout.rstrip(b"\r\n").decode("utf-8") if result.returncode == 0 else ""
    if not TOKEN_PATTERN.fullmatch(token):
        raise Phase4Error("E_TOKEN")
    return token


class Client:
    def __init__(self, cfg: dict[str, object], token: str):
        self.api = str(cfg["api_base_url"]).rstrip("/")
        self.allowed = {urllib.parse.urlsplit(str(x)).netloc for x in cfg["allowed_download_origins"]}
        self.auth = f"Bearer {token}"
        if "repository_id" in cfg:
            self.repo = str(cfg["repository_id"])
        else:
            needle = str(cfg["repository_name_contains_casefold"]).casefold()
            with self.request(f"{self.api}/api2/repos/") as response:
                repositories = json.loads(response.read(4 * 1024 * 1024))
            matches = [row["id"] for row in repositories if needle in str(row.get("name", "")).casefold()]
            if len(matches) != 1:
                raise Phase4Error("E_REPOSITORY_DISCOVERY")
            self.repo = matches[0]

    def request(self, url: str):
        req = urllib.request.Request(url, headers={"Authorization": self.auth, "Accept-Encoding": "identity"})
        return urllib.request.urlopen(req, timeout=120)

    def api_value(self, endpoint: str, locator: str):
        path = locator[len("/ChildLens"):] if locator.startswith("/ChildLens/") else locator
        query = urllib.parse.urlencode({"p": path, "reuse": "1"})
        url = f"{self.api}/api2/repos/{urllib.parse.quote(self.repo, safe='')}/{endpoint}/?{query}"
        with self.request(url) as response:
            return json.loads(response.read(1024 * 1024))

    def metadata(self, locator: str) -> int:
        value = self.api_value("file/detail", locator)
        size = value.get("size") if isinstance(value, dict) else None
        if not isinstance(size, int) or size <= 0:
            raise Phase4Error("E_REMOTE_SIZE")
        return size

    def download(self, locator: str, target: Path, expected: int) -> str:
        link = self.api_value("file", locator)
        if not isinstance(link, str) or urllib.parse.urlsplit(link).netloc not in self.allowed:
            raise Phase4Error("E_DOWNLOAD_ORIGIN")
        h = hashlib.sha256(); size = 0
        pending = target.parent / f".pending-{secrets.token_hex(8)}"
        with self.request(link) as response, pending.open("wb") as handle:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                size += len(block); h.update(block); handle.write(block)
        if size != expected:
            pending.unlink(missing_ok=True)
            raise Phase4Error("E_DOWNLOAD_SIZE")
        os.chmod(pending, 0o600); os.replace(pending, target)
        return h.hexdigest()


def acquire(private_root: Path, transfer_config: Path, workers: int) -> dict[str, object]:
    ledger = json.loads((private_root / "restricted_allocation.json").read_text())
    cfg = json.loads(transfer_config.read_text())
    client = Client(cfg, read_token())
    media_root = private_root / "evaluation_media"; private_dir(media_root)
    checkpoint_path = private_root / "restricted_acquisition.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"items": {}}
    pending_rows = []
    for index, row in enumerate(ledger["evaluation_media"]):
        key = digest({"participant_key": row["participant_key"], "media_key": row["media_key"]})
        if checkpoint["items"].get(key, {}).get("status") == "COMPLETE":
            continue
        pending_rows.append((index, row, key))

    def fetch(item):
        index, row, key = item
        size = client.metadata(row["source_locator"])
        target = media_root / f"{key}.bin"
        sha = client.download(row["source_locator"], target, size)
        return key, {"status": "COMPLETE", "size_bytes": size, "sha256": sha, "file": target.name}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, item) for item in pending_rows]
        for future in as_completed(futures):
            key, result = future.result()
            checkpoint["items"][key] = result
            write_private(checkpoint_path, checkpoint)
    completed = sum(x.get("status") == "COMPLETE" for x in checkpoint["items"].values())
    return {"status": "PASS", "evaluation_recordings": len(ledger["evaluation_media"]), "completed": completed, "bytes": sum(x["size_bytes"] for x in checkpoint["items"].values()), "acquisition_commitment": digest(checkpoint)}


def audit_overlap(mount: Path, private_root: Path) -> dict[str, object]:
    cfg = json.loads(CONFIG.read_text())
    catalog = json.loads(locate_catalog(mount, cfg["catalog_sha256"]).read_text())
    allocation_path = private_root / "restricted_allocation.json"
    allocation = json.loads(allocation_path.read_text())
    calibration = calibration_children(mount)
    child_role = {child: "calibration" for child in calibration}
    for role, children in allocation["assignments"].items():
        for child in children:
            if child in child_role:
                raise Phase4Error("E_CHILD_ROLE_OVERLAP")
            child_role[child] = role
    objects = {row["object_key"]: row for row in catalog["objects"]}
    role_order = cfg["overlap_audit"]["earliest_role_order"]
    priority = {role: index for index, role in enumerate(role_order)}
    rows = []
    for media in catalog["media"]:
        obj = objects[media["object_key"]]
        rows.append({
            "role": child_role[media["participant_key"]], "media_key": media["media_key"],
            "object_key": media["object_key"], "session_key": media["session_key"],
            "duration_milliseconds": media["duration_milliseconds"],
            "content_sha256": obj.get("source_checksum_sha256") or obj.get("local_sha256"),
        })
    quarantine = set()
    pre_counts = {"object_key": 0, "session_time_interval": 0, "content_sha256": 0}
    for field in ("object_key", "session_key", "content_sha256"):
        groups = {}
        for row in rows:
            value = row[field]
            if value:
                groups.setdefault(value, []).append(row)
        label = "session_time_interval" if field == "session_key" else field
        for members in groups.values():
            roles = {row["role"] for row in members}
            if len(roles) < 2:
                continue
            pre_counts[label] += 1
            earliest = min(roles, key=priority.__getitem__)
            quarantine.update(row["media_key"] for row in members if row["role"] != earliest)
    quarantined_by_role = {role: sum(row["media_key"] in quarantine and row["role"] == role for row in rows) for role in role_order}
    audit = {
        "schema_version": 1, "status": "PASS", "earliest_role_order": role_order,
        "pre_resolution_cross_role_group_counts": pre_counts,
        "quarantined_media_keys": sorted(quarantine), "quarantined_by_role": quarantined_by_role,
        "post_resolution_unresolved_exact_or_temporal_count": 0,
        "perceptual_hash": {"status": "PENDING_STAGED_MEDIA_DIAGNOSTIC", "blocking": False, "hamming_max": cfg["overlap_audit"]["perceptual_hash_hamming_max_diagnostic"]},
        "embedding": {"status": "PENDING_PUBLIC_MODEL_QUALIFICATION", "blocking": False},
    }
    write_private(private_root / "restricted_overlap_audit.json", audit)
    allocation["quarantined_media_keys"] = sorted(quarantine)
    allocation["evaluation_media"] = [row for row in allocation["evaluation_media"] if row["media_key"] not in quarantine]
    write_private(allocation_path, allocation)
    return {
        "status": "PASS", "pre_resolution_cross_role_group_counts": pre_counts,
        "quarantined_by_role": quarantined_by_role,
        "post_resolution_unresolved_exact_or_temporal_count": 0,
        "allocation_commitment_after_resolution": digest(allocation),
    }


def stage(mount: Path, private_root: Path) -> dict[str, object]:
    plan = checkpoint = None
    for path in mount.rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        items = value.get("items", []) if isinstance(value, dict) else []
        if len(items) != 58 or not items:
            continue
        keys = set(items[0])
        if {"participant_key", "source_object_key", "media_key"} <= keys:
            plan = value
        if {"source_object_key", "clips", "status"} <= keys:
            checkpoint = value
    if plan is None or checkpoint is None:
        raise Phase4Error("E_CALIBRATION_MANIFEST")
    participant = {row["source_object_key"]: row for row in plan["items"]}
    media_by_signature: dict[tuple[int, str], Path] = {}
    expected = {(clip["bytes"], clip["sha256"]) for row in checkpoint["items"] for clip in row["clips"]}
    expected_sizes = {x[0] for x in expected}
    for path in mount.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink() and path.stat().st_size in expected_sizes:
                sha = file_digest(path)
                if (path.stat().st_size, sha) in expected:
                    media_by_signature[(path.stat().st_size, sha)] = path
        except OSError:
            continue
    calibration_root = private_root / "stage/calibration"
    private_dir(calibration_root)
    calibration = []
    index = 0
    for source in checkpoint["items"]:
        meta = participant[source["source_object_key"]]
        for clip in source["clips"]:
            src = media_by_signature.get((clip["bytes"], clip["sha256"]))
            if src is None:
                raise Phase4Error("E_CALIBRATION_CLIP")
            target = calibration_root / f"{index:04d}.bin"
            if not target.exists():
                shutil.copyfile(src, target); os.chmod(target, 0o600)
            calibration.append({"asset_key": digest([source["source_object_key"], clip["ordinal"]]), "child_key": meta["participant_key"], "session_key": meta["media_key"], "file": str(target.relative_to(private_root))})
            index += 1
    allocation = json.loads((private_root / "restricted_allocation.json").read_text())
    acquisition = json.loads((private_root / "restricted_acquisition.json").read_text())
    evaluation = []
    quarantine = set(allocation.get("quarantined_media_keys", []))
    for row in allocation["evaluation_media"]:
        if row["media_key"] in quarantine:
            continue
        key = digest({"participant_key": row["participant_key"], "media_key": row["media_key"]})
        item = acquisition["items"].get(key)
        if not item or item.get("status") != "COMPLETE":
            raise Phase4Error("E_EVALUATION_ACQUISITION")
        evaluation.append({"asset_key": key, "child_key": row["participant_key"], "session_key": row["session_key"], "file": f"evaluation_media/{item['file']}"})
    manifest = {"schema_version": 1, "calibration": calibration, "evaluation": evaluation}
    write_private(private_root / "restricted_stage_manifest.json", manifest)
    return {"status": "PASS", "calibration_clips": len(calibration), "evaluation_recordings": len(evaluation), "stage_commitment": digest(manifest)}


def verify_acquisition(private_root: Path) -> dict[str, object]:
    allocation=json.loads((private_root/"restricted_allocation.json").read_text())
    acquisition=json.loads((private_root/"restricted_acquisition.json").read_text())
    verified=[]
    for row in allocation["evaluation_media"]:
        key=digest({"participant_key":row["participant_key"],"media_key":row["media_key"]})
        item=acquisition["items"].get(key)
        if not item or item.get("status")!="COMPLETE": raise Phase4Error("E_EVALUATION_ACQUISITION")
        media_path=private_root/"evaluation_media"/item["file"]
        if not media_path.is_file() or media_path.stat().st_size!=item["size_bytes"] or file_digest(media_path)!=item["sha256"]:
            raise Phase4Error("E_EVALUATION_VERIFY")
        verified.append({"asset_key":key,"size_bytes":item["size_bytes"],"sha256":item["sha256"]})
    record={"schema_version":1,"status":"PASS","items":verified}
    write_private(private_root/"restricted_acquisition_verification.json",record)
    return {"status":"PASS","verified_recordings":len(verified),"verified_bytes":sum(x["size_bytes"] for x in verified),"verification_commitment":digest(record)}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("allocate"); a.add_argument("--mount", type=Path, required=True); a.add_argument("--private-root", type=Path, required=True)
    q = sub.add_parser("acquire"); q.add_argument("--private-root", type=Path, required=True); q.add_argument("--transfer-config", type=Path, required=True); q.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    o = sub.add_parser("audit-overlap"); o.add_argument("--mount", type=Path, required=True); o.add_argument("--private-root", type=Path, required=True)
    v = sub.add_parser("verify-acquisition"); v.add_argument("--private-root", type=Path, required=True)
    s = sub.add_parser("stage"); s.add_argument("--mount", type=Path, required=True); s.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "allocate":
        result = allocate(args.mount.resolve(), args.private_root.resolve())
    elif args.command == "acquire":
        result = acquire(args.private_root.resolve(), args.transfer_config.resolve(), args.workers)
    elif args.command == "audit-overlap":
        result = audit_overlap(args.mount.resolve(), args.private_root.resolve())
    elif args.command == "verify-acquisition":
        result = verify_acquisition(args.private_root.resolve())
    else:
        result = stage(args.mount.resolve(), args.private_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
