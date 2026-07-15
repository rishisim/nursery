from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.request
import zipfile

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.manifest import MPS_COMPONENT_PREFIX, load_safe_manifest


CHUNK_BYTES = 4 * 1024 * 1024


def _load_private_manifest(path: str | Path) -> Mapping[str, Any]:
    # This is the only acquisition boundary that retains signed URLs.  They are
    # never returned from a reporting function or written to the repository.
    with Path(path).open() as handle:
        value = json.load(handle)
    if "sequences" not in value or "sequence_config" not in value:
        raise ValueError("invalid AEA download-links JSON")
    return value


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _open_resumable(url: str, offset: int):
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(request, timeout=120)


def _download(entry: Mapping[str, Any], part: Path, label: str) -> None:
    expected = int(entry["file_size_bytes"])
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected:
        part.unlink()
        offset = 0
    if offset == expected:
        if _sha1(part) == str(entry["sha1sum"]):
            return
        part.unlink()
        offset = 0
    try:
        response = _open_resumable(str(entry["download_url"]), offset)
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        last_report = time.monotonic()
        with response, part.open(mode) as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                if time.monotonic() - last_report > 30:
                    fraction = part.stat().st_size / max(expected, 1)
                    print(f"{label}: {fraction:.1%} ({part.stat().st_size / 1024**3:.2f} GiB)")
                    last_report = time.monotonic()
    except Exception:
        # Never include an exception string: urllib errors may echo the signed URL.
        raise RuntimeError(f"download failed for {label}; signed URL suppressed") from None
    if part.stat().st_size != expected:
        raise RuntimeError(
            f"size mismatch for {label}: got {part.stat().st_size}, expected {expected}"
        )
    if _sha1(part) != str(entry["sha1sum"]):
        raise RuntimeError(f"checksum mismatch for {label}")


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("unsafe path in downloaded archive")
        handle.extractall(destination)


def _install_component(
    entry: Mapping[str, Any],
    component: str,
    sequence_dir: Path,
    recording_filename: str,
    label: str,
) -> None:
    source_name = str(entry["filename"])
    if component == "annotations" and all(
        (sequence_dir / filename).is_file() for filename in ("metadata.json", "speech.csv")
    ):
        print(f"{label}: extracted files already present")
        return
    if component == "main_vrs":
        destination = sequence_dir / recording_filename
    else:
        destination = sequence_dir / source_name
    if destination.is_file() and destination.stat().st_size == int(entry["file_size_bytes"]):
        if _sha1(destination) == str(entry["sha1sum"]):
            print(f"{label}: already verified")
            return
    part = sequence_dir / f".{source_name}.part"
    _download(entry, part, label)
    if zipfile.is_zipfile(part):
        _safe_extract(part, sequence_dir)
        part.unlink()
    else:
        os.replace(part, destination)
    print(f"{label}: complete")


def _required_bytes(
    raw: Mapping[str, Any], sequences: Sequence[str], components: Sequence[str]
) -> int:
    return sum(
        int(raw["sequences"][name][component]["file_size_bytes"])
        for name in sequences for component in components
    )


def _remaining_bytes(
    raw: Mapping[str, Any],
    sequences: Sequence[str],
    components: Sequence[str],
    output: Path,
) -> int:
    remaining = 0
    recording_filename = str(raw["sequence_config"]["main"]["recording"])
    for name in sequences:
        sequence_dir = output / name
        for component in components:
            entry = raw["sequences"][name][component]
            expected = int(entry["file_size_bytes"])
            if component == "annotations" and all(
                (sequence_dir / filename).is_file() for filename in ("metadata.json", "speech.csv")
            ):
                continue
            destination = sequence_dir / (
                recording_filename if component == "main_vrs" else str(entry["filename"])
            )
            if destination.is_file() and destination.stat().st_size == expected:
                continue
            part = sequence_dir / f".{entry['filename']}.part"
            partial = min(part.stat().st_size, expected) if part.is_file() else 0
            remaining += expected - partial
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a license-authorized selective AEA subset without logging signed URLs."
    )
    parser.add_argument("--links", required=True, help="User-supplied expiring AEA links JSON.")
    parser.add_argument("--plan", default="configs/aea_subset_40.yaml")
    parser.add_argument("--out", default="data/aea_raw")
    parser.add_argument(
        "--components", nargs="+", default=["annotations", "main_vrs"],
        help="Initial recommendation: annotations main_vrs (no MPS).",
    )
    parser.add_argument(
        "--all-annotations", action="store_true",
        help="Acquire tiny annotations for all 143 recordings while limiting other components to the plan.",
    )
    parser.add_argument("--allow-mps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-license-accepted-by-user", action="store_true",
        help="Acknowledges the user already accepted the license; this script never accepts it.",
    )
    args = parser.parse_args()

    if not args.confirm_license_accepted_by_user:
        raise SystemExit(
            "Refusing acquisition: the user must personally accept the AEA license, then pass "
            "--confirm-license-accepted-by-user."
        )
    if any(component.startswith(MPS_COMPONENT_PREFIX) for component in args.components) and not args.allow_mps:
        raise SystemExit("MPS components require explicit --allow-mps; they are excluded initially.")

    safe = load_safe_manifest(args.links)
    plan = yaml.safe_load(Path(args.plan).read_text())
    if (
        str(plan.get("dataset", {}).get("name")) != safe.dataset_name
        or str(plan.get("dataset", {}).get("release")) != safe.release
    ):
        raise SystemExit("subset plan dataset/release does not match the supplied link manifest")
    selected = list(plan["selection"]["sequences"])
    if args.all_annotations and set(args.components) == {"annotations"}:
        selected = sorted(safe.sequences)
    unknown = set(selected) - set(safe.sequences)
    if unknown:
        raise SystemExit(f"plan contains unknown sequence ids: {sorted(unknown)}")
    available = {key for row in safe.sequences.values() for key in row.assets}
    if set(args.components) - available:
        raise SystemExit(f"unknown components: {sorted(set(args.components) - available)}")

    raw = _load_private_manifest(args.links)
    planned = _required_bytes(raw, selected, args.components)
    output = Path(args.out)
    remaining = _remaining_bytes(raw, selected, args.components, output)
    free = shutil.disk_usage(output.resolve().anchor).free
    reserve = 40 * 1024**3
    summary = {
        "sequence_count": len(selected),
        "components": list(args.components),
        "planned_gib": round(planned / 1024**3, 3),
        "remaining_gib": round(remaining / 1024**3, 3),
        "free_gib": round(free / 1024**3, 3),
        "post_download_reserve_gib": 40,
        "mps_included": any(value.startswith(MPS_COMPONENT_PREFIX) for value in args.components),
        "signed_urls_printed_or_copied": False,
    }
    print(json.dumps(summary, indent=2))
    if remaining + reserve > free:
        raise SystemExit("insufficient disk while preserving the 40 GiB reserve")
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=True)
    recording_filename = str(raw["sequence_config"]["main"]["recording"])
    for sequence in selected:
        sequence_dir = output / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        for component in args.components:
            entry = raw["sequences"][sequence][component]
            _install_component(
                entry, component, sequence_dir, recording_filename,
                f"{sequence}/{component}",
            )


if __name__ == "__main__":
    main()
