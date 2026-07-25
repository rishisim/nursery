#!/usr/bin/env python3
"""Fetch a bounded Poly Haven CC0 pilot subset into ignored local storage."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tmp" / "childlens_asset_rich" / "assets"
ASSETS = ("rubber_duck_toy", "food_apple_01", "baseball_01", "wooden_bowl_01", "wooden_spoon")
API = "https://api.polyhaven.com/files/{asset}"
AGENT = "nursery-childlens-asset-fetch/1.0"


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request) as source, destination.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = []
    for asset in ASSETS:
        request = urllib.request.Request(API.format(asset=asset), headers={"User-Agent": AGENT})
        with urllib.request.urlopen(request) as response:
            metadata = json.load(response)
        entry = metadata["gltf"]["1k"]["gltf"]
        base = DEST / asset
        files = {"model.gltf": entry}
        files.update(entry.get("include", {}))
        hashes = {}
        for relative, description in files.items():
            destination = base / relative
            hashes[relative] = {
                "sha256": download(description["url"], destination),
                "source_url": description["url"],
                "declared_md5": description["md5"],
            }
        manifest.append({
            "asset_id": asset,
            "source": "Poly Haven official API",
            "license": "CC0",
            "files": hashes,
        })
    out = DEST.parent / "asset_manifest.local.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
