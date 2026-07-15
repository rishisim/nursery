from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.manifest import (
    build_balanced_subset_plan,
    load_safe_manifest,
    manifest_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely inspect an AEA link manifest and lock the balanced 40-sequence plan."
    )
    parser.add_argument("--links", required=True, help="Expiring AEA download-links JSON (never copied).")
    parser.add_argument("--out", default="configs/aea_subset_40.yaml")
    parser.add_argument(
        "--annotations-root", default="data/aea_raw",
        help="Optional extracted speech/metadata root used only for predeclared action-anchor support.",
    )
    args = parser.parse_args()

    manifest = load_safe_manifest(args.links)
    annotations = Path(args.annotations_root)
    plan = build_balanced_subset_plan(
        manifest, annotations if annotations.is_dir() else None
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(plan, sort_keys=False))
    print(json.dumps({
        "safe_manifest_summary": manifest_summary(manifest),
        "plan_path": str(out),
        "selection": plan["selection"],
        "signed_urls_loaded_into_output": False,
    }, indent=2))


if __name__ == "__main__":
    main()
