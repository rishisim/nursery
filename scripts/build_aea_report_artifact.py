from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.report_artifact import write_report_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical artifact JSON for the portable AEA technical report."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--preprocess")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.results).read_text())
    preprocess = json.loads(Path(args.preprocess).read_text()) if args.preprocess else None
    artifact, notes = write_report_artifact(result, preprocess, args.out)
    print(json.dumps({"artifact": str(artifact), "source_notes": str(notes)}, indent=2))


if __name__ == "__main__":
    main()
