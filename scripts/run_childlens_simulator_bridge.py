from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.childlens_simulator_bridge.calibration import run_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen ChildLens-calibrated simulator bridge.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = args.contract or args.root / "configs" / "childlens_simulator_bridge.json"
    report = run_validation(args.root, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["decision"])


if __name__ == "__main__":
    main()
