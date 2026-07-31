"""Compile a manual EpisodeIntent into a deterministic ResolvedEpisodeSpec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import write_json
from .spec_kernel import CalibrationPolicy, compile_episode_intent


def compile_file(
    intent_path: Path,
    protocol_path: Path,
    calibration_path: Path,
    asset_registry_path: Path,
    output_path: Path,
) -> dict:
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assets = json.loads(asset_registry_path.read_text(encoding="utf-8"))
    calibration = CalibrationPolicy.load(calibration_path)
    resolved = compile_episode_intent(
        intent, protocol, calibration, asset_registry=assets
    )
    write_json(output_path, resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("intent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/childlens_mimo_molmospaces_spec_kernel.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/childlens_simulator_bridge.json"),
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("configs/childlens_mimo_molmospaces_asset_registry.json"),
    )
    args = parser.parse_args()
    resolved = compile_file(
        args.intent, args.protocol, args.calibration, args.assets, args.output
    )
    print(json.dumps({"spec_sha256": resolved["spec_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
