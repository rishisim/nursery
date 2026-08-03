"""Focused validator for the native MuJoCo + Filament truth-control episode."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main(output_root: Path) -> int:
    rows = list(csv.DictReader((output_root / "synchronized_trace.csv").open()))
    if len(rows) != 500:
        raise SystemExit(f"expected 500 synchronized samples, found {len(rows)}")
    times = [float(row["time_s"]) for row in rows]
    if any(abs((b - a) - 0.032) > 1e-9 for a, b in zip(times, times[1:])):
        raise SystemExit("fixed sample clock is not 31.25 Hz")
    contact_rows = [row for row in rows if row["left_contact"] == "1" and row["right_contact"] == "1"]
    if not contact_rows:
        raise SystemExit("no bilateral target contact")
    min_penetration_m = min(
        float(row[key]) for row in contact_rows for key in ("left_distance", "right_distance")
    )
    object_z = [float(row["target_z"]) for row in rows]
    lift_m = max(object_z) - object_z[0]
    if lift_m < 0.04 or object_z[-1] > object_z[0] + 0.02:
        raise SystemExit(f"physical lift/release gate failed: lift={lift_m}, final_z={object_z[-1]}")
    if any(row["object_id"] != "target_block_001" for row in rows):
        raise SystemExit("object identity did not persist")
    source = Path(__file__).with_name("native_filament_smoke.cpp").read_text()
    forbidden = ["<equality", "<weld", "<connect", "mocap=\"true\"", "xfrc_applied"]
    if any(token in source for token in forbidden):
        raise SystemExit("forbidden attachment/assist mechanism present")
    manifest = json.loads((output_root / "stream_manifest.json").read_text())
    if manifest["same_machine_replay_maximum_error"] != 0:
        raise SystemExit("same-machine replay was not exact")
    evidence = {
        "pass": True,
        "clock_samples": len(rows),
        "bilateral_contact_samples": len(contact_rows),
        "first_bilateral_contact_s": float(contact_rows[0]["time_s"]),
        "maximum_target_lift_m": lift_m,
        "release": True,
        "persistent_identity": "target_block_001",
        "same_machine_replay_maximum_error": manifest["same_machine_replay_maximum_error"],
        "minimum_contact_distance_m": min_penetration_m,
        "penetration_diagnostic": "repair" if min_penetration_m < -0.005 else "pass",
        "rendered_channels": {name: len(list((output_root / name).glob("*.ppm"))) for name in ("rgb", "depth", "object_id")},
        "no_attachment_or_assist": True,
    }
    (output_root / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
