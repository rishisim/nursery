"""Focused validator for the native MuJoCo + Filament truth-control episode."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def main(output_root: Path) -> int:
    rows = list(csv.DictReader((output_root / "synchronized_trace.csv").open()))
    if len(rows) != 500:
        raise SystemExit(f"expected 500 synchronized samples, found {len(rows)}")
    times = [float(row["time_s"]) for row in rows]
    if any(abs((b - a) - 0.032) > 1e-9 for a, b in zip(times, times[1:])):
        raise SystemExit("fixed sample clock is not 31.25 Hz")
    bilateral = [row for row in rows if row["left_contact"] == "1" and row["right_contact"] == "1"]
    if not bilateral:
        raise SystemExit("no bilateral target contact")
    hold = [row for row in rows if 6.6 <= float(row["time_s"]) <= 11.46]
    if not hold or any(row not in bilateral for row in hold):
        raise SystemExit("bilateral support did not persist through the declared hold window")
    if min(float(row["target_z"]) for row in hold) < 0.40:
        raise SystemExit("target was not elevated throughout the declared hold window")
    post_release = [row for row in rows if float(row["time_s"]) >= 11.524]
    if any(row["left_contact"] == "1" and row["right_contact"] == "1" for row in post_release):
        raise SystemExit("bilateral contact persisted after the 11.5 s opening command")
    if float(post_release[-1]["target_z"]) >= float(post_release[0]["target_z"]) - 0.02:
        raise SystemExit("target did not descend after the release command")
    min_penetration_m = min(
        float(row[key]) for row in bilateral for key in ("left_distance", "right_distance")
    )
    table_penetration_m = min(float(row["target_table_distance"]) for row in rows)
    if min_penetration_m < -0.005 or table_penetration_m < -0.005:
        raise SystemExit(
            f"penetration gate failed: finger={min_penetration_m}, table={table_penetration_m}"
        )
    object_z = [float(row["target_z"]) for row in rows]
    lift_m = max(object_z) - object_z[0]
    if lift_m < 0.20 or object_z[-1] > object_z[0] + 0.02:
        raise SystemExit(f"physical lift/release gate failed: lift={lift_m}, final_z={object_z[-1]}")
    yaw = [math.atan2(float(row["camera_r10"]), float(row["camera_r00"])) for row in rows]
    yaw_range_deg = math.degrees(max(yaw) - min(yaw))
    if yaw_range_deg < 10.0:
        raise SystemExit(f"head-derived reorientation is not visible: {yaw_range_deg} degrees")
    shake = [float(row["target_x"]) for row in rows if 7.2 <= float(row["time_s"]) <= 9.2]
    if max(shake) - min(shake) < 0.03:
        raise SystemExit("inspect shake is not visually legible")
    if any(row["object_id"] != "target_block_001" for row in rows):
        raise SystemExit("object identity did not persist")
    source = Path(__file__).with_name("native_filament_smoke.cpp").read_text()
    forbidden = ["<equality", "<weld", "<connect", "mocap=\"true\"", "xfrc_applied"]
    if any(token in source for token in forbidden):
        raise SystemExit("forbidden attachment/assist mechanism present")
    manifest = json.loads((output_root / "stream_manifest.json").read_text())
    if manifest["same_machine_replay_maximum_error"] != 0 or "std::memcmp(&a, &b, sizeof(TraceSample))" not in source:
        raise SystemExit("same-machine replay was not exact")
    rendered_channels = {name: len(list((output_root / name).glob("*.ppm"))) for name in ("rgb", "depth", "object_id")}
    if any(count != 500 for count in rendered_channels.values()):
        raise SystemExit(f"rendered channel counts are not exactly 500: {rendered_channels}")
    evidence = {
        "pass": True,
        "clock_samples": len(rows),
        "bilateral_contact_samples": len(bilateral),
        "first_bilateral_contact_s": float(bilateral[0]["time_s"]),
        "last_bilateral_contact_s": float(bilateral[-1]["time_s"]),
        "hold_window": [6.6, 11.46],
        "maximum_target_lift_m": lift_m,
        "release": True,
        "persistent_identity": "target_block_001",
        "same_machine_replay_maximum_error": manifest["same_machine_replay_maximum_error"],
        "minimum_finger_target_distance_m": min_penetration_m,
        "minimum_target_table_distance_m": table_penetration_m,
        "penetration_diagnostic": "pass",
        "camera_yaw_range_deg": yaw_range_deg,
        "inspect_shake_x_range_m": max(shake) - min(shake),
        "rendered_channels": rendered_channels,
        "no_attachment_or_assist": True,
    }
    (output_root / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
