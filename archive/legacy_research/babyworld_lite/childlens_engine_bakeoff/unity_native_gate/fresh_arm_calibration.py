from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def run_candidate(unity: Path, project: Path, scene: Path, root: Path, index: int, targets: list[float]) -> dict:
    output = root / "evaluations" / f"candidate_{index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(UNITY_NATIVE_GATE_OUTPUT=str(output), UNITY_NATIVE_SCENE_SPEC=str(scene), UNITY_NATIVE_ARM_TARGETS=",".join(f"{x:.6f}" for x in targets))
    command = [str(unity), "-batchmode", "-nographics", "-quit", "-projectPath", str(project), "-executeMethod", "UnitySkinRegistrationBuilder.RunFreshArmCandidate", "-logFile", str(output / "unity.log")]
    result = subprocess.run(command, env=env, check=False)
    receipt_path = output / "fresh_candidate.json"
    if result.returncode or not receipt_path.exists():
        raise RuntimeError(f"fresh candidate {index} failed with Unity exit {result.returncode}")
    receipt = json.loads(receipt_path.read_text())
    receipt["evaluation_id"] = index
    return receipt


def calibrate(unity: Path, project: Path, scene: Path, output: Path) -> dict:
    current = [0.0] * 7
    evaluations: list[dict] = []
    accepted: list[dict] = []
    initial_hash: str | None = None

    def evaluate(targets: list[float]) -> dict:
        nonlocal initial_hash
        receipt = run_candidate(unity, project, scene, output, len(evaluations), targets)
        evaluations.append(receipt)
        if initial_hash is None:
            initial_hash = receipt["initial_state_hash"]
        elif receipt["initial_state_hash"] != initial_hash:
            raise RuntimeError("fresh candidate initial-state hash mismatch")
        return receipt

    for increment in (20.0, 5.0):
        for dof in range(7):
            baseline = evaluate(current.copy())
            plus_targets, minus_targets = current.copy(), current.copy()
            plus_targets[dof] += increment
            minus_targets[dof] -= increment
            plus, minus = evaluate(plus_targets), evaluate(minus_targets)
            best_targets, best = min(((current.copy(), baseline), (plus_targets, plus), (minus_targets, minus)), key=lambda row: row[1]["final_error_m"])
            if best["final_error_m"] + 0.001 < baseline["final_error_m"]:
                confirmation = evaluate(best_targets)
                if confirmation["final_error_m"] + 0.001 < baseline["final_error_m"] and abs(confirmation["final_error_m"] - best["final_error_m"]) <= 0.002:
                    current = best_targets
                    accepted.append({"increment_deg": increment, "dof": dof, "baseline_evaluation": baseline["evaluation_id"], "candidate_evaluation": best["evaluation_id"], "confirmation_evaluation": confirmation["evaluation_id"], "baseline_error_m": baseline["final_error_m"], "candidate_error_m": best["final_error_m"], "confirmation_error_m": confirmation["final_error_m"], "targets_deg": current.copy()})

    final = evaluate(current.copy())
    aggregate = {"schema": "embodied.unity_native.fresh_process_arm_calibration.v1", "scene_spec_path": str(scene), "fixed_steps": 480, "fixed_hz": 240, "initial_state_hash": initial_hash, "all_initial_hashes_identical": True, "coarse_fine_increments_deg": [20.0, 5.0], "selected_dofs": ["shoulder_x", "shoulder_y", "shoulder_z", "elbow_x", "wrist_x", "wrist_y", "wrist_z"], "frozen_targets_deg": current, "accepted_updates": accepted, "final_evaluation_id": final["evaluation_id"], "final_error_m": final["final_error_m"], "precontact_threshold_m": 0.03, "passed": final["final_error_m"] <= 0.03, "evaluation_count": len(evaluations)}
    encoded = json.dumps(aggregate, indent=2) + "\n"
    aggregate["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    (output / "fresh_arm_calibration.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(calibrate(args.unity, args.project, args.scene, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
