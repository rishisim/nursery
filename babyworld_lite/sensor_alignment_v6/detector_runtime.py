from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.sensor_alignment_v2 import detector as frozen_v2_detector

from .protocol import (
    IdentifierFirewall,
    IdentifierReference,
    canonical_digest,
    sha256_file,
)


class DetectorRuntime:
    def __init__(self, model: frozen_v2_detector.SensorEventDetector, file_digest: str):
        if not isinstance(model, frozen_v2_detector.SensorEventDetector):
            raise TypeError("v6 requires the actual frozen v2 SensorEventDetector")
        self.model = model
        self.file_digest = file_digest
        self.model_digest = canonical_digest(model.serializable())
        self.load_calls = 1
        self.inference_calls = 0
        self.call_digests: list[str] = []

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "DetectorRuntime":
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(f"frozen v2 detector digest mismatch: {observed}")
        value = json.loads(Path(path).read_text())
        model = frozen_v2_detector.SensorEventDetector(
            feature_mean=tuple(map(float, value["feature_mean"])),
            feature_scale=tuple(map(float, value["feature_scale"])),
            activity_weights=tuple(map(float, value["activity_weights"])),
            boundary_weights=tuple(map(float, value["boundary_weights"])),
            training_digest=str(value["training_digest"]),
        )
        if canonical_digest(model.serializable()) != canonical_digest(value):
            raise ValueError("detector adapter changed numeric serialization")
        return cls(model, observed)

    def infer_episode(self, row: Mapping[str, Any]) -> dict[str, Any]:
        intervals = [
            {"start": int(event["start"]), "end": int(event["end"])} for event in row["events"]
        ]
        value = frozen_v2_detector.candidate_evidence(self.model, row["raw_stream"], intervals)
        serializable = {
            "event_logits": list(map(float, value["event_logits"])),
            "null_logit": float(value["null_logit"]),
            "owner_probabilities": list(map(float, value["owner_probabilities"])),
            "quality": float(value["quality"]),
            "availability": float(value["availability"]),
            "top_event_index": value["top_event_index"],
        }
        self.inference_calls += 1
        self.call_digests.append(canonical_digest(serializable))
        return serializable

    def infer_episodes(
        self,
        rows: Sequence[Mapping[str, Any]],
        firewall: IdentifierFirewall,
        *,
        corpus_seed: int,
        purpose: str,
    ) -> dict[str, dict[str, Any]]:
        firewall.authorize(
            "detector_infer", [IdentifierReference("corpus", corpus_seed)], purpose=purpose
        )
        return {str(row["episode_id"]): self.infer_episode(row) for row in rows}

    def perturbation_check(self, row: Mapping[str, Any]) -> dict[str, Any]:
        original = self.infer_episode(row)
        changed = copy.deepcopy(row)
        changed["raw_stream"]["imu"][int(row["events"][0]["start"])][0] += 3.0
        perturbed = self.infer_episode(changed)
        different = canonical_digest(original) != canonical_digest(perturbed)
        return {
            "status": "PASS" if different else "FAIL",
            "changed_only_visible_raw_sensor_byte_value": True,
            "original_digest": canonical_digest(original),
            "perturbed_digest": canonical_digest(perturbed),
            "output_changed": different,
        }

    def runtime_audit(self, *, minimum_calls: int, perturbation: Mapping[str, Any]) -> dict[str, Any]:
        implementation_class = f"{type(self.model).__module__}.{type(self.model).__name__}"
        implementation_function = "babyworld_lite.sensor_alignment_v2.detector.candidate_evidence"
        passed = (
            self.load_calls == 1
            and self.inference_calls >= minimum_calls
            and len(self.call_digests) == self.inference_calls
            and implementation_class == "babyworld_lite.sensor_alignment_v2.detector.SensorEventDetector"
            and perturbation["status"] == "PASS"
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "file_sha256": self.file_digest,
            "detector_model_digest": self.model_digest,
            "implementation_class": implementation_class,
            "implementation_function": implementation_function,
            "load_calls": self.load_calls,
            "inference_calls": self.inference_calls,
            "minimum_required_calls": minimum_calls,
            "hash_only_placeholder": self.inference_calls == 0,
            "evidence_digest": canonical_digest(self.call_digests),
            "perturbation": dict(perturbation),
        }


def detector_capacity_audit(
    evidence: Mapping[str, Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    by_stratum: defaultdict[str, list[float]] = defaultdict(list)
    for key in oracle:
        if key["family"] != "action":
            continue
        value = evidence[str(key["episode_id"])]
        scores = np.asarray([*value["event_logits"], value["null_logit"]], dtype=float)
        answer = int(key["target_event_index"]) if key["grounded"] else len(scores) - 1
        maxima = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0.0))
        credit = 1.0 / len(maxima) if answer in set(map(int, maxima)) else 0.0
        stratum = str(key["factor_values"]["sensor_stratum"])
        by_stratum[stratum].append(credit)
        rows.append(
            {
                "episode_id": key["episode_id"],
                "stratum": stratum,
                "grounded": bool(key["grounded"]),
                "fractional_top_event_or_null_credit": credit,
            }
        )
    stratum_metrics = {}
    strata_pass = True
    for stratum, values in sorted(by_stratum.items()):
        accuracy = float(np.mean(values))
        bounds = config["detector"]["strata_capacity_bounds"][stratum]
        passed = float(bounds["minimum"]) <= accuracy <= float(bounds["maximum"])
        strata_pass &= passed
        stratum_metrics[stratum] = {
            "count": len(values),
            "fractional_accuracy": accuracy,
            "minimum": float(bounds["minimum"]),
            "maximum": float(bounds["maximum"]),
            "status": "PASS" if passed else "FAIL",
        }
    overall = float(np.mean([row["fractional_top_event_or_null_credit"] for row in rows]))
    overall_pass = float(config["detector"]["overall_minimum"]) <= overall <= float(
        config["detector"]["overall_maximum"]
    )
    informative_help = (
        stratum_metrics.get("informative", {}).get("fractional_accuracy", 0.0)
        > stratum_metrics.get("zero_information", {}).get("fractional_accuracy", 1.0)
    )
    passed = strata_pass and overall_pass and informative_help and overall not in {0.0, 1.0}
    return {
        "status": "PASS" if passed else "FAIL",
        "overall_fractional_top_event_or_null_accuracy": overall,
        "overall_minimum": float(config["detector"]["overall_minimum"]),
        "overall_maximum": float(config["detector"]["overall_maximum"]),
        "strata": stratum_metrics,
        "informative_better_than_zero_information": informative_help,
        "not_perfect_everywhere": overall < 1.0,
        "not_chance_everywhere": informative_help,
        "rows": rows,
    }
