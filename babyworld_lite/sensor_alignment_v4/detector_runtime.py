from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from babyworld_lite.sensor_alignment_v2 import detector as frozen_v2_detector

from .protocol import (
    SeedFirewall,
    SeedReference,
    canonical_digest,
    sha256_file,
)


class DetectorRuntime:
    def __init__(self, model: frozen_v2_detector.SensorEventDetector, file_digest: str):
        if not isinstance(model, frozen_v2_detector.SensorEventDetector):
            raise TypeError("v4 requires the actual frozen v2 SensorEventDetector")
        self.model = model
        self.file_digest = str(file_digest)
        self.model_digest = canonical_digest(model.serializable())
        self.load_calls = 1
        self.inference_calls = 0
        self.evidence_call_digests: list[str] = []

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "DetectorRuntime":
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(f"frozen detector digest mismatch: {observed}")
        value = json.loads(Path(path).read_text())
        if value.get("schema_version") != "synthetic-sensor-event-detector-v2":
            raise ValueError("unexpected frozen detector schema")
        model = frozen_v2_detector.SensorEventDetector(
            feature_mean=tuple(map(float, value["feature_mean"])),
            feature_scale=tuple(map(float, value["feature_scale"])),
            activity_weights=tuple(map(float, value["activity_weights"])),
            boundary_weights=tuple(map(float, value["boundary_weights"])),
            training_digest=str(value["training_digest"]),
        )
        if canonical_digest(model.serializable()) != canonical_digest(value):
            raise ValueError("compatibility adapter changed frozen detector serialization")
        return cls(model, observed)

    def infer_episode(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if "raw_stream" not in row:
            raise ValueError("detector inference requires a raw stream")
        intervals = [
            {"start": int(event["start"]), "end": int(event["end"])}
            for event in row["events"]
        ]
        value = frozen_v2_detector.candidate_evidence(
            self.model, row["raw_stream"], intervals
        )
        serializable = {
            "event_logits": list(map(float, value["event_logits"])),
            "null_logit": float(value["null_logit"]),
            "owner_probabilities": list(map(float, value["owner_probabilities"])),
            "quality": float(value["quality"]),
            "availability": float(value["availability"]),
            "top_event_index": value["top_event_index"],
        }
        self.inference_calls += 1
        self.evidence_call_digests.append(canonical_digest(serializable))
        return serializable

    def infer_episodes(
        self,
        rows: Sequence[Mapping[str, Any]],
        firewall: SeedFirewall,
        *,
        corpus_seed: int,
        purpose: str,
    ) -> dict[str, dict[str, Any]]:
        firewall.authorize(
            "detector_infer",
            [SeedReference("corpus", corpus_seed)],
            purpose=purpose,
        )
        output: dict[str, dict[str, Any]] = {}
        for row in rows:
            output[str(row["episode_id"])] = self.infer_episode(row)
        return output

    def audit(self, *, expected_minimum_calls: int) -> dict[str, Any]:
        implementation_class = (
            f"{type(self.model).__module__}.{type(self.model).__name__}"
        )
        implementation_function = (
            "babyworld_lite.sensor_alignment_v2.detector.candidate_evidence"
        )
        sufficient = self.inference_calls >= int(expected_minimum_calls)
        status = (
            "PASS"
            if self.load_calls == 1
            and sufficient
            and self.inference_calls == len(self.evidence_call_digests)
            and implementation_class
            == "babyworld_lite.sensor_alignment_v2.detector.SensorEventDetector"
            else "FAIL"
        )
        return {
            "status": status,
            "file_sha256": self.file_digest,
            "detector_model_digest": self.model_digest,
            "load_calls": self.load_calls,
            "inference_calls": self.inference_calls,
            "minimum_required_inference_calls": int(expected_minimum_calls),
            "inference_call_count_sufficient": sufficient,
            "implementation_class": implementation_class,
            "implementation_function": implementation_function,
            "evidence_outputs_digest": canonical_digest(self.evidence_call_digests),
            "hash_only_placeholder": self.inference_calls == 0,
            "compatibility_adapter": "JSON fields mapped without numeric transformation",
        }
