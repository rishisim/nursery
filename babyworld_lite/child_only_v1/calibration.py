"""Corpus-specific calibration-spec and post-access adapter gates."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .policy import CHILD_CORPORA, PolicyViolation, reject_forbidden_source_markers


_VALUE_FIELD_NAMES = {
    "value",
    "values",
    "estimate",
    "estimates",
    "empirical_value",
    "mean",
    "median",
    "quantile_value",
    "distribution_parameters",
}


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def load_measurement_spec(path: str | Path, expected_corpus: str) -> dict[str, Any]:
    """Load a definitions-only specification and reject embedded empirical values."""

    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    reject_forbidden_source_markers(spec)
    if expected_corpus not in CHILD_CORPORA or spec.get("corpus") != expected_corpus:
        raise PolicyViolation("calibration spec is not bound to the expected child corpus")
    if spec.get("empirical_values_status") != "ABSENT_DEFINITIONS_ONLY":
        raise PolicyViolation("pre-access calibration specs cannot contain empirical values")
    for mapping in _walk_mappings(spec):
        if _VALUE_FIELD_NAMES.intersection(str(key).lower() for key in mapping):
            raise PolicyViolation("empirical value field found in definitions-only calibration spec")
    metric_ids = [metric["metric_id"] for metric in spec.get("measurements", [])]
    if not metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise PolicyViolation("calibration metrics must be nonempty and uniquely named")
    required = {
        "metric_id",
        "definition",
        "numerator",
        "denominator",
        "unit",
        "required_inputs",
        "estimator",
        "uncertainty",
        "missingness",
        "export_constraint",
    }
    for metric in spec["measurements"]:
        if set(metric) != required:
            raise PolicyViolation("calibration measurement definition is incomplete")
    return spec


class SelectedCorpusCalibrationAdapter:
    """Pluggable only after access and explicit selection of one child corpus."""

    def __init__(
        self,
        *,
        selected_corpus: str | None,
        corpus_instance_id: str | None,
        restricted_access_available: bool,
        measurement_spec: Mapping[str, Any],
    ) -> None:
        if selected_corpus not in CHILD_CORPORA:
            raise PolicyViolation("calibration adapter requires exactly one selected child corpus")
        if not restricted_access_available or not corpus_instance_id:
            raise PolicyViolation("calibration adapter is disabled until restricted access and instance binding exist")
        if measurement_spec.get("corpus") != selected_corpus:
            raise PolicyViolation("calibration adapter/spec corpus mismatch")
        self.selected_corpus = selected_corpus
        self.corpus_instance_id = corpus_instance_id
        self.metric_ids = frozenset(metric["metric_id"] for metric in measurement_spec["measurements"])

    def validate_export_package(self, package: Mapping[str, Any]) -> None:
        """Validate metadata around an eventual values payload before ingestion."""

        reject_forbidden_source_markers(package)
        expected = {
            "export_version",
            "selected_corpus",
            "corpus_instance_id",
            "measurement_spec_version",
            "export_constraints_applied",
            "contains_raw_transcripts",
            "contains_example_frames",
            "contains_exact_timestamps",
            "contains_participant_or_episode_ids",
            "measurements",
        }
        if set(package) != expected:
            raise PolicyViolation("calibration export package has unknown or missing fields")
        if package["selected_corpus"] != self.selected_corpus or package["corpus_instance_id"] != self.corpus_instance_id:
            raise PolicyViolation("calibration export crosses its selected corpus instance")
        if package["export_constraints_applied"] is not True:
            raise PolicyViolation("calibration export constraints were not applied")
        sensitive_flags = (
            "contains_raw_transcripts",
            "contains_example_frames",
            "contains_exact_timestamps",
            "contains_participant_or_episode_ids",
        )
        if any(package[flag] is not False for flag in sensitive_flags):
            raise PolicyViolation("calibration export contains prohibited sensitive content")
        if set(package["measurements"]) != self.metric_ids:
            raise PolicyViolation("calibration export must exactly cover its corpus-specific measurement spec")
