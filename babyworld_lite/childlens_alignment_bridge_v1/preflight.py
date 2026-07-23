"""Pure, testable mechanics for the ChildLens alignment-bridge preflight.

Raw ChildLens material is never accepted by these helpers. The executable
controller is responsible for keeping restricted manifests, text, intervals,
and row-level scores inside the existing owner-private quarantine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import random
import re
import unicodedata
from typing import Any


class BridgeError(RuntimeError):
    """A fixed-code fail-closed error."""


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\wäöüß]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_similarity(left: str, right: str) -> float:
    """Bounded normalized character similarity after transparent normalization."""

    left_norm = canonical_text(left)
    right_norm = canonical_text(right)
    denominator = max(len(left_norm), len(right_norm))
    if denominator == 0:
        return 1.0
    return 1.0 - (_levenshtein(left_norm, right_norm) / denominator)


def _interval_center(interval: Mapping[str, Any]) -> float:
    return (float(interval["start_seconds"]) + float(interval["end_seconds"])) / 2.0


def interval_boundary_f1(
    primary: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    *,
    tolerance_seconds: float = 0.5,
) -> float:
    """Greedy one-to-one center matching F1 for model-model timing sensitivity."""

    if not primary and not sensitivity:
        return 1.0
    if not primary or not sensitivity:
        return 0.0
    unmatched = set(range(len(sensitivity)))
    matches = 0
    for interval in sorted(primary, key=_interval_center):
        center = _interval_center(interval)
        candidates = [
            (abs(center - _interval_center(sensitivity[index])), index)
            for index in unmatched
            if abs(center - _interval_center(sensitivity[index])) <= tolerance_seconds
        ]
        if candidates:
            _, index = min(candidates)
            unmatched.remove(index)
            matches += 1
    precision = matches / len(primary)
    recall = matches / len(sensitivity)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def participant_bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float,
    replicates: int,
    seed: int,
    statistic: str = "median",
) -> tuple[float, float]:
    """Percentile participant bootstrap with a deterministic seed."""

    if not values:
        raise BridgeError("E_EMPTY_BOOTSTRAP")
    if statistic not in {"mean", "median"}:
        raise BridgeError("E_BOOTSTRAP_STATISTIC")
    if not 0.0 < confidence < 1.0 or replicates < 100:
        raise BridgeError("E_BOOTSTRAP_CONFIG")
    ordered = sorted(float(value) for value in values)
    rng = random.Random(seed)

    def summarize(sample: list[float]) -> float:
        sample.sort()
        if statistic == "mean":
            return sum(sample) / len(sample)
        middle = len(sample) // 2
        return (
            sample[middle]
            if len(sample) % 2
            else (sample[middle - 1] + sample[middle]) / 2.0
        )

    estimates = []
    for _ in range(replicates):
        estimates.append(summarize([rng.choice(ordered) for _ in ordered]))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(alpha * (replicates - 1)))
    upper_index = min(replicates - 1, math.ceil((1.0 - alpha) * (replicates - 1)))
    return estimates[lower_index], estimates[upper_index]


def _digest(protocol_sha256: str, key: str) -> str:
    return hashlib.sha256(f"{protocol_sha256}|{key}".encode("utf-8")).hexdigest()


def build_control_assignment(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
) -> list[dict[str, str]]:
    """Build a deterministic cross-participant derangement.

    Rows must already encode the frozen fallback match stratum in ``stratum``.
    Strata with fewer than two participants are merged by the caller according
    to the protocol's fixed hierarchy.
    """

    if len(rows) < 2:
        raise BridgeError("E_CONTROL_SUPPORT")
    ordered = sorted(rows, key=lambda row: _digest(protocol_sha256, str(row["utterance_key"])))
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for row in ordered:
        by_stratum.setdefault(str(row["stratum"]), []).append(row)
    assignments: list[dict[str, str]] = []
    for stratum, group in sorted(by_stratum.items()):
        participants = {str(row["participant_key"]) for row in group}
        if len(group) < 2 or len(participants) < 2:
            raise BridgeError("E_CONTROL_SUPPORT")
        donor_order = None
        for offset in range(1, len(group)):
            candidate_order = group[offset:] + group[:offset]
            if all(
                receiver["participant_key"] != donor["participant_key"]
                for receiver, donor in zip(group, candidate_order)
            ):
                donor_order = candidate_order
                break
        if donor_order is None:
            raise BridgeError("E_CONTROL_DERANGEMENT")
        for receiver, donor in zip(group, donor_order):
            assignments.append(
                {
                    "utterance_key": str(receiver["utterance_key"]),
                    "donor_utterance_key": str(donor["utterance_key"]),
                    "stratum": stratum,
                }
            )
    if len({row["utterance_key"] for row in assignments}) != len(rows):
        raise BridgeError("E_CONTROL_DUPLICATE")
    if len({row["donor_utterance_key"] for row in assignments}) != len(rows):
        raise BridgeError("E_CONTROL_INVENTORY")
    return assignments
