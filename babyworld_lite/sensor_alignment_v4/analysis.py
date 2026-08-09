from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _rng(seed: int, label: str) -> np.random.Generator:
    value = int.from_bytes(
        hashlib.sha256(f"v4-analysis|{seed}|{label}".encode()).digest()[:8], "little"
    )
    return np.random.default_rng(value)


def infer_mean(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    alpha: float,
    zero_variance_tolerance: float,
) -> dict[str, Any]:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) < 2:
        raise ValueError("inference requires at least two independent units")
    if not np.all(np.isfinite(sample)):
        raise ValueError("inference sample contains a non-finite value")
    mean = float(sample.mean())
    standard_error = float(sample.std(ddof=1) / math.sqrt(len(sample)))
    summary = {
        "mean": mean,
        "standard_error": standard_error,
        "minimum": float(sample.min()),
        "q1": float(np.quantile(sample, 0.25)),
        "median": float(np.median(sample)),
        "q3": float(np.quantile(sample, 0.75)),
        "maximum": float(sample.max()),
        "n_independent_units": len(sample),
    }
    if sample.var() <= zero_variance_tolerance:
        point = float(sample[0])
        return {
            **summary,
            "mean": point,
            "method": "degenerate_point_mass",
            "degenerate": True,
            "ci_low": point,
            "ci_high": point,
            "bootstrap_interval_computed": False,
            "valid_studentized_resamples": 0,
            "population_uncertainty_estimated": False,
            "warning": "All observed independent-unit values are identical; the interval is the exact empirical point mass and is not a population-uncertainty estimate.",
        }
    rng = _rng(seed, "studentized-bootstrap")
    statistics: list[float] = []
    for _ in range(int(resamples)):
        bootstrap = sample[rng.integers(0, len(sample), len(sample))]
        bootstrap_se = float(bootstrap.std(ddof=1) / math.sqrt(len(sample)))
        if bootstrap_se > zero_variance_tolerance:
            statistics.append((float(bootstrap.mean()) - mean) / bootstrap_se)
    if not statistics:
        raise RuntimeError("non-degenerate sample produced no valid studentized resamples")
    low_quantile, high_quantile = np.quantile(
        statistics, [1.0 - alpha / 2.0, alpha / 2.0]
    )
    return {
        **summary,
        "method": "studentized_bootstrap_t",
        "degenerate": False,
        "ci_low": float(mean - low_quantile * standard_error),
        "ci_high": float(mean - high_quantile * standard_error),
        "bootstrap_interval_computed": True,
        "requested_resamples": int(resamples),
        "valid_studentized_resamples": len(statistics),
        "population_uncertainty_estimated": True,
        "warning": None,
    }


def noun_detector_selectivity_audit(
    evidence: Mapping[str, Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    *,
    maximum_gap: float,
) -> dict[str, Any]:
    target_scores: list[float] = []
    distractor_scores: list[float] = []
    owner_scores: list[float] = []
    nonowner_scores: list[float] = []
    for row in oracle:
        if row["family"] != "noun":
            continue
        value = evidence[str(row["episode_id"])]
        scores = list(map(float, value["owner_probabilities"]))
        target = int(row["target_event_index"])
        owners = list(map(bool, row["event_owners"]))
        target_scores.append(scores[target])
        distractor_scores.extend(score for index, score in enumerate(scores) if index != target)
        owner_scores.extend(score for score, owner in zip(scores, owners) if owner)
        nonowner_scores.extend(score for score, owner in zip(scores, owners) if not owner)
    target_mean = float(np.mean(target_scores))
    distractor_mean = float(np.mean(distractor_scores))
    gap = target_mean - distractor_mean
    return {
        "status": "PASS" if abs(gap) <= maximum_gap else "FAIL",
        "noun_target_detector_mean": target_mean,
        "noun_distractor_detector_mean": distractor_mean,
        "target_minus_distractor_gap": gap,
        "maximum_absolute_gap": float(maximum_gap),
        "owner_detector_mean": float(np.mean(owner_scores)),
        "nonowner_detector_mean": float(np.mean(nonowner_scores)),
        "structural_interpretation": "Owner detection may be strong, but target status is exactly factorially independent of ownership.",
    }


def factor_stratification_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factors = sorted({str(row["factor"]) for row in rows})
    checks: dict[str, Any] = {}
    for factor in factors:
        selected = [row for row in rows if row["factor"] == factor]
        levels = {str(row["level"]) for row in selected}
        episode_digests = {str(row["training_episode_ids_digest"]) for row in selected}
        model_digests = {str(row["model_digest"]) for row in selected}
        passed = (
            len(levels) >= 2
            and "pooled_training_distribution" not in levels
            and len(episode_digests) >= 2
            and len(model_digests) >= 2
            and all(int(row["training_episode_count"]) > 0 for row in selected)
        )
        checks[factor] = {
            "status": "PASS" if passed else "FAIL",
            "levels": sorted(levels),
            "distinct_training_episode_sets": len(episode_digests),
            "distinct_model_digests": len(model_digests),
        }
    return {
        "status": "PASS" if checks and all(value["status"] == "PASS" for value in checks.values()) else "FAIL",
        "factors": checks,
    }
