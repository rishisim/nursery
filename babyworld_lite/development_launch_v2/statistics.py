from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import IdentifierFirewall, IdentifierReference, canonical_digest


def average_model_replicates(
    model_results: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    purpose: str,
) -> dict[str, Any]:
    expected_models = set(
        map(int, config["resolved_registries"][purpose]["model"])
    )
    grouped: defaultdict[
        tuple[int, str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    model_digests: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    for row in model_results:
        corpus_seed = int(row["corpus_seed"])
        model_seed = int(row["model_seed"])
        condition = str(row["condition"])
        model_digests[(corpus_seed, condition)].add(str(row["model_digest"]))
        for kind in ("primitive", "manner", "action", "noun"):
            for presence in ("present", "null"):
                grouped[(corpus_seed, condition, kind, presence)].append(
                    row["metrics"]["by_kind_presence"][kind][presence]
                )
    rows = []
    for (corpus_seed, condition, kind, presence), values in sorted(
        grouped.items()
    ):
        if len(values) != len(expected_models):
            raise RuntimeError(
                f"missing model replicate for {corpus_seed}/{condition}/{kind}/{presence}"
            )
        numeric_fields = sorted(
            key
            for key, value in values[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        averaged = {
            field: float(np.mean([float(value[field]) for value in values]))
            for field in numeric_fields
        }
        rows.append(
            {
                "corpus_seed": corpus_seed,
                "condition": condition,
                "kind": kind,
                "presence": presence,
                "model_replicates_averaged": len(values),
                "metrics": averaged,
            }
        )
    distinct_replicates = all(
        len(digests) == len(expected_models)
        for digests in model_digests.values()
    )
    return {
        "status": "PASS" if rows and distinct_replicates else "FAIL",
        "independent_unit": "corpus_seed",
        "model_replicates_averaged_within_corpus": True,
        "expected_model_seeds": sorted(expected_models),
        "distinct_model_digests_per_corpus_condition": distinct_replicates,
        "rows": rows,
    }


def _sign_test_p_greater(differences: Sequence[float]) -> tuple[float, int, int]:
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0, positive, nonzero
    probability = sum(
        math.comb(nonzero, count)
        for count in range(positive, nonzero + 1)
    ) / (2**nonzero)
    return float(probability), positive, nonzero


def bounded_paired_inference(
    differences: Sequence[float],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    purpose: str,
    inference_seed: int,
    contrast: str,
    lower_bound_threshold: float,
) -> dict[str, Any]:
    corpus_seeds = list(
        map(int, config["resolved_registries"][purpose]["corpus"])
    )
    model_seeds = list(
        map(int, config["resolved_registries"][purpose]["model"])
    )
    firewall.authorize(
        "inference",
        [
            *[
                IdentifierReference("corpus", value)
                for value in corpus_seeds
            ],
            *[
                IdentifierReference("model", value)
                for value in model_seeds
            ],
            IdentifierReference("inference", int(inference_seed)),
        ],
        purpose=purpose,
    )
    values = np.asarray(list(map(float, differences)), dtype=float)
    if len(values) != len(corpus_seeds):
        raise ValueError("paired inference must contain one value per corpus")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("bounded paired differences must lie in [-1, 1]")
    analysis = config["analysis"]
    tolerance = 1e-15
    shifted = values - float(lower_bound_threshold)
    nonzero_values = shifted[np.abs(shifted) > tolerance]
    unique = np.unique(values)
    sign_p, positive_count, nonzero_count = _sign_test_p_greater(shifted)
    positive_fraction = float(np.mean(shifted > 0.0))
    bootstrap_count = int(analysis["bootstrap_replicates"])
    seed_bytes = hashlib.sha256(
        f"development-v2|{inference_seed}|{contrast}".encode()
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    if len(unique) == 1:
        lower = float(unique[0])
        upper = float(unique[0])
        if abs(float(unique[0])) <= tolerance:
            mode = "FAIL_DEGENERATE_ZERO"
        elif float(unique[0]) > lower_bound_threshold:
            mode = "PASS_POINT_MASS_WITH_EXACT_SIGN_TEST"
        else:
            mode = "FAIL_POINT_MASS"
    else:
        samples = rng.integers(
            0, len(values), size=(bootstrap_count, len(values))
        )
        means = values[samples].mean(axis=1)
        alpha = 1.0 - float(analysis["confidence_level_one_sided"])
        lower = float(np.quantile(means, alpha, method="linear"))
        upper = float(np.quantile(means, 1.0 - alpha, method="linear"))
        mode = "NON_DEGENERATE_BOOTSTRAP"
    enough_nonzero = nonzero_count >= int(
        analysis["minimum_nonzero_corpora_for_non_degenerate_inference"]
    )
    if mode == "NON_DEGENERATE_BOOTSTRAP" and not enough_nonzero:
        mode = "FAIL_INSUFFICIENT_NONZERO_CORPORA"
    passed = (
        mode
        in {
            "NON_DEGENERATE_BOOTSTRAP",
            "PASS_POINT_MASS_WITH_EXACT_SIGN_TEST",
        }
        and float(values.mean()) > float(lower_bound_threshold)
        and lower > float(lower_bound_threshold)
        and sign_p <= float(analysis["maximum_one_sided_sign_test_p"])
        and positive_fraction
        >= float(analysis["minimum_positive_corpus_fraction"])
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "contrast": contrast,
        "independent_unit": "corpus_seed",
        "corpus_count": len(values),
        "model_replicates_averaged_before_inference": True,
        "bounded_support": [-1.0, 1.0],
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "minimum_difference": float(values.min()),
        "maximum_difference": float(values.max()),
        "zero_fraction": float(np.mean(np.abs(values) <= tolerance)),
        "threshold_equality_fraction": float(
            np.mean(np.abs(shifted) <= tolerance)
        ),
        "boundary_fraction": float(
            np.mean((values <= -1.0 + tolerance) | (values >= 1.0 - tolerance))
        ),
        "positive_fraction": positive_fraction,
        "positive_relative_to_required_bound": True,
        "positive_count": int(positive_count),
        "nonzero_count": int(nonzero_count),
        "exact_one_sided_sign_test_p": sign_p,
        "bootstrap_replicates": bootstrap_count,
        "one_sided_confidence_level": float(
            analysis["confidence_level_one_sided"]
        ),
        "lower_confidence_bound": lower,
        "upper_reference_quantile": upper,
        "required_lower_bound": float(lower_bound_threshold),
        "degenerate_mode": mode,
        "inference_seed": int(inference_seed),
        "values_digest": canonical_digest(list(map(float, values))),
    }


def development_inference(
    averaged: Mapping[str, Any],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    purpose: str,
) -> dict[str, Any]:
    rows = {
        (
            int(row["corpus_seed"]),
            str(row["condition"]),
            str(row["kind"]),
            str(row["presence"]),
        ): row["metrics"]
        for row in averaged["rows"]
    }
    corpus_seeds = list(
        map(int, config["resolved_registries"][purpose]["corpus"])
    )
    inference_seed = int(
        config["resolved_registries"][purpose]["inference"][0]
    )
    primary = {}
    for comparator in ("absent_channel", "randomized_shuffle"):
        name = f"synchronized_minus_{comparator}"
        differences = [
            float(
                rows[(seed, "synchronized", "action", "present")][
                    "mean_correct_probability"
                ]
            )
            - float(
                rows[(seed, comparator, "action", "present")][
                    "mean_correct_probability"
                ]
            )
            for seed in corpus_seeds
        ]
        primary[name] = bounded_paired_inference(
            differences,
            config,
            firewall,
            purpose=purpose,
            inference_seed=inference_seed,
            contrast=name,
            lower_bound_threshold=float(
                config["analysis"]["lower_confidence_bound_must_exceed"]
            ),
        )
    null_noninferiority = {}
    for comparator in config["analysis"]["null_comparators"]:
        name = f"synchronized_minus_{comparator}_null"
        differences = [
            float(
                rows[(seed, "synchronized", "action", "null")][
                    "mean_correct_probability"
                ]
            )
            - float(
                rows[(seed, str(comparator), "action", "null")][
                    "mean_correct_probability"
                ]
            )
            for seed in corpus_seeds
        ]
        null_noninferiority[name] = bounded_paired_inference(
            differences,
            config,
            firewall,
            purpose=purpose,
            inference_seed=inference_seed,
            contrast=name,
            lower_bound_threshold=float(
                config["analysis"]["null_noninferiority_margin"]
            ),
        )
    return {
        "status": (
            "PASS"
            if all(value["status"] == "PASS" for value in primary.values())
            and all(
                value["status"] == "PASS"
                for value in null_noninferiority.values()
            )
            else "FAIL"
        ),
        "intersection_union_rule": True,
        "co_primary": primary,
        "null_noninferiority": null_noninferiority,
        "development_outcome": purpose == "development",
        "confirmation_outcome": False,
    }
