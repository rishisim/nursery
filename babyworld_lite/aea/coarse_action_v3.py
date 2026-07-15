from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "aea-coarse-action-v3"
MODELED_ACTIONS = ("gross_body_motion", "object_or_body_interaction")
ALL_ACTIONS = (*MODELED_ACTIONS, "no_goal_directed_visible_action", "uncertain")
ASR_REFERENTS = ("wearer_action", "nonwearer_or_nonliteral", "unclear")
TEMPORAL_RELATIONS = ("aligned", "before", "after", "none", "unclear")
CONFIDENCES = ("high", "medium", "low")
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.75, "low": 0.5}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def salted_digest(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def verify_protocol_freeze(
    protocol_path: str | Path,
    freeze_receipt_path: str | Path,
    preregistration_path: str | Path,
    codebook_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    receipt = load_json(freeze_receipt_path)
    checks = {
        "protocol_id": protocol.get("protocol_id") == PROTOCOL_ID,
        "receipt_protocol_id": receipt.get("protocol_id") == PROTOCOL_ID,
        "protocol_hash": sha256_file(protocol_path)
        == receipt["machine_readable_protocol"]["sha256"],
        "preregistration_hash": sha256_file(preregistration_path)
        == receipt["preregistration_markdown"]["sha256"],
        "codebook_hash": sha256_file(codebook_path)
        == receipt["annotation_codebook"]["sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"v3 protocol freeze verification failed: {checks}")
    return protocol, {"checks": checks}


def validate_fixed_dense_source(
    protocol: Mapping[str, Any],
    manifest_path: str | Path,
    access_receipt_path: str | Path,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = protocol["sources"]
    manifest = load_json(manifest_path)
    receipt = load_json(access_receipt_path)
    root = Path(repository_root).resolve()
    dense_root = (root / "output/aea_visible_action_v2/dense_evidence").resolve()
    manifest_root = Path(manifest_path).resolve().parent
    rows = list(manifest.get("rows", []))
    example_ids = [str(row.get("example_id", "")) for row in rows]
    sorted_id_digest = hashlib.sha256(
        ("\n".join(sorted(example_ids)) + "\n").encode()
    ).hexdigest()
    reserve_groups = set(map(str, source["reserve_event_groups"]))
    path_checks = []
    referenced_frames = 0
    for row in rows:
        paths = [
            *row.get("dense_frame_paths", []),
            row.get("contact_sheet", ""),
            *row.get("detail_sheets", []),
        ]
        for relative in paths:
            path = (manifest_root / str(relative)).resolve()
            path_checks.append(
                path.is_file() and (path == dense_root or dense_root in path.parents)
            )
        referenced_frames += len(row.get("dense_frame_paths", []))
    checks = {
        "manifest_hash": sha256_file(manifest_path)
        == source["v2_dense_manifest_sha256"],
        "access_receipt_hash": sha256_file(access_receipt_path)
        == source["v2_reserve_access_receipt_sha256"],
        "manifest_protocol": manifest.get("protocol_id") == "aea-visible-action-v2",
        "sample_size": len(rows) == int(source["sample_size"]) == 72,
        "manifest_sample_size": manifest.get("sample_size") == 72,
        "sample_digest": manifest.get("sample_digest") == source["sample_digest_v2"],
        "unique_ids": len(example_ids) == len(set(example_ids)),
        "sorted_example_id_digest": sorted_id_digest
        == source["sorted_example_id_digest"],
        "all_18_development_groups": len({str(row["event_group"]) for row in rows})
        == int(source["development_event_groups"]),
        "zero_reserve_groups": not any(
            str(row["event_group"]) in reserve_groups for row in rows
        ),
        "manifest_declares_zero_reserve": manifest.get("reserve_groups_present") == [],
        "source_receipt_zero_reserve_rgb": receipt.get("reserve_rgb_files_opened") == 0,
        "source_receipt_zero_reserve_imu": receipt.get("reserve_imu_arrays_opened") == 0,
        "source_receipt_zero_signed_url_exposure": receipt.get(
            "signed_urls_loaded_printed_or_copied"
        )
        is False,
        "exactly_31_frames_each": referenced_frames == 72 * 31,
        "all_evidence_paths_existing_and_beneath_v2_root": all(path_checks),
    }
    if not all(checks.values()):
        raise ValueError(f"v3 fixed dense source verification failed: {checks}")
    return manifest, {
        "checks": checks,
        "referenced_dense_frames": referenced_frames,
        "evidence_paths_checked": len(path_checks),
        "additional_rgb_frames_queried": 0,
        "development_imu_arrays_opened": 0,
        "reserve_rgb_files_opened": 0,
        "reserve_imu_arrays_opened": 0,
        "signed_urls_loaded_printed_copied_or_used": False,
    }


def reblind_fixed_rows(
    v2_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    blind_ids: set[str] = set()
    salt = str(protocol["sample"]["blind_id_salt"])
    for source in sorted(v2_rows, key=lambda row: str(row["example_id"])):
        blind_id = "CA3-" + salted_digest(salt, str(source["example_id"]))[:12].upper()
        if blind_id in blind_ids:
            raise AssertionError("v3 blind ID collision")
        blind_ids.add(blind_id)
        output.append({
            "blind_id": blind_id,
            "example_id": str(source["example_id"]),
            "sequence_id": str(source["sequence_id"]),
            "event_group": str(source["event_group"]),
            "location": int(source["location"]),
            "anchored_asr_verb": str(source["asr_action_verb"]),
            "transcript": str(source["transcript"]),
            "window": dict(source["window"]),
            "source_v2_blind_id": str(source["blind_id"]),
            "dense_frame_paths": [
                str(Path("output/aea_visible_action_v2") / str(path))
                for path in source["dense_frame_paths"]
            ],
            "contact_sheet": str(
                Path("output/aea_visible_action_v2") / str(source["contact_sheet"])
            ),
            "detail_sheets": [
                str(Path("output/aea_visible_action_v2") / str(path))
                for path in source["detail_sheets"]
            ],
        })
    if len(output) != 72:
        raise AssertionError("v3 fixed sample must contain exactly 72 rows")
    return output


def packet_order(
    rows: Sequence[Mapping[str, Any]], salt: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                salted_digest(salt, str(row["blind_id"])),
                str(row["blind_id"]),
            ),
        )
    ]


def _rationale_valid(value: Any, maximum: int) -> bool:
    return isinstance(value, str) and 1 <= len(value.split()) <= maximum


def validate_stage1_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_blind_ids: Iterable[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = set(map(str, expected_blind_ids))
    observed = [str(row.get("blind_id", "")) for row in rows]
    if len(rows) != 72 or len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("stage-1 pass IDs are incomplete, duplicate, or unexpected")
    maximum = int(protocol["annotation_interface"]["rationale_max_words"])
    normalized = []
    for source in rows:
        row = dict(source)
        action = row.get("observable_action")
        start = row.get("evidence_frame_start")
        end = row.get("evidence_frame_end")
        nullable = action in ("no_goal_directed_visible_action", "uncertain")
        null_pair = start is None and end is None
        valid_interval = (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start <= end <= 30
        )
        checks = {
            "action": action in ALL_ACTIONS,
            "confidence": row.get("visible_confidence") in CONFIDENCES,
            "evidence": valid_interval or (nullable and null_pair),
            "rationale": _rationale_valid(row.get("visible_rationale"), maximum),
            "no_language_fields": not any(
                key in row
                for key in (
                    "transcript",
                    "anchored_asr_verb",
                    "asr_referent",
                    "temporal_relation",
                    "language_confidence",
                    "language_rationale",
                )
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"invalid v3 stage-1 row {row.get('blind_id')}: {checks}")
        normalized.append(row)
    return normalized


def validate_stage2_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_blind_ids: Iterable[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = set(map(str, expected_blind_ids))
    observed = [str(row.get("blind_id", "")) for row in rows]
    if len(rows) != 72 or len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("stage-2 pass IDs are incomplete, duplicate, or unexpected")
    maximum = int(protocol["annotation_interface"]["rationale_max_words"])
    normalized = []
    for source in rows:
        row = dict(source)
        checks = {
            "referent": row.get("asr_referent") in ASR_REFERENTS,
            "temporal": row.get("temporal_relation") in TEMPORAL_RELATIONS,
            "confidence": row.get("language_confidence") in CONFIDENCES,
            "rationale": _rationale_valid(row.get("language_rationale"), maximum),
            "no_visible_mutation_fields": not any(
                key in row
                for key in (
                    "observable_action",
                    "visible_confidence",
                    "evidence_frame_start",
                    "evidence_frame_end",
                    "visible_rationale",
                )
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"invalid v3 stage-2 row {row.get('blind_id')}: {checks}")
        normalized.append(row)
    return normalized


def merge_annotation_stages(
    stage1: Sequence[Mapping[str, Any]], stage2: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    language = {str(row["blind_id"]): row for row in stage2}
    return [
        {**dict(row), **dict(language[str(row["blind_id"])])}
        for row in stage1
    ]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {
            "successes": int(successes),
            "total": int(total),
            "rate": None,
            "ci95_low": None,
            "ci95_high": None,
            "method": "two-sided Wilson 95% interval",
        }
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(p),
        "ci95_low": float(max(0.0, center - radius)),
        "ci95_high": float(min(1.0, center + radius)),
        "method": "two-sided Wilson 95% interval",
    }


def agreement_metric(
    values_a: Sequence[str],
    values_b: Sequence[str],
    labels: Sequence[str],
    confidences_a: Sequence[str],
    confidences_b: Sequence[str],
) -> dict[str, Any]:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("agreement requires paired nonempty values")
    order = list(labels)
    index = {label: position for position, label in enumerate(order)}
    confusion = np.zeros((len(order), len(order)), dtype=np.int64)
    for left, right in zip(values_a, values_b):
        confusion[index[left], index[right]] += 1
    matches = np.asarray(values_a, dtype=object) == np.asarray(values_b, dtype=object)
    exact = float(np.mean(matches))
    left_marginal = confusion.sum(axis=1) / confusion.sum()
    right_marginal = confusion.sum(axis=0) / confusion.sum()
    expected = float(left_marginal @ right_marginal)
    kappa = (
        None
        if np.isclose(1.0 - expected, 0.0)
        else float((exact - expected) / (1.0 - expected))
    )
    weights = np.asarray([
        min(CONFIDENCE_WEIGHT[left], CONFIDENCE_WEIGHT[right])
        for left, right in zip(confidences_a, confidences_b)
    ])
    return {
        "labels": order,
        "confusion_a_rows_b_columns": confusion.tolist(),
        "exact_agreement": exact,
        "cohens_kappa_unweighted": kappa,
        "confidence_weighted_exact_agreement": float(
            np.sum(weights * matches) / np.sum(weights)
        ),
        "n": len(values_a),
    }


def support_summary(
    rows: Sequence[Mapping[str, Any]], label_key: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in sorted({str(row[label_key]) for row in rows}):
        selected = [row for row in rows if str(row[label_key]) == label]
        output[label] = {
            "windows": len(selected),
            "event_groups": len({str(row["event_group"]) for row in selected}),
            "locations": len({int(row["location"]) for row in selected}),
            "by_event_group": dict(sorted(Counter(
                str(row["event_group"]) for row in selected
            ).items())),
            "by_location": dict(sorted(Counter(
                str(row["location"]) for row in selected
            ).items())),
        }
    return output


def summarize_annotations(
    pass_a: Sequence[Mapping[str, Any]],
    pass_b: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {str(row["blind_id"]): row for row in manifest_rows}
    expected = set(manifest)
    if len(pass_a) != len(pass_b) or len(pass_a) != 72:
        raise ValueError("v3 combined passes must each contain 72 rows")
    map_a = {str(row["blind_id"]): dict(row) for row in pass_a}
    map_b = {str(row["blind_id"]): dict(row) for row in pass_b}
    if set(map_a) != expected or set(map_b) != expected:
        raise ValueError("v3 combined pass IDs do not match fixed manifest")
    ordered = sorted(expected)
    visible_conf_a = [str(map_a[item]["visible_confidence"]) for item in ordered]
    visible_conf_b = [str(map_b[item]["visible_confidence"]) for item in ordered]
    language_conf_a = [str(map_a[item]["language_confidence"]) for item in ordered]
    language_conf_b = [str(map_b[item]["language_confidence"]) for item in ordered]
    agreement = {
        "observable_action": agreement_metric(
            [str(map_a[item]["observable_action"]) for item in ordered],
            [str(map_b[item]["observable_action"]) for item in ordered],
            ALL_ACTIONS,
            visible_conf_a,
            visible_conf_b,
        ),
        "asr_referent": agreement_metric(
            [str(map_a[item]["asr_referent"]) for item in ordered],
            [str(map_b[item]["asr_referent"]) for item in ordered],
            ASR_REFERENTS,
            language_conf_a,
            language_conf_b,
        ),
        "temporal_relation": agreement_metric(
            [str(map_a[item]["temporal_relation"]) for item in ordered],
            [str(map_b[item]["temporal_relation"]) for item in ordered],
            TEMPORAL_RELATIONS,
            language_conf_a,
            language_conf_b,
        ),
    }
    consensus_rows = []
    paired_rows = []
    language_aligned_rows = []
    allowed_temporal = set(
        protocol["language_alignment_gate"]["aligned_consensus_definition"]
        ["both_pass_temporal_relation_in"]
    )
    for blind_id in ordered:
        left = map_a[blind_id]
        right = map_b[blind_id]
        source = manifest[blind_id]
        action = str(left["observable_action"])
        modeled_consensus = (
            action == str(right["observable_action"])
            and action in MODELED_ACTIONS
            and str(left["visible_confidence"]) in ("high", "medium")
            and str(right["visible_confidence"]) in ("high", "medium")
        )
        language_aligned = (
            str(left["asr_referent"]) == str(right["asr_referent"]) == "wearer_action"
            and str(left["observable_action"]) in MODELED_ACTIONS
            and str(right["observable_action"]) in MODELED_ACTIONS
            and str(left["temporal_relation"]) in allowed_temporal
            and str(right["temporal_relation"]) in allowed_temporal
        )
        shared = {
            "blind_id": blind_id,
            "example_id": str(source["example_id"]),
            "sequence_id": str(source["sequence_id"]),
            "event_group": str(source["event_group"]),
            "location": int(source["location"]),
        }
        paired_rows.append({
            **shared,
            "pass_a": left,
            "pass_b": right,
            "modeled_action_consensus": modeled_consensus,
            "consensus_observable_action": action if modeled_consensus else None,
            "language_aligned_consensus": language_aligned,
        })
        if modeled_consensus:
            consensus_rows.append({**shared, "observable_action": action})
        if language_aligned:
            language_aligned_rows.append({**shared})

    n = len(ordered)
    judgeable_a = sum(map_a[item]["observable_action"] != "uncertain" for item in ordered)
    judgeable_b = sum(map_b[item]["observable_action"] != "uncertain" for item in ordered)
    wearer_a = sum(map_a[item]["asr_referent"] == "wearer_action" for item in ordered)
    wearer_b = sum(map_b[item]["asr_referent"] == "wearer_action" for item in ordered)
    support = support_summary(consensus_rows, "observable_action")
    coarse = protocol["coarse_annotation_gate"]
    action_agreement = agreement["observable_action"]
    support_checks = {
        label: (
            label in support
            and support[label]["windows"] >= int(coarse["minimum_windows_each_modeled_label"])
            and support[label]["event_groups"]
            >= int(coarse["minimum_event_groups_each_modeled_label"])
        )
        for label in MODELED_ACTIONS
    }
    coarse_checks = {
        "complete_valid_passes": len(map_a) == len(map_b) == n == 72,
        "judgeable_pass_a": judgeable_a / n >= float(coarse["judgeable_minimum_each_pass"]),
        "judgeable_pass_b": judgeable_b / n >= float(coarse["judgeable_minimum_each_pass"]),
        "action_exact": action_agreement["exact_agreement"]
        >= float(coarse["action_exact_minimum"]),
        "action_kappa": action_agreement["cohens_kappa_unweighted"] is not None
        and action_agreement["cohens_kappa_unweighted"]
        >= float(coarse["action_kappa_minimum"]),
        "modeled_consensus_yield": len(consensus_rows) / n
        >= float(coarse["modeled_consensus_yield_minimum"]),
        **{f"support_{label}": passed for label, passed in support_checks.items()},
    }
    language_gate = protocol["language_alignment_gate"]
    referent_agreement = agreement["asr_referent"]
    language_checks = {
        "consensus_alignment_rate": len(language_aligned_rows)
        >= int(language_gate["minimum_consensus_rows"]),
        "wearer_action_rate_pass_a": wearer_a
        >= int(language_gate["minimum_wearer_action_rows_each_pass"]),
        "wearer_action_rate_pass_b": wearer_b
        >= int(language_gate["minimum_wearer_action_rows_each_pass"]),
        "referent_exact": referent_agreement["exact_agreement"]
        >= float(language_gate["referent_exact_minimum"]),
        "referent_kappa": referent_agreement["cohens_kappa_unweighted"] is not None
        and referent_agreement["cohens_kappa_unweighted"]
        >= float(language_gate["referent_kappa_minimum"]),
    }
    return {
        "schema_version": "aea-coarse-action-agreement-v3",
        "protocol_id": PROTOCOL_ID,
        "scientific_role": protocol["scientific_role"],
        "annotation_role": protocol["annotation_interface"]["pass_role"],
        "n": n,
        "agreement": agreement,
        "rates": {
            "judgeable_pass_a": wilson(judgeable_a, n),
            "judgeable_pass_b": wilson(judgeable_b, n),
            "uncertain_pass_a": wilson(n - judgeable_a, n),
            "uncertain_pass_b": wilson(n - judgeable_b, n),
            "modeled_consensus_yield": wilson(len(consensus_rows), n),
            "wearer_action_pass_a": wilson(wearer_a, n),
            "wearer_action_pass_b": wilson(wearer_b, n),
            "language_aligned_consensus": wilson(len(language_aligned_rows), n),
        },
        "support": support,
        "support_checks": support_checks,
        "coarse_gate_checks": coarse_checks,
        "coarse_annotation_gate_passed": all(coarse_checks.values()),
        "language_gate_checks": language_checks,
        "language_alignment_gate_passed": all(language_checks.values()),
        "language_conclusion": (
            "LANGUAGE_ROUTE_VIABLE"
            if all(language_checks.values())
            else "STOP_LANGUAGE_ROUTE"
        ),
        "consensus_rows": consensus_rows,
        "language_aligned_rows": language_aligned_rows,
        "paired_rows": paired_rows,
    }


def group_safe_bijection(
    indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    salt: str,
) -> dict[str, Any]:
    indices = list(map(int, indices))
    groups = {index: str(rows[index]["event_group"]) for index in indices}
    counts = Counter(groups.values())
    half_bound = max(counts.values(), default=0) <= len(indices) / 2
    recipients = sorted(
        indices,
        key=lambda index: salted_digest(
            f"{salt}-recipient-{seed}", str(rows[index]["example_id"])
        ),
    )
    donors = sorted(
        indices,
        key=lambda index: salted_digest(
            f"{salt}-donor-{seed}", str(rows[index]["example_id"])
        ),
    )
    donor_to_recipient: dict[int, int] = {}

    def augment(recipient: int, seen: set[int]) -> bool:
        for donor in donors:
            if donor in seen or groups[donor] == groups[recipient]:
                continue
            seen.add(donor)
            if donor not in donor_to_recipient or augment(donor_to_recipient[donor], seen):
                donor_to_recipient[donor] = recipient
                return True
        return False

    matched = sum(augment(recipient, set()) for recipient in recipients)
    mapping_indices = {
        recipient: donor for donor, recipient in donor_to_recipient.items()
    }
    feasible = matched == len(indices)
    if feasible != half_bound:
        raise AssertionError("half-size theorem and matching disagree")
    donor_map = None
    if feasible:
        donor_map = {
            str(rows[index]["example_id"]): str(rows[mapping_indices[index]]["example_id"])
            for index in sorted(indices)
        }
    return {
        "feasible": feasible,
        "windows": len(indices),
        "event_group_counts": dict(sorted(counts.items())),
        "largest_event_group": max(counts.values(), default=0),
        "half_size_theorem_passed": half_bound,
        "matching_size": int(matched),
        "whole_window_bijection": feasible
        and len(set(mapping_indices.values())) == len(indices),
        "self_match_count": None
        if not feasible
        else sum(index == mapping_indices[index] for index in indices),
        "same_event_group_match_count": None
        if not feasible
        else sum(groups[index] == groups[mapping_indices[index]] for index in indices),
        "donor_map": donor_map,
        "donor_map_sha256": None if donor_map is None else canonical_digest(donor_map),
    }


def construct_constrained_folds(
    consensus_rows: Sequence[Mapping[str, Any]],
    all_sample_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from scipy.optimize import Bounds, LinearConstraint, milp

    labels = MODELED_ACTIONS
    endpoint_rows = [
        dict(row)
        for row in consensus_rows
        if str(row["observable_action"]) in labels
    ]
    groups = sorted({str(row["event_group"]) for row in all_sample_rows})
    if len(groups) != int(protocol["sources"]["development_event_groups"]):
        raise ValueError("v3 split requires all 18 development event groups")
    folds_config = protocol["folds"]
    fold_count = int(folds_config["count"])
    group_index = {group: position for position, group in enumerate(groups)}
    location_by_group: dict[str, int] = {}
    for row in all_sample_rows:
        group = str(row["event_group"])
        location = int(row["location"])
        if group in location_by_group and location_by_group[group] != location:
            raise ValueError("event group has inconsistent location")
        location_by_group[group] = location
    locations = sorted(set(location_by_group.values()))
    by_group_label = Counter(
        (str(row["event_group"]), str(row["observable_action"]))
        for row in endpoint_rows
    )
    by_group = Counter(str(row["event_group"]) for row in endpoint_rows)
    by_group_location = Counter(
        (str(row["event_group"]), int(row["location"])) for row in endpoint_rows
    )
    total_by_label = Counter(str(row["observable_action"]) for row in endpoint_rows)
    label_groups = {
        label: {group for group in groups if by_group_label[(group, label)] > 0}
        for label in labels
    }
    total_by_location = Counter(int(row["location"]) for row in endpoint_rows)

    n_x = len(groups) * fold_count
    label_slack_start = n_x
    n_label_slack = len(labels) * fold_count
    location_slack_start = label_slack_start + n_label_slack
    n_location_slack = len(locations) * fold_count
    n_variables = n_x + n_label_slack + n_location_slack

    def x_index(group: str, fold: int) -> int:
        return group_index[group] * fold_count + fold

    def label_slack_index(label: str, fold: int) -> int:
        return label_slack_start + labels.index(label) * fold_count + fold

    def location_slack_index(location: int, fold: int) -> int:
        return location_slack_start + locations.index(location) * fold_count + fold

    matrix_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: Mapping[int, float], low: float, high: float) -> None:
        row = np.zeros(n_variables, dtype=np.float64)
        for index, value in coefficients.items():
            row[int(index)] = float(value)
        matrix_rows.append(row)
        lower.append(float(low))
        upper.append(float(high))

    for group in groups:
        add({x_index(group, fold): 1 for fold in range(fold_count)}, 1, 1)
    for fold in range(fold_count):
        add(
            {x_index(group, fold): 1 for group in groups},
            int(folds_config["development_event_groups_per_fold"]),
            int(folds_config["development_event_groups_per_fold"]),
        )
    for label in labels:
        total_windows = total_by_label[label]
        total_groups = len(label_groups[label])
        for fold in range(fold_count):
            windows = {
                x_index(group, fold): by_group_label[(group, label)]
                for group in groups
                if by_group_label[(group, label)]
            }
            label_group_terms = {
                x_index(group, fold): 1 for group in label_groups[label]
            }
            add(
                windows,
                int(folds_config["minimum_test_windows_per_label_per_fold"]),
                total_windows
                - int(folds_config["minimum_train_windows_per_label_per_fold"]),
            )
            add(
                label_group_terms,
                int(folds_config["minimum_test_event_groups_per_label_per_fold"]),
                total_groups
                - int(folds_config["minimum_train_event_groups_per_label_per_fold"]),
            )
            slack = label_slack_index(label, fold)
            add(
                {**{index: 3 * value for index, value in windows.items()}, slack: -1},
                -np.inf,
                total_windows,
            )
            add(
                {**{index: -3 * value for index, value in windows.items()}, slack: -1},
                -np.inf,
                -total_windows,
            )
    total_endpoint = len(endpoint_rows)
    for fold in range(fold_count):
        test_total = {
            x_index(group, fold): by_group[group]
            for group in groups
            if by_group[group]
        }
        for group in groups:
            coefficients = dict(test_total)
            coefficients[x_index(group, fold)] = (
                coefficients.get(x_index(group, fold), 0.0) - 2 * by_group[group]
            )
            add(coefficients, 0, np.inf)
        for group in groups:
            coefficients = {index: -value for index, value in test_total.items()}
            coefficients[x_index(group, fold)] = (
                coefficients.get(x_index(group, fold), 0.0) + 2 * by_group[group]
            )
            add(coefficients, 2 * by_group[group] - total_endpoint, np.inf)
    for location in locations:
        for fold in range(fold_count):
            counts = {
                x_index(group, fold): by_group_location[(group, location)]
                for group in groups
                if by_group_location[(group, location)]
            }
            slack = location_slack_index(location, fold)
            add(
                {**{index: 3 * value for index, value in counts.items()}, slack: -1},
                -np.inf,
                total_by_location[location],
            )
            add(
                {**{index: -3 * value for index, value in counts.items()}, slack: -1},
                -np.inf,
                -total_by_location[location],
            )

    objective = np.zeros(n_variables, dtype=np.float64)
    objective[label_slack_start:location_slack_start] = 1e9
    objective[location_slack_start:] = 1e5
    for group in groups:
        for fold in range(fold_count):
            objective[x_index(group, fold)] = int(
                salted_digest(str(folds_config["tie_salt"]), f"{group}|{fold}")[:12],
                16,
            ) / float(16**12)
    variable_lower = np.zeros(n_variables, dtype=np.float64)
    variable_upper = np.full(n_variables, np.inf, dtype=np.float64)
    variable_upper[:n_x] = 1.0
    integrality = np.zeros(n_variables, dtype=np.int8)
    integrality[:n_x] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            np.vstack(matrix_rows), np.asarray(lower), np.asarray(upper)
        ),
        options={"presolve": True, "disp": False},
    )
    solver = {
        "implementation": "scipy.optimize.milp_HiGHS",
        "runs": 1,
        "status_code": int(result.status),
        "message": str(result.message),
        "success": bool(result.success),
        "objective": None if result.fun is None else float(result.fun),
        "binary_variables": n_x,
        "continuous_slack_variables": n_label_slack + n_location_slack,
        "linear_constraints": len(matrix_rows),
    }
    if not result.success:
        return {
            "schema_version": "aea-coarse-action-split-donor-v3",
            "protocol_id": PROTOCOL_ID,
            "status": "infeasible_certified"
            if int(result.status) == 2
            else "solver_integrity_failure",
            "retained_labels": list(labels),
            "endpoint_rows": len(endpoint_rows),
            "global_support": support_summary(endpoint_rows, "observable_action"),
            "solver": solver,
            "constraints": dict(folds_config),
            "relaxation_or_retry_performed": False,
            "split_gate_passed": False,
        }

    assignments = {}
    for group in groups:
        values = [result.x[x_index(group, fold)] for fold in range(fold_count)]
        fold = int(np.argmax(values))
        if not np.isclose(values[fold], 1.0, atol=1e-6):
            raise AssertionError("MILP returned non-integral v3 assignment")
        assignments[group] = fold
    held_out = np.zeros(len(endpoint_rows), dtype=np.int64)
    fold_rows = []
    for fold in range(fold_count):
        test_groups = sorted(
            group for group, assigned in assignments.items() if assigned == fold
        )
        train_groups = sorted(set(groups) - set(test_groups))
        test_indices = [
            index
            for index, row in enumerate(endpoint_rows)
            if str(row["event_group"]) in test_groups
        ]
        train_indices = [
            index
            for index, row in enumerate(endpoint_rows)
            if str(row["event_group"]) in train_groups
        ]
        held_out[test_indices] += 1
        training_donors = {
            str(seed): group_safe_bijection(
                train_indices,
                endpoint_rows,
                int(seed),
                str(protocol["donors"]["training_salt"]),
            )
            for seed in protocol["donors"]["training_intervention_seeds"]
        }
        test_donors = {
            str(seed): group_safe_bijection(
                test_indices,
                endpoint_rows,
                int(seed),
                str(protocol["donors"]["test_salt"]),
            )
            for seed in protocol["donors"]["imu_diagnostic_test_seeds"]
        }
        fold_rows.append({
            "fold": fold,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_example_ids": [str(endpoint_rows[index]["example_id"]) for index in train_indices],
            "test_example_ids": [str(endpoint_rows[index]["example_id"]) for index in test_indices],
            "train_event_groups": train_groups,
            "test_event_groups": test_groups,
            "event_group_overlap": sorted(set(train_groups) & set(test_groups)),
            "sequence_overlap": sorted(
                {str(endpoint_rows[index]["sequence_id"]) for index in train_indices}
                & {str(endpoint_rows[index]["sequence_id"]) for index in test_indices}
            ),
            "train_support": support_summary(
                [endpoint_rows[index] for index in train_indices], "observable_action"
            ),
            "test_support": support_summary(
                [endpoint_rows[index] for index in test_indices], "observable_action"
            ),
            "training_intervention_donors": training_donors,
            "imu_diagnostic_test_donors": test_donors,
        })
    checks = {
        "all_18_groups_assigned_once": len(assignments) == 18,
        "exactly_six_groups_per_fold": all(
            len(row["test_event_groups"])
            == int(folds_config["development_event_groups_per_fold"])
            for row in fold_rows
        ),
        "each_endpoint_row_held_out_once": bool(np.all(held_out == 1)),
        "zero_event_group_leakage": all(not row["event_group_overlap"] for row in fold_rows),
        "zero_sequence_leakage": all(not row["sequence_overlap"] for row in fold_rows),
        "all_training_donors_feasible": all(
            donor["feasible"]
            for row in fold_rows
            for donor in row["training_intervention_donors"].values()
        ),
        "all_imu_test_donors_feasible": all(
            donor["feasible"]
            for row in fold_rows
            for donor in row["imu_diagnostic_test_donors"].values()
        ),
        "zero_donor_self_matches": all(
            donor["self_match_count"] == 0
            for row in fold_rows
            for family in ("training_intervention_donors", "imu_diagnostic_test_donors")
            for donor in row[family].values()
        ),
        "zero_donor_same_group_matches": all(
            donor["same_event_group_match_count"] == 0
            for row in fold_rows
            for family in ("training_intervention_donors", "imu_diagnostic_test_donors")
            for donor in row[family].values()
        ),
    }
    return {
        "schema_version": "aea-coarse-action-split-donor-v3",
        "protocol_id": PROTOCOL_ID,
        "status": "feasible" if all(checks.values()) else "postsolver_integrity_failure",
        "retained_labels": list(labels),
        "endpoint_rows": len(endpoint_rows),
        "endpoint_row_manifest": endpoint_rows,
        "global_support": support_summary(endpoint_rows, "observable_action"),
        "solver": solver,
        "constraints": dict(folds_config),
        "group_assignments": assignments,
        "folds": fold_rows,
        "checks": checks,
        "relaxation_or_retry_performed": False,
        "split_gate_passed": all(checks.values()),
    }
