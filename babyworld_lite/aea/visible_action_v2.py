from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "aea-visible-action-v2"
MODELED_ACTIONS = (
    "locomotion_posture",
    "reach_grasp",
    "transport_place",
    "state_change_operate",
    "food_material_handling",
    "clean_groom",
)
ALL_ACTIONS = (*MODELED_ACTIONS, "other_observable", "none_visible", "uncertain")
ASR_REFERENT = ("yes", "no", "unclear")
AGENCIES = (
    "wearer",
    "other_person",
    "narrated_media_or_phone",
    "figurative_or_nonaction",
    "unclear",
)
TEMPORAL_ALIGNMENTS = (
    "aligned_within_window",
    "action_precedes_anchor",
    "action_follows_anchor",
    "no_corresponding_action",
    "unclear",
)
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


def apply_protocol_amendment(
    protocol: Mapping[str, Any], amendment: Mapping[str, Any]
) -> dict[str, Any]:
    if amendment.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("protocol amendment ID does not match parent")
    if int(amendment.get("amendment", 0)) != 1:
        raise ValueError("only frozen AEA visible-action amendment 1 is supported")
    if not amendment.get("all_other_protocol_fields_unchanged"):
        raise ValueError("protocol amendment attempts an unbounded change")
    effective = copy.deepcopy(dict(protocol))
    change = amendment["changes"]["sample"]
    if int(change["sample_size"]) != int(effective["sample"]["size"]):
        raise ValueError("amendment changed frozen sample size")
    if int(change["prior_audit_ids_included"]) != int(
        effective["sample"]["prior_audit_ids_included"]
    ):
        raise ValueError("amendment changed frozen prior-audit inclusion")
    effective["sample"]["target_capped_by_total_group_support"] = bool(
        change["target_capped_by_total_group_support"]
    )
    effective["sample"]["minimum_fill"] = str(change["minimum_fill"])
    effective["sample"]["remainder_until_sample_size"] = list(
        change["remainder_until_sample_size"]
    )
    effective["effective_amendment"] = {
        "number": 1,
        "frozen_at_utc": amendment["frozen_at_utc"],
        "reason": amendment["reason"],
    }
    return effective


def validate_frozen_development_inputs(
    development_path: str | Path,
    partition_path: str | Path,
    prior_prelabel_path: str | Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate membership and hashes before any media path is constructed."""
    source = protocol["sources"]
    paths = {
        "development": Path(development_path),
        "partition": Path(partition_path),
        "prior_prelabel": Path(prior_prelabel_path),
    }
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    hash_checks = {
        "development": observed_hashes["development"]
        == source["development_examples_sha256"],
        "partition": observed_hashes["partition"]
        == source["partition_manifest_sha256"],
        "prior_prelabel": observed_hashes["prior_prelabel"]
        == source["prior_audit_prelabel_sha256"],
    }
    if not all(hash_checks.values()):
        raise ValueError(f"frozen v2 input hash mismatch: {hash_checks}")

    rows = load_jsonl(paths["development"])
    partition = load_json(paths["partition"])
    prelabel = load_json(paths["prior_prelabel"])
    prior_rows = list(prelabel["rows"])
    entries = {str(row["example_id"]): row for row in partition["entries"]}
    expected_ids = {
        example_id
        for example_id, row in entries.items()
        if row["partition"] == "development"
    }
    reserve_ids = set(entries) - expected_ids
    reserve_groups = set(map(str, source["reserve_event_groups"]))
    observed_ids = [str(row["example_id"]) for row in rows]
    prior_ids = [str(row["example_id"]) for row in prior_rows]
    prior_id_digest = hashlib.sha256(
        ("\n".join(sorted(prior_ids)) + "\n").encode()
    ).hexdigest()
    checks = {
        "protocol_id": protocol["protocol_id"] == PROTOCOL_ID,
        "row_count": len(rows) == int(source["development_windows"]),
        "unique_development_ids": len(observed_ids) == len(set(observed_ids)),
        "exact_development_membership": set(observed_ids) == expected_ids,
        "zero_reserve_id_overlap": not bool(set(observed_ids) & reserve_ids),
        "zero_reserve_group_overlap": not any(
            str(row["event_group"]) in reserve_groups for row in rows
        ),
        "development_metadata_matches_partition": all(
            str(row["event_group"])
            == str(entries[str(row["example_id"])]["event_group"])
            and str(row["sequence_id"])
            == str(entries[str(row["example_id"])]["sequence_id"])
            for row in rows
        ),
        "prior_prelabel_count": len(prior_rows)
        == int(protocol["sample"]["prior_audit_ids_included"]),
        "unique_prior_ids": len(prior_ids) == len(set(prior_ids)),
        "prior_ids_are_development": set(prior_ids) <= expected_ids,
        "prior_id_set_digest": prior_id_digest
        == source["prior_audit_id_set_sha256"],
        "prior_labels_absent": all(
            row.get("audit_label") is None and row.get("rationale") is None
            for row in prior_rows
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"v2 development/reserve preflight failed: {checks}")
    return rows, prior_rows, {
        "checks": checks,
        "observed_hashes": observed_hashes,
        "reserve_rgb_files_opened": 0,
        "reserve_imu_files_opened": 0,
        "media_access_started": False,
    }


def select_frozen_sample(
    rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(row["example_id"]): row for row in rows}
    selected_ids = {str(row["example_id"]) for row in prior_rows}
    target = int(protocol["sample"]["target_per_event_group"])
    sample_size = int(protocol["sample"]["size"])
    selection_salt = str(protocol["sample"]["selection_salt"])
    group_salt = str(protocol["sample"]["group_order_salt"])
    blind_salt = str(protocol["sample"]["blind_id_salt"])
    groups = sorted({str(row["event_group"]) for row in rows})
    total_group_support = Counter(str(row["event_group"]) for row in rows)
    group_count = Counter(str(by_id[item]["event_group"]) for item in selected_ids)
    action_count = Counter(
        str(by_id[item]["evaluation_targets"]["action_verb"])
        for item in selected_ids
    )
    total_action_support = Counter(
        str(row["evaluation_targets"]["action_verb"]) for row in rows
    )
    group_order = sorted(groups, key=lambda group: (salted_digest(group_salt, group), group))

    def choose_from_group(group: str) -> Mapping[str, Any]:
        candidates = [
            row
            for row in rows
            if str(row["event_group"]) == group
            and str(row["example_id"]) not in selected_ids
        ]
        if not candidates:
            raise ValueError(f"event group {group} has no unselected candidate")
        return min(
            candidates,
            key=lambda row: (
                action_count[str(row["evaluation_targets"]["action_verb"])],
                total_action_support[str(row["evaluation_targets"]["action_verb"])],
                salted_digest(selection_salt, str(row["example_id"])),
                str(row["example_id"]),
            ),
        )

    def add(chosen: Mapping[str, Any]) -> None:
        example_id = str(chosen["example_id"])
        action = str(chosen["evaluation_targets"]["action_verb"])
        group = str(chosen["event_group"])
        selected_ids.add(example_id)
        group_count[group] += 1
        action_count[action] += 1

    for group in group_order:
        minimum_target = (
            min(target, total_group_support[group])
            if protocol["sample"].get("target_capped_by_total_group_support")
            else target
        )
        while group_count[group] < minimum_target:
            add(choose_from_group(group))

    while len(selected_ids) < sample_size:
        eligible_groups = [
            group for group in groups
            if group_count[group] < total_group_support[group]
        ]
        if not eligible_groups:
            raise ValueError("development source exhausted before frozen sample size")
        group = min(
            eligible_groups,
            key=lambda item: (
                group_count[item],
                group_count[item] / total_group_support[item],
                salted_digest(group_salt, item),
                item,
            ),
        )
        add(choose_from_group(group))
    if len(selected_ids) != sample_size:
        raise AssertionError(
            f"frozen sample expected {sample_size} rows, selected {len(selected_ids)}"
        )
    if any(
        group_count[group] < min(target, total_group_support[group])
        for group in groups
    ):
        raise AssertionError("amended frozen minimum event-group targets were not met")

    output = []
    blind_ids: set[str] = set()
    for example_id in sorted(selected_ids):
        row = by_id[example_id]
        blind_id = "VA2-" + salted_digest(blind_salt, example_id)[:12].upper()
        if blind_id in blind_ids:
            raise AssertionError("opaque blind ID collision")
        blind_ids.add(blind_id)
        output.append({
            "blind_id": blind_id,
            "example_id": example_id,
            "sequence_id": str(row["sequence_id"]),
            "event_group": str(row["event_group"]),
            "location": int(row["location"]),
            "asr_action_verb": str(row["evaluation_targets"]["action_verb"]),
            "transcript": str(row["model_inputs"]["transcript"]),
            "window": dict(row["window"]),
            "prior_v1_audit_id": example_id
            in {str(item["example_id"]) for item in prior_rows},
        })
    return output


def packet_order(
    sample_rows: Sequence[Mapping[str, Any]], salt: str
) -> list[dict[str, Any]]:
    ordered = sorted(
        sample_rows,
        key=lambda row: (
            salted_digest(salt, str(row["blind_id"])),
            str(row["blind_id"]),
        ),
    )
    return [dict(row) for row in ordered]


def validate_annotation_pass(
    rows: Sequence[Mapping[str, Any]],
    expected_blind_ids: Iterable[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = set(map(str, expected_blind_ids))
    observed = [str(row.get("blind_id", "")) for row in rows]
    if len(rows) != int(protocol["sample"]["size"]):
        raise ValueError("annotation pass does not contain 72 rows")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("annotation pass blind IDs are duplicate, missing, or unexpected")
    normalized = []
    for source in rows:
        row = dict(source)
        checks = {
            "asr": row.get("asr_refers_to_visible_wearer_action") in ASR_REFERENT,
            "agency": row.get("agency") in AGENCIES,
            "temporal": row.get("temporal_alignment") in TEMPORAL_ALIGNMENTS,
            "action": row.get("observable_action") in ALL_ACTIONS,
            "confidence": row.get("confidence") in CONFIDENCES,
            "rationale": isinstance(row.get("rationale"), str)
            and 1 <= len(str(row["rationale"]).split())
            <= int(protocol["annotation_schema"]["rationale_max_words"]),
        }
        start = row.get("evidence_frame_start")
        end = row.get("evidence_frame_end")
        action = row.get("observable_action")
        if action in ("none_visible", "uncertain") and start is None and end is None:
            evidence_valid = True
        else:
            evidence_valid = (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start <= end <= int(protocol["annotation_schema"]["evidence_frame_max"])
            )
        checks["evidence"] = evidence_valid
        if not all(checks.values()):
            raise ValueError(f"invalid annotation row {row.get('blind_id')}: {checks}")
        normalized.append(row)
    return normalized


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"successes": int(successes), "total": int(total), "rate": None,
                "ci95_low": None, "ci95_high": None, "method": "Wilson 95%"}
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(p),
        "ci95_low": float(max(0.0, center - radius)),
        "ci95_high": float(min(1.0, center + radius)),
        "method": "two-sided Wilson 95% interval",
    }


def agreement_metric(
    values_a: Sequence[str], values_b: Sequence[str], labels: Sequence[str],
    confidences_a: Sequence[str], confidences_b: Sequence[str],
) -> dict[str, Any]:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("agreement needs paired nonempty values")
    label_order = list(labels)
    index = {label: number for number, label in enumerate(label_order)}
    confusion = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
    for left, right in zip(values_a, values_b):
        confusion[index[left], index[right]] += 1
    exact = float(np.mean(np.asarray(values_a, dtype=object) == np.asarray(values_b, dtype=object)))
    observed = exact
    left_marginal = confusion.sum(axis=1) / confusion.sum()
    right_marginal = confusion.sum(axis=0) / confusion.sum()
    expected = float(left_marginal @ right_marginal)
    kappa = None if np.isclose(1.0 - expected, 0.0) else float((observed - expected) / (1.0 - expected))
    weights = np.asarray([
        min(CONFIDENCE_WEIGHT[left], CONFIDENCE_WEIGHT[right])
        for left, right in zip(confidences_a, confidences_b)
    ])
    matches = np.asarray(values_a, dtype=object) == np.asarray(values_b, dtype=object)
    weighted_exact = float(np.sum(weights * matches) / np.sum(weights))
    return {
        "labels": label_order,
        "confusion_a_rows_b_columns": confusion.tolist(),
        "exact_agreement": exact,
        "cohens_kappa_unweighted": kappa,
        "confidence_weighted_exact_agreement": weighted_exact,
        "n": len(values_a),
    }


def support_summary(rows: Sequence[Mapping[str, Any]], label_key: str) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    labels = sorted({str(row[label_key]) for row in rows})
    for label in labels:
        selected = [row for row in rows if str(row[label_key]) == label]
        by_label[label] = {
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
    return by_label


def summarize_annotations(
    pass_a: Sequence[Mapping[str, Any]],
    pass_b: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    by_blind = {str(row["blind_id"]): row for row in sample_rows}
    valid_a = validate_annotation_pass(pass_a, by_blind, protocol)
    valid_b = validate_annotation_pass(pass_b, by_blind, protocol)
    map_a = {str(row["blind_id"]): row for row in valid_a}
    map_b = {str(row["blind_id"]): row for row in valid_b}
    ordered_ids = sorted(by_blind)
    agreements = {}
    field_specs = {
        "observable_action": ALL_ACTIONS,
        "agency": AGENCIES,
        "temporal_alignment": TEMPORAL_ALIGNMENTS,
        "asr_refers_to_visible_wearer_action": ASR_REFERENT,
    }
    confidences_a = [str(map_a[item]["confidence"]) for item in ordered_ids]
    confidences_b = [str(map_b[item]["confidence"]) for item in ordered_ids]
    for field, labels in field_specs.items():
        agreements[field] = agreement_metric(
            [str(map_a[item][field]) for item in ordered_ids],
            [str(map_b[item][field]) for item in ordered_ids],
            labels,
            confidences_a,
            confidences_b,
        )

    consensus_rows = []
    paired_rows = []
    for blind_id in ordered_ids:
        left = map_a[blind_id]
        right = map_b[blind_id]
        manifest = by_blind[blind_id]
        action = str(left["observable_action"])
        modeled_consensus = (
            action == str(right["observable_action"])
            and action in MODELED_ACTIONS
            and str(left["confidence"]) in ("high", "medium")
            and str(right["confidence"]) in ("high", "medium")
        )
        paired = {
            "blind_id": blind_id,
            "example_id": str(manifest["example_id"]),
            "sequence_id": str(manifest["sequence_id"]),
            "event_group": str(manifest["event_group"]),
            "location": int(manifest["location"]),
            "pass_a": left,
            "pass_b": right,
            "modeled_action_consensus": modeled_consensus,
            "consensus_observable_action": action if modeled_consensus else None,
        }
        paired_rows.append(paired)
        if modeled_consensus:
            consensus_rows.append({
                "blind_id": blind_id,
                "example_id": str(manifest["example_id"]),
                "sequence_id": str(manifest["sequence_id"]),
                "event_group": str(manifest["event_group"]),
                "location": int(manifest["location"]),
                "observable_action": action,
            })

    n = len(ordered_ids)
    judgeable_a = sum(map_a[item]["observable_action"] != "uncertain" for item in ordered_ids)
    judgeable_b = sum(map_b[item]["observable_action"] != "uncertain" for item in ordered_ids)
    action_agreement = agreements["observable_action"]
    agency_agreement = agreements["agency"]
    temporal_agreement = agreements["temporal_alignment"]
    asr_agreement = agreements["asr_refers_to_visible_wearer_action"]
    support = support_summary(consensus_rows, "observable_action")
    retained_support = {
        label: values for label, values in support.items()
        if label in MODELED_ACTIONS
        and values["windows"] >= int(protocol["support"]["minimum_windows_per_retained_label"])
        and values["event_groups"] >= int(protocol["support"]["minimum_event_groups_per_retained_label"])
    }
    low_support_count = sum(
        values["windows"] >= 6 and values["event_groups"] >= 3
        for label, values in support.items() if label in MODELED_ACTIONS
    )
    gate_checks = {
        "complete_valid_passes": len(valid_a) == len(valid_b) == n == int(protocol["sample"]["size"]),
        "judgeable_pass_a": judgeable_a / n >= float(protocol["agreement"]["judgeable_minimum_each_pass"]),
        "judgeable_pass_b": judgeable_b / n >= float(protocol["agreement"]["judgeable_minimum_each_pass"]),
        "action_exact": action_agreement["exact_agreement"] >= float(protocol["agreement"]["action_exact_minimum"]),
        "action_kappa": action_agreement["cohens_kappa_unweighted"] is not None
        and action_agreement["cohens_kappa_unweighted"] >= float(protocol["agreement"]["action_kappa_minimum"]),
        "agency_exact": agency_agreement["exact_agreement"] >= float(protocol["agreement"]["agency_exact_minimum"]),
        "agency_kappa": agency_agreement["cohens_kappa_unweighted"] is not None
        and agency_agreement["cohens_kappa_unweighted"] >= float(protocol["agreement"]["agency_kappa_minimum"]),
        "temporal_exact": temporal_agreement["exact_agreement"] >= float(protocol["agreement"]["temporal_exact_minimum"]),
        "temporal_kappa": temporal_agreement["cohens_kappa_unweighted"] is not None
        and temporal_agreement["cohens_kappa_unweighted"] >= float(protocol["agreement"]["temporal_kappa_minimum"]),
        "asr_referent_exact": asr_agreement["exact_agreement"] >= float(protocol["agreement"]["asr_referent_exact_minimum"]),
        "asr_referent_kappa": asr_agreement["cohens_kappa_unweighted"] is not None
        and asr_agreement["cohens_kappa_unweighted"] >= float(protocol["agreement"]["asr_referent_kappa_minimum"]),
        "modeled_consensus_yield": len(consensus_rows) / n >= float(protocol["agreement"]["modeled_consensus_yield_minimum"]),
        "retained_label_count": len(retained_support) >= int(protocol["support"]["minimum_retained_labels"]),
    }
    severe = {
        "modeled_consensus_yield_below_0_40": len(consensus_rows) / n
        < float(protocol["agreement"]["severe_stop"]["modeled_consensus_yield_below"]),
        "action_exact_below_0_50": action_agreement["exact_agreement"]
        < float(protocol["agreement"]["severe_stop"]["action_exact_below"]),
        "fewer_than_two_labels_with_six_windows_three_groups": low_support_count
        < int(protocol["agreement"]["severe_stop"]["minimum_labels_with_six_windows_three_groups"]),
    }
    return {
        "schema_version": "aea-visible-action-agreement-v2",
        "protocol_id": PROTOCOL_ID,
        "scientific_role": protocol["scientific_role"],
        "annotation_role": protocol["annotation_schema"]["pass_role"],
        "n": n,
        "agreement": agreements,
        "rates": {
            "judgeable_pass_a": wilson(judgeable_a, n),
            "judgeable_pass_b": wilson(judgeable_b, n),
            "modeled_consensus_yield": wilson(len(consensus_rows), n),
            "uncertain_pass_a": wilson(n - judgeable_a, n),
            "uncertain_pass_b": wilson(n - judgeable_b, n),
            "asr_refers_yes_pass_a": wilson(sum(map_a[item]["asr_refers_to_visible_wearer_action"] == "yes" for item in ordered_ids), n),
            "asr_refers_yes_pass_b": wilson(sum(map_b[item]["asr_refers_to_visible_wearer_action"] == "yes" for item in ordered_ids), n),
        },
        "support": support,
        "retained_support": retained_support,
        "gate_checks": gate_checks,
        "annotation_gate_passed": all(gate_checks.values()),
        "severe_failure_checks": severe,
        "severe_annotation_failure": any(severe.values()),
        "consensus_rows": consensus_rows,
        "paired_rows": paired_rows,
    }


def group_safe_bijection(
    indices: Sequence[int], rows: Sequence[Mapping[str, Any]], seed: int, salt: str
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
        raise AssertionError("half-size theorem and bipartite matching disagree")
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
        "self_match_count": None if not feasible else sum(
            index == mapping_indices[index] for index in indices
        ),
        "same_event_group_match_count": None if not feasible else sum(
            groups[index] == groups[mapping_indices[index]] for index in indices
        ),
        "donor_map": donor_map,
        "donor_map_sha256": None if donor_map is None else canonical_digest(donor_map),
    }


def construct_constrained_folds(
    consensus_rows: Sequence[Mapping[str, Any]],
    all_sample_rows: Sequence[Mapping[str, Any]],
    retained_labels: Sequence[str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve the frozen three-fold group assignment once with integer constraints."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    labels = tuple(sorted(map(str, retained_labels)))
    endpoint_rows = [
        dict(row) for row in consensus_rows
        if str(row["observable_action"]) in labels
    ]
    groups = sorted({str(row["event_group"]) for row in all_sample_rows})
    if len(groups) != int(protocol["sources"]["development_event_groups"]):
        raise ValueError("constrained split requires all 18 development event groups")
    fold_count = int(protocol["folds"]["count"])
    if len(labels) < int(protocol["support"]["minimum_retained_labels"]):
        return {
            "schema_version": "aea-visible-action-split-donor-v2",
            "protocol_id": PROTOCOL_ID,
            "status": "not_run_insufficient_retained_labels",
            "retained_labels": list(labels),
            "split_gate_passed": False,
        }
    group_index = {group: number for number, group in enumerate(groups)}
    location_by_group: dict[str, int] = {}
    for row in all_sample_rows:
        group = str(row["event_group"])
        location = int(row["location"])
        if group in location_by_group and location_by_group[group] != location:
            raise ValueError("event group has inconsistent locations")
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

    rows_a: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: Mapping[int, float], low: float, high: float) -> None:
        row = np.zeros(n_variables, dtype=np.float64)
        for index, value in coefficients.items():
            row[int(index)] = float(value)
        rows_a.append(row)
        lower.append(float(low))
        upper.append(float(high))

    for group in groups:
        add({x_index(group, fold): 1 for fold in range(fold_count)}, 1, 1)
    groups_per_fold = int(protocol["folds"]["development_event_groups_per_fold"])
    for fold in range(fold_count):
        add({x_index(group, fold): 1 for group in groups}, groups_per_fold, groups_per_fold)

    for label in labels:
        total_windows = total_by_label[label]
        total_groups = len(label_groups[label])
        for fold in range(fold_count):
            window_coefficients = {
                x_index(group, fold): by_group_label[(group, label)]
                for group in groups if by_group_label[(group, label)]
            }
            group_coefficients = {
                x_index(group, fold): 1
                for group in label_groups[label]
            }
            add(
                window_coefficients,
                int(protocol["folds"]["minimum_test_windows_per_label_per_fold"]),
                total_windows - int(protocol["folds"]["minimum_train_windows_per_label_per_fold"]),
            )
            add(
                group_coefficients,
                int(protocol["folds"]["minimum_test_event_groups_per_label_per_fold"]),
                total_groups - int(protocol["folds"]["minimum_train_event_groups_per_label_per_fold"]),
            )
            slack = label_slack_index(label, fold)
            add(
                {**{index: 3 * value for index, value in window_coefficients.items()}, slack: -1},
                -np.inf,
                total_windows,
            )
            add(
                {**{index: -3 * value for index, value in window_coefficients.items()}, slack: -1},
                -np.inf,
                -total_windows,
            )

    total_endpoint = len(endpoint_rows)
    for fold in range(fold_count):
        test_total_coefficients = {
            x_index(group, fold): by_group[group]
            for group in groups if by_group[group]
        }
        # Test-side complete multipartite derangement bound.
        for group in groups:
            coefficients = dict(test_total_coefficients)
            coefficients[x_index(group, fold)] = (
                coefficients.get(x_index(group, fold), 0.0) - 2 * by_group[group]
            )
            add(coefficients, 0, np.inf)
        # Training-side bound: total - test_total >= 2*c_g when g is in train.
        for group in groups:
            coefficients = {
                index: -value for index, value in test_total_coefficients.items()
            }
            coefficients[x_index(group, fold)] = (
                coefficients.get(x_index(group, fold), 0.0) + 2 * by_group[group]
            )
            add(coefficients, 2 * by_group[group] - total_endpoint, np.inf)

    for location in locations:
        for fold in range(fold_count):
            count_coefficients = {
                x_index(group, fold): by_group_location[(group, location)]
                for group in groups if by_group_location[(group, location)]
            }
            slack = location_slack_index(location, fold)
            add(
                {**{index: 3 * value for index, value in count_coefficients.items()}, slack: -1},
                -np.inf,
                total_by_location[location],
            )
            add(
                {**{index: -3 * value for index, value in count_coefficients.items()}, slack: -1},
                -np.inf,
                -total_by_location[location],
            )

    objective = np.zeros(n_variables, dtype=np.float64)
    objective[label_slack_start:location_slack_start] = 1e9
    objective[location_slack_start:] = 1e5
    tie_salt = str(protocol["folds"]["tie_salt"])
    for group in groups:
        for fold in range(fold_count):
            objective[x_index(group, fold)] = int(
                salted_digest(tie_salt, f"{group}|{fold}")[:12], 16
            ) / float(16 ** 12)

    variable_lower = np.zeros(n_variables, dtype=np.float64)
    variable_upper = np.full(n_variables, np.inf, dtype=np.float64)
    variable_upper[:n_x] = 1.0
    integrality = np.zeros(n_variables, dtype=np.int8)
    integrality[:n_x] = 1
    constraint = LinearConstraint(np.vstack(rows_a), np.asarray(lower), np.asarray(upper))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=constraint,
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
        "linear_constraints": len(rows_a),
    }
    if not result.success:
        status = "infeasible_certified" if int(result.status) == 2 else "solver_integrity_failure"
        return {
            "schema_version": "aea-visible-action-split-donor-v2",
            "protocol_id": PROTOCOL_ID,
            "status": status,
            "retained_labels": list(labels),
            "endpoint_rows": len(endpoint_rows),
            "global_support": support_summary(endpoint_rows, "observable_action"),
            "solver": solver,
            "constraints": dict(protocol["folds"]),
            "relaxation_or_retry_performed": False,
            "split_gate_passed": False,
        }

    assignments: dict[str, int] = {}
    for group in groups:
        values = [result.x[x_index(group, fold)] for fold in range(fold_count)]
        fold = int(np.argmax(values))
        if not np.isclose(values[fold], 1.0, atol=1e-6) or sum(value > 0.5 for value in values) != 1:
            raise AssertionError("MILP returned non-integral group assignment")
        assignments[group] = fold

    fold_rows = []
    held_out_counts = np.zeros(len(endpoint_rows), dtype=np.int64)
    donor_checks_pass = True
    for fold in range(fold_count):
        test_groups = sorted(group for group, assigned in assignments.items() if assigned == fold)
        train_groups = sorted(set(groups) - set(test_groups))
        test_indices = [
            index for index, row in enumerate(endpoint_rows)
            if str(row["event_group"]) in test_groups
        ]
        train_indices = [
            index for index, row in enumerate(endpoint_rows)
            if str(row["event_group"]) in train_groups
        ]
        held_out_counts[test_indices] += 1
        train_sequences = {str(endpoint_rows[index]["sequence_id"]) for index in train_indices}
        test_sequences = {str(endpoint_rows[index]["sequence_id"]) for index in test_indices}
        training_donors = {
            str(seed): group_safe_bijection(
                train_indices, endpoint_rows, int(seed), "aea-visible-action-v2-train"
            )
            for seed in protocol["donors"]["training_intervention_seeds"]
        }
        test_donors = {
            str(seed): group_safe_bijection(
                test_indices, endpoint_rows, int(seed), "aea-visible-action-v2-test"
            )
            for seed in protocol["donors"]["imu_diagnostic_test_seeds"]
        }
        donor_checks_pass &= all(item["feasible"] for item in training_donors.values())
        donor_checks_pass &= all(item["feasible"] for item in test_donors.values())
        fold_rows.append({
            "fold": fold,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_example_ids": [str(endpoint_rows[index]["example_id"]) for index in train_indices],
            "test_example_ids": [str(endpoint_rows[index]["example_id"]) for index in test_indices],
            "train_event_groups": train_groups,
            "test_event_groups": test_groups,
            "event_group_overlap": sorted(set(train_groups) & set(test_groups)),
            "sequence_overlap": sorted(train_sequences & test_sequences),
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
            len(row["test_event_groups"]) == int(protocol["folds"]["development_event_groups_per_fold"])
            for row in fold_rows
        ),
        "each_endpoint_row_held_out_once": bool(np.all(held_out_counts == 1)),
        "zero_event_group_leakage": all(not row["event_group_overlap"] for row in fold_rows),
        "zero_sequence_leakage": all(not row["sequence_overlap"] for row in fold_rows),
        "all_training_donors_feasible": all(
            item["feasible"]
            for row in fold_rows for item in row["training_intervention_donors"].values()
        ),
        "all_imu_test_donors_feasible": all(
            item["feasible"]
            for row in fold_rows for item in row["imu_diagnostic_test_donors"].values()
        ),
        "zero_donor_self_matches": all(
            item["self_match_count"] == 0
            for row in fold_rows
            for family in ("training_intervention_donors", "imu_diagnostic_test_donors")
            for item in row[family].values()
        ),
        "zero_donor_same_group_matches": all(
            item["same_event_group_match_count"] == 0
            for row in fold_rows
            for family in ("training_intervention_donors", "imu_diagnostic_test_donors")
            for item in row[family].values()
        ),
    }
    if donor_checks_pass != (
        checks["all_training_donors_feasible"] and checks["all_imu_test_donors_feasible"]
    ):
        raise AssertionError("donor feasibility aggregation disagrees")
    return {
        "schema_version": "aea-visible-action-split-donor-v2",
        "protocol_id": PROTOCOL_ID,
        "status": "feasible" if all(checks.values()) else "postsolver_integrity_failure",
        "retained_labels": list(labels),
        "endpoint_rows": len(endpoint_rows),
        "endpoint_row_manifest": endpoint_rows,
        "global_support": support_summary(endpoint_rows, "observable_action"),
        "solver": solver,
        "constraints": dict(protocol["folds"]),
        "group_assignments": assignments,
        "folds": fold_rows,
        "checks": checks,
        "relaxation_or_retry_performed": False,
        "split_gate_passed": all(checks.values()),
    }
