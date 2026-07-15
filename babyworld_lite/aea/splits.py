from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AEASplit:
    name: str
    family: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    supported_actions: tuple[str, ...]
    purged_indices: tuple[int, ...]
    audit: Mapping[str, Any]


def _target(record: Mapping[str, Any]) -> Mapping[str, str]:
    return record["evaluation_targets"]


def _finalize(
    name: str,
    family: str,
    records: Sequence[Mapping[str, Any]],
    train: Sequence[int],
    test: Sequence[int],
    purged: Sequence[int],
    minimum_test_examples_per_action: int,
    details: Mapping[str, Any],
) -> AEASplit | None:
    train_actions = Counter(str(_target(records[index])["action_verb"]) for index in train)
    test_actions = Counter(str(_target(records[index])["action_verb"]) for index in test)
    supported = tuple(sorted(
        action for action, count in test_actions.items()
        if count >= minimum_test_examples_per_action and train_actions[action] >= 1
    ))
    if len(supported) < 2 and family != "held_out_composition":
        return None
    filtered_test = tuple(
        index for index in test if str(_target(records[index])["action_verb"]) in supported
    )
    if not train or not filtered_test:
        return None
    train_sequences = {str(records[index]["sequence_id"]) for index in train}
    test_sequences = {str(records[index]["sequence_id"]) for index in filtered_test}
    train_events = {str(records[index]["event_group"]) for index in train}
    test_events = {str(records[index]["event_group"]) for index in filtered_test}
    sequence_overlap = sorted(train_sequences & test_sequences)
    event_overlap = sorted(train_events & test_events)
    audit = {
        **details,
        "train_count": len(train),
        "test_count_before_action_support_filter": len(test),
        "test_count": len(filtered_test),
        "purged_count": len(purged),
        "supported_actions": list(supported),
        "train_action_counts": dict(train_actions),
        "test_action_counts": dict(test_actions),
        "sequence_overlap": sequence_overlap,
        "concurrent_event_overlap": event_overlap,
        "valid": not sequence_overlap and not event_overlap,
    }
    if not audit["valid"]:
        raise AssertionError(f"leakage in {name}: {audit}")
    return AEASplit(
        name=name,
        family=family,
        train_indices=tuple(train),
        test_indices=filtered_test,
        supported_actions=supported,
        purged_indices=tuple(purged),
        audit=audit,
    )


def held_out_location_splits(
    records: Sequence[Mapping[str, Any]],
    locations: Sequence[int],
    minimum_test_examples_per_action: int,
) -> list[AEASplit]:
    output: list[AEASplit] = []
    for location in locations:
        test = [i for i, row in enumerate(records) if int(row["location"]) == int(location)]
        train = [i for i, row in enumerate(records) if int(row["location"]) != int(location)]
        split = _finalize(
            f"held_out_location_{location}", "held_out_location", records,
            train, test, (), minimum_test_examples_per_action,
            {"held_out_location": int(location), "location_5_script_scope_limited": int(location) == 5},
        )
        if split is not None:
            output.append(split)
    return output


def held_out_wearer_session_splits(
    records: Sequence[Mapping[str, Any]],
    minimum_test_examples_per_action: int,
    maximum_folds: int = 5,
) -> list[AEASplit]:
    """Leave out release-visible wearer-session proxies and purge concurrent views."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_group[str(row["wearer_session_group"])].append(index)
    eligible = sorted(
        by_group,
        key=lambda group: (-len(by_group[group]), group),
    )
    # Prefer at most one high-support wearer-session proxy per location.
    selected: list[str] = []
    seen_locations: set[int] = set()
    for group in eligible:
        location = int(records[by_group[group][0]]["location"])
        actions = Counter(_target(records[i])["action_verb"] for i in by_group[group])
        if location in seen_locations or sum(count >= minimum_test_examples_per_action for count in actions.values()) < 2:
            continue
        selected.append(group)
        seen_locations.add(location)
        if len(selected) >= maximum_folds:
            break
    output: list[AEASplit] = []
    all_indices = set(range(len(records)))
    for group in selected:
        test = set(by_group[group])
        test_events = {str(records[index]["event_group"]) for index in test}
        concurrent = {
            index for index, row in enumerate(records)
            if str(row["event_group"]) in test_events and index not in test
        }
        train = sorted(all_indices - test - concurrent)
        split = _finalize(
            f"held_out_wearer_session_{group}", "held_out_wearer_session", records,
            train, sorted(test), sorted(concurrent), minimum_test_examples_per_action,
            {
                "held_out_wearer_session_proxy": group,
                "persistent_person_identity_available": False,
                "concurrent_partner_windows_purged": len(concurrent),
            },
        )
        if split is not None:
            output.append(split)
    return output


def held_out_composition_splits(
    records: Sequence[Mapping[str, Any]],
    compositions: Sequence[tuple[str, str]],
    minimum_test_examples: int = 2,
) -> list[AEASplit]:
    output: list[AEASplit] = []
    all_indices = set(range(len(records)))
    for action, obj in compositions:
        target = {
            index for index, row in enumerate(records)
            if _target(row)["action_verb"] == action and _target(row)["object_noun"] == obj
        }
        if len(target) < minimum_test_examples:
            continue
        test_events = {str(records[index]["event_group"]) for index in target}
        if len(test_events) < 2:
            continue
        purged = {
            index for index, row in enumerate(records)
            if str(row["event_group"]) in test_events and index not in target
        }
        train = sorted(all_indices - target - purged)
        train_actions = {_target(records[index])["action_verb"] for index in train}
        train_objects = {_target(records[index])["object_noun"] for index in train}
        if action not in train_actions or obj not in train_objects:
            continue
        # Composition tests score the held-out action against every separately
        # supported training action, even though the target windows share one label.
        supported = tuple(sorted(train_actions))
        train_sequences = {records[index]["sequence_id"] for index in train}
        test_sequences = {records[index]["sequence_id"] for index in target}
        train_events = {records[index]["event_group"] for index in train}
        audit = {
            "held_out_action": action,
            "held_out_object": obj,
            "train_count": len(train),
            "test_count": len(target),
            "purged_count": len(purged),
            "supported_actions": list(supported),
            "exact_composition_absent_from_train": all(
                not (_target(records[index])["action_verb"] == action and _target(records[index])["object_noun"] == obj)
                for index in train
            ),
            "action_supported_separately": action in train_actions,
            "object_supported_separately": obj in train_objects,
            "sequence_overlap": sorted(set(train_sequences) & set(test_sequences)),
            "concurrent_event_overlap": sorted(train_events & test_events),
        }
        audit["valid"] = all((
            audit["exact_composition_absent_from_train"],
            audit["action_supported_separately"],
            audit["object_supported_separately"],
            not audit["sequence_overlap"],
            not audit["concurrent_event_overlap"],
        ))
        if not audit["valid"]:
            raise AssertionError(f"leakage in composition {action}/{obj}: {audit}")
        output.append(AEASplit(
            name=f"held_out_composition_{action}_{obj}",
            family="held_out_composition",
            train_indices=tuple(train),
            test_indices=tuple(sorted(target)),
            supported_actions=supported,
            purged_indices=tuple(sorted(purged)),
            audit=audit,
        ))
    return output
