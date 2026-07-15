from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SEQUENCE_RE = re.compile(
    r"^loc(?P<location>\d+)_script(?P<script>\d+)_seq(?P<sequence>\d+)_rec(?P<recording>\d+)$"
)
MPS_COMPONENT_PREFIX = "mps_"
ACTION_ANCHOR_FORMS = {
    "put", "putting", "take", "taking", "took", "get", "getting", "got",
    "grab", "grabbing", "grabbed", "make", "making", "made", "pour", "pouring",
    "poured", "open", "opening", "opened", "close", "closing", "closed", "wash",
    "washing", "washed", "clean", "cleaning", "cleaned", "cook", "cooking", "cooked",
    "eat", "eating", "ate", "drink", "drinking", "drank", "fold", "folding", "folded",
    "play", "playing", "played", "read", "reading", "watch", "watching", "watched",
    "move", "moving", "moved", "turn", "turning", "turned", "brush", "brushing",
    "brushed", "wipe", "wiping", "wiped", "set", "setting", "cut", "cutting", "serve",
    "serving", "served", "pick", "picking", "picked", "bring", "bringing", "brought",
    "walk", "walking", "walked", "sit", "sitting", "sat", "stand", "standing", "stood",
    "press", "pressing", "pressed", "remove", "removing", "removed", "vacuum",
    "vacuuming", "vacuumed", "stack", "stacking", "stacked",
}


@dataclass(frozen=True, order=True)
class AEASequenceId:
    location: int
    script: int
    sequence: int
    recording: int

    @classmethod
    def parse(cls, value: str) -> "AEASequenceId":
        match = SEQUENCE_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid AEA sequence id: {value!r}")
        return cls(**{key: int(value) for key, value in match.groupdict().items()})

    @property
    def name(self) -> str:
        return (
            f"loc{self.location}_script{self.script}_seq{self.sequence}_rec{self.recording}"
        )

    @property
    def event_group(self) -> str:
        """Concurrent glasses share this event/session group."""
        return f"loc{self.location}_script{self.script}_seq{self.sequence}"

    @property
    def wearer_session_group(self) -> str:
        """Release-visible wearer-session proxy, not a persistent person identity."""
        return f"loc{self.location}_script{self.script}_rec{self.recording}"


@dataclass(frozen=True)
class SafeAsset:
    filename: str
    sha1sum: str
    file_size_bytes: int


@dataclass(frozen=True)
class SafeSequence:
    sequence_id: AEASequenceId
    assets: Mapping[str, SafeAsset]


@dataclass(frozen=True)
class SafeManifest:
    dataset_name: str
    release: str
    sequence_config: Mapping[str, Any]
    sequences: Mapping[str, SafeSequence]


def load_safe_manifest(path: str | Path) -> SafeManifest:
    """Load only non-secret metadata from an expiring AEA link manifest.

    Signed ``download_url`` values are deliberately discarded at this boundary
    so callers used for planning and reporting cannot accidentally serialize or
    print them.
    """
    with Path(path).open() as handle:
        raw = json.load(handle)
    if set(raw) != {"sequence_config", "sequences"}:
        raise ValueError("unexpected AEA download manifest top-level schema")
    config = raw["sequence_config"]
    safe: dict[str, SafeSequence] = {}
    for name, components in raw["sequences"].items():
        sequence_id = AEASequenceId.parse(name)
        assets: dict[str, SafeAsset] = {}
        for component, values in components.items():
            required = {"filename", "sha1sum", "file_size_bytes", "download_url"}
            if not required <= set(values):
                raise ValueError(f"{name}/{component} is missing required metadata")
            assets[str(component)] = SafeAsset(
                filename=str(values["filename"]),
                sha1sum=str(values["sha1sum"]),
                file_size_bytes=int(values["file_size_bytes"]),
            )
        safe[name] = SafeSequence(sequence_id, assets)
    return SafeManifest(
        dataset_name=str(config.get("dataset_name", "AriaEverydayActivities")),
        release=str(config.get("release", "unknown")),
        # Keep the planning representation intentionally narrow. The private
        # downloader reads operational fields at its isolated boundary; audit
        # and planning code only need these two non-secret identifiers.
        sequence_config={
            "dataset_name": str(config.get("dataset_name", "AriaEverydayActivities")),
            "release": str(config.get("release", "unknown")),
        },
        sequences=safe,
    )


def component_budget(
    manifest: SafeManifest,
    sequence_names: Sequence[str],
    components: Sequence[str],
) -> dict[str, Any]:
    by_component = {
        component: sum(
            manifest.sequences[name].assets[component].file_size_bytes
            for name in sequence_names
        )
        for component in components
    }
    return {
        "bytes_by_component": by_component,
        "total_bytes": sum(by_component.values()),
        "total_gib": round(sum(by_component.values()) / 1024**3, 3),
    }


def manifest_summary(manifest: SafeManifest) -> dict[str, Any]:
    names = sorted(manifest.sequences)
    components = sorted({key for row in manifest.sequences.values() for key in row.assets})
    return {
        "dataset_name": manifest.dataset_name,
        "release": manifest.release,
        "sequence_count": len(names),
        "location_counts": dict(sorted(Counter(
            manifest.sequences[name].sequence_id.location for name in names
        ).items())),
        "component_totals_bytes": {
            component: sum(
                manifest.sequences[name].assets[component].file_size_bytes for name in names
            )
            for component in components
        },
    }


def action_anchor_counts(
    annotation_root: str | Path,
    action_forms: Sequence[str] = tuple(ACTION_ANCHOR_FORMS),
) -> dict[str, int]:
    """Count predeclared action forms without reading any restricted VRS data."""
    root = Path(annotation_root)
    allowed = {str(value).lower() for value in action_forms}
    counts: dict[str, int] = {}
    for speech_path in root.glob("*/speech.csv"):
        total = 0
        with speech_path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                word = re.sub(r"[^a-z']", "", str(row.get("written", "")).lower())
                total += word in allowed
        counts[speech_path.parent.name] = total
    return counts


def _spread_select(
    rows: Sequence[SafeSequence], count: int, anchor_counts: Mapping[str, int]
) -> list[SafeSequence]:
    if len(rows) < count:
        raise ValueError(f"need {count} rows but only {len(rows)} are available")
    chosen: list[SafeSequence] = []
    by_support: dict[int, list[SafeSequence]] = defaultdict(list)
    for row in rows:
        by_support[int(anchor_counts.get(row.sequence_id.name, 0))].append(row)
    for _support, group in sorted(by_support.items(), reverse=True):
        remaining = count - len(chosen)
        if remaining <= 0:
            break
        if len(group) <= remaining:
            chosen.extend(group)
            continue
        ordered = sorted(group, key=lambda row: (
            row.assets["main_vrs"].file_size_bytes, row.sequence_id.name
        ))
        # Midpoints of equal quantile bins provide duration/size spread among
        # rows tied on the annotation support criterion.
        positions = [
            min(len(ordered) - 1, math.floor((i + 0.5) * len(ordered) / remaining))
            for i in range(remaining)
        ]
        chosen.extend(ordered[position] for position in positions)
    return chosen


def _select_concurrent_pair(
    rows: Sequence[SafeSequence], anchor_counts: Mapping[str, int]
) -> list[SafeSequence] | None:
    grouped: dict[int, dict[int, SafeSequence]] = defaultdict(dict)
    for row in rows:
        grouped[row.sequence_id.sequence][row.sequence_id.recording] = row
    pairs = [
        [by_recording[1], by_recording[2]]
        for _sequence, by_recording in sorted(grouped.items())
        if {1, 2} <= set(by_recording)
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda pair: (
        -sum(anchor_counts.get(row.sequence_id.name, 0) for row in pair),
        sum(row.assets["main_vrs"].file_size_bytes for row in pair),
        pair[0].sequence_id.sequence,
    ))
    return pairs[0]


def build_balanced_subset_plan(
    manifest: SafeManifest,
    annotation_root: str | Path | None = None,
    action_forms: Sequence[str] = tuple(ACTION_ANCHOR_FORMS),
) -> dict[str, Any]:
    """Lock a 40-recording subset balanced over locations and scripts.

    Locations 1--4 contribute two script-1 recordings, one concurrent pair
    from each of scripts 2 and 3, and one recording from scripts 4 and 5.
    Location 5 contains only scripts 4 and 5, so it contributes four of each.
    This yields eight recordings per location and eight per script.
    """
    strata: dict[tuple[int, int], list[SafeSequence]] = defaultdict(list)
    for row in manifest.sequences.values():
        strata[(row.sequence_id.location, row.sequence_id.script)].append(row)

    anchors = (
        action_anchor_counts(annotation_root, action_forms)
        if annotation_root is not None else {}
    )
    selected: list[SafeSequence] = []
    for location in range(1, 5):
        selected.extend(_spread_select(strata[(location, 1)], 2, anchors))
        for script in (2, 3):
            pair = _select_concurrent_pair(strata[(location, script)], anchors)
            selected.extend(pair or _spread_select(strata[(location, script)], 2, anchors))
        selected.extend(_spread_select(strata[(location, 4)], 1, anchors))
        selected.extend(_spread_select(strata[(location, 5)], 1, anchors))
    selected.extend(_spread_select(strata[(5, 4)], 4, anchors))
    selected.extend(_spread_select(strata[(5, 5)], 4, anchors))

    names = sorted({row.sequence_id.name for row in selected})
    if len(names) != 40:
        raise AssertionError(f"balanced selection should contain 40 unique rows, got {len(names)}")
    locations = Counter(AEASequenceId.parse(name).location for name in names)
    scripts = Counter(AEASequenceId.parse(name).script for name in names)
    if set(locations.values()) != {8} or set(scripts.values()) != {8}:
        raise AssertionError(f"selection is not balanced: locations={locations}, scripts={scripts}")

    return {
        "schema_version": "aea-subset-plan-v1",
        "dataset": {
            "name": manifest.dataset_name,
            "release": manifest.release,
            "scientific_role": "adult_partly_scripted_sensor_format_analogue_not_developmental_evidence",
        },
        "selection": {
            "strategy": "location_script_balanced_action_support_then_size_spread_with_concurrent_pairs",
            "sequence_count": len(names),
            "sequences": names,
            "location_counts": dict(sorted(locations.items())),
            "script_counts": dict(sorted(scripts.items())),
            "components_initial": ["annotations", "main_vrs"],
            "mps_included": False,
            "selected_action_anchor_count": sum(anchors.get(name, 0) for name in names),
            "annotation_screening": (
                "broad predeclared high-recall action-form counts only; the locked "
                "preprocessing lexicon is narrower; no VRS inspected"
                if annotation_root is not None else "not_available"
            ),
            "budget": component_budget(manifest, names, ("annotations", "main_vrs")),
        },
        "split_policy": {
            "group_all_windows_by_sequence": True,
            "group_concurrent_recordings_by_event": True,
            "wearer_group_proxy": "location+script+recording; AEA does not expose persistent person identity",
            "held_out_location_folds": [1, 2, 3, 4, 5],
            "location_5_caveat": "only scripts 4 and 5 are present",
        },
    }


def plan_safe_json(plan: Mapping[str, Any]) -> str:
    """Canonical safe serialization used for tests and CLI output."""
    return json.dumps(plan, indent=2, sort_keys=False) + "\n"
