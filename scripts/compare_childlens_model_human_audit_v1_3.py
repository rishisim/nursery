#!/usr/bin/env python3
"""Post-lock ChildLens v1.3 model--human agreement comparator.

The zero-argument entry point discovers the already initialized owner-private
quarantine, proves that AUTHOR_AUDIT_A is irreversibly locked, constructs and
attaches the exact local pseudo-prediction store, and computes only the frozen
aggregate feasibility gates. Restricted rows, text, labels, times, paths, and
identifiers never enter the repository receipt or stdout.

The pure evaluation helpers are intentionally usable with synthetic fixtures.
They do not discover a runtime or open media. Machine outputs remain
measurement-instrument hypotheses; simulator oracle labels remain the only
permitted primary truth for a later causal evaluation.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import os
import random
import re
import sqlite3
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import childlens_author_audit_v1_3 as workflow  # noqa: E402


VERSION = "childlens-model-human-comparator-v1.3.0"
RECEIPT_SCHEMA = "childlens-v1.3-author-audit-result-receipt-v1"
PREDICTION_STORE_SCHEMA = "childlens-author-audit-prediction-store-v1.3.0"
RESERVE_ACTIVATION_SCHEMA = "childlens-author-audit-reserve-activation-v1.3.0"
PROTOCOL_PATH = (
    workflow.REPO_ROOT
    / "docs/childlens_feasibility_v1_3/frozen_model_assisted_author_audit_protocol_v1_3.json"
)
SAMPLING_RECEIPT_PATH = (
    workflow.REPO_ROOT
    / "output/childlens_feasibility_v1_3/author_audit_sampling_receipt.json"
)
PSEUDO_RECEIPT_PATH = (
    workflow.REPO_ROOT
    / "output/childlens_feasibility_v1_3/pseudo_annotation_receipt.json"
)
PUBLIC_RESULT_PATH = (
    workflow.REPO_ROOT
    / "output/childlens_feasibility_v1_3/author_audit_result_receipt.json"
)

COMPARISON_DIRECTORY = "model_assisted_v1_3/author_comparison"
PREDICTION_STORE_FILE = "primary_prediction_store.json"
PREDICTION_BINDING_FILE = "primary_prediction_binding.json"
RESERVE_ACTIVATION_FILE = "reserve_activation_seal.json"
AUDIO_DIRECTORY = "model_assisted_v1_3/pseudo_annotations/audio"
REFERENTIAL_DIRECTORY = "model_assisted_v1_3/pseudo_annotations/referential"

PRIMARY_SECONDS = 900
RESERVE_SECONDS = 900
EXPECTED_ITEMS = 15
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.90
MIN_DEFINED_BOOTSTRAP_FRACTION = 0.80
MAX_SINGLE_ITEM_SUPPORT_SHARE = 0.30
SMALL_CELL_LIMIT = 5
MAX_PRIVATE_JSON_BYTES = 256 * 1024 * 1024

GATE_NAMES = (
    "speech_utterance_boundary",
    "source_role_non_child_priority",
    "non_child_transcript_error_and_coverage",
    "coarse_referential_status",
    "noun_object_candidate_precision_coverage",
    "verb_action_candidate_precision_coverage",
)
PROTOCOL_GATE_KEYS = {
    "speech_utterance_boundary": "boundary_matching",
    "source_role_non_child_priority": "source_role",
    "non_child_transcript_error_and_coverage": "non_child_transcription",
    "coarse_referential_status": "coarse_referential",
    "noun_object_candidate_precision_coverage": "noun_object_candidates",
    "verb_action_candidate_precision_coverage": "verb_action_candidates",
}
COARSE_CLASSES = (
    "VISIBLE",
    "NULL",
    "IRRELEVANT",
    "UNDECIDABLE_OR_UNUSABLE",
)
ROLE_DECIDABLE = frozenset({"NON_CHILD", "CHILD", "OVERLAP"})
LANGUAGE_COMPETENT = frozenset({"NATIVE", "FLUENT", "PROFICIENT"})
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "participant_id", "child_id", "session_id", "recording_id",
        "media_id", "speaker_id", "filename", "source_filename",
        "relative_path", "absolute_path", "media_path", "audio_path",
        "frame_path", "transcript_text", "utterance_text", "lexical_string",
        "raw_hypothesis", "model_label_payload", "exact_timestamp", "timecode",
        "start_time", "end_time", "items", "author_rows", "model_rows",
    }
)


class ComparatorError(RuntimeError):
    """A stable, payload-free error safe to emit outside quarantine."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ComparatorError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_CANONICAL_JSON")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        _fail("E_PRIVATE_FILE_READ")
    return digest.hexdigest()


def _private_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _private_json(path: Path, root: Path, code: str) -> Mapping[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        confined = _inside(resolved, root.resolve(strict=True))
        size = resolved.stat().st_size
    except OSError:
        _fail(code)
    if not confined or not _private_regular(resolved) or not 0 < size <= MAX_PRIVATE_JSON_BYTES:
        _fail(code)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _public_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _atomic_private(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = _canonical(value) + b"\n"
    if path.exists():
        if not _private_regular(path) or path.read_bytes() != payload:
            _fail("E_PRIVATE_OUTPUT_IMMUTABLE")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _atomic_public(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    temporary = path.with_suffix(".json.pending")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _normalized_key(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _walk_public(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield _normalized_key(key), child
            yield from _walk_public(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk_public(child)


def validate_public_receipt(value: Mapping[str, Any]) -> None:
    """Reject restricted payload shapes and unsuppressed small support cells."""

    if (
        value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("audit_complete") is not True
        or set(value.get("threshold_results", {})) != set(GATE_NAMES)
    ):
        _fail("E_PUBLIC_RECEIPT_SCHEMA")
    encoded = _canonical(value).decode("utf-8")
    if re.search(
        r"(?i)(?:/Users/|/home/|file://|\\Users\\|\.(?:mp4|mov|mkv|wav|m4a|jpg|png)\b)",
        encoded,
    ):
        _fail("E_PUBLIC_RECEIPT_PRIVACY")
    for key, child in _walk_public(value):
        if key in FORBIDDEN_PUBLIC_KEYS:
            _fail("E_PUBLIC_RECEIPT_PRIVACY")
        if key == "observed" and isinstance(child, (int, float)) and child < SMALL_CELL_LIMIT:
            _fail("E_PUBLIC_RECEIPT_SMALL_CELL")


def normalize_text(value: str) -> str:
    """Apply the exact frozen language-agnostic normalization."""

    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value).casefold()
    text = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in text
    )
    return " ".join(text.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Memory-bounded Levenshtein distance."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for index, left in enumerate(reference, start=1):
        current = [index]
        for right_index, right in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _intersection(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    overlap = _intersection(left, right)
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def _coalesce(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = sorted((start, end) for start, end in intervals if end > start)
    merged: list[list[int]] = []
    for start, end in rows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _coalesce(intervals))


def _intersection_duration(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    a = _coalesce(left)
    b = _coalesce(right)
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        total += _intersection(a[i], b[j])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _clip_intervals(
    intervals: Iterable[tuple[int, int]], masks: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for interval in intervals:
        for mask in masks:
            start = max(interval[0], mask[0])
            end = min(interval[1], mask[1])
            if end > start:
                clipped.append((start, end))
    return _coalesce(clipped)


def maximum_cardinality_temporal_matching(
    human: Sequence[tuple[int, int]],
    machine: Sequence[tuple[int, int]],
    *,
    tolerance_ms: int,
) -> list[tuple[int, int]]:
    """Maximum-cardinality matching, then maximum total temporal IoU.

    Successive shortest augmenting paths operate over the complete residual
    network. Multiplying IoU by one billion gives a stable integral tie-break;
    edge insertion order supplies the final deterministic tie.
    """

    human_count = len(human)
    machine_count = len(machine)
    source = human_count + machine_count
    sink = source + 1
    node_count = sink + 1
    graph: list[list[list[int]]] = [[] for _ in range(node_count)]
    pair_edges: dict[tuple[int, int], tuple[int, int]] = {}

    def add_edge(start: int, end: int, capacity: int, cost: int) -> tuple[int, int]:
        forward = [end, len(graph[end]), capacity, cost]
        reverse = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(reverse)
        return start, len(graph[start]) - 1

    for index in range(human_count):
        add_edge(source, index, 1, 0)
    for index in range(machine_count):
        add_edge(human_count + index, sink, 1, 0)
    for human_index, human_interval in enumerate(human):
        for machine_index, machine_interval in enumerate(machine):
            if (
                abs(human_interval[0] - machine_interval[0]) <= tolerance_ms
                and abs(human_interval[1] - machine_interval[1]) <= tolerance_ms
            ):
                score = int(round(_interval_iou(human_interval, machine_interval) * 1_000_000_000))
                pair_edges[(human_index, machine_index)] = add_edge(
                    human_index, human_count + machine_index, 1, -score
                )

    while True:
        infinity = 10**30
        distance = [infinity] * node_count
        predecessor: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0
        # Bellman-Ford is small, deterministic, and handles negative residuals.
        for _ in range(node_count - 1):
            changed = False
            for start in range(node_count):
                if distance[start] == infinity:
                    continue
                for edge_index, edge in enumerate(graph[start]):
                    end, _reverse, capacity, cost = edge
                    candidate = distance[start] + cost
                    if capacity > 0 and candidate < distance[end]:
                        distance[end] = candidate
                        predecessor[end] = (start, edge_index)
                        changed = True
            if not changed:
                break
        if predecessor[sink] is None:
            break
        node = sink
        while node != source:
            start, edge_index = predecessor[node]  # type: ignore[misc]
            edge = graph[start][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = start

    matches: list[tuple[int, int]] = []
    for pair, (node, edge_index) in pair_edges.items():
        if graph[node][edge_index][2] == 0:
            matches.append(pair)
    return sorted(matches)


def _maximum_bipartite_matching(
    left_count: int, right_count: int, eligible: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    adjacency = [[] for _ in range(left_count)]
    for left, right in sorted(set(eligible)):
        if 0 <= left < left_count and 0 <= right < right_count:
            adjacency[left].append(right)
    right_owner = [-1] * right_count

    def visit(left: int, seen: set[int]) -> bool:
        for right in adjacency[left]:
            if right in seen:
                continue
            seen.add(right)
            if right_owner[right] < 0 or visit(right_owner[right], seen):
                right_owner[right] = left
                return True
        return False

    for left in range(left_count):
        visit(left, set())
    return sorted((left, right) for right, left in enumerate(right_owner) if left >= 0)


@dataclass(frozen=True)
class HumanMention:
    family: str
    text: str
    visible_interval: tuple[int, int]


@dataclass(frozen=True)
class HumanUtterance:
    interval: tuple[int, int]
    text: str
    role: str
    referential_status: str
    mentions: tuple[HumanMention, ...] = ()


@dataclass(frozen=True)
class MachineUtterance:
    interval: tuple[int, int]
    text: str
    role: str


@dataclass(frozen=True)
class MachineWindow:
    interval: tuple[int, int]
    referential_status: str
    role: str
    noun_candidates: tuple[str, ...] = ()
    verb_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ItemEvidence:
    masks: tuple[tuple[int, int], ...]
    human_utterances: tuple[HumanUtterance, ...]
    machine_utterances: tuple[MachineUtterance, ...]
    machine_vad: tuple[tuple[int, int], ...]
    machine_windows: tuple[MachineWindow, ...]
    wer_applicable: bool


def _coarse(value: str) -> str:
    return {
        "VISIBLE_CANDIDATE": "VISIBLE",
        "NULL_NOT_VISIBLE": "NULL",
        "IRRELEVANT": "IRRELEVANT",
        "UNDECIDABLE": "UNDECIDABLE_OR_UNUSABLE",
        "UNUSABLE": "UNDECIDABLE_OR_UNUSABLE",
    }.get(value, "UNDECIDABLE_OR_UNUSABLE")


def _best_window(
    interval: tuple[int, int], windows: Sequence[MachineWindow]
) -> MachineWindow | None:
    candidates = [
        (_intersection(interval, window.interval), -index, window)
        for index, window in enumerate(windows)
        if _intersection(interval, window.interval) > 0
    ]
    return max(candidates, default=(0, 0, None), key=lambda row: (row[0], row[1]))[2]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _f1(tp: float, fp: float, fn: float) -> float | None:
    return _safe_ratio(2 * tp, 2 * tp + fp + fn)


def _empty_stats() -> dict[str, float]:
    keys = {
        "boundary_human_n", "boundary_machine_n", "boundary_tp1000",
        "boundary_tp500", "boundary_human_ms", "boundary_machine_ms",
        "boundary_overlap_ms", "role_human_total", "role_human_nonchild",
        "role_human_child", "role_decided", "role_nc_tp", "role_nc_fp",
        "role_nc_fn", "role_child_tp", "role_child_fp", "role_child_fn",
        "transcript_char_errors", "transcript_ref_chars", "transcript_word_errors",
        "transcript_ref_words", "transcript_covered_ms", "transcript_total_ms",
        "transcript_wer_applicable", "coarse_total", "coarse_exact",
        "coarse_decided", "coarse_human_visible", "coarse_pred_visible",
        "coarse_true_visible", "noun_human", "noun_machine", "noun_match",
        "verb_human", "verb_machine", "verb_match",
    }
    for actual in COARSE_CLASSES:
        for predicted in COARSE_CLASSES:
            keys.add(f"coarse_{actual}_{predicted}")
    return {key: 0.0 for key in keys}


def _item_stats(item: ItemEvidence) -> dict[str, float]:
    stats = _empty_stats()
    humans = [row for row in item.human_utterances if row.role != "NONSPEECH"]
    machines = list(item.machine_utterances)
    human_intervals = [row.interval for row in humans]
    machine_intervals = [row.interval for row in machines]
    matches1000 = maximum_cardinality_temporal_matching(
        human_intervals, machine_intervals, tolerance_ms=1000
    )
    matches500 = maximum_cardinality_temporal_matching(
        human_intervals, machine_intervals, tolerance_ms=500
    )
    human_to_machine = {human: machine for human, machine in matches1000}
    machine_to_human = {machine: human for human, machine in matches1000}

    stats["boundary_human_n"] = len(humans)
    stats["boundary_machine_n"] = len(machines)
    stats["boundary_tp1000"] = len(matches1000)
    stats["boundary_tp500"] = len(matches500)
    human_union = _coalesce(human_intervals)
    machine_vad = _clip_intervals(item.machine_vad, item.masks)
    stats["boundary_human_ms"] = _duration(human_union)
    stats["boundary_machine_ms"] = _duration(machine_vad)
    stats["boundary_overlap_ms"] = _intersection_duration(human_union, machine_vad)

    # Role: unmatched or uncertain machine outputs are abstentions. They lower
    # coverage and recall but are never silently reassigned.
    for human_index, human in enumerate(humans):
        if human.role not in ROLE_DECIDABLE:
            continue
        actual = "NON_CHILD" if human.role == "NON_CHILD" else "CHILD"
        stats["role_human_total"] += 1
        stats[f"role_human_{'nonchild' if actual == 'NON_CHILD' else 'child'}"] += 1
        machine_index = human_to_machine.get(human_index)
        predicted: str | None = None
        if machine_index is not None:
            machine_role = machines[machine_index].role
            if machine_role == "NON_CHILD":
                predicted = "NON_CHILD"
            elif machine_role in {"CHILD", "OVERLAP"}:
                predicted = "CHILD"
        if predicted is not None:
            stats["role_decided"] += 1
        if actual == "NON_CHILD":
            if predicted == "NON_CHILD":
                stats["role_nc_tp"] += 1
            else:
                stats["role_nc_fn"] += 1
            if predicted == "CHILD":
                stats["role_child_fp"] += 1
        else:
            if predicted == "CHILD":
                stats["role_child_tp"] += 1
            else:
                stats["role_child_fn"] += 1
            if predicted == "NON_CHILD":
                stats["role_nc_fp"] += 1
    for machine_index, machine in enumerate(machines):
        if machine_index in machine_to_human:
            continue
        if machine.role == "NON_CHILD":
            stats["role_nc_fp"] += 1
        elif machine.role in {"CHILD", "OVERLAP"}:
            stats["role_child_fp"] += 1

    # Non-child CER/WER. Every unmatched human reference is a deletion; every
    # unmatched predicted non-child hypothesis is an insertion.
    for human_index, human in enumerate(humans):
        if human.role != "NON_CHILD":
            continue
        reference = normalize_text(human.text)
        stats["transcript_ref_chars"] += len(reference)
        if item.wer_applicable:
            stats["transcript_ref_words"] += len(reference.split())
        stats["transcript_total_ms"] += human.interval[1] - human.interval[0]
        machine_index = human_to_machine.get(human_index)
        hypothesis = ""
        if machine_index is not None:
            hypothesis = normalize_text(machines[machine_index].text)
            if hypothesis:
                stats["transcript_covered_ms"] += human.interval[1] - human.interval[0]
        stats["transcript_char_errors"] += edit_distance(reference, hypothesis)
        if item.wer_applicable:
            stats["transcript_word_errors"] += edit_distance(
                reference.split(), hypothesis.split()
            )
    for machine_index, machine in enumerate(machines):
        if machine_index not in machine_to_human and machine.role == "NON_CHILD":
            hypothesis = normalize_text(machine.text)
            stats["transcript_char_errors"] += len(hypothesis)
            if item.wer_applicable:
                stats["transcript_word_errors"] += len(hypothesis.split())
    stats["transcript_wer_applicable"] = 1.0 if item.wer_applicable else 0.0

    # Coarse reference is compared on every human non-child utterance, retaining
    # undecidable/unusable as its declared fourth class.
    for human in humans:
        if human.role != "NON_CHILD":
            continue
        actual = _coarse(human.referential_status)
        window = _best_window(human.interval, item.machine_windows)
        predicted = _coarse(window.referential_status) if window else "UNDECIDABLE_OR_UNUSABLE"
        stats["coarse_total"] += 1
        stats[f"coarse_{actual}_{predicted}"] += 1
        if actual == predicted:
            stats["coarse_exact"] += 1
        if predicted != "UNDECIDABLE_OR_UNUSABLE":
            stats["coarse_decided"] += 1
        if actual == "VISIBLE":
            stats["coarse_human_visible"] += 1
        if predicted == "VISIBLE":
            stats["coarse_pred_visible"] += 1
        if actual == predicted == "VISIBLE":
            stats["coarse_true_visible"] += 1

    # Candidate proposals are auditable only inside a human non-child utterance.
    # A match additionally requires the exact normalized source mention to occur
    # in the temporally aligned ASR hypothesis and a visible-band intersection
    # with the frozen 1000 ms collar.
    for family, prefix in (("NOUN_OBJECT", "noun"), ("VERB_ACTION", "verb")):
        human_candidates: list[tuple[int, HumanMention]] = []
        for human_index, human in enumerate(humans):
            if human.role != "NON_CHILD":
                continue
            for mention in human.mentions:
                if family == mention.family:
                    human_candidates.append((human_index, mention))
        model_candidates: list[tuple[int, str, tuple[int, int]]] = []
        for window in item.machine_windows:
            parent_humans = [
                index
                for index, human in enumerate(humans)
                if human.role == "NON_CHILD" and _intersection(human.interval, window.interval) > 0
            ]
            if not parent_humans:
                continue
            strings = window.noun_candidates if family == "NOUN_OBJECT" else window.verb_candidates
            for value in strings:
                normalized = normalize_text(value)
                if normalized:
                    # The maximally overlapping human supplies a deterministic
                    # audit context; it is not used to select the sample.
                    parent = max(
                        parent_humans,
                        key=lambda index: (_intersection(humans[index].interval, window.interval), -index),
                    )
                    model_candidates.append((parent, normalized, window.interval))
        edges: list[tuple[int, int]] = []
        for model_index, (parent, candidate, model_interval) in enumerate(model_candidates):
            machine_index = human_to_machine.get(parent)
            aligned_machine_text = (
                normalize_text(machines[machine_index].text) if machine_index is not None else ""
            )
            if not aligned_machine_text or candidate not in aligned_machine_text:
                continue
            for human_index, (owner, mention) in enumerate(human_candidates):
                normalized_mention = normalize_text(mention.text)
                collared = (mention.visible_interval[0] - 1000, mention.visible_interval[1] + 1000)
                if (
                    owner == parent
                    and candidate == normalized_mention
                    and _intersection(model_interval, collared) > 0
                ):
                    edges.append((model_index, human_index))
        matches = _maximum_bipartite_matching(
            len(model_candidates), len(human_candidates), edges
        )
        stats[f"{prefix}_human"] = len(human_candidates)
        stats[f"{prefix}_machine"] = len(model_candidates)
        stats[f"{prefix}_match"] = len(matches)
    return stats


def _sum_stats(rows: Sequence[Mapping[str, float]], indices: Sequence[int] | None = None) -> dict[str, float]:
    selected = range(len(rows)) if indices is None else indices
    result = _empty_stats()
    for index in selected:
        for key, value in rows[index].items():
            result[key] = result.get(key, 0.0) + float(value)
    return result


def _metrics(stats: Mapping[str, float], *, wer_applicable: bool) -> dict[str, float | None]:
    role_nc_precision = _safe_ratio(stats["role_nc_tp"], stats["role_nc_tp"] + stats["role_nc_fp"])
    role_nc_recall = _safe_ratio(stats["role_nc_tp"], stats["role_nc_tp"] + stats["role_nc_fn"])
    role_macro_parts = (
        _f1(stats["role_nc_tp"], stats["role_nc_fp"], stats["role_nc_fn"]),
        _f1(stats["role_child_tp"], stats["role_child_fp"], stats["role_child_fn"]),
    )
    coarse_f1: list[float] = []
    for target in COARSE_CLASSES:
        tp = stats[f"coarse_{target}_{target}"]
        fp = sum(stats[f"coarse_{actual}_{target}"] for actual in COARSE_CLASSES if actual != target)
        fn = sum(stats[f"coarse_{target}_{predicted}"] for predicted in COARSE_CLASSES if predicted != target)
        value = _f1(tp, fp, fn)
        coarse_f1.append(0.0 if value is None else value)
    return {
        "segment_f1_at_1000ms": _safe_ratio(
            2 * stats["boundary_tp1000"],
            stats["boundary_human_n"] + stats["boundary_machine_n"],
        ),
        "matched_both_edges_within_500ms": _safe_ratio(
            stats["boundary_tp500"], stats["boundary_tp1000"]
        ),
        "human_speech_time_recall": _safe_ratio(
            stats["boundary_overlap_ms"], stats["boundary_human_ms"]
        ),
        "machine_speech_time_precision": _safe_ratio(
            stats["boundary_overlap_ms"], stats["boundary_machine_ms"]
        ),
        "non_child_precision": role_nc_precision,
        "non_child_recall": role_nc_recall,
        "binary_macro_f1": (
            sum(value for value in role_macro_parts if value is not None) / 2
            if all(value is not None for value in role_macro_parts)
            else None
        ),
        "role_decision_coverage": _safe_ratio(stats["role_decided"], stats["role_human_total"]),
        "cer": _safe_ratio(stats["transcript_char_errors"], stats["transcript_ref_chars"]),
        "wer": (
            _safe_ratio(stats["transcript_word_errors"], stats["transcript_ref_words"])
            if wer_applicable
            else None
        ),
        "usable_transcript_coverage": _safe_ratio(
            stats["transcript_covered_ms"], stats["transcript_total_ms"]
        ),
        "four_class_exact_agreement": _safe_ratio(stats["coarse_exact"], stats["coarse_total"]),
        "four_class_macro_f1": sum(coarse_f1) / len(COARSE_CLASSES) if len(coarse_f1) == len(COARSE_CLASSES) else None,
        "visible_precision": _safe_ratio(stats["coarse_true_visible"], stats["coarse_pred_visible"]),
        "visible_recall": _safe_ratio(stats["coarse_true_visible"], stats["coarse_human_visible"]),
        "referential_decision_coverage": _safe_ratio(stats["coarse_decided"], stats["coarse_total"]),
        "noun_candidate_precision": _safe_ratio(stats["noun_match"], stats["noun_machine"]),
        "noun_candidate_coverage": _safe_ratio(stats["noun_match"], stats["noun_human"]),
        "verb_candidate_precision": _safe_ratio(stats["verb_match"], stats["verb_machine"]),
        "verb_candidate_coverage": _safe_ratio(stats["verb_match"], stats["verb_human"]),
    }


def _gate_metric_names(gate: str) -> tuple[str, ...]:
    return {
        "speech_utterance_boundary": (
            "segment_f1_at_1000ms",
            "matched_both_edges_within_500ms",
            "human_speech_time_recall",
            "machine_speech_time_precision",
        ),
        "source_role_non_child_priority": (
            "non_child_precision", "non_child_recall", "binary_macro_f1", "role_decision_coverage"
        ),
        "non_child_transcript_error_and_coverage": (
            "cer", "wer", "usable_transcript_coverage"
        ),
        "coarse_referential_status": (
            "four_class_exact_agreement", "four_class_macro_f1", "visible_precision",
            "visible_recall", "referential_decision_coverage",
        ),
        "noun_object_candidate_precision_coverage": (
            "noun_candidate_precision", "noun_candidate_coverage"
        ),
        "verb_action_candidate_precision_coverage": (
            "verb_candidate_precision", "verb_candidate_coverage"
        ),
    }[gate]


def _protocol_metric_name(gate: str, name: str) -> str:
    if gate == "noun_object_candidate_precision_coverage":
        return name.removeprefix("noun_")
    if gate == "verb_action_candidate_precision_coverage":
        return name.removeprefix("verb_")
    return name


def _support(
    gate: str, rows: Sequence[Mapping[str, float]], pooled: Mapping[str, float], *, wer_applicable: bool
) -> tuple[dict[str, tuple[float, float]], list[float]]:
    if gate == "speech_utterance_boundary":
        checks = {
            "human_linguistic_utterances": (pooled["boundary_human_n"], 30),
            "represented_items": (sum(row["boundary_human_n"] > 0 for row in rows), 10),
            "human_linguistic_speech_ms": (pooled["boundary_human_ms"], 300_000),
        }
        contributions = [row["boundary_human_n"] for row in rows]
    elif gate == "source_role_non_child_priority":
        checks = {
            "human_decidable_linguistic_utterances": (pooled["role_human_total"], 30),
            "represented_items": (sum(row["role_human_total"] > 0 for row in rows), 10),
            "non_child_utterances": (pooled["role_human_nonchild"], 10),
            "child_or_overlap_utterances": (pooled["role_human_child"], 10),
        }
        contributions = [row["role_human_total"] for row in rows]
    elif gate == "non_child_transcript_error_and_coverage":
        checks = {
            "normalized_reference_characters": (pooled["transcript_ref_chars"], 500),
            "represented_items": (sum(row["transcript_ref_chars"] > 0 for row in rows), 10),
            "human_non_child_speech_ms": (pooled["transcript_total_ms"], 180_000),
        }
        if wer_applicable:
            checks["normalized_reference_words_if_wer_applicable"] = (
                pooled["transcript_ref_words"], 100
            )
        contributions = [row["transcript_ref_chars"] for row in rows]
    elif gate == "coarse_referential_status":
        checks = {
            "auditable_non_child_utterances": (pooled["coarse_total"], 20),
            "represented_items": (sum(row["coarse_total"] > 0 for row in rows), 8),
            "human_visible_candidates": (pooled["coarse_human_visible"], 8),
        }
        contributions = [row["coarse_total"] for row in rows]
    elif gate == "noun_object_candidate_precision_coverage":
        checks = {
            "human_eligible_visible_candidates": (pooled["noun_human"], 20),
            "represented_items": (sum(row["noun_human"] > 0 for row in rows), 8),
        }
        contributions = [row["noun_human"] for row in rows]
    else:
        checks = {
            "human_eligible_visible_candidates": (pooled["verb_human"], 15),
            "represented_items": (sum(row["verb_human"] > 0 for row in rows), 6),
        }
        contributions = [row["verb_human"] for row in rows]
    return checks, contributions


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("undefined quantile")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rules(protocol_gate: Mapping[str, Any], gate: str, wer_applicable: bool) -> dict[str, dict[str, tuple[str, float]]]:
    result: dict[str, dict[str, tuple[str, float]]] = {"pass": {}, "bound": {}, "hard": {}}
    for key, value in protocol_gate.get("pass_thresholds", {}).items():
        if not isinstance(value, (int, float)):
            _fail("E_PROTOCOL_THRESHOLDS")
        applicable = not key.endswith("_if_applicable") or wer_applicable
        name = key.removesuffix("_if_applicable")
        direction = "min" if name.endswith("_min") else "max"
        name = name.removesuffix("_min").removesuffix("_max")
        if applicable:
            result["pass"][_metric_internal_name(gate, name)] = (direction, float(value))
    for key, value in protocol_gate.get("cluster_bound_floors", {}).items():
        if not isinstance(value, (int, float)):
            _fail("E_PROTOCOL_THRESHOLDS")
        applicable = not key.endswith("_if_applicable") or wer_applicable
        name = key.removesuffix("_if_applicable")
        if name.endswith("_lcb90_min"):
            direction = "min"
            name = name.removesuffix("_lcb90_min")
        elif name.endswith("_ucb90_max"):
            direction = "max"
            name = name.removesuffix("_ucb90_max")
        else:
            _fail("E_PROTOCOL_THRESHOLDS")
        if applicable:
            result["bound"][_metric_internal_name(gate, name)] = (direction, float(value))
    for key, value in protocol_gate.get("hard_fail_thresholds", {}).items():
        if not isinstance(value, (int, float)):
            _fail("E_PROTOCOL_THRESHOLDS")
        applicable = not key.endswith("_if_applicable") or wer_applicable
        name = key.removesuffix("_if_applicable")
        if name.endswith("_below"):
            direction = "below"
            name = name.removesuffix("_below")
        elif name.endswith("_above"):
            direction = "above"
            name = name.removesuffix("_above")
        else:
            _fail("E_PROTOCOL_THRESHOLDS")
        if applicable:
            result["hard"][_metric_internal_name(gate, name)] = (direction, float(value))
    return result


def _metric_internal_name(gate: str, protocol_name: str) -> str:
    if gate == "noun_object_candidate_precision_coverage":
        return "noun_" + protocol_name
    if gate == "verb_action_candidate_precision_coverage":
        return "verb_" + protocol_name
    return protocol_name


def _observed_support(observed: float, required: float) -> Mapping[str, Any]:
    integral = float(observed).is_integer()
    small = observed < SMALL_CELL_LIMIT
    return {
        "required_minimum": int(required) if float(required).is_integer() else required,
        "met": observed >= required,
        "observed": None if small else (int(observed) if integral else observed),
        "small_cell_suppressed": small,
    }


def evaluate_items(
    items: Sequence[ItemEvidence],
    protocol: Mapping[str, Any],
    *,
    protocol_digest: str,
    sample_digest: str,
    phase: str = "PRIMARY",
) -> tuple[dict[str, Any], bool]:
    """Compute all frozen gate dispositions from in-memory evidence.

    Returns ``(threshold_results, reserve_eligible)``. ``phase`` is PRIMARY or
    POOLED_AFTER_RESERVE; primary support shortfalls are BORDERLINE because the
    one pre-frozen reserve may recover them.
    """

    if len(items) != EXPECTED_ITEMS or phase not in {"PRIMARY", "POOLED_AFTER_RESERVE"}:
        _fail("E_EVALUATION_SCOPE")
    if not HEX64_RE.fullmatch(protocol_digest) or not HEX64_RE.fullmatch(sample_digest):
        _fail("E_EVALUATION_BINDING")
    agreement = protocol.get("agreement_gates")
    uncertainty = protocol.get("cluster_aware_uncertainty")
    if not isinstance(agreement, Mapping) or not isinstance(uncertainty, Mapping):
        _fail("E_PROTOCOL_THRESHOLDS")
    if (
        uncertainty.get("replicates") != BOOTSTRAP_REPLICATES
        or uncertainty.get("confidence_level") != BOOTSTRAP_CONFIDENCE
        or uncertainty.get("minimum_defined_replicate_fraction") != MIN_DEFINED_BOOTSTRAP_FRACTION
        or uncertainty.get("maximum_single_item_support_share_for_PASS") != MAX_SINGLE_ITEM_SUPPORT_SHARE
    ):
        _fail("E_PROTOCOL_THRESHOLDS")

    rows = [_item_stats(item) for item in items]
    pooled = _sum_stats(rows)
    wer_applicable = any(item.wer_applicable for item in items)
    point = _metrics(pooled, wer_applicable=wer_applicable)
    seed = int.from_bytes(
        hashlib.sha256((protocol_digest + sample_digest).encode("ascii")).digest()[:16],
        "big",
    )
    generator = random.Random(seed)
    bootstrap_values: dict[str, list[float]] = {
        name: [] for gate in GATE_NAMES for name in _gate_metric_names(gate)
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(len(rows)) for _ in range(len(rows))]
        metrics = _metrics(_sum_stats(rows, indices), wer_applicable=wer_applicable)
        for name, value in metrics.items():
            if name in bootstrap_values and value is not None and math.isfinite(value):
                bootstrap_values[name].append(value)

    threshold_results: dict[str, Any] = {}
    any_borderline = False
    any_hard = False
    alpha = (1 - BOOTSTRAP_CONFIDENCE) / 2
    for gate in GATE_NAMES:
        protocol_key = PROTOCOL_GATE_KEYS[gate]
        protocol_gate = agreement.get(protocol_key)
        if not isinstance(protocol_gate, Mapping):
            _fail("E_PROTOCOL_THRESHOLDS")
        rules = _rules(protocol_gate, gate, wer_applicable)
        checks, contributions = _support(gate, rows, pooled, wer_applicable=wer_applicable)
        support_met = all(observed >= required for observed, required in checks.values())
        total_support = sum(contributions)
        maximum_share = (
            max(contributions, default=0.0) / total_support if total_support > 0 else None
        )
        concentration_pass = maximum_share is not None and maximum_share <= MAX_SINGLE_ITEM_SUPPORT_SHARE

        metric_receipts: dict[str, Any] = {}
        point_pass = True
        hard_fail = False
        bounds_pass = True
        defined_pass = True
        for metric in _gate_metric_names(gate):
            if metric == "wer" and not wer_applicable:
                metric_receipts[metric] = {
                    "applicability": "NOT_APPLICABLE_BY_BLIND_AUTHOR_LOCK",
                    "small_cell_suppressed": False,
                }
                continue
            value = point.get(metric)
            values = bootstrap_values[metric]
            defined_fraction = len(values) / BOOTSTRAP_REPLICATES
            lower = _quantile(values, alpha) if values else None
            upper = _quantile(values, 1 - alpha) if values else None
            if value is None or defined_fraction < MIN_DEFINED_BOOTSTRAP_FRACTION:
                defined_pass = False
            pass_rule = rules["pass"].get(metric)
            if pass_rule and value is not None:
                direction, threshold = pass_rule
                point_pass = point_pass and (
                    value >= threshold if direction == "min" else value <= threshold
                )
            elif pass_rule:
                point_pass = False
            hard_rule = rules["hard"].get(metric)
            if hard_rule and value is not None:
                direction, threshold = hard_rule
                hard_fail = hard_fail or (
                    value < threshold if direction == "below" else value > threshold
                )
            bound_rule = rules["bound"].get(metric)
            selected_bound: float | None = None
            if bound_rule:
                direction, threshold = bound_rule
                selected_bound = lower if direction == "min" else upper
                bounds_pass = bounds_pass and (
                    selected_bound is not None
                    and (selected_bound >= threshold if direction == "min" else selected_bound <= threshold)
                    and defined_fraction >= MIN_DEFINED_BOOTSTRAP_FRACTION
                )
            # Metrics are exported only when the frozen support floor is met;
            # unsupported ratios can reconstruct small cells.
            metric_receipts[metric] = {
                "applicability": "APPLICABLE",
                "point_estimate": round(float(value), 6) if support_met and value is not None else None,
                "lower_90": round(float(lower), 6) if support_met and lower is not None else None,
                "upper_90": round(float(upper), 6) if support_met and upper is not None else None,
                "defined_bootstrap_fraction": round(defined_fraction, 6),
                "support_suppressed_until_minimum_met": not support_met,
            }

        # Aggregate leave-one-item-out influence, with no cluster identity.
        loo_changes: dict[str, float | None] = {}
        for metric in _gate_metric_names(gate):
            if metric == "wer" and not wer_applicable:
                loo_changes[metric] = None
                continue
            base = point.get(metric)
            changes: list[float] = []
            if base is not None:
                for excluded in range(len(rows)):
                    loo = _metrics(
                        _sum_stats(rows, [index for index in range(len(rows)) if index != excluded]),
                        wer_applicable=wer_applicable,
                    ).get(metric)
                    if loo is not None:
                        changes.append(abs(loo - base))
            loo_changes[metric] = round(max(changes), 6) if support_met and changes else None

        if support_met and hard_fail:
            status = "HARD_FAIL"
            any_hard = True
        elif (
            support_met
            and point_pass
            and bounds_pass
            and defined_pass
            and concentration_pass
        ):
            status = "PASS"
        elif phase == "PRIMARY":
            status = "BORDERLINE"
            any_borderline = True
        elif not support_met:
            status = "NOT_ESTIMABLE"
        else:
            status = "BORDERLINE"
            any_borderline = True

        threshold_results[gate] = {
            "status": status,
            "threshold_frozen": True,
            "support_met": support_met,
            "support": {
                name: _observed_support(observed, required)
                for name, (observed, required) in checks.items()
            },
            "metrics": metric_receipts,
            "bootstrap": {
                "method": "WHOLE_ITEM_PERCENTILE_CLUSTER_BOOTSTRAP",
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence_level": BOOTSTRAP_CONFIDENCE,
                "minimum_defined_fraction": MIN_DEFINED_BOOTSTRAP_FRACTION,
            },
            "maximum_single_item_support_share": (
                round(maximum_share, 6) if maximum_share is not None else None
            ),
            "maximum_single_item_support_share_pass": concentration_pass,
            "leave_one_item_out": {
                "reported": True,
                "maximum_absolute_metric_change": loo_changes,
                "item_identity_exported": False,
            },
            "small_sample_limitation": "DESCRIPTIVE_FEASIBILITY_ONLY_NOT_POPULATION_INFERENCE",
        }
    reserve_eligible = phase == "PRIMARY" and any_borderline and not any_hard
    return threshold_results, reserve_eligible


def _ms(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(code)
    return int(round(float(value) * 1000))


def _source_intervals(value: Any, code: str) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        _fail(code)
    intervals: list[tuple[int, int]] = []
    for row in value:
        if not isinstance(row, Mapping):
            _fail(code)
        start = _ms(row.get("start_seconds"), code)
        end = _ms(row.get("end_seconds"), code)
        if start < 0 or end <= start:
            _fail(code)
        intervals.append((start, end))
    return intervals


def _validate_pseudo_item(
    audio: Mapping[str, Any], referential: Mapping[str, Any]
) -> None:
    if (
        audio.get("schema_version") != "childlens-restricted-audio-pseudo-labels-v1.3.0"
        or audio.get("pseudo_labels_are_ground_truth") is not False
        or audio.get("primary_evaluation_truth_allowed") is not False
        or not isinstance(audio.get("vad_boundary_hypotheses"), list)
        or not isinstance(audio.get("asr_hypotheses"), list)
        or not isinstance(audio.get("speaker_role_hypotheses"), list)
        or referential.get("schema_version")
        != "childlens-restricted-referential-pseudo-labels-v1.3.0"
        or referential.get("pseudo_labels_are_ground_truth") is not False
        or referential.get("primary_evaluation_truth_allowed") is not False
        or referential.get("frame_selection_prediction_independent") is not True
        or referential.get("confidence_adaptive_resampling") is not False
        or not isinstance(referential.get("candidates"), list)
    ):
        _fail("E_PSEUDO_STORE_SCHEMA")


def _locked_context(root: Path) -> tuple[sqlite3.Connection, sqlite3.Row, bytes]:
    database = root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
    secret_path = root / workflow.WORKFLOW_DIR / workflow.SECRET_FILE
    if not _private_regular(database) or not _private_regular(secret_path):
        _fail("E_AUTHOR_AUDIT_RUNTIME")
    secret = secret_path.read_bytes()
    if len(secret) != 32:
        _fail("E_AUTHOR_AUDIT_RUNTIME")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    meta = connection.execute("SELECT * FROM workflow_meta WHERE singleton=1").fetchone()
    if (
        not meta
        or meta["workflow_state"] not in {"AUTHOR_LOCKED", "PREDICTION_JOIN_ENABLED"}
        or not meta["author_pass_locked_at_utc"]
        or not meta["author_record_hmac"]
    ):
        connection.close()
        _fail("E_AUTHOR_AUDIT_NOT_LOCKED")
    return connection, meta, secret


def _verify_locked_author(connection: sqlite3.Connection, meta: sqlite3.Row, secret: bytes) -> None:
    payload = workflow._author_record_payload(connection)
    expected = hmac.new(
        secret, b"author-record\0" + workflow._canonical(payload), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, meta["author_record_hmac"]):
        _fail("E_AUTHOR_RECORD_CHANGED_AFTER_LOCK")
    labels = connection.execute("SELECT * FROM item_labels ORDER BY item_id").fetchall()
    if len(labels) != EXPECTED_ITEMS or any(row["locked"] != 1 for row in labels):
        _fail("E_AUTHOR_AUDIT_INCOMPLETE")
    annotated = [row for row in labels if row["disposition"] == "ANNOTATED"]
    if not annotated or any(row["language_competence"] not in LANGUAGE_COMPETENT for row in annotated):
        _fail("E_AUTHOR_QUALIFICATION_NOT_CONFIRMED")
    events = connection.execute(
        "SELECT event_id,event FROM audit_log ORDER BY event_id"
    ).fetchall()
    lock_ids = [row["event_id"] for row in events if row["event"] == "AUTHOR_PASS_IRREVERSIBLY_LOCKED"]
    join_ids = [row["event_id"] for row in events if row["event"] == "PREDICTION_JOIN_ENABLED_AFTER_AUTHOR_LOCK"]
    if len(lock_ids) != 1 or any(join <= lock_ids[0] for join in join_ids):
        _fail("E_PREDICTION_REVEAL_ORDER")


def _build_prediction_store(root: Path, connection: sqlite3.Connection, meta: sqlite3.Row) -> tuple[Path, Path]:
    comparison = root / COMPARISON_DIRECTORY
    comparison.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(comparison, 0o700)
    items: list[dict[str, Any]] = []
    rows = connection.execute(
        "SELECT prediction_join_key,media_sha256 FROM audit_items ORDER BY item_id"
    ).fetchall()
    if len(rows) != EXPECTED_ITEMS:
        _fail("E_AUTHOR_AUDIT_INCOMPLETE")
    for row in rows:
        digest = row["media_sha256"]
        if not isinstance(digest, str) or not HEX64_RE.fullmatch(digest):
            _fail("E_PSEUDO_BINDING")
        audio = _private_json(root / AUDIO_DIRECTORY / f"{digest}.json", root, "E_PSEUDO_STORE_MISSING")
        referential = _private_json(
            root / REFERENTIAL_DIRECTORY / f"{digest}.json", root, "E_PSEUDO_STORE_MISSING"
        )
        _validate_pseudo_item(audio, referential)
        items.append(
            {
                "prediction_join_key": row["prediction_join_key"],
                "media_sha256": digest,
                "audio": audio,
                "referential": referential,
            }
        )
    store = {
        "schema_version": PREDICTION_STORE_SCHEMA,
        "protocol_digest": meta["protocol_digest"],
        "sample_digest": meta["sample_digest"],
        "item_count": EXPECTED_ITEMS,
        "pseudo_labels_are_ground_truth": False,
        "unaudited_pseudo_labels_are_primary_evaluation_truth": False,
        "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
        "items": items,
    }
    store_path = comparison / PREDICTION_STORE_FILE
    _atomic_private(store_path, store)
    binding = {
        "schema_version": workflow.PREDICTION_BINDING_VERSION,
        "sample_digest": meta["sample_digest"],
        "prediction_store_relpath": os.fspath(store_path.relative_to(root)),
        "prediction_store_sha256": _sha256_file(store_path),
        "local_offline_inference_attested": True,
        "network_disabled_during_inference": True,
        "pseudo_labels_are_ground_truth": False,
        "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
    }
    binding_path = comparison / PREDICTION_BINDING_FILE
    _atomic_private(binding_path, binding)
    return store_path, binding_path


def _evidence_from_locked_store(
    connection: sqlite3.Connection, store: Mapping[str, Any]
) -> list[ItemEvidence]:
    stored_items = store.get("items")
    if not isinstance(stored_items, list) or len(stored_items) != EXPECTED_ITEMS:
        _fail("E_PREDICTION_STORE_INVALID")
    by_join = {
        row.get("prediction_join_key"): row
        for row in stored_items
        if isinstance(row, Mapping) and isinstance(row.get("prediction_join_key"), str)
    }
    if len(by_join) != EXPECTED_ITEMS:
        _fail("E_PREDICTION_STORE_INVALID")
    result: list[ItemEvidence] = []
    item_rows = connection.execute("SELECT * FROM audit_items ORDER BY item_id").fetchall()
    for item_row in item_rows:
        sample_segments = connection.execute(
            "SELECT segment_index,start_ms,end_ms FROM audit_segments WHERE item_id=? ORDER BY segment_index",
            (item_row["item_id"],),
        ).fetchall()
        masks = [(row["start_ms"], row["end_ms"]) for row in sample_segments]
        segment_map = {row["segment_index"]: (row["start_ms"], row["end_ms"]) for row in sample_segments}
        mentions_by_utterance: dict[int, list[HumanMention]] = {}
        mention_rows = connection.execute(
            """SELECT m.*,u.item_id FROM author_mentions m
               JOIN author_utterances u ON u.utterance_id=m.utterance_id
               WHERE u.item_id=? ORDER BY m.mention_id""",
            (item_row["item_id"],),
        ).fetchall()
        utterance_text = {
            row["utterance_id"]: row["source_text"]
            for row in connection.execute(
                "SELECT utterance_id,source_text FROM author_utterances WHERE item_id=?",
                (item_row["item_id"],),
            )
        }
        for mention in mention_rows:
            if mention["referential_status"] != "VISIBLE_CANDIDATE":
                continue
            visible_segment = segment_map.get(mention["visible_segment_index"])
            source = utterance_text.get(mention["utterance_id"], "")
            if (
                visible_segment is None
                or mention["visible_onset_ms"] is None
                or mention["visible_offset_ms"] is None
                or not isinstance(source, str)
            ):
                continue
            start_char = mention["mention_start_char"]
            end_char = mention["mention_end_char"]
            if not 0 <= start_char < end_char <= len(source):
                _fail("E_AUTHOR_RECORD_SCHEMA")
            visible = (
                visible_segment[0] + mention["visible_onset_ms"],
                visible_segment[0] + mention["visible_offset_ms"],
            )
            mentions_by_utterance.setdefault(mention["utterance_id"], []).append(
                HumanMention(
                    family=mention["mention_family"],
                    text=source[start_char:end_char],
                    visible_interval=visible,
                )
            )
        human_rows = connection.execute(
            "SELECT * FROM author_utterances WHERE item_id=? ORDER BY segment_index,onset_ms,utterance_id",
            (item_row["item_id"],),
        ).fetchall()
        humans: list[HumanUtterance] = []
        for row in human_rows:
            segment = segment_map.get(row["segment_index"])
            if segment is None:
                _fail("E_AUTHOR_RECORD_SCHEMA")
            humans.append(
                HumanUtterance(
                    interval=(segment[0] + row["onset_ms"], segment[0] + row["offset_ms"]),
                    text=row["source_text"],
                    role=row["speaker_role"],
                    referential_status=row["referential_status"],
                    mentions=tuple(mentions_by_utterance.get(row["utterance_id"], ())),
                )
            )
        label = connection.execute(
            "SELECT wer_applicability FROM item_labels WHERE item_id=?",
            (item_row["item_id"],),
        ).fetchone()
        private = by_join.get(item_row["prediction_join_key"])
        if not isinstance(private, Mapping):
            _fail("E_PREDICTION_STORE_INVALID")
        audio = private.get("audio")
        referential = private.get("referential")
        if not isinstance(audio, Mapping) or not isinstance(referential, Mapping):
            _fail("E_PREDICTION_STORE_INVALID")
        windows: list[MachineWindow] = []
        for row in referential.get("candidates", []):
            if not isinstance(row, Mapping):
                _fail("E_PREDICTION_STORE_INVALID")
            interval = (
                _ms(row.get("window_start_seconds"), "E_PREDICTION_STORE_INVALID"),
                _ms(row.get("window_end_seconds"), "E_PREDICTION_STORE_INVALID"),
            )
            if interval[1] <= interval[0] or not _clip_intervals([interval], masks):
                continue
            noun = tuple(value for value in row.get("noun_object_candidates", []) if isinstance(value, str))
            verb = tuple(value for value in row.get("verb_action_candidates", []) if isinstance(value, str))
            windows.append(
                MachineWindow(
                    interval=interval,
                    referential_status=str(row.get("referential_status", "UNDECIDABLE")),
                    role=str(row.get("source_role", "UNCERTAIN")),
                    noun_candidates=noun,
                    verb_candidates=verb,
                )
            )
        machines: list[MachineUtterance] = []
        asr_rows = audio.get("asr_hypotheses")
        role_rows = audio.get("speaker_role_hypotheses")
        if not isinstance(asr_rows, list) or not isinstance(role_rows, list):
            _fail("E_PREDICTION_STORE_INVALID")
        for index, row in enumerate(asr_rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("text"), str):
                _fail("E_PREDICTION_STORE_INVALID")
            fragments = _clip_intervals(
                _source_intervals(row.get("source_intervals"), "E_PREDICTION_STORE_INVALID"),
                masks,
            )
            if not fragments:
                continue
            longest = max(range(len(fragments)), key=lambda offset: (fragments[offset][1] - fragments[offset][0], -offset))
            audio_role = "UNCERTAIN"
            if index < len(role_rows) and isinstance(role_rows[index], Mapping):
                audio_role = str(role_rows[index].get("role", "UNCERTAIN"))
            for fragment_index, fragment in enumerate(fragments):
                window = _best_window(fragment, windows)
                visual_role = window.role if window else "UNCERTAIN"
                role = audio_role if audio_role in ROLE_DECIDABLE else visual_role
                machines.append(
                    MachineUtterance(
                        interval=fragment,
                        text=row["text"] if fragment_index == longest else "",
                        role=role if role in workflow.ROLE_VALUES else "UNCERTAIN",
                    )
                )
        vad: list[tuple[int, int]] = []
        for row in audio.get("vad_boundary_hypotheses", []):
            if not isinstance(row, Mapping):
                _fail("E_PREDICTION_STORE_INVALID")
            vad.extend(
                _source_intervals(row.get("source_intervals"), "E_PREDICTION_STORE_INVALID")
            )
        result.append(
            ItemEvidence(
                masks=tuple(masks),
                human_utterances=tuple(humans),
                machine_utterances=tuple(machines),
                machine_vad=tuple(vad),
                machine_windows=tuple(windows),
                wer_applicable=bool(label and label["wer_applicability"] == "APPLICABLE"),
            )
        )
    if len(result) != EXPECTED_ITEMS:
        _fail("E_EVALUATION_SCOPE")
    return result


def _locate_reserve_packet(root: Path, expected_digest: str) -> tuple[Path, Mapping[str, Any]]:
    if not HEX64_RE.fullmatch(expected_digest):
        _fail("E_RESERVE_BINDING")
    matches: list[tuple[Path, Mapping[str, Any]]] = []
    excluded = {"raw_v1_2", "pseudo_annotations", "browser_profile", "browser_cache", "browser_downloads"}
    for directory, directories, files in os.walk(root, followlinks=False):
        current = Path(directory)
        directories[:] = [
            name for name in directories
            if name not in excluded and not (current / name).is_symlink()
        ]
        for name in files:
            candidate = current / name
            if candidate.suffix.lower() != ".json" or not _private_regular(candidate):
                continue
            try:
                if candidate.stat().st_size > 16 * 1024 * 1024:
                    continue
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and _sha256_bytes(_canonical(value)) == expected_digest:
                matches.append((candidate.resolve(strict=True), value))
    if len(matches) != 1:
        _fail("E_RESERVE_PACKET_AMBIGUOUS" if matches else "E_RESERVE_PACKET_MISSING")
    path, packet = matches[0]
    if (
        packet.get("schema_version") != "childlens-author-audit-reserve-sample-v1.3.0"
        or packet.get("sample_kind") != "BORDERLINE_ESCALATION_RESERVE"
        or packet.get("activation") != "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK"
        or packet.get("maximum_activation_count") != 1
        or packet.get("total_duration_ms") != RESERVE_SECONDS * 1000
        or packet.get("item_count") != EXPECTED_ITEMS
        or packet.get("model_predictions_present") is not False
    ):
        _fail("E_RESERVE_PACKET_SCHEMA")
    return path, packet


def _activate_reserve_once(
    root: Path,
    meta: sqlite3.Row,
    threshold_results: Mapping[str, Any],
) -> bool:
    try:
        locked_at = meta["author_pass_locked_at_utc"]
        workflow_state = meta["workflow_state"]
    except (IndexError, KeyError):
        _fail("E_RESERVE_REQUIRES_AUTHOR_LOCK")
    if not locked_at or workflow_state not in {"AUTHOR_LOCKED", "PREDICTION_JOIN_ENABLED"}:
        _fail("E_RESERVE_REQUIRES_AUTHOR_LOCK")
    sampling = _public_json(SAMPLING_RECEIPT_PATH, "E_SAMPLING_RECEIPT")
    reserve_digest = sampling.get("reserve_packet_sha256")
    if not isinstance(reserve_digest, str):
        _fail("E_RESERVE_BINDING")
    _path, packet = _locate_reserve_packet(root, reserve_digest)
    if packet.get("frozen_selection_digest") != sampling.get("frozen_v1_2_selection_digest"):
        _fail("E_RESERVE_BINDING")
    if set(threshold_results) != set(GATE_NAMES):
        _fail("E_RESERVE_ACTIVATION_NOT_ALLOWED")
    statuses = {name: result.get("status") for name, result in threshold_results.items()}
    if (
        any(status == "HARD_FAIL" for status in statuses.values())
        or not any(status == "BORDERLINE" for status in statuses.values())
        or any(
            status not in {"PASS", "BORDERLINE", "HARD_FAIL", "NOT_ESTIMABLE"}
            for status in statuses.values()
        )
    ):
        _fail("E_RESERVE_ACTIVATION_NOT_ALLOWED")
    stable = {
        "schema_version": RESERVE_ACTIVATION_SCHEMA,
        "activation_count": 1,
        "maximum_activation_count": 1,
        "activation_basis": "PRIMARY_BORDERLINE_WITHOUT_HARD_FAIL",
        "primary_author_record_hmac": meta["author_record_hmac"],
        "primary_sample_digest": meta["sample_digest"],
        "protocol_digest": meta["protocol_digest"],
        "reserve_packet_sha256": reserve_digest,
        "reserve_total_ms": RESERVE_SECONDS * 1000,
        "reserve_predictions_mounted_before_lock": False,
        "reserve_author_record_state": "BLINDED_AUTHOR_LOCK_PENDING",
        "pseudo_labels_are_ground_truth": False,
        "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
    }
    path = root / COMPARISON_DIRECTORY / RESERVE_ACTIVATION_FILE
    _atomic_private(path, stable)
    return True


def compare_locked_primary() -> Mapping[str, Any]:
    """Zero-argument restricted comparison and aggregate receipt export."""

    if len(sys.argv) != 1:
        _fail("E_ZERO_ARGUMENT_ONLY")
    root = workflow.discover_runtime_root()
    workflow.validate_workflow(root)
    protocol = _public_json(PROTOCOL_PATH, "E_PROTOCOL")
    protocol_digest = _sha256_file(PROTOCOL_PATH)
    sampling = _public_json(SAMPLING_RECEIPT_PATH, "E_SAMPLING_RECEIPT")
    pseudo_receipt = _public_json(PSEUDO_RECEIPT_PATH, "E_PSEUDO_RECEIPT")
    if (
        pseudo_receipt.get("status") != "COMPLETE"
        or pseudo_receipt.get("pseudo_labels_marked_not_ground_truth") is not True
        or pseudo_receipt.get("simulator_oracle_labels_primary_evaluation_truth") is not True
    ):
        _fail("E_PSEUDO_ANNOTATION_INCOMPLETE")
    connection, meta, secret = _locked_context(root)
    try:
        _verify_locked_author(connection, meta, secret)
        if meta["protocol_digest"] != protocol_digest:
            _fail("E_PROTOCOL_DIGEST_MISMATCH")
        sample = _private_json(
            root / workflow.WORKFLOW_DIR / workflow.SAMPLE_SNAPSHOT_FILE,
            root,
            "E_SAMPLE_SNAPSHOT",
        )
        packet_digest = _sha256_bytes(_canonical(sample))
        if (
            packet_digest != meta["sample_digest"]
            or packet_digest != sampling.get("primary_packet_sha256")
        ):
            _fail("E_SAMPLE_DIGEST_MISMATCH")
        store_path, binding_path = _build_prediction_store(root, connection, meta)
    finally:
        connection.close()

    # This is the first operation capable of joining predictions, and it occurs
    # only after the immutable author HMAC and event ordering were verified.
    workflow.attach_prediction_store_after_lock(root, binding_path)
    workflow.validate_workflow(root)
    connection, meta, secret = _locked_context(root)
    try:
        _verify_locked_author(connection, meta, secret)
        store = _private_json(store_path, root, "E_PREDICTION_STORE_INVALID")
        if (
            store.get("schema_version") != PREDICTION_STORE_SCHEMA
            or store.get("sample_digest") != meta["sample_digest"]
            or store.get("protocol_digest") != meta["protocol_digest"]
            or store.get("pseudo_labels_are_ground_truth") is not False
            or store.get("primary_evaluation_truth") != "SIMULATOR_ORACLE_ONLY"
        ):
            _fail("E_PREDICTION_STORE_INVALID")
        evidence = _evidence_from_locked_store(connection, store)
        threshold_results, reserve_eligible = evaluate_items(
            evidence,
            protocol,
            protocol_digest=protocol_digest,
            sample_digest=meta["sample_digest"],
            phase="PRIMARY",
        )
        reserve_activated = (
            _activate_reserve_once(root, meta, threshold_results) if reserve_eligible else False
        )
        hard_failure = any(
            result.get("status") == "HARD_FAIL" for result in threshold_results.values()
        )
        receipt: Mapping[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "audit_complete": True,
            "audit_phase": "PRIMARY_900_SECONDS",
            "qualified_author_confirmed": True,
            "author_used_raw_audio_video": True,
            "author_blinded_until_irreversible_lock": True,
            "author_record_irreversibly_locked": True,
            "model_predictions_revealed_only_after_lock": True,
            "sample_selection_digest_match": True,
            "thresholds_digest_match": True,
            "cluster_aware_uncertainty_reported": True,
            "small_sample_limitation_reported": True,
            "author_record_changed_after_lock": False,
            "thresholds_changed_after_author_labels": False,
            "sample_changed_after_author_labels": False,
            "human_evidence_fabricated": False,
            "inter_human_reliability_claimed": False,
            "inter_human_reliability_available": False,
            "model_human_agreement_only": True,
            "pseudo_labels_are_ground_truth": False,
            "unaudited_pseudo_labels_are_primary_evaluation_truth": False,
            "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_confidence_level": BOOTSTRAP_CONFIDENCE,
            "additional_audit_speech_seconds_used": 0,
            "reserve_activation": {
                "eligible": reserve_eligible,
                "activated_once": reserve_activated,
                "activation_count": 1 if reserve_activated else 0,
                "maximum_activation_count": 1,
                "pending_author_lock": reserve_activated,
                "prediction_based_selection": False,
            },
            "fundamental_calibration_failure_established": hard_failure,
            "fundamental_failure_evidence_grade": (
                "LOCKED_QUALIFIED_AUTHOR_AUDIT" if hard_failure else "NOT_ESTABLISHED"
            ),
            "threshold_results": threshold_results,
            "privacy": {
                "item_rows_exported": False,
                "text_exported": False,
                "exact_times_exported": False,
                "identifiers_exported": False,
                "model_payloads_exported": False,
                "restricted_paths_exported": False,
                "small_cells_suppressed": True,
            },
        }
    finally:
        connection.close()
    validate_public_receipt(receipt)
    _atomic_public(PUBLIC_RESULT_PATH, receipt)
    return receipt


def main() -> int:
    try:
        compare_locked_primary()
        print("CHILDLENS_V13_AUTHOR_AUDIT_COMPARISON_COMPLETE")
        return 0
    except (ComparatorError, workflow.WorkflowError) as exc:
        print(getattr(exc, "code", "E_FAIL_CLOSED"))
        return 1
    except Exception:
        print("E_FAIL_CLOSED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
