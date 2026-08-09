from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_childlens_restricted_manifest_v1_1.py"
SPEC = importlib.util.spec_from_file_location("build_childlens_restricted_manifest_v1_1", SCRIPT)
assert SPEC and SPEC.loader
manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest
SPEC.loader.exec_module(manifest)


def _key(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _document(
    count: int = 30,
    *,
    participant_count: int | None = None,
    object_size: int = 1_000,
    complete_checksums: bool = True,
) -> dict:
    participant_count = participant_count or count
    objects = []
    annotations = []
    media = []
    for index in range(count):
        media_key = _key(f"media-key-{index}")
        video_object_key = _key(f"video-object-key-{index}")
        annotation_key = _key(f"annotation-key-{index}")
        annotation_object_key = _key(f"annotation-object-key-{index}")
        participant_index = index % participant_count
        participant_key = _key(f"participant-key-{participant_index}")
        session_key = _key(f"session-key-{participant_index}-{index}")
        objects.extend(
            [
                {
                    "object_key": video_object_key,
                    "source_locator": f"restricted/sensitive-child-clip-{index:03d}.mp4",
                    "top_level_class": "VIDEO",
                    "size_bytes": object_size,
                    "source_checksum_sha256": (
                        _key(f"video-content-{index}") if complete_checksums else None
                    ),
                    "local_sha256": None,
                    "remote_version": f"restricted-version-{index}",
                    "remote_etag": f"restricted-etag-{index}",
                    "available_for_selective_copy": True,
                    "container_parseable": None,
                },
                {
                    "object_key": annotation_object_key,
                    "source_locator": f"restricted/sensitive-annotation-{index:03d}.json",
                    "top_level_class": "ANNOTATION",
                    "size_bytes": 100,
                    "source_checksum_sha256": (
                        _key(f"annotation-content-{index}") if complete_checksums else None
                    ),
                    "local_sha256": None,
                    "remote_version": f"restricted-annotation-version-{index}",
                    "remote_etag": None,
                    "available_for_selective_copy": True,
                    "container_parseable": None,
                },
            ]
        )
        annotations.append(
            {
                "annotation_key": annotation_key,
                "object_key": annotation_object_key,
                "linked_media_key": media_key,
                "representation_kind": "FINAL" if index % 2 == 0 else "PLATFORM",
            }
        )
        media.append(
            {
                "media_key": media_key,
                "object_key": video_object_key,
                "participant_key": participant_key,
                "session_key": session_key,
                "duration_milliseconds": 60_000 + index * 1_000,
                "coarse_activity_label": f"ACTIVITY_{index % 3}",
                "speech_presence_bin": "PRESENT" if index % 2 == 0 else "ABSENT",
                "location_label": f"LOCATION_{index % 2}",
            }
        )
    return {
        "schema_version": manifest.INPUT_SCHEMA,
        "release": {
            "doi": "10.17617/4.fe",
            "keeper_library_id": "c8ed0104-b793-4c35-817e-302afd4e036b",
            "source_role": "LIVE_REVISION",
            "public_doi_snapshot_commit": "4856662653b2fa183e53268d088cffba02a33443",
            "observed_revision_commit": None,
            "observation_date": "2026-07-21",
            "observation_receipt_sha256": _key("observation-receipt"),
            "doi_snapshot_object_inventory_sha256": None,
            "doi_snapshot_inventory_receipt_sha256": None,
            "session_definition": "PARTICIPANT_PLUS_RECORDING_DATE",
        },
        "attestations": {
            "internal_keys_are_hmac_sha256": True,
            "key_derivation_receipt_sha256": _key("key-derivation-receipt"),
            "metadata_frozen_before_content_inspection": True,
            "no_content_fields_used": True,
        },
        "objects": objects,
        "annotations": annotations,
        "media": media,
    }


def _preflight(
    *,
    raw_cap_bytes: int = 1_000_000,
    namespace_peak_bytes: int = 10 * manifest.GIB,
    free_before_bytes: int = 100 * manifest.GIB,
) -> manifest.ResourcePreflight:
    return manifest.ResourcePreflight(
        raw_cap_bytes=raw_cap_bytes,
        projected_namespace_peak_bytes=namespace_peak_bytes,
        volume_free_before_bytes=free_before_bytes,
    )


def _build(document: dict | None = None, **kwargs):
    return manifest.build_artifacts(
        document or _document(),
        preflight=kwargs.pop("preflight", _preflight()),
        protocol_sha256=_key("synthetic-frozen-protocol-receipt"),
        **kwargs,
    )


def _error_code(document: dict, **kwargs) -> str:
    with pytest.raises(manifest.ManifestError) as caught:
        _build(document, **kwargs)
    return caught.value.code


def test_builds_deterministic_manifest_and_target_fifteen_selection() -> None:
    document = _document()
    restricted_a, reportable_a = _build(document)
    shuffled = copy.deepcopy(document)
    random.Random(47).shuffle(shuffled["objects"])
    random.Random(48).shuffle(shuffled["annotations"])
    random.Random(49).shuffle(shuffled["media"])
    restricted_b, reportable_b = _build(shuffled)

    assert restricted_a == restricted_b
    assert reportable_a == reportable_b
    assert len(restricted_a["pilot_selection"]) == 15
    assert reportable_a["frozen_pilot_selection"]["selected_video_count"] == 15
    assert reportable_a["frozen_pilot_selection"]["target_adjusted"] is False


def test_selection_hash_implements_frozen_concatenation_exactly() -> None:
    restricted, _ = _build()
    release_digest = restricted["canonical_restricted_manifest_sha256"]
    for row in restricted["pilot_selection"]:
        expected = hashlib.sha256(
            (
                release_digest
                + manifest.FROZEN_PROTOCOL_ID
                + row["media_key"]
            ).encode("utf-8")
        ).hexdigest()
        assert row["selection_hash"] == expected


def test_prefers_distinct_participants_and_never_exceeds_two() -> None:
    restricted, reportable = _build(_document(count=30, participant_count=30))
    participants = [row["participant_key"] for row in restricted["pilot_selection"]]
    assert len(participants) == len(set(participants)) == 15
    assert reportable["frozen_pilot_selection"]["all_selected_participants_distinct"] is True

    restricted_reuse, reportable_reuse = _build(_document(count=16, participant_count=8))
    counts = Counter(row["participant_key"] for row in restricted_reuse["pilot_selection"])
    assert len(restricted_reuse["pilot_selection"]) == 15
    assert max(counts.values()) == 2
    assert reportable_reuse["frozen_pilot_selection"]["maximum_videos_per_participant"] == 2


def test_balances_metadata_strata_before_reusing_a_stratum() -> None:
    restricted, _ = _build()
    strata = [json.dumps(row["stratum"], sort_keys=True) for row in restricted["pilot_selection"]]
    counts = Counter(strata)
    assert len(counts) >= 9
    assert max(counts.values()) - min(counts.values()) <= 1


def test_raw_cap_may_reduce_target_only_within_frozen_range() -> None:
    document = _document(object_size=100)
    restricted, reportable = _build(document, preflight=_preflight(raw_cap_bytes=1_200))
    assert len(restricted["pilot_selection"]) == 12
    assert reportable["frozen_pilot_selection"]["target_adjusted"] is True
    assert reportable["frozen_pilot_selection"]["adjustment_reasons"] == ["RAW_BYTE_CAP"]
    assert reportable["frozen_pilot_selection"]["selected_raw_bytes"] == 1_200

    assert (
        _error_code(document, preflight=_preflight(raw_cap_bytes=1_100))
        == "PILOT_MINIMUM_NOT_REACHABLE"
    )


@pytest.mark.parametrize("target", [11, 19])
def test_rejects_target_outside_frozen_range(target: int) -> None:
    assert _error_code(_document(), target=target) == "PILOT_TARGET_OUTSIDE_FROZEN_RANGE"


def test_enforces_all_three_storage_admission_caps() -> None:
    assert (
        _error_code(_document(), preflight=_preflight(raw_cap_bytes=manifest.HARD_RAW_CAP_BYTES + 1))
        == "RAW_CAP_EXCEEDS_FROZEN_LIMIT"
    )
    assert (
        _error_code(
            _document(),
            preflight=_preflight(namespace_peak_bytes=manifest.HARD_NAMESPACE_CAP_BYTES + 1),
        )
        == "NAMESPACE_PEAK_EXCEEDS_FROZEN_LIMIT"
    )
    assert (
        _error_code(
            _document(),
            preflight=_preflight(
                namespace_peak_bytes=20 * manifest.GIB,
                free_before_bytes=69 * manifest.GIB,
            ),
        )
        == "POST_PEAK_FREE_SPACE_BELOW_FLOOR"
    )


def test_rejects_duplicate_object_and_locator_records() -> None:
    duplicate_key = _document()
    duplicate_key["objects"][1]["object_key"] = duplicate_key["objects"][0]["object_key"]
    assert _error_code(duplicate_key) == "DUPLICATE_OBJECT_KEY"

    duplicate_locator = _document()
    duplicate_locator["objects"][1]["source_locator"] = duplicate_locator["objects"][0][
        "source_locator"
    ]
    assert _error_code(duplicate_locator) == "DUPLICATE_SOURCE_LOCATOR"


def test_allows_byte_duplicates_but_counts_known_content_distinctly() -> None:
    document = _document()
    video_rows = [row for row in document["objects"] if row["top_level_class"] == "VIDEO"]
    video_rows[1]["source_checksum_sha256"] = video_rows[0]["source_checksum_sha256"]
    _, reportable = _build(document)
    video_summary = next(
        row for row in reportable["inventory_summary"] if row["top_level_class"] == "VIDEO"
    )
    assert video_summary["file_count"] == 30
    assert video_summary["distinct_known_content_count"] == 29


def test_rejects_broken_annotation_linkage_without_echoing_value() -> None:
    document = _document()
    sentinel = _key("nonexistent-sensitive-media-key")
    document["annotations"][0]["linked_media_key"] = sentinel
    code = _error_code(document)
    assert code == "ANNOTATION_MEDIA_LINK_INVALID"
    assert sentinel not in code


def test_rejects_incomplete_or_cross_participant_grouping() -> None:
    incomplete = _document()
    incomplete["media"][0]["participant_key"] = None
    incomplete["media"][0]["session_key"] = None
    assert _error_code(incomplete) == "GROUPING_INCOMPLETE_FOR_ANNOTATED_MEDIA"

    crossing = _document()
    crossing["media"][1]["session_key"] = crossing["media"][0]["session_key"]
    assert _error_code(crossing) == "SESSION_CROSSES_PARTICIPANTS"


def test_rejects_content_or_unrecognised_input_fields() -> None:
    document = _document()
    document["media"][0]["transcript_text"] = "restricted lexical content"
    assert _error_code(document) == "MEDIA_FIELDS_INVALID"


def test_reportable_receipt_contains_no_restricted_rows_or_sentinels() -> None:
    document = _document()
    restricted, reportable = _build(document)
    encoded = json.dumps(reportable, sort_keys=True)
    for row in document["objects"]:
        assert row["source_locator"] not in encoded
        assert row["object_key"] not in encoded
        if row["source_checksum_sha256"] is not None:
            assert row["source_checksum_sha256"] not in encoded
    for row in document["media"]:
        assert row["media_key"] not in encoded
        assert row["participant_key"] not in encoded
        assert row["session_key"] not in encoded
    assert "sensitive-child-clip" not in encoded
    assert reportable["privacy"] == {
        "contains_filenames_or_paths": False,
        "contains_row_level_keys": False,
        "contains_per_object_checksums": False,
        "contains_exact_media_timestamps": False,
        "contains_transcript_or_lexical_content": False,
        "cell_suppression_applied": True,
    }
    assert restricted["pilot_selection"]


def test_small_inventory_cell_and_complement_are_suppressed() -> None:
    document = _document()
    document["objects"].append(
        {
            "object_key": _key("one-small-other-object"),
            "source_locator": "restricted/small-sensitive-object.bin",
            "top_level_class": "OTHER",
            "size_bytes": 7,
            "source_checksum_sha256": _key("small-sensitive-content"),
            "local_sha256": None,
            "remote_version": None,
            "remote_etag": None,
            "available_for_selective_copy": False,
            "container_parseable": None,
        }
    )
    _, reportable = _build(document)
    by_class = {row["top_level_class"]: row for row in reportable["inventory_summary"]}
    assert by_class["OTHER"]["file_count"] == "SUPPRESSED"
    assert by_class["VIDEO"]["file_count"] == "SUPPRESSED"
    assert by_class["ANNOTATION"]["file_count"] == 30


def test_release_binding_names_unpinned_live_limitation() -> None:
    _, reportable = _build()
    binding = reportable["release_binding"]
    assert binding["binding_mode"] == "RESTRICTED_MANIFEST_DIGEST_PLUS_OBSERVATION_RECEIPT"
    assert binding["remote_revision_immutable"] is False
    assert binding["doi_snapshot_equivalence"] == "NOT_PROVEN"
    assert len(binding["limitations"]) == 2


def test_doi_snapshot_binding_requires_the_exact_public_commit() -> None:
    document = _document()
    document["release"]["source_role"] = "DOI_SNAPSHOT"
    document["release"]["observed_revision_commit"] = document["release"][
        "public_doi_snapshot_commit"
    ]
    _, reportable = _build(document)
    assert reportable["release_binding"]["binding_mode"] == (
        "DOI_SNAPSHOT_COMMIT_PLUS_RESTRICTED_MANIFEST"
    )
    assert reportable["release_binding"]["remote_revision_immutable"] is True

    bad = copy.deepcopy(document)
    bad["release"]["observed_revision_commit"] = "0" * 40
    assert _error_code(bad) == "DOI_SNAPSHOT_COMMIT_BINDING_INVALID"


def test_snapshot_inventory_comparison_proves_equivalence_only_with_complete_checksums() -> None:
    document = _document(complete_checksums=True)
    _, first_report = _build(document)
    document["release"]["doi_snapshot_object_inventory_sha256"] = first_report[
        "release_binding"
    ]["canonical_object_inventory_sha256"]
    document["release"]["doi_snapshot_inventory_receipt_sha256"] = _key(
        "independent-snapshot-manifest-receipt"
    )
    _, reportable = _build(document)
    assert reportable["release_binding"]["doi_snapshot_equivalence"] == (
        "PROVEN_BY_COMPLETE_SIZE_AND_SHA256_MULTISET_DIGEST"
    )

    incomplete = _document(complete_checksums=False)
    incomplete["release"]["doi_snapshot_object_inventory_sha256"] = _key(
        "independent-complete-snapshot-inventory"
    )
    incomplete["release"]["doi_snapshot_inventory_receipt_sha256"] = _key(
        "independent-incomplete-snapshot-receipt"
    )
    _, incomplete_report = _build(incomplete)
    assert incomplete_report["release_binding"]["doi_snapshot_equivalence"] == (
        "NOT_PROVEN_CURRENT_OBJECT_CHECKSUMS_INCOMPLETE"
    )


def test_snapshot_manifest_digest_mismatch_fails_closed() -> None:
    document = _document()
    document["release"]["doi_snapshot_object_inventory_sha256"] = "0" * 64
    document["release"]["doi_snapshot_inventory_receipt_sha256"] = _key("comparison-receipt")
    assert _error_code(document) == "DOI_SNAPSHOT_OBJECT_INVENTORY_MISMATCH"


def test_restricted_output_inside_repository_or_symlink_alias_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    restricted, reportable = _build()
    inside = repo / "restricted.json"
    public = tmp_path / "public.json"
    with pytest.raises(manifest.ManifestError, match="RESTRICTED_OUTPUT_INSIDE_REPOSITORY"):
        manifest.write_artifacts(inside, public, restricted, reportable, repo_root=repo)
    assert not inside.exists()

    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(manifest.ManifestError, match="RESTRICTED_OUTPUT_INSIDE_REPOSITORY"):
        manifest.write_artifacts(alias / "restricted.json", public, restricted, reportable, repo_root=repo)


def test_writes_restricted_mode_outside_repo_and_public_receipt_without_rows(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    quarantine = tmp_path / "quarantine"
    reports = repo / "reports"
    repo.mkdir()
    quarantine.mkdir()
    reports.mkdir()
    restricted, reportable = _build()
    restricted_path = quarantine / "restricted.json"
    reportable_path = reports / "receipt.json"
    manifest.write_artifacts(
        restricted_path,
        reportable_path,
        restricted,
        reportable,
        repo_root=repo,
    )
    assert restricted_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(restricted_path.read_text())["pilot_selection"]
    public_text = reportable_path.read_text()
    assert "media_key" not in public_text
    assert "source_locator" not in public_text


def test_frozen_protocol_file_still_matches_selector_contract() -> None:
    digest = manifest._load_and_validate_frozen_protocol()
    assert len(digest) == 64
