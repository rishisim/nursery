from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_childlens_local_media_v1_2.py"
SPEC = importlib.util.spec_from_file_location("childlens_local_media_v1_2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_manifest(root: Path, media_relpath: str, *, extra_item: dict[str, object] | None = None) -> Path:
    media = root / media_relpath
    item: dict[str, object] = {
        "blinded_item_key": "SECRET_BLINDED_KEY",
        "media_relpath": media_relpath,
        "expected_size_bytes": media.stat().st_size if media.is_file() else 1,
        "expected_media_sha256": _sha256(media) if media.is_file() else "b" * 64,
        "reference_duration_seconds": 1.0,
        "reference_duration_basis": "AUTHORITATIVE_MEDIA_DURATION",
        "speech_presence_expectation": "PRESENT",
        "speech_windows": [{"start_seconds": 0.1, "end_seconds": 0.8}],
    }
    if extra_item:
        item.update(extra_item)
    manifest = {
        "schema_version": "childlens-restricted-measurement-manifest-v1.2.1",
        "pilot_selection_sha256": "a" * 64,
        "items": [item],
    }
    path = root / "SECRET_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.chmod(0o600)
    return path


def _quarantine(tmp_path: Path) -> Path:
    root = tmp_path / "SECRET_quarantine"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _make_synthetic_media(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=16000:duration=1",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_binary_metric_suppresses_small_nonzero_cell() -> None:
    value = MODULE._binary_metric(14, 15, 5)
    assert value == {
        "status": "MIXED_SMALL_CELL_SUPPRESSED",
        "pass_count": None,
        "fail_count": None,
        "cell_suppressed": True,
    }


def test_binary_metric_exports_all_pass() -> None:
    value = MODULE._binary_metric(15, 15, 5)
    assert value["status"] == "ALL_PASS"
    assert value["pass_count"] == 15
    assert value["fail_count"] == 0


def test_synthetic_media_audit_returns_only_aggregate_receipt(tmp_path: Path) -> None:
    ffmpeg_value = shutil.which("ffmpeg")
    ffprobe_value = shutil.which("ffprobe")
    if not ffmpeg_value or not ffprobe_value:
        pytest.skip("ffmpeg toolchain is not installed")
    root = _quarantine(tmp_path)
    media = root / "SECRET_source_media.mp4"
    _make_synthetic_media(media)
    media.chmod(0o600)
    manifest = _write_manifest(root, media.name)
    receipt = MODULE.audit(
        root,
        manifest,
        Path(ffprobe_value),
        Path(ffmpeg_value),
        timeout_seconds=30,
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert receipt["status"] == "ALL_STRUCTURAL_CHECKS_PASS"
    assert receipt["media_count"] == 1
    assert receipt["container_audio_decode_metrics"]["audio_stream_present"]["status"] == "ALL_PASS"
    assert receipt["container_audio_decode_metrics"]["full_decode_success"]["status"] == "ALL_PASS"
    assert (
        receipt["container_audio_decode_metrics"]["corruption_absence_evidenced"]["status"]
        == "ALL_PASS"
    )
    assert (
        receipt["container_audio_decode_metrics"]["duration_consistency_evidenced"]["status"]
        == "ALL_PASS"
    )
    assert (
        receipt["container_audio_decode_metrics"]
        ["speech_presence_window_expectation_satisfied"]["status"]
        == "ALL_PASS"
    )
    assert receipt["speech_window_structure"]["window_validity"]["status"] == "ALL_PASS"
    assert "SECRET" not in serialized
    assert str(tmp_path) not in serialized
    assert media.name not in serialized
    assert "0.1" not in serialized
    assert "0.8" not in serialized
    assert '"PRESENT"' not in serialized


def test_annotation_max_end_basis_checks_containment_without_claiming_authoritative_duration(
    tmp_path: Path,
) -> None:
    ffmpeg_value = shutil.which("ffmpeg")
    ffprobe_value = shutil.which("ffprobe")
    if not ffmpeg_value or not ffprobe_value:
        pytest.skip("ffmpeg toolchain is not installed")
    root = _quarantine(tmp_path)
    media = root / "opaque_media.bin"
    _make_synthetic_media(media)
    media.chmod(0o600)
    manifest = _write_manifest(
        root,
        media.name,
        extra_item={
            "reference_duration_seconds": 0.8,
            "reference_duration_basis": "ANNOTATION_MAX_END",
        },
    )
    receipt = MODULE.audit(
        root,
        manifest,
        Path(ffprobe_value),
        Path(ffmpeg_value),
        timeout_seconds=30,
    )
    metrics = receipt["container_audio_decode_metrics"]
    assert receipt["status"] == "ALL_STRUCTURAL_CHECKS_PASS"
    assert metrics["duration_reference_structurally_valid"]["status"] == "ALL_PASS"
    assert metrics["authoritative_duration_reference_coverage"]["status"] == "NONE_PASS"


def test_absent_speech_expectation_accepts_zero_windows_without_requiring_every_item_to_have_one(
    tmp_path: Path,
) -> None:
    ffmpeg_value = shutil.which("ffmpeg")
    ffprobe_value = shutil.which("ffprobe")
    if not ffmpeg_value or not ffprobe_value:
        pytest.skip("ffmpeg toolchain is not installed")
    root = _quarantine(tmp_path)
    media = root / "opaque_media.bin"
    _make_synthetic_media(media)
    media.chmod(0o600)
    manifest = _write_manifest(
        root,
        media.name,
        extra_item={"speech_presence_expectation": "ABSENT", "speech_windows": []},
    )
    receipt = MODULE.audit(
        root,
        manifest,
        Path(ffprobe_value),
        Path(ffmpeg_value),
        timeout_seconds=30,
    )
    metrics = receipt["container_audio_decode_metrics"]
    assert receipt["status"] == "ALL_STRUCTURAL_CHECKS_PASS"
    assert metrics["media_with_speech_windows"]["status"] == "NONE_PASS"
    assert metrics["speech_presence_window_expectation_satisfied"]["status"] == "ALL_PASS"
    assert receipt["speech_window_structure"]["window_validity"]["status"] == "NO_ELIGIBLE_ITEMS"


@pytest.mark.parametrize(
    ("expectation", "windows"),
    [
        ("PRESENT", []),
        ("PRESENT", [{"start_seconds": 0.9, "end_seconds": 1.8}]),
        ("ABSENT", [{"start_seconds": 0.1, "end_seconds": 0.8}]),
    ],
)
def test_speech_presence_expectation_mismatch_requires_review(
    tmp_path: Path, expectation: str, windows: list[dict[str, float]]
) -> None:
    ffmpeg_value = shutil.which("ffmpeg")
    ffprobe_value = shutil.which("ffprobe")
    if not ffmpeg_value or not ffprobe_value:
        pytest.skip("ffmpeg toolchain is not installed")
    root = _quarantine(tmp_path)
    media = root / "opaque_media.bin"
    _make_synthetic_media(media)
    media.chmod(0o600)
    manifest = _write_manifest(
        root,
        media.name,
        extra_item={"speech_presence_expectation": expectation, "speech_windows": windows},
    )
    receipt = MODULE.audit(
        root,
        manifest,
        Path(ffprobe_value),
        Path(ffmpeg_value),
        timeout_seconds=30,
    )
    assert receipt["status"] == "REVIEW_REQUIRED"
    assert (
        receipt["container_audio_decode_metrics"]
        ["speech_presence_window_expectation_satisfied"]["status"]
        == "NONE_PASS"
    )


def test_unknown_speech_presence_expectation_fails_closed(tmp_path: Path) -> None:
    root = _quarantine(tmp_path)
    media = root / "opaque.bin"
    media.write_bytes(b"not media")
    media.chmod(0o600)
    manifest = _write_manifest(
        root, media.name, extra_item={"speech_presence_expectation": "UNKNOWN"}
    )
    with pytest.raises(ValueError, match="E_SPEECH_PRESENCE_EXPECTATION"):
        MODULE._parse_manifest(manifest, root)


def test_unknown_item_field_fails_closed(tmp_path: Path) -> None:
    root = _quarantine(tmp_path)
    media = root / "opaque.bin"
    media.write_bytes(b"not media")
    media.chmod(0o600)
    manifest = _write_manifest(root, media.name, extra_item={"transcript_text": "SECRET WORDS"})
    with pytest.raises(ValueError, match="E_ITEM_SCHEMA"):
        MODULE._parse_manifest(manifest, root)


def test_path_traversal_fails_closed(tmp_path: Path) -> None:
    root = _quarantine(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not media")
    manifest = _write_manifest(root, "../outside.bin")
    with pytest.raises(ValueError, match="E_MEDIA_RELPATH"):
        MODULE._parse_manifest(manifest, root)


def test_symlink_fails_closed(tmp_path: Path) -> None:
    root = _quarantine(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not media")
    linked = root / "linked.bin"
    os.symlink(outside, linked)
    manifest = _write_manifest(root, linked.name)
    with pytest.raises(ValueError, match="E_SYMLINK"):
        MODULE._parse_manifest(manifest, root)


def test_manifest_permissions_fail_closed(tmp_path: Path) -> None:
    root = _quarantine(tmp_path)
    media = root / "opaque.bin"
    media.write_bytes(b"not media")
    media.chmod(0o600)
    manifest = _write_manifest(root, media.name)
    manifest.chmod(0o644)
    with pytest.raises(ValueError, match="E_MANIFEST_PERMISSIONS"):
        MODULE._parse_manifest(manifest, root)


def test_media_permissions_fail_closed(tmp_path: Path) -> None:
    root = _quarantine(tmp_path)
    media = root / "opaque.bin"
    media.write_bytes(b"not media")
    media.chmod(0o644)
    manifest = _write_manifest(root, media.name)
    with pytest.raises(ValueError, match="E_MEDIA_PERMISSIONS"):
        MODULE._parse_manifest(manifest, root)


def test_quarantine_inside_repository_is_rejected(tmp_path: Path) -> None:
    inside = ROOT / ".synthetic_childlens_v1_2_test"
    try:
        inside.mkdir(mode=0o700, exist_ok=False)
        inside.chmod(0o700)
        with pytest.raises(ValueError, match="E_QUARANTINE_INSIDE_REPOSITORY"):
            MODULE.audit(inside, inside / "absent.json", Path("/bin/false"), Path("/bin/false"))
    finally:
        if inside.exists():
            inside.rmdir()


def test_fixed_error_never_reflects_input() -> None:
    value = MODULE._fixed_error("E_MANIFEST_SCHEMA")
    assert value == {
        "schema_version": "childlens-local-media-audit-v1.2.1",
        "status": "error",
        "error_code": "E_MANIFEST_SCHEMA",
    }
