from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "childlens_native_transfer_v1_2.py"
SPEC = importlib.util.spec_from_file_location("childlens_native_transfer_v1_2", SCRIPT)
assert SPEC and SPEC.loader
transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transfer
SPEC.loader.exec_module(transfer)


def _key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _plan() -> dict:
    selected = []
    for index in range(15):
        selected.append(
            {
                "selection_rank": index + 1,
                "media_key": _key(f"synthetic-media-{index}"),
                "object_key": _key(f"synthetic-object-{index}"),
                "selection_hash": _key(f"synthetic-selection-{index}"),
                "source_locator": f"/ChildLens/videos/synthetic-fixture-{index:02d}.bin",
                "expected_size_bytes": None,
                "local_sha256": None,
            }
        )
    return {
        "schema_version": transfer.PLAN_SCHEMA,
        "canonical_restricted_manifest_sha256": _key("synthetic-manifest"),
        "pilot_selection_sha256": _key("synthetic-pilot-selection"),
        "video_size_status": "EXACT_BYTES_UNRESOLVED",
        "selected": selected,
    }


def _config(plan: dict, *, mode: str = "NATIVE_EXACT") -> dict:
    if mode == "CONSERVATIVE_ROUNDED":
        # The per-object upper bound is 998,012,416 bytes.  Fifteen objects
        # reproduce the coordinator-frozen aggregate of 14,970,186,240 bytes.
        bounds = [
            {
                "object_key": row["object_key"],
                "rounded_display_size_bytes": 997_000_000,
                "rounding_quantum_bytes": 1_000_000,
                "transfer_overhead_bytes": 12_416,
            }
            for row in plan["selected"]
        ]
        amendment = True
    else:
        bounds = []
        amendment = False
    return {
        "schema_version": transfer.CONFIG_SCHEMA,
        "api_base_url": "https://keeper.invalid",
        "repository_id": "11111111-2222-3333-4444-555555555555",
        "immutable_commit_id": None,
        "allowed_download_origins": ["https://keeper.invalid"],
        "authentication": {
            "environment_variable": "CHILDLENS_SEAFILE_TOKEN",
            "scheme": "Token",
            "send_authorization_to_download": False,
        },
        "request_timeout_seconds": 120,
        "metadata_strategy": "FILE_DETAIL",
        "expected_download_plan_sha256": transfer._digest(plan),
        "quarantine_attestations": {
            "owner_only_access_verified": True,
            "outside_git_repository_verified": True,
            "spotlight_excluded_verified": True,
            "backup_excluded_or_encrypted_local_only_verified": True,
            "retention_deadline": "2027-07-31",
            "signed_agreement_controls_verified": True,
        },
        "admission": {
            "mode": mode,
            "amendment_frozen_before_media_open": amendment,
            "predeclared_nonraw_reserve_bytes": 0,
            "conservative_bounds": bounds,
        },
    }


def _write_private(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _workspace(tmp_path: Path, config: dict | None = None):
    root = tmp_path / "restricted"
    root.mkdir(mode=0o700)
    plan = _plan()
    config = config or _config(plan)
    plan_path = root / "plan.json"
    config_path = root / "config.json"
    receipt_path = root / "receipt.json"
    aggregate_path = tmp_path / "aggregate.json"
    _write_private(plan_path, plan)
    _write_private(config_path, config)
    return root, plan, config, plan_path, config_path, receipt_path, aggregate_path


class FakeResponse:
    def __init__(self, payload: bytes, tracker: dict, *, etag: str = '"synthetic"'):
        self.payload = payload
        self.offset = 0
        self.tracker = tracker
        self.headers = {
            "Content-Length": str(len(payload)),
            "Content-Encoding": "identity",
            "ETag": etag,
        }
        tracker["active"] += 1
        tracker["maximum_active"] = max(tracker["maximum_active"], tracker["active"])

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        result = self.payload[self.offset : self.offset + size]
        self.offset += len(result)
        return result

    def close(self) -> None:
        if self.tracker["active"]:
            self.tracker["active"] -= 1


class FakeClient:
    def __init__(self, plan: dict, *, duplicate: bool = False):
        self.payloads = {}
        for index, row in enumerate(plan["selected"]):
            self.payloads[row["source_locator"]] = (
                b"same synthetic payload" if duplicate else bytes([index + 1]) * (index + 5)
            )
        self.tracker = {"active": 0, "maximum_active": 0}
        self.metadata_calls = []
        self.download_calls = []

    def exact_metadata(self, source_locator: str):
        self.metadata_calls.append(source_locator)
        return transfer.RemoteMetadata(len(self.payloads[source_locator]), None, False)

    def open_download(self, source_locator: str):
        self.download_calls.append(source_locator)
        return FakeResponse(self.payloads[source_locator], self.tracker)


def _error(callable_, *args, **kwargs) -> str:
    with pytest.raises(transfer.TransferError) as caught:
        callable_(*args, **kwargs)
    return caught.value.code


def test_frozen_plan_requires_exactly_fifteen_unmodified_rows() -> None:
    plan = _plan()
    assert len(transfer.validate_plan(plan)["selected"]) == 15
    plan["selected"].pop()
    assert _error(transfer.validate_plan, plan) == "E_FROZEN_SELECTION_COUNT"

    plan = _plan()
    plan["selected"][0]["expected_size_bytes"] = 1
    assert _error(transfer.validate_plan, plan) == "E_FROZEN_PLAN_MUTATED"


def test_config_binds_to_exact_frozen_plan_digest() -> None:
    plan = transfer.validate_plan(_plan())
    config = _config(plan)
    config["expected_download_plan_sha256"] = _key("wrong")
    assert (
        _error(transfer.validate_config, config, plan)
        == "E_FROZEN_PLAN_DIGEST_MISMATCH"
    )


def test_conservative_amendment_matches_frozen_aggregate_and_margin() -> None:
    plan = transfer.validate_plan(_plan())
    config = transfer.validate_config(_config(plan, mode="CONSERVATIVE_ROUNDED"), plan)
    total = sum(row["upper_bound_bytes"] for row in config["admission"]["conservative_bounds"])
    assert total == 14_970_186_240
    assert total <= transfer.CONSERVATIVE_ADMISSION_LIMIT_BYTES
    assert (
        transfer.HARD_RAW_CAP_BYTES - transfer.CONSERVATIVE_ADMISSION_LIMIT_BYTES
        == 2 * transfer.GIB
    )
    assert transfer.HARD_STREAM_LIMIT_BYTES == 21_474_836_479


def test_conservative_amendment_fails_above_eighteen_gib() -> None:
    plan = transfer.validate_plan(_plan())
    config = _config(plan, mode="CONSERVATIVE_ROUNDED")
    row = config["admission"]["conservative_bounds"][0]
    row["rounded_display_size_bytes"] += (
        transfer.CONSERVATIVE_ADMISSION_LIMIT_BYTES - 14_970_186_240 + 1
    )
    assert _error(transfer.validate_config, config, plan) == "E_CONSERVATIVE_MARGIN"


def test_exact_prepare_and_acquire_are_sequential_atomic_and_identifier_free_publicly(
    tmp_path: Path,
) -> None:
    root, plan, _, plan_path, config_path, receipt_path, aggregate_path = _workspace(tmp_path)
    client = FakeClient(plan)
    prepared = transfer.run(
        phase="prepare",
        quarantine_root=root,
        plan_path=plan_path,
        config_path=config_path,
        restricted_receipt_path=receipt_path,
        aggregate_output_path=aggregate_path,
        client=client,
    )
    assert prepared["status"] == "PREPARED"
    assert len(client.metadata_calls) == 15

    completed = transfer.run(
        phase="acquire",
        quarantine_root=root,
        plan_path=plan_path,
        config_path=config_path,
        restricted_receipt_path=receipt_path,
        aggregate_output_path=aggregate_path,
        client=client,
    )
    assert completed["status"] == "COMPLETE"
    assert completed["transfer"]["completed_count"] == 15
    assert completed["transfer"]["unique_content_count"] == 15
    assert client.tracker["maximum_active"] == 1
    assert len(list((root / "raw_v1_2").glob("*.bin"))) == 15
    assert not list((root / "raw_v1_2").glob(".partial-*"))
    assert stat_mode(receipt_path) == 0o600

    public_text = aggregate_path.read_text()
    restricted_sentinels = [
        plan["selected"][0]["source_locator"],
        plan["selected"][0]["object_key"],
        plan["selected"][0]["media_key"],
        plan["pilot_selection_sha256"],
        "synthetic-fixture",
    ]
    assert all(sentinel not in public_text for sentinel in restricted_sentinels)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_duplicate_payloads_store_one_content_addressed_copy(tmp_path: Path) -> None:
    root, plan, _, plan_path, config_path, receipt_path, aggregate_path = _workspace(tmp_path)
    client = FakeClient(plan, duplicate=True)
    transfer.run(
        phase="prepare",
        quarantine_root=root,
        plan_path=plan_path,
        config_path=config_path,
        restricted_receipt_path=receipt_path,
        aggregate_output_path=aggregate_path,
        client=client,
    )
    aggregate = transfer.run(
        phase="acquire",
        quarantine_root=root,
        plan_path=plan_path,
        config_path=config_path,
        restricted_receipt_path=receipt_path,
        aggregate_output_path=aggregate_path,
        client=client,
    )
    assert aggregate["transfer"]["unique_content_count"] == 1
    assert len(list((root / "raw_v1_2").glob("*.bin"))) == 1


def test_response_larger_than_admission_bound_aborts_before_body(tmp_path: Path) -> None:
    root, plan, _, plan_path, config_path, receipt_path, aggregate_path = _workspace(
        tmp_path, _config(_plan(), mode="CONSERVATIVE_ROUNDED")
    )
    transfer.run(
        phase="prepare",
        quarantine_root=root,
        plan_path=plan_path,
        config_path=config_path,
        restricted_receipt_path=receipt_path,
        aggregate_output_path=aggregate_path,
        client=None,
    )

    class OversizeClient(FakeClient):
        def open_download(self, source_locator: str):
            response = super().open_download(source_locator)
            response.headers["Content-Length"] = str(998_012_417)
            return response

    client = OversizeClient(plan)
    assert (
        _error(
            transfer.run,
            phase="acquire",
            quarantine_root=root,
            plan_path=plan_path,
            config_path=config_path,
            restricted_receipt_path=receipt_path,
            aggregate_output_path=aggregate_path,
            client=client,
        )
        == "E_OBJECT_BOUND_EXCEEDED"
    )
    assert not list((root / "raw_v1_2").glob(".partial-*"))


def test_tampered_restricted_receipt_fails_closed(tmp_path: Path) -> None:
    root, plan, _, plan_path, config_path, receipt_path, aggregate_path = _workspace(tmp_path)
    client = FakeClient(plan)
    transfer.run(
        phase="prepare",
        quarantine_root=root,
        plan_path=plan_path,
        config_path=config_path,
        restricted_receipt_path=receipt_path,
        aggregate_output_path=aggregate_path,
        client=client,
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["admission_upper_bound_bytes"] += 1
    _write_private(receipt_path, receipt)
    config = transfer.validate_config(_config(transfer.validate_plan(plan)), transfer.validate_plan(plan))
    assert (
        _error(transfer._validate_receipt, receipt, transfer.validate_plan(plan), config)
        == "E_RECEIPT_DIGEST_MISMATCH"
    )


def test_partial_aggregate_suppresses_small_cells() -> None:
    plan = transfer.validate_plan(_plan())
    config = transfer.validate_config(_config(plan, mode="CONSERVATIVE_ROUNDED"), plan)
    receipt = transfer._new_receipt(plan, config, None)
    item = receipt["items"][0]
    item.update(
        {
            "status": "COMPLETE",
            "transferred_bytes": 10,
            "local_sha256": _key("payload"),
            "stored_relative_path": f"raw_v1_2/{_key('payload')}.bin",
        }
    )
    receipt["status"] = "IN_PROGRESS"
    receipt = transfer._seal_receipt(receipt)
    aggregate = transfer.aggregate_receipt(receipt, None)
    assert aggregate["transfer"]["completed_count"] is None
    assert aggregate["transfer"]["completed_count_suppressed"] is True
    assert aggregate["transfer"]["total_bytes"] is None
    assert aggregate["transfer"]["unique_content_count"] is None


def test_download_origin_rejects_non_https_and_unlisted_hosts() -> None:
    allowed = frozenset({"https://keeper.invalid"})
    assert (
        _error(transfer._download_origin, "http://keeper.invalid/object", allowed)
        == "E_DOWNLOAD_URL"
    )
    assert (
        _error(transfer._download_origin, "https://outside.invalid/object", allowed)
        == "E_DOWNLOAD_ORIGIN"
    )


def test_official_detail_and_revision_routes_are_constructed_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = transfer.validate_plan(_plan())
    config = transfer.validate_config(_config(plan), plan)
    monkeypatch.setenv("CHILDLENS_SEAFILE_TOKEN", "synthetic-token-not-a-secret")
    client = transfer.SeafileNativeClient(config)
    observed = []

    def detail(url: str):
        observed.append(url)
        return {"size": 123, "name": "ignored", "last_modifier_name": "ignored"}

    monkeypatch.setattr(client, "_read_api_json", detail)
    metadata = client.exact_metadata(plan["selected"][0]["source_locator"])
    assert metadata.size_bytes == 123
    assert "/api2/repos/11111111-2222-3333-4444-555555555555/file/detail/" in observed[0]
    assert "p=%2Fvideos%2F" in observed[0]
    assert "%2FChildLens%2F" not in observed[0]

    raw_config = _config(plan)
    raw_config["immutable_commit_id"] = "a" * 40
    raw_config["metadata_strategy"] = "DOWNLOAD_HEAD"
    revision_config = transfer.validate_config(raw_config, plan)
    revision_client = transfer.SeafileNativeClient(revision_config)
    revision_urls = []

    def revision(url: str):
        revision_urls.append(url)
        return "https://keeper.invalid/native/opaque"

    monkeypatch.setattr(revision_client, "_read_api_json", revision)
    link = revision_client._download_link(plan["selected"][0]["source_locator"])
    assert link == "https://keeper.invalid/native/opaque"
    assert "/file/revision/" in revision_urls[0]
    assert "commit_id=" + ("a" * 40) in revision_urls[0]


def test_release_namespace_maps_deterministically_to_repo_root() -> None:
    assert transfer._repository_path("/ChildLens/videos/opaque.bin") == "/videos/opaque.bin"
    assert transfer._repository_path("/videos/opaque.bin") == "/videos/opaque.bin"
    assert (
        _error(transfer._repository_path, "/ChildLens/videos/../opaque.bin")
        == "E_REPOSITORY_PATH"
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "E_HTTP_AUTHENTICATION"),
        (403, "E_HTTP_AUTHORIZATION"),
        (404, "E_REMOTE_OBJECT_NOT_FOUND"),
        (500, "E_HTTP_REQUEST"),
    ],
)
def test_http_failures_expose_only_fixed_operator_categories(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: str
) -> None:
    plan = transfer.validate_plan(_plan())
    config = transfer.validate_config(_config(plan), plan)
    monkeypatch.setenv("CHILDLENS_SEAFILE_TOKEN", "synthetic-token-not-a-secret")
    client = transfer.SeafileNativeClient(config)

    class FailingOpener:
        def open(self, request, timeout):
            del request, timeout
            raise transfer.urllib.error.HTTPError(
                "https://keeper.invalid/restricted-sentinel",
                status,
                "restricted-response-sentinel",
                None,
                None,
            )

    client.api_opener = FailingOpener()
    assert (
        _error(
            client._request,
            "https://keeper.invalid/api2/safe-probe",
            download=False,
        )
        == expected
    )


def test_restricted_paths_must_be_outside_repository() -> None:
    assert (
        _error(
            transfer._validate_quarantine_paths,
            REPO_ROOT,
            REPO_ROOT / "output/childlens_feasibility_v1_1/pilot_preselection_receipt.json",
            REPO_ROOT / "output/childlens_feasibility_v1_1/pilot_preselection_receipt.json",
            REPO_ROOT / "restricted-receipt.json",
        )
        == "E_QUARANTINE_INSIDE_REPOSITORY"
    )


def test_cli_parse_failure_never_echoes_restricted_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "restricted-path-sentinel-never-echo"
    assert transfer.main(["--unknown", sentinel]) == 2
    captured = capsys.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert json.loads(captured.out) == {"status": "error", "error_code": "E_ARGUMENTS"}
