#!/usr/bin/env python3
"""Fail-closed, sequential native transfer for the frozen ChildLens pilot.

The restricted download plan, network locators, authorization token, exact
per-object metadata, hashes, ETags, and stored object paths never enter the
reportable output.  The command prints only a fixed success tuple or a constant
error code.  It performs no media decode, annotation, learner work, or remote
write operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol


VERSION = "childlens-native-transfer-controller-v1.2.0"
PLAN_SCHEMA = "childlens-restricted-download-plan-v1.1.0"
CONFIG_SCHEMA = "childlens-native-transfer-config-v1.2.0"
RECEIPT_SCHEMA = "childlens-native-transfer-restricted-receipt-v1.2.0"
AGGREGATE_SCHEMA = "childlens-native-transfer-aggregate-receipt-v1.2.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
HARD_RAW_CAP_BYTES = 20 * GIB
HARD_STREAM_LIMIT_BYTES = HARD_RAW_CAP_BYTES - 1
CONSERVATIVE_ADMISSION_LIMIT_BYTES = 18 * GIB
HARD_NAMESPACE_CAP_BYTES = 73 * GIB
HARD_FREE_SPACE_FLOOR_BYTES = 50 * GIB
EXPECTED_SELECTED_COUNT = 15
CHUNK_BYTES = 4 * 1024 * 1024
MAX_JSON_RESPONSE_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 4096

HEX_64 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

PLAN_KEYS = frozenset(
    {
        "schema_version",
        "canonical_restricted_manifest_sha256",
        "pilot_selection_sha256",
        "video_size_status",
        "selected",
    }
)
PLAN_ROW_KEYS = frozenset(
    {
        "selection_rank",
        "media_key",
        "object_key",
        "selection_hash",
        "source_locator",
        "expected_size_bytes",
        "local_sha256",
    }
)
CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "api_base_url",
        "repository_id",
        "immutable_commit_id",
        "allowed_download_origins",
        "authentication",
        "request_timeout_seconds",
        "metadata_strategy",
        "expected_download_plan_sha256",
        "quarantine_attestations",
        "admission",
    }
)
AUTH_KEYS = frozenset(
    {"environment_variable", "scheme", "send_authorization_to_download"}
)
ATTESTATION_KEYS = frozenset(
    {
        "owner_only_access_verified",
        "outside_git_repository_verified",
        "spotlight_excluded_verified",
        "backup_excluded_or_encrypted_local_only_verified",
        "retention_deadline",
        "signed_agreement_controls_verified",
    }
)
ADMISSION_KEYS = frozenset(
    {
        "mode",
        "amendment_frozen_before_media_open",
        "predeclared_nonraw_reserve_bytes",
        "conservative_bounds",
    }
)
BOUND_KEYS = frozenset(
    {
        "object_key",
        "rounded_display_size_bytes",
        "rounding_quantum_bytes",
        "transfer_overhead_bytes",
    }
)
RECEIPT_ITEM_KEYS = frozenset(
    {
        "selection_rank",
        "object_key",
        "admission_bound_bytes",
        "expected_exact_bytes",
        "metadata_etag",
        "metadata_content_length_present",
        "response_content_length",
        "response_etag",
        "transferred_bytes",
        "local_sha256",
        "stored_relative_path",
        "status",
    }
)


class TransferError(RuntimeError):
    """A failure with a constant, non-sensitive diagnostic code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise TransferError(code)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        _fail("E_ARGUMENTS")


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
        _fail("E_JSON_CANONICALIZATION")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], keys: frozenset[str], code: str) -> None:
    if set(value) != keys:
        _fail(code)


def _dict(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _list(value: Any, code: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _bool(value: Any, code: str) -> bool:
    if not isinstance(value, bool):
        _fail(code)
    return value


def _integer(value: Any, code: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(code)
    if maximum is not None and value > maximum:
        _fail(code)
    return value


def _string(value: Any, code: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail(code)
    if "\r" in value or "\n" in value or "\x00" in value:
        _fail(code)
    return value


def _hex(value: Any, code: str) -> str:
    text = _string(value, code, maximum=64)
    if not HEX_64.fullmatch(text):
        _fail(code)
    return text


def _read_json(path: Path, code: str) -> Any:
    try:
        if path.is_symlink() or not path.is_file():
            _fail(code)
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            _fail("E_RESTRICTED_FILE_PERMISSIONS")
        data = path.read_bytes()
        if len(data) > 16 * 1024 * 1024:
            _fail(code)
        return json.loads(data)
    except TransferError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)


def _atomic_json(path: Path, value: Any, mode: int) -> None:
    encoded = _canonical(value) + b"\n"
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    except TransferError:
        raise
    except OSError:
        _fail("E_ATOMIC_WRITE")


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                return True
        except OSError:
            _fail("E_QUARANTINE_PATH")
    return False


def _validate_quarantine_paths(
    root_path: Path,
    plan_path: Path,
    config_path: Path,
    receipt_path: Path,
) -> tuple[Path, Path, Path, Path]:
    try:
        root = root_path.resolve(strict=True)
        plan = plan_path.resolve(strict=True)
        config = config_path.resolve(strict=True)
        receipt_parent = receipt_path.parent.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_PATH")
    if _inside(root, REPO_ROOT) or _inside(REPO_ROOT, root):
        _fail("E_QUARANTINE_INSIDE_REPOSITORY")
    if _has_symlink_component(root_path) or not root.is_dir():
        _fail("E_QUARANTINE_ROOT_TYPE")
    if stat.S_IMODE(root.stat().st_mode) & 0o077:
        _fail("E_QUARANTINE_ROOT_PERMISSIONS")
    for value in (plan, config, receipt_parent):
        if not _inside(value, root):
            _fail("E_RESTRICTED_PATH_OUTSIDE_QUARANTINE")
    if receipt_path.exists() and receipt_path.is_symlink():
        _fail("E_RESTRICTED_RECEIPT_SYMLINK")
    return root, plan, config, receipt_parent / receipt_path.name


def _validate_public_output(path: Path, quarantine_root: Path) -> Path:
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        _fail("E_AGGREGATE_OUTPUT_PARENT")
    result = parent / path.name
    if _inside(result, quarantine_root):
        _fail("E_AGGREGATE_OUTPUT_RESTRICTED")
    if path.exists() and path.is_symlink():
        _fail("E_AGGREGATE_OUTPUT_SYMLINK")
    return result


def validate_plan(document: Any) -> dict[str, Any]:
    root = _dict(document, "E_PLAN_ROOT")
    _exact_keys(root, PLAN_KEYS, "E_PLAN_FIELDS")
    if root["schema_version"] != PLAN_SCHEMA:
        _fail("E_PLAN_SCHEMA")
    manifest_digest = _hex(root["canonical_restricted_manifest_sha256"], "E_MANIFEST_DIGEST")
    selection_digest = _hex(root["pilot_selection_sha256"], "E_SELECTION_DIGEST")
    if root["video_size_status"] != "EXACT_BYTES_UNRESOLVED":
        _fail("E_FROZEN_PLAN_SIZE_STATUS")
    rows = _list(root["selected"], "E_PLAN_SELECTED")
    if len(rows) != EXPECTED_SELECTED_COUNT:
        _fail("E_FROZEN_SELECTION_COUNT")
    normalised: list[dict[str, Any]] = []
    seen_media: set[str] = set()
    seen_objects: set[str] = set()
    seen_locators: set[str] = set()
    for expected_rank, raw in enumerate(rows, 1):
        row = _dict(raw, "E_PLAN_ROW")
        _exact_keys(row, PLAN_ROW_KEYS, "E_PLAN_ROW_FIELDS")
        rank = _integer(row["selection_rank"], "E_SELECTION_RANK", minimum=1)
        if rank != expected_rank:
            _fail("E_SELECTION_RANK_ORDER")
        media_key = _hex(row["media_key"], "E_MEDIA_KEY")
        object_key = _hex(row["object_key"], "E_OBJECT_KEY")
        selection_hash = _hex(row["selection_hash"], "E_SELECTION_HASH")
        locator = _string(row["source_locator"], "E_SOURCE_LOCATOR")
        if not locator.startswith("/ChildLens/videos/") or locator.endswith("/"):
            _fail("E_SOURCE_LOCATOR_SCOPE")
        if row["expected_size_bytes"] is not None or row["local_sha256"] is not None:
            _fail("E_FROZEN_PLAN_MUTATED")
        if media_key in seen_media or object_key in seen_objects or locator in seen_locators:
            _fail("E_FROZEN_PLAN_DUPLICATE")
        seen_media.add(media_key)
        seen_objects.add(object_key)
        seen_locators.add(locator)
        normalised.append(
            {
                "selection_rank": rank,
                "media_key": media_key,
                "object_key": object_key,
                "selection_hash": selection_hash,
                "source_locator": locator,
                "expected_size_bytes": None,
                "local_sha256": None,
            }
        )
    return {
        "schema_version": PLAN_SCHEMA,
        "canonical_restricted_manifest_sha256": manifest_digest,
        "pilot_selection_sha256": selection_digest,
        "video_size_status": "EXACT_BYTES_UNRESOLVED",
        "selected": normalised,
    }


def _origin(url: str, code: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        _fail(code)
    if parsed.scheme != "https" or not parsed.hostname:
        _fail(code)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        _fail(code)
    if parsed.path not in ("", "/"):
        _fail(code)
    port = f":{parsed.port}" if parsed.port else ""
    return f"https://{parsed.hostname.lower()}{port}"


def validate_config(document: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    root = _dict(document, "E_CONFIG_ROOT")
    _exact_keys(root, CONFIG_KEYS, "E_CONFIG_FIELDS")
    if root["schema_version"] != CONFIG_SCHEMA:
        _fail("E_CONFIG_SCHEMA")
    api_origin = _origin(_string(root["api_base_url"], "E_API_BASE_URL"), "E_API_BASE_URL")
    repository_id = _string(root["repository_id"], "E_REPOSITORY_ID", maximum=36)
    if not UUID.fullmatch(repository_id):
        _fail("E_REPOSITORY_ID")
    commit = root["immutable_commit_id"]
    if commit is not None:
        commit = _string(commit, "E_COMMIT_ID", maximum=64)
        if not HEX_COMMIT.fullmatch(commit):
            _fail("E_COMMIT_ID")
    origins = _list(root["allowed_download_origins"], "E_DOWNLOAD_ORIGINS")
    allowed_origins: list[str] = []
    for raw in origins:
        value = _origin(_string(raw, "E_DOWNLOAD_ORIGIN"), "E_DOWNLOAD_ORIGIN")
        if value in allowed_origins:
            _fail("E_DOWNLOAD_ORIGIN_DUPLICATE")
        allowed_origins.append(value)
    if not allowed_origins or len(allowed_origins) > 4:
        _fail("E_DOWNLOAD_ORIGINS")

    auth = _dict(root["authentication"], "E_AUTH")
    _exact_keys(auth, AUTH_KEYS, "E_AUTH_FIELDS")
    env_name = _string(auth["environment_variable"], "E_AUTH_ENV", maximum=64)
    if not ENV_NAME.fullmatch(env_name):
        _fail("E_AUTH_ENV")
    scheme = auth["scheme"]
    if scheme not in ("Token", "Bearer"):
        _fail("E_AUTH_SCHEME")
    send_download_auth = _bool(
        auth["send_authorization_to_download"], "E_DOWNLOAD_AUTH_FLAG"
    )
    timeout = _integer(
        root["request_timeout_seconds"], "E_REQUEST_TIMEOUT", minimum=10, maximum=3600
    )
    strategy = root["metadata_strategy"]
    if strategy not in ("FILE_DETAIL", "DOWNLOAD_HEAD"):
        _fail("E_METADATA_STRATEGY")
    if commit is not None and strategy != "DOWNLOAD_HEAD":
        _fail("E_REVISION_METADATA_NOT_BOUND")
    expected_plan_digest = _hex(
        root["expected_download_plan_sha256"], "E_EXPECTED_PLAN_DIGEST"
    )
    if expected_plan_digest != _digest(plan):
        _fail("E_FROZEN_PLAN_DIGEST_MISMATCH")

    attestations = _dict(root["quarantine_attestations"], "E_ATTESTATIONS")
    _exact_keys(attestations, ATTESTATION_KEYS, "E_ATTESTATION_FIELDS")
    for key in ATTESTATION_KEYS - {"retention_deadline"}:
        if _bool(attestations[key], "E_ATTESTATION_VALUE") is not True:
            _fail("E_ATTESTATION_FALSE")
    if attestations["retention_deadline"] != "2027-07-31":
        _fail("E_RETENTION_DEADLINE")

    admission = _dict(root["admission"], "E_ADMISSION")
    _exact_keys(admission, ADMISSION_KEYS, "E_ADMISSION_FIELDS")
    mode = admission["mode"]
    if mode not in ("NATIVE_EXACT", "CONSERVATIVE_ROUNDED"):
        _fail("E_ADMISSION_MODE")
    frozen_amendment = _bool(
        admission["amendment_frozen_before_media_open"], "E_AMENDMENT_ATTESTATION"
    )
    nonraw_reserve = _integer(
        admission["predeclared_nonraw_reserve_bytes"],
        "E_NONRAW_RESERVE",
        maximum=HARD_NAMESPACE_CAP_BYTES,
    )
    bounds_raw = _list(admission["conservative_bounds"], "E_CONSERVATIVE_BOUNDS")
    bounds: list[dict[str, int | str]] = []
    if mode == "NATIVE_EXACT":
        if frozen_amendment or bounds_raw:
            _fail("E_EXACT_MODE_AMENDMENT_PRESENT")
    else:
        if not frozen_amendment or len(bounds_raw) != EXPECTED_SELECTED_COUNT:
            _fail("E_CONSERVATIVE_AMENDMENT_INCOMPLETE")
        expected_keys = {row["object_key"] for row in plan["selected"]}
        seen: set[str] = set()
        for raw in bounds_raw:
            row = _dict(raw, "E_CONSERVATIVE_BOUND")
            _exact_keys(row, BOUND_KEYS, "E_CONSERVATIVE_BOUND_FIELDS")
            key = _hex(row["object_key"], "E_CONSERVATIVE_OBJECT_KEY")
            if key in seen or key not in expected_keys:
                _fail("E_CONSERVATIVE_OBJECT_LINK")
            seen.add(key)
            display = _integer(row["rounded_display_size_bytes"], "E_DISPLAY_BYTES", minimum=1)
            quantum = _integer(row["rounding_quantum_bytes"], "E_ROUNDING_QUANTUM", minimum=1)
            overhead = _integer(row["transfer_overhead_bytes"], "E_TRANSFER_OVERHEAD")
            upper = display + quantum + overhead
            if upper > HARD_STREAM_LIMIT_BYTES:
                _fail("E_CONSERVATIVE_OBJECT_BOUND")
            bounds.append(
                {
                    "object_key": key,
                    "rounded_display_size_bytes": display,
                    "rounding_quantum_bytes": quantum,
                    "transfer_overhead_bytes": overhead,
                    "upper_bound_bytes": upper,
                }
            )
        if seen != expected_keys:
            _fail("E_CONSERVATIVE_OBJECT_LINK")
        bounds.sort(key=lambda row: str(row["object_key"]))
        if sum(int(row["upper_bound_bytes"]) for row in bounds) > CONSERVATIVE_ADMISSION_LIMIT_BYTES:
            _fail("E_CONSERVATIVE_MARGIN")

    return {
        "schema_version": CONFIG_SCHEMA,
        "api_base_url": api_origin,
        "repository_id": repository_id,
        "immutable_commit_id": commit,
        "allowed_download_origins": sorted(allowed_origins),
        "authentication": {
            "environment_variable": env_name,
            "scheme": scheme,
            "send_authorization_to_download": send_download_auth,
        },
        "request_timeout_seconds": timeout,
        "metadata_strategy": strategy,
        "expected_download_plan_sha256": expected_plan_digest,
        "quarantine_attestations": dict(attestations),
        "admission": {
            "mode": mode,
            "amendment_frozen_before_media_open": frozen_amendment,
            "predeclared_nonraw_reserve_bytes": nonraw_reserve,
            "conservative_bounds": bounds,
        },
    }


def _token(config: Mapping[str, Any]) -> str:
    name = config["authentication"]["environment_variable"]
    value = os.environ.get(name)
    if value is None or not value or len(value.encode("utf-8")) > MAX_TOKEN_BYTES:
        _fail("E_AUTH_TOKEN_UNAVAILABLE")
    if any(char in value for char in ("\r", "\n", "\x00")):
        _fail("E_AUTH_TOKEN_INVALID")
    return value


@dataclass(frozen=True)
class RemoteMetadata:
    size_bytes: int
    etag: str | None
    content_length_present: bool


class StreamResponse(Protocol):
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class NativeClient(Protocol):
    def exact_metadata(self, source_locator: str) -> RemoteMetadata: ...

    def open_download(self, source_locator: str) -> StreamResponse: ...


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_origins: frozenset[str], authorization: str | None):
        self.allowed_origins = allowed_origins
        self.authorization = authorization
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        del fp, msg, headers
        if code not in (301, 302, 303, 307, 308):
            _fail("E_HTTP_REDIRECT_STATUS")
        origin = _download_origin(newurl, self.allowed_origins)
        redirected = urllib.request.Request(
            newurl,
            method="HEAD" if req.get_method() == "HEAD" else "GET",
            headers={"Accept-Encoding": "identity", "User-Agent": VERSION},
        )
        if self.authorization is not None and origin in self.allowed_origins:
            redirected.add_header("Authorization", self.authorization)
        return redirected


def _download_origin(url: str, allowed: frozenset[str]) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            _fail("E_DOWNLOAD_URL")
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        _fail("E_DOWNLOAD_URL")
    origin = f"https://{parsed.hostname.lower()}{port}"
    if origin not in allowed:
        _fail("E_DOWNLOAD_ORIGIN")
    if parsed.fragment:
        _fail("E_DOWNLOAD_URL")
    return origin


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 1024 or any(c in value for c in "\r\n\x00"):
        _fail("E_RESPONSE_HEADER")
    return value


def _repository_path(source_locator: str) -> str:
    """Map the frozen logical locator to the path inside its bound library.

    The v1.1 plan intentionally retained the release-level ``/ChildLens``
    namespace label.  Seafile's repo-scoped API is already rooted at that
    library, so the same selected object is addressed without the redundant
    first component.  No basename, ordering, selection key, or content-derived
    value participates in this deterministic mapping.
    """

    prefix = "/ChildLens/"
    path = source_locator[len("/ChildLens") :] if source_locator.startswith(prefix) else source_locator
    parts = path.split("/")
    if not path.startswith("/") or any(part in (".", "..") for part in parts):
        _fail("E_REPOSITORY_PATH")
    return path


def _response_metadata(headers: Mapping[str, str], *, require_length: bool) -> RemoteMetadata:
    encoding = _header(headers, "Content-Encoding")
    if encoding is not None and encoding.lower() not in ("identity", ""):
        _fail("E_CONTENT_ENCODING")
    length = _header(headers, "Content-Length")
    if length is None:
        if require_length:
            _fail("E_CONTENT_LENGTH_MISSING")
        size = -1
    elif not length.isdigit():
        _fail("E_CONTENT_LENGTH_INVALID")
    else:
        size = int(length)
    return RemoteMetadata(
        size_bytes=size,
        etag=_header(headers, "ETag"),
        content_length_present=length is not None,
    )


class SeafileNativeClient:
    """Read-only Seafile Web API client with a fixed endpoint allowlist."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        token = _token(config)
        self.authorization = f"{config['authentication']['scheme']} {token}"
        download_authorization = (
            self.authorization
            if config["authentication"]["send_authorization_to_download"]
            else None
        )
        self.allowed_origins = frozenset(config["allowed_download_origins"])
        self.api_opener = urllib.request.build_opener(
            _SafeRedirect(frozenset({config["api_base_url"]}), self.authorization)
        )
        self.download_opener = urllib.request.build_opener(
            _SafeRedirect(self.allowed_origins, download_authorization)
        )
        self.download_authorization = download_authorization

    def _request(self, url: str, *, download: bool, method: str = "GET"):
        headers = {"Accept-Encoding": "identity", "User-Agent": VERSION}
        if not download or self.download_authorization is not None:
            headers["Authorization"] = (
                self.authorization if not download else self.download_authorization
            )
        request = urllib.request.Request(url, method=method, headers=headers)
        opener = self.download_opener if download else self.api_opener
        try:
            return opener.open(request, timeout=self.config["request_timeout_seconds"])
        except TransferError:
            raise
        except urllib.error.HTTPError as exc:
            # Preserve only the status class needed for a safe operator
            # correction.  Never surface the response body, reason, URL, or
            # headers because any of those may contain restricted locators or
            # authorization material.
            if exc.code == 401:
                _fail("E_HTTP_AUTHENTICATION")
            if exc.code == 403:
                _fail("E_HTTP_AUTHORIZATION")
            if exc.code == 404:
                _fail("E_REMOTE_OBJECT_NOT_FOUND")
            _fail("E_HTTP_REQUEST")
        except (urllib.error.URLError, TimeoutError, OSError):
            _fail("E_HTTP_REQUEST")

    def _api_url(self, endpoint: str, query: Mapping[str, str]) -> str:
        return (
            f"{self.config['api_base_url']}{endpoint}?"
            f"{urllib.parse.urlencode(query, quote_via=urllib.parse.quote)}"
        )

    def _read_api_json(self, url: str) -> Any:
        response = self._request(url, download=False)
        try:
            payload = response.read(MAX_JSON_RESPONSE_BYTES + 1)
        finally:
            response.close()
        if len(payload) > MAX_JSON_RESPONSE_BYTES:
            _fail("E_API_RESPONSE_TOO_LARGE")
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _fail("E_API_RESPONSE_JSON")

    def _download_link(self, locator: str) -> str:
        locator = _repository_path(locator)
        repo = urllib.parse.quote(self.config["repository_id"], safe="")
        commit = self.config["immutable_commit_id"]
        if commit is None:
            endpoint = f"/api2/repos/{repo}/file/"
            query = {"p": locator, "reuse": "1"}
        else:
            endpoint = f"/api2/repos/{repo}/file/revision/"
            query = {"p": locator, "commit_id": commit}
        value = self._read_api_json(self._api_url(endpoint, query))
        link = _string(value, "E_DOWNLOAD_LINK_RESPONSE", maximum=8192)
        _download_origin(link, self.allowed_origins)
        return link

    def exact_metadata(self, source_locator: str) -> RemoteMetadata:
        source_locator = _repository_path(source_locator)
        if self.config["metadata_strategy"] == "FILE_DETAIL":
            repo = urllib.parse.quote(self.config["repository_id"], safe="")
            endpoint = f"/api2/repos/{repo}/file/detail/"
            value = _dict(
                self._read_api_json(self._api_url(endpoint, {"p": source_locator})),
                "E_FILE_DETAIL_RESPONSE",
            )
            size = _integer(value.get("size"), "E_FILE_DETAIL_SIZE", minimum=1)
            return RemoteMetadata(size, None, False)
        link = self._download_link(source_locator)
        response = self._request(link, download=True, method="HEAD")
        try:
            return _response_metadata(response.headers, require_length=True)
        finally:
            response.close()

    def open_download(self, source_locator: str) -> StreamResponse:
        return self._request(self._download_link(source_locator), download=True, method="GET")


def _namespace_bytes(root: Path) -> int:
    total = 0
    try:
        for directory, dirs, files in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in dirs:
                candidate = directory_path / name
                if candidate.is_symlink():
                    _fail("E_QUARANTINE_SYMLINK")
            for name in files:
                candidate = directory_path / name
                if candidate.is_symlink():
                    _fail("E_QUARANTINE_SYMLINK")
                info = candidate.stat()
                if not stat.S_ISREG(info.st_mode):
                    _fail("E_QUARANTINE_NONREGULAR")
                total += info.st_blocks * 512
        return total
    except TransferError:
        raise
    except OSError:
        _fail("E_NAMESPACE_MEASUREMENT")


def _capacity_check(root: Path, raw_upper_bound: int, nonraw_reserve: int) -> dict[str, int]:
    try:
        stats = os.statvfs(root)
    except OSError:
        _fail("E_VOLUME_MEASUREMENT")
    free = stats.f_bavail * stats.f_frsize
    namespace_now = _namespace_bytes(root)
    projected_delta = raw_upper_bound + nonraw_reserve
    if namespace_now + projected_delta > HARD_NAMESPACE_CAP_BYTES:
        _fail("E_NAMESPACE_CAP")
    if free - projected_delta < HARD_FREE_SPACE_FLOOR_BYTES:
        _fail("E_FREE_SPACE_FLOOR")
    return {
        "namespace_before_bytes": namespace_now,
        "volume_free_before_bytes": free,
        "projected_delta_bytes": projected_delta,
    }


def _new_receipt(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    exact_metadata: Mapping[str, RemoteMetadata] | None,
) -> dict[str, Any]:
    bounds_by_key = {
        row["object_key"]: row for row in config["admission"]["conservative_bounds"]
    }
    items = []
    for row in plan["selected"]:
        key = row["object_key"]
        if config["admission"]["mode"] == "NATIVE_EXACT":
            if exact_metadata is None or key not in exact_metadata:
                _fail("E_EXACT_METADATA_INCOMPLETE")
            metadata = exact_metadata[key]
            expected = metadata.size_bytes
            bound = expected
            metadata_etag = metadata.etag
            metadata_content_length = metadata.content_length_present
        else:
            expected = None
            bound = int(bounds_by_key[key]["upper_bound_bytes"])
            metadata_etag = None
            metadata_content_length = False
        items.append(
            {
                "selection_rank": row["selection_rank"],
                "object_key": key,
                "admission_bound_bytes": bound,
                "expected_exact_bytes": expected,
                "metadata_etag": metadata_etag,
                "metadata_content_length_present": metadata_content_length,
                "response_content_length": None,
                "response_etag": None,
                "transferred_bytes": None,
                "local_sha256": None,
                "stored_relative_path": None,
                "status": "PENDING",
            }
        )
    admission_upper = sum(item["admission_bound_bytes"] for item in items)
    if config["admission"]["mode"] == "NATIVE_EXACT":
        if admission_upper > HARD_STREAM_LIMIT_BYTES:
            _fail("E_RAW_CAP")
    elif admission_upper > CONSERVATIVE_ADMISSION_LIMIT_BYTES:
        _fail("E_CONSERVATIVE_MARGIN")
    return _seal_receipt(
        {
            "schema_version": RECEIPT_SCHEMA,
            "controller_version": VERSION,
            "download_plan_sha256": _digest(plan),
            "canonical_restricted_manifest_sha256": plan[
                "canonical_restricted_manifest_sha256"
            ],
            "pilot_selection_sha256": plan["pilot_selection_sha256"],
            "immutable_commit_id": config["immutable_commit_id"],
            "metadata_strategy": config["metadata_strategy"],
            "admission_mode": config["admission"]["mode"],
            "amendment_frozen_before_media_open": config["admission"][
                "amendment_frozen_before_media_open"
            ],
            "admission_upper_bound_bytes": admission_upper,
            "hard_raw_cap_bytes": HARD_RAW_CAP_BYTES,
            "hard_stream_limit_bytes": HARD_STREAM_LIMIT_BYTES,
            "conservative_admission_limit_bytes": CONSERVATIVE_ADMISSION_LIMIT_BYTES,
            "status": "PREPARED",
            "items": items,
            "restricted_receipt_sha256": None,
        }
    )


def _seal_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    payload = dict(receipt)
    payload["restricted_receipt_sha256"] = None
    payload["restricted_receipt_sha256"] = _digest(payload)
    return payload


def _validate_receipt(
    value: Any, plan: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _dict(value, "E_RECEIPT_ROOT")
    expected_keys = frozenset(
        {
            "schema_version",
            "controller_version",
            "download_plan_sha256",
            "canonical_restricted_manifest_sha256",
            "pilot_selection_sha256",
            "immutable_commit_id",
            "metadata_strategy",
            "admission_mode",
            "amendment_frozen_before_media_open",
            "admission_upper_bound_bytes",
            "hard_raw_cap_bytes",
            "hard_stream_limit_bytes",
            "conservative_admission_limit_bytes",
            "status",
            "items",
            "restricted_receipt_sha256",
        }
    )
    _exact_keys(receipt, expected_keys, "E_RECEIPT_FIELDS")
    claimed = _hex(receipt["restricted_receipt_sha256"], "E_RECEIPT_DIGEST")
    resealed = dict(receipt)
    resealed["restricted_receipt_sha256"] = None
    if _digest(resealed) != claimed:
        _fail("E_RECEIPT_DIGEST_MISMATCH")
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA
        or receipt["controller_version"] != VERSION
        or receipt["download_plan_sha256"] != _digest(plan)
        or receipt["canonical_restricted_manifest_sha256"]
        != plan["canonical_restricted_manifest_sha256"]
        or receipt["pilot_selection_sha256"] != plan["pilot_selection_sha256"]
        or receipt["immutable_commit_id"] != config["immutable_commit_id"]
        or receipt["metadata_strategy"] != config["metadata_strategy"]
        or receipt["admission_mode"] != config["admission"]["mode"]
    ):
        _fail("E_RECEIPT_BINDING")
    if receipt["status"] not in ("PREPARED", "IN_PROGRESS", "COMPLETE"):
        _fail("E_RECEIPT_STATUS")
    items = _list(receipt["items"], "E_RECEIPT_ITEMS")
    if len(items) != EXPECTED_SELECTED_COUNT:
        _fail("E_RECEIPT_ITEMS")
    plan_keys = [row["object_key"] for row in plan["selected"]]
    conservative_bounds = {
        row["object_key"]: int(row["upper_bound_bytes"])
        for row in config["admission"]["conservative_bounds"]
    }
    item_keys: list[str] = []
    completed_count = 0
    admission_sum = 0
    for expected_rank, raw in enumerate(items, 1):
        item = _dict(raw, "E_RECEIPT_ITEM")
        _exact_keys(item, RECEIPT_ITEM_KEYS, "E_RECEIPT_ITEM_FIELDS")
        rank = _integer(item["selection_rank"], "E_RECEIPT_ITEM_RANK", minimum=1)
        if rank != expected_rank:
            _fail("E_RECEIPT_ITEM_RANK")
        key = _hex(item["object_key"], "E_RECEIPT_ITEM_KEY")
        item_keys.append(key)
        bound = _integer(
            item["admission_bound_bytes"], "E_RECEIPT_ITEM_BOUND", minimum=1
        )
        admission_sum += bound
        expected = item["expected_exact_bytes"]
        if receipt["admission_mode"] == "NATIVE_EXACT":
            expected = _integer(expected, "E_RECEIPT_EXPECTED_BYTES", minimum=1)
            if expected != bound:
                _fail("E_RECEIPT_EXPECTED_BYTES")
        else:
            if expected is not None or conservative_bounds.get(key) != bound:
                _fail("E_RECEIPT_CONSERVATIVE_BOUND")
        if item["metadata_etag"] is not None:
            _string(item["metadata_etag"], "E_RECEIPT_METADATA_ETAG", maximum=1024)
        _bool(item["metadata_content_length_present"], "E_RECEIPT_METADATA_LENGTH")
        status_value = item["status"]
        if status_value not in ("PENDING", "COMPLETE"):
            _fail("E_RECEIPT_ITEM_STATUS")
        completion_fields = (
            item["response_content_length"],
            item["response_etag"],
            item["transferred_bytes"],
            item["local_sha256"],
            item["stored_relative_path"],
        )
        if status_value == "PENDING":
            if any(value is not None for value in completion_fields):
                _fail("E_RECEIPT_PENDING_FIELDS")
            continue
        completed_count += 1
        response_length = item["response_content_length"]
        if response_length is not None:
            _integer(response_length, "E_RECEIPT_RESPONSE_LENGTH", minimum=1)
        if item["response_etag"] is not None:
            _string(item["response_etag"], "E_RECEIPT_RESPONSE_ETAG", maximum=1024)
        transferred = _integer(
            item["transferred_bytes"], "E_RECEIPT_TRANSFERRED_BYTES", minimum=1
        )
        if transferred > bound:
            _fail("E_RECEIPT_TRANSFERRED_BYTES")
        digest = _hex(item["local_sha256"], "E_RECEIPT_LOCAL_SHA")
        if item["stored_relative_path"] != f"raw_v1_2/{digest}.bin":
            _fail("E_RECEIPT_STORED_PATH")
    if item_keys != plan_keys:
        _fail("E_RECEIPT_ITEM_BINDING")
    if admission_sum != receipt["admission_upper_bound_bytes"]:
        _fail("E_RECEIPT_ADMISSION_SUM")
    if receipt["status"] == "PREPARED" and completed_count != 0:
        _fail("E_RECEIPT_STATUS_COHERENCE")
    if receipt["status"] == "COMPLETE" and completed_count != EXPECTED_SELECTED_COUNT:
        _fail("E_RECEIPT_STATUS_COHERENCE")
    if receipt["amendment_frozen_before_media_open"] != config["admission"][
        "amendment_frozen_before_media_open"
    ]:
        _fail("E_RECEIPT_BINDING")
    if receipt["hard_raw_cap_bytes"] != HARD_RAW_CAP_BYTES:
        _fail("E_RECEIPT_CAP")
    if receipt["hard_stream_limit_bytes"] != HARD_STREAM_LIMIT_BYTES:
        _fail("E_RECEIPT_CAP")
    if receipt["conservative_admission_limit_bytes"] != CONSERVATIVE_ADMISSION_LIMIT_BYTES:
        _fail("E_RECEIPT_CAP")
    return receipt


def prepare_transfer(
    root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    client: NativeClient | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    exact: dict[str, RemoteMetadata] | None = None
    if config["admission"]["mode"] == "NATIVE_EXACT":
        if client is None:
            _fail("E_NATIVE_CLIENT_REQUIRED")
        exact = {}
        for row in plan["selected"]:
            metadata = client.exact_metadata(row["source_locator"])
            if metadata.size_bytes <= 0 or metadata.size_bytes > HARD_STREAM_LIMIT_BYTES:
                _fail("E_EXACT_REMOTE_SIZE")
            exact[row["object_key"]] = metadata
    receipt = _new_receipt(plan, config, exact)
    capacity = _capacity_check(
        root,
        receipt["admission_upper_bound_bytes"],
        config["admission"]["predeclared_nonraw_reserve_bytes"],
    )
    return receipt, capacity


def _verify_content_object(path: Path, expected_size: int, expected_hash: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            _fail("E_CONTENT_OBJECT_TYPE")
        if path.stat().st_size != expected_size:
            _fail("E_CONTENT_OBJECT_SIZE")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            _fail("E_CONTENT_OBJECT_HASH")
    except TransferError:
        raise
    except OSError:
        _fail("E_CONTENT_OBJECT_READ")


def acquire_transfer(
    root: Path,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    receipt: dict[str, Any],
    client: NativeClient,
    receipt_path: Path,
) -> dict[str, Any]:
    raw_dir = root / "raw_v1_2"
    try:
        raw_dir.mkdir(mode=0o700, exist_ok=True)
        if raw_dir.is_symlink() or stat.S_IMODE(raw_dir.stat().st_mode) & 0o077:
            _fail("E_RAW_DIRECTORY")
    except TransferError:
        raise
    except OSError:
        _fail("E_RAW_DIRECTORY")

    rows_by_key = {row["object_key"]: row for row in plan["selected"]}
    completed_bytes = sum(
        int(item["transferred_bytes"] or 0)
        for item in receipt["items"]
        if item["status"] == "COMPLETE"
    )
    if completed_bytes > HARD_STREAM_LIMIT_BYTES:
        _fail("E_HARD_CUMULATIVE_CAP")
    receipt["status"] = "IN_PROGRESS"
    receipt = _seal_receipt(receipt)
    _atomic_json(receipt_path, receipt, 0o600)

    for item in receipt["items"]:
        if item["status"] == "COMPLETE":
            stored = _string(item["stored_relative_path"], "E_STORED_PATH", maximum=128)
            if not stored.startswith("raw_v1_2/") or "/" in stored[len("raw_v1_2/") :]:
                _fail("E_STORED_PATH")
            _verify_content_object(
                root / stored,
                _integer(item["transferred_bytes"], "E_TRANSFERRED_BYTES", minimum=1),
                _hex(item["local_sha256"], "E_LOCAL_SHA"),
            )
            continue

        remaining_bound = sum(
            int(other["admission_bound_bytes"])
            for other in receipt["items"]
            if other["status"] != "COMPLETE"
        )
        _capacity_check(
            root,
            remaining_bound,
            config["admission"]["predeclared_nonraw_reserve_bytes"],
        )
        response = client.open_download(rows_by_key[item["object_key"]]["source_locator"])
        temporary: str | None = None
        try:
            response_meta = _response_metadata(response.headers, require_length=False)
            expected = item["expected_exact_bytes"]
            bound = int(item["admission_bound_bytes"])
            if response_meta.content_length_present:
                if response_meta.size_bytes > bound:
                    _fail("E_OBJECT_BOUND_EXCEEDED")
                if expected is not None and response_meta.size_bytes != expected:
                    _fail("E_CONTENT_LENGTH_MISMATCH")
            fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=raw_dir)
            os.fchmod(fd, 0o600)
            digest = hashlib.sha256()
            transferred = 0
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    transferred += len(chunk)
                    if transferred > bound:
                        _fail("E_OBJECT_BOUND_EXCEEDED")
                    if completed_bytes + transferred > HARD_STREAM_LIMIT_BYTES:
                        _fail("E_HARD_CUMULATIVE_CAP")
                    handle.write(chunk)
                    digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if transferred <= 0:
                _fail("E_EMPTY_OBJECT")
            if expected is not None and transferred != expected:
                _fail("E_TRANSFER_SIZE_MISMATCH")
            if response_meta.content_length_present and transferred != response_meta.size_bytes:
                _fail("E_RESPONSE_TRUNCATED")
            sha256 = digest.hexdigest()
            destination = raw_dir / f"{sha256}.bin"
            if destination.exists():
                _verify_content_object(destination, transferred, sha256)
                os.unlink(temporary)
                temporary = None
            else:
                os.replace(temporary, destination)
                temporary = None
                os.chmod(destination, 0o600)
                directory_fd = os.open(raw_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            item.update(
                {
                    "response_content_length": (
                        response_meta.size_bytes if response_meta.content_length_present else None
                    ),
                    "response_etag": response_meta.etag,
                    "transferred_bytes": transferred,
                    "local_sha256": sha256,
                    "stored_relative_path": f"raw_v1_2/{sha256}.bin",
                    "status": "COMPLETE",
                }
            )
            completed_bytes += transferred
            receipt = _seal_receipt(receipt)
            _atomic_json(receipt_path, receipt, 0o600)
        except TransferError:
            raise
        except OSError:
            _fail("E_TRANSFER_IO")
        finally:
            response.close()
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    receipt["status"] = "COMPLETE"
    receipt = _seal_receipt(receipt)
    _atomic_json(receipt_path, receipt, 0o600)
    return receipt


def _suppress_small(value: int) -> int | None:
    return value if value == 0 or value >= 5 else None


def aggregate_receipt(
    receipt: Mapping[str, Any],
    capacity: Mapping[str, int] | None,
) -> dict[str, Any]:
    completed = [item for item in receipt["items"] if item["status"] == "COMPLETE"]
    complete = receipt["status"] == "COMPLETE" and len(completed) == EXPECTED_SELECTED_COUNT
    completed_count = len(completed)
    total_bytes = sum(int(item["transferred_bytes"] or 0) for item in completed)
    sha_values = {item["local_sha256"] for item in completed}
    etag_all = bool(completed) and all(item["response_etag"] is not None for item in completed)
    content_length_all = bool(completed) and all(
        item["response_content_length"] is not None for item in completed
    )
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "controller_version": VERSION,
        "status": receipt["status"],
        "release_binding": {
            "download_plan_sha256": receipt["download_plan_sha256"],
            "canonical_restricted_manifest_sha256": receipt[
                "canonical_restricted_manifest_sha256"
            ],
            "immutable_revision_download_used": receipt["immutable_commit_id"] is not None,
        },
        "admission": {
            "mode": receipt["admission_mode"],
            "storage_only_amendment_used": receipt[
                "amendment_frozen_before_media_open"
            ],
            "selected_count": EXPECTED_SELECTED_COUNT,
            "admission_upper_bound_bytes": receipt["admission_upper_bound_bytes"],
            "hard_raw_cap_bytes": HARD_RAW_CAP_BYTES,
            "hard_stream_limit_bytes": HARD_STREAM_LIMIT_BYTES,
            "conservative_admission_limit_bytes": CONSERVATIVE_ADMISSION_LIMIT_BYTES,
            "minimum_margin_bytes": HARD_RAW_CAP_BYTES
            - CONSERVATIVE_ADMISSION_LIMIT_BYTES,
        },
        "transfer": {
            "complete": complete,
            "completed_count": _suppress_small(completed_count),
            "completed_count_suppressed": 0 < completed_count < 5,
            "total_bytes": total_bytes if completed_count >= 5 else (0 if completed_count == 0 else None),
            "unique_content_count": (
                len(sha_values) if completed_count >= 5 else (0 if completed_count == 0 else None)
            ),
            "all_response_content_lengths_present": content_length_all if completed else None,
            "all_response_etags_present": etag_all if completed else None,
            "sequential_single_stream": True,
            "atomic_content_addressed_storage": True,
        },
        "capacity_observation": (
            {
                "namespace_before_bytes": capacity["namespace_before_bytes"],
                "volume_free_before_bytes": capacity["volume_free_before_bytes"],
                "projected_delta_bytes": capacity["projected_delta_bytes"],
            }
            if capacity is not None
            else None
        ),
        "restricted_receipt_sha256": receipt["restricted_receipt_sha256"],
        "restricted_payload_absence": {
            "source_filenames": True,
            "source_paths": True,
            "participant_or_media_keys": True,
            "per_object_sizes_hashes_or_etags": True,
            "authorization_material": True,
            "timestamps_transcripts_or_frames": True,
        },
        "scientific_operations": {
            "media_decode": False,
            "learner_training": False,
            "causal_arm_run": False,
        },
    }
    return result


def run(
    *,
    phase: str,
    quarantine_root: Path,
    plan_path: Path,
    config_path: Path,
    restricted_receipt_path: Path,
    aggregate_output_path: Path,
    client: NativeClient | None = None,
) -> dict[str, Any]:
    root, safe_plan_path, safe_config_path, safe_receipt_path = _validate_quarantine_paths(
        quarantine_root, plan_path, config_path, restricted_receipt_path
    )
    safe_aggregate_path = _validate_public_output(aggregate_output_path, root)
    plan = validate_plan(_read_json(safe_plan_path, "E_PLAN_READ"))
    config = validate_config(_read_json(safe_config_path, "E_CONFIG_READ"), plan)
    network = client
    if network is None and (phase == "acquire" or config["admission"]["mode"] == "NATIVE_EXACT"):
        network = SeafileNativeClient(config)

    if phase == "prepare":
        receipt, capacity = prepare_transfer(root, plan, config, network)
        _atomic_json(safe_receipt_path, receipt, 0o600)
        aggregate = aggregate_receipt(receipt, capacity)
        _atomic_json(safe_aggregate_path, aggregate, 0o644)
        return aggregate
    if phase != "acquire":
        _fail("E_PHASE")
    if not safe_receipt_path.exists():
        _fail("E_PREPARED_RECEIPT_REQUIRED")
    receipt = _validate_receipt(
        _read_json(safe_receipt_path, "E_RECEIPT_READ"), plan, config
    )
    if network is None:
        _fail("E_NATIVE_CLIENT_REQUIRED")
    receipt = acquire_transfer(root, plan, config, receipt, network, safe_receipt_path)
    aggregate = aggregate_receipt(receipt, None)
    _atomic_json(safe_aggregate_path, aggregate, 0o644)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = _SafeArgumentParser(add_help=True)
    parser.add_argument("--phase", choices=("prepare", "acquire"), required=True)
    parser.add_argument("--quarantine-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--restricted-receipt", required=True)
    parser.add_argument("--aggregate-output", required=True)
    old_umask = os.umask(0o077)
    try:
        args = parser.parse_args(argv)
        result = run(
            phase=args.phase,
            quarantine_root=Path(args.quarantine_root),
            plan_path=Path(args.plan),
            config_path=Path(args.config),
            restricted_receipt_path=Path(args.restricted_receipt),
            aggregate_output_path=Path(args.aggregate_output),
        )
    except TransferError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)
    print(json.dumps({"status": "ok", "phase": args.phase, "state": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
