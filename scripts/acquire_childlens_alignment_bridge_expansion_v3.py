#!/usr/bin/env python3
"""Acquire ten frozen ChildLens v3 expansion clips into private quarantine."""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_expansion_v3.json"
SELECTION_RECEIPT = (
    ROOT
    / "output/childlens_alignment_bridge_expansion_v3/"
    "selection_freeze_receipt.json"
)
PUBLIC_RECEIPT = (
    ROOT
    / "output/childlens_alignment_bridge_expansion_v3/"
    "acquisition_receipt.json"
)
PRIVATE_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3"
)
PRIVATE_PLAN = PRIVATE_RELATIVE / "restricted_expansion_plan.json"
PRIVATE_CHECKPOINT = PRIVATE_RELATIVE / "restricted_acquisition_checkpoint.json"
PRIVATE_CLIPS = PRIVATE_RELATIVE / "clips"
PRIVATE_TRANSIENT = PRIVATE_RELATIVE / "transient"

PROTOCOL_SHA256 = "787f64eba92a6a2f206e09a447b2f595691230349ba8f17c800faa0e50108f02"
SELECTION_RECEIPT_SHA256 = (
    "9decd2b5ad832c29c24d97b416711032df0950aa39fcf82302a75be0cc91d081"
)
RESTRICTED_PLAN_SHA256 = (
    "796eccc748cd61590bd0c9d4499e92e81277327f11e6ad4a19ce106afa4b4cb6"
)
EXPECTED_COUNT = 10
RAW_CAP_BYTES = 20 * 1024**3
DERIVED_CAP_BYTES = 4 * 1024**3
NAMESPACE_CAP_BYTES = 10 * 1024**3
FREE_FLOOR_BYTES = 50 * 1024**3
CHUNK_BYTES = 4 * 1024**2
MAX_JSON_RESPONSE_BYTES = 1024 * 1024
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE = Path("/opt/homebrew/bin/ffprobe")
KEYCHAIN_TOOL = Path("/usr/bin/security")
KEYCHAIN_SERVICE = "ChildLens-v1.2-Keeper-Repo-Token"
KEYCHAIN_ACCOUNT = "childlens-v1.2-read-only"
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~+/=-]{16,1024}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PATH_TOKEN = re.compile(r"(?i)(?:/users/|file://|\\users\\)")
MEDIA_TOKEN = re.compile(r"(?i)\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a)\b")


class AcquisitionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise AcquisitionError(code)


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
        _fail("E_CANONICAL")


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
    except OSError:
        _fail("E_FILE")
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_FILE")


def _private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _private_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _discover_runtime() -> Path:
    search = ROOT.parent
    candidates: list[Path] = []
    for hidden in search.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            for manifest in hidden.rglob("restricted_manifest/preselection_manifest.json"):
                if _private_file(manifest):
                    candidates.append(manifest.parent)
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if not _private_directory(runtime) or not _private_file(
        runtime / ".metadata_never_index"
    ):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _write_atomic(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600 if private else 0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, 0o600 if private else 0o644)
    except OSError:
        _fail("E_WRITE")
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    _write_atomic(path, value, private=private)


def _read_token() -> str:
    try:
        result = subprocess.run(
            [
                str(KEYCHAIN_TOOL),
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("E_KEYCHAIN_READ")
    if result.returncode != 0:
        _fail("E_KEYCHAIN_ITEM_UNAVAILABLE")
    try:
        token = result.stdout.rstrip(b"\r\n").decode("utf-8")
    except UnicodeDecodeError:
        _fail("E_KEYCHAIN_TOKEN")
    if not TOKEN_PATTERN.fullmatch(token):
        _fail("E_KEYCHAIN_TOKEN")
    return token


def _delete_token() -> None:
    try:
        result = subprocess.run(
            [
                str(KEYCHAIN_TOOL),
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("E_KEYCHAIN_DELETE")
    if result.returncode != 0:
        _fail("E_KEYCHAIN_DELETE")


def _origin(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        _fail("E_URL")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        _fail("E_URL")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"https://{parsed.hostname.lower()}{port}"


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, origins: frozenset[str], authorization: str | None):
        super().__init__()
        self.origins = origins
        self.authorization = authorization

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if _origin(newurl) not in self.origins:
            _fail("E_REDIRECT_ORIGIN")
        redirected = urllib.request.Request(
            newurl,
            method=req.get_method(),
            headers={"Accept-Encoding": "identity", "User-Agent": "ChildLens-v3"},
        )
        if self.authorization is not None:
            redirected.add_header("Authorization", self.authorization)
        return redirected


def _repository_path(source_locator: str) -> str:
    prefix = "/ChildLens/"
    path = (
        source_locator[len("/ChildLens") :]
        if source_locator.startswith(prefix)
        else source_locator
    )
    if not path.startswith("/") or any(part in (".", "..") for part in path.split("/")):
        _fail("E_REPOSITORY_PATH")
    return path


class SeafileClient:
    def __init__(self, config: Mapping[str, Any], token: str):
        api_base = config.get("api_base_url")
        repository_id = config.get("repository_id")
        origins = config.get("allowed_download_origins")
        authentication = config.get("authentication")
        timeout = config.get("request_timeout_seconds")
        if (
            not isinstance(api_base, str)
            or not isinstance(repository_id, str)
            or not isinstance(origins, list)
            or not all(isinstance(value, str) for value in origins)
            or not isinstance(authentication, Mapping)
            or authentication.get("scheme") != "Bearer"
            or authentication.get("send_authorization_to_download") is not True
            or type(timeout) is not int
            or timeout <= 0
            or config.get("metadata_strategy") != "FILE_DETAIL"
        ):
            _fail("E_TRANSFER_CONFIG")
        self.api_base = api_base.rstrip("/")
        self.repository_id = repository_id
        self.timeout = timeout
        self.authorization = f"Bearer {token}"
        self.api_origin = _origin(self.api_base)
        self.download_origins = frozenset(_origin(value) for value in origins)
        self.api_opener = urllib.request.build_opener(
            _SafeRedirect(frozenset({self.api_origin}), self.authorization)
        )
        self.download_opener = urllib.request.build_opener(
            _SafeRedirect(self.download_origins, self.authorization)
        )

    def _request(
        self,
        url: str,
        *,
        download: bool,
        method: str = "GET",
    ):
        expected = self.download_origins if download else frozenset({self.api_origin})
        if _origin(url) not in expected:
            _fail("E_REQUEST_ORIGIN")
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "Accept-Encoding": "identity",
                "Authorization": self.authorization,
                "User-Agent": "ChildLens-v3",
            },
        )
        opener = self.download_opener if download else self.api_opener
        try:
            return opener.open(request, timeout=self.timeout)
        except AcquisitionError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                _fail("E_HTTP_AUTHENTICATION")
            if exc.code == 403:
                _fail("E_HTTP_AUTHORIZATION")
            if exc.code == 404:
                _fail("E_REMOTE_OBJECT_NOT_FOUND")
            _fail("E_HTTP_REQUEST")
        except (urllib.error.URLError, TimeoutError, OSError):
            _fail("E_HTTP_REQUEST")

    def _api_json(self, endpoint: str, query: Mapping[str, str]) -> Any:
        url = (
            self.api_base
            + endpoint
            + "?"
            + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        )
        response = self._request(url, download=False)
        try:
            payload = response.read(MAX_JSON_RESPONSE_BYTES + 1)
        finally:
            response.close()
        if len(payload) > MAX_JSON_RESPONSE_BYTES:
            _fail("E_API_RESPONSE_SIZE")
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            _fail("E_API_RESPONSE_JSON")

    def exact_size(self, source_locator: str) -> int:
        path = _repository_path(source_locator)
        repo = urllib.parse.quote(self.repository_id, safe="")
        value = self._api_json(
            f"/api2/repos/{repo}/file/detail/",
            {"p": path},
        )
        size = value.get("size") if isinstance(value, Mapping) else None
        if type(size) is not int or size <= 0:
            _fail("E_REMOTE_SIZE")
        return size

    def open_download(self, source_locator: str):
        path = _repository_path(source_locator)
        repo = urllib.parse.quote(self.repository_id, safe="")
        link = self._api_json(
            f"/api2/repos/{repo}/file/",
            {"p": path, "reuse": "1"},
        )
        if not isinstance(link, str) or len(link) > 8192:
            _fail("E_DOWNLOAD_LINK")
        if _origin(link) not in self.download_origins:
            _fail("E_DOWNLOAD_ORIGIN")
        return self._request(link, download=True)


def _run_json(command: list[str], *, timeout: int) -> Mapping[str, Any]:
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("E_MEDIA_TOOL")
    if result.returncode != 0:
        _fail("E_MEDIA_TOOL")
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("E_MEDIA_TOOL")
    if not isinstance(value, Mapping):
        _fail("E_MEDIA_TOOL")
    return value


def _probe(path: Path) -> tuple[float, bool, bool]:
    value = _run_json(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        timeout=300,
    )
    try:
        duration = float(value["format"]["duration"])
        types = {
            stream.get("codec_type")
            for stream in value["streams"]
            if isinstance(stream, Mapping)
        }
    except (KeyError, TypeError, ValueError):
        _fail("E_MEDIA_PROBE")
    if not math.isfinite(duration) or duration <= 0:
        _fail("E_MEDIA_PROBE")
    return duration, "video" in types, "audio" in types


def _namespace_bytes(path: Path) -> int:
    total = 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for name in dirs:
            if (directory_path / name).is_symlink():
                _fail("E_NAMESPACE_SYMLINK")
        for name in files:
            candidate = directory_path / name
            if candidate.is_symlink():
                _fail("E_NAMESPACE_SYMLINK")
            total += candidate.stat().st_size
    return total


def _capacity(control_root: Path, namespace: Path, incoming: int) -> None:
    usage = shutil.disk_usage(control_root)
    if (
        incoming <= 0
        or incoming > RAW_CAP_BYTES
        or usage.free - incoming < FREE_FLOOR_BYTES
        or _namespace_bytes(namespace) + incoming > NAMESPACE_CAP_BYTES
    ):
        _fail("E_CAPACITY")


def _validate_plan(value: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version")
        != "childlens-alignment-bridge-expansion-plan-v3.0.0"
        or value.get("protocol_sha256") != PROTOCOL_SHA256
        or value.get("selection_used_media_or_model_outcomes") is not False
        or not isinstance(value.get("items"), list)
        or len(value["items"]) != EXPECTED_COUNT
    ):
        _fail("E_PLAN")
    rows = [dict(row) for row in value["items"] if isinstance(row, Mapping)]
    if len(rows) != EXPECTED_COUNT:
        _fail("E_PLAN")
    participants: set[str] = set()
    for rank, row in enumerate(rows, 1):
        if (
            row.get("selection_rank") != rank
            or not isinstance(row.get("participant_key"), str)
            or row["participant_key"] in participants
            or not isinstance(row.get("source_locator"), str)
            or type(row.get("clip_source_start_ms")) is not int
            or type(row.get("clip_source_end_ms")) is not int
            or row["clip_source_end_ms"] <= row["clip_source_start_ms"]
            or type(row.get("duration_milliseconds")) is not int
            or not isinstance(row.get("sample_segments_source_ms"), list)
        ):
            _fail("E_PLAN")
        sample = sum(
            int(segment["end_ms"]) - int(segment["start_ms"])
            for segment in row["sample_segments_source_ms"]
        )
        if sample != 60_000:
            _fail("E_PLAN")
        participants.add(row["participant_key"])
    return rows


def _checkpoint_template(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "childlens-alignment-bridge-expansion-acquisition-v3.0.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_plan_sha256": RESTRICTED_PLAN_SHA256,
        "status": "IN_PROGRESS",
        "items": [
            {
                "selection_rank": row["selection_rank"],
                "object_key": row["object_key"],
                "source_exact_bytes": None,
                "source_sha256": None,
                "clip_relative_path": None,
                "clip_bytes": None,
                "clip_sha256": None,
                "clip_duration_seconds": None,
                "status": "PENDING",
            }
            for row in rows
        ],
    }


def _verify_completed(runtime: Path, item: Mapping[str, Any]) -> None:
    relative = item.get("clip_relative_path")
    sha256 = item.get("clip_sha256")
    size = item.get("clip_bytes")
    if (
        not isinstance(relative, str)
        or not relative.startswith(PRIVATE_CLIPS.as_posix() + "/")
        or ".." in Path(relative).parts
        or not isinstance(sha256, str)
        or not HEX64.fullmatch(sha256)
        or type(size) is not int
        or size <= 0
    ):
        _fail("E_CHECKPOINT")
    path = (runtime / relative).resolve()
    if (
        not _inside(path, runtime)
        or not _private_file(path)
        or path.stat().st_size != size
        or _sha256_file(path) != sha256
    ):
        _fail("E_CHECKPOINT")
    _, video, audio = _probe(path)
    if not video or not audio:
        _fail("E_CHECKPOINT")


def _public_guard(value: Any) -> None:
    encoded = _canonical(value).decode("utf-8")
    if PATH_TOKEN.search(encoded) or MEDIA_TOKEN.search(encoded):
        _fail("E_PUBLIC_PRIVACY")
    forbidden = {
        "participant_key",
        "object_key",
        "source_locator",
        "source_sha256",
        "clip_sha256",
        "clip_relative_path",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if forbidden.intersection(current):
                _fail("E_PUBLIC_PRIVACY")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def execute() -> Mapping[str, Any]:
    if (
        _sha256_file(CONFIG) != PROTOCOL_SHA256
        or _sha256_file(SELECTION_RECEIPT) != SELECTION_RECEIPT_SHA256
        or not FFMPEG.is_file()
        or not FFPROBE.is_file()
        or not KEYCHAIN_TOOL.is_file()
    ):
        _fail("E_PUBLIC_BINDING")
    runtime = _discover_runtime()
    control_root = runtime.parent.resolve()
    namespace = runtime / PRIVATE_RELATIVE
    if not _private_directory(namespace):
        _fail("E_NAMESPACE")
    plan_path = runtime / PRIVATE_PLAN
    if (
        not _private_file(plan_path)
        or _sha256_file(plan_path) != RESTRICTED_PLAN_SHA256
    ):
        _fail("E_PLAN_BINDING")
    rows = _validate_plan(_read_json(plan_path))
    clips = runtime / PRIVATE_CLIPS
    transient = runtime / PRIVATE_TRANSIENT
    for directory in (clips, transient):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        if not _private_directory(directory):
            _fail("E_NAMESPACE")
    checkpoint_path = runtime / PRIVATE_CHECKPOINT
    template = _checkpoint_template(rows)
    if checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path)
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("schema_version") != template["schema_version"]
            or checkpoint.get("protocol_sha256") != PROTOCOL_SHA256
            or checkpoint.get("restricted_plan_sha256") != RESTRICTED_PLAN_SHA256
            or not isinstance(checkpoint.get("items"), list)
            or len(checkpoint["items"]) != EXPECTED_COUNT
        ):
            _fail("E_CHECKPOINT")
        checkpoint = dict(checkpoint)
    else:
        checkpoint = template
    for expected, actual in zip(template["items"], checkpoint["items"]):
        if (
            actual.get("selection_rank") != expected["selection_rank"]
            or actual.get("object_key") != expected["object_key"]
            or actual.get("status") not in {"PENDING", "COMPLETE"}
        ):
            _fail("E_CHECKPOINT")
        if actual["status"] == "COMPLETE":
            _verify_completed(runtime, actual)
    transient_source = transient / "source.bin"
    transient_clip = transient / "derived.mp4"
    for path in (transient_source, transient_clip):
        with contextlib.suppress(OSError):
            path.unlink()
    token = _read_token()
    completed_all = False
    try:
        transfer_config = _read_json(runtime / "transfer_v1_2/config.json")
        if not isinstance(transfer_config, Mapping):
            _fail("E_TRANSFER_CONFIG")
        client = SeafileClient(transfer_config, token)
        for row, item in zip(rows, checkpoint["items"]):
            if item["status"] == "COMPLETE":
                continue
            exact_size = client.exact_size(row["source_locator"])
            _capacity(control_root, namespace, exact_size)
            source_digest = hashlib.sha256()
            transferred = 0
            response = client.open_download(row["source_locator"])
            try:
                descriptor = os.open(
                    transient_source,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    while True:
                        block = response.read(CHUNK_BYTES)
                        if not block:
                            break
                        transferred += len(block)
                        if transferred > exact_size or transferred > RAW_CAP_BYTES:
                            _fail("E_STREAM_CAP")
                        handle.write(block)
                        source_digest.update(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                response.close()
            if (
                transferred != exact_size
                or not _private_file(transient_source)
            ):
                _fail("E_TRANSFER_SIZE")
            source_duration, source_video, source_audio = _probe(transient_source)
            expected_duration = row["duration_milliseconds"] / 1000.0
            if (
                not source_video
                or not source_audio
                or abs(source_duration - expected_duration)
                > max(2.0, expected_duration * 0.02)
            ):
                _fail("E_SOURCE_MEDIA")
            clip_start = row["clip_source_start_ms"] / 1000.0
            clip_duration = (
                row["clip_source_end_ms"] - row["clip_source_start_ms"]
            ) / 1000.0
            try:
                result = subprocess.run(
                    [
                        str(FFMPEG),
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{clip_start:.3f}",
                        "-i",
                        str(transient_source),
                        "-t",
                        f"{clip_duration:.3f}",
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a:0",
                        "-vf",
                        (
                            "scale=640:640:force_original_aspect_ratio=decrease:"
                            "force_divisible_by=2,fps=15,format=yuv420p"
                        ),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "24",
                        "-c:a",
                        "aac",
                        "-ac",
                        "1",
                        "-b:a",
                        "96k",
                        "-movflags",
                        "+faststart",
                        "-y",
                        str(transient_clip),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=7200,
                )
            except (OSError, subprocess.SubprocessError):
                _fail("E_CLIP")
            if result.returncode != 0 or not _private_file(transient_clip):
                _fail("E_CLIP")
            actual_duration, clip_video, clip_audio = _probe(transient_clip)
            if (
                not clip_video
                or not clip_audio
                or abs(actual_duration - clip_duration)
                > max(1.0, clip_duration * 0.02)
            ):
                _fail("E_CLIP_MEDIA")
            clip_sha256 = _sha256_file(transient_clip)
            destination = clips / f"{clip_sha256}.mp4"
            clip_bytes = transient_clip.stat().st_size
            if destination.exists():
                if (
                    not _private_file(destination)
                    or destination.stat().st_size != clip_bytes
                    or _sha256_file(destination) != clip_sha256
                ):
                    _fail("E_CLIP_CONFLICT")
                transient_clip.unlink()
            else:
                os.replace(transient_clip, destination)
                os.chmod(destination, 0o600)
            transient_source.unlink()
            item.update(
                {
                    "source_exact_bytes": transferred,
                    "source_sha256": source_digest.hexdigest(),
                    "clip_relative_path": destination.relative_to(runtime).as_posix(),
                    "clip_bytes": clip_bytes,
                    "clip_sha256": clip_sha256,
                    "clip_duration_seconds": round(actual_duration, 3),
                    "status": "COMPLETE",
                }
            )
            checkpoint["status"] = (
                "COMPLETE"
                if all(state["status"] == "COMPLETE" for state in checkpoint["items"])
                else "IN_PROGRESS"
            )
            _write_atomic(checkpoint_path, checkpoint, private=True)
            if any(transient.iterdir()):
                _fail("E_TRANSIENT_NOT_EMPTY")
        completed_all = all(
            item["status"] == "COMPLETE" for item in checkpoint["items"]
        )
    finally:
        token = ""
        for path in (transient_source, transient_clip):
            with contextlib.suppress(OSError):
                path.unlink()
    if not completed_all:
        _fail("E_INCOMPLETE")
    _delete_token()
    total_source = sum(item["source_exact_bytes"] for item in checkpoint["items"])
    total_clips = sum(item["clip_bytes"] for item in checkpoint["items"])
    if total_clips > DERIVED_CAP_BYTES:
        _fail("E_DERIVED_CAP")
    receipt = {
        "schema_version": "childlens-alignment-bridge-expansion-acquisition-receipt-v3.0.0",
        "status": "TEN_PARTICIPANT_EXPANSION_ACQUIRED",
        "protocol_sha256": PROTOCOL_SHA256,
        "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
        "restricted_plan_sha256": RESTRICTED_PLAN_SHA256,
        "restricted_checkpoint_sha256": _sha256_file(checkpoint_path),
        "selected_recording_count": EXPECTED_COUNT,
        "completed_recording_count": EXPECTED_COUNT,
        "participant_distinct": True,
        "overlap_with_prior_30_participants": 0,
        "locked_outcomes_loaded_or_evaluated": 0,
        "source_transfer_bytes_rounded_up_gib": math.ceil(total_source / 1024**3),
        "retained_clip_bytes_rounded_up_gib": math.ceil(total_clips / 1024**3),
        "maximum_concurrent_full_source_recordings": 1,
        "transient_full_sources_removed": True,
        "all_retained_clips_have_video_and_audio": True,
        "exact_remote_metadata_used": True,
        "external_volume_used": False,
        "restricted_payload_exported": False,
        "measurement_outcome_opened": False,
        "retention_review_deadline": "2027-07-31",
    }
    _public_guard(receipt)
    _write_once(PUBLIC_RECEIPT, receipt, private=False)
    return receipt


def main() -> int:
    old_umask = os.umask(0o077)
    try:
        result = execute()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "state": result["status"],
                    "completed_recording_count": result["completed_recording_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    except AcquisitionError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
