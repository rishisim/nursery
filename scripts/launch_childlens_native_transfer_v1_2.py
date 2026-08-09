#!/usr/bin/env python3
"""Credential-safe launcher for the frozen ChildLens v1.2 native transfer.

This command takes no path or credential arguments.  It discovers one matching
private plan/config bundle in a ChildLens-labelled hidden quarantine below the
repository parent, reads a fixed temporary Keychain item into memory, runs the
prepare and acquire phases, and deletes the Keychain item only after a verified
complete result.  Terminal output is limited to fixed status/error values.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SEARCH_ROOT = REPO_ROOT.parent
CONTROLLER_PATH = SCRIPT_PATH.with_name("childlens_native_transfer_v1_2.py")

if "childlens_native_transfer_v1_2" in sys.modules:
    controller = sys.modules["childlens_native_transfer_v1_2"]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "childlens_native_transfer_v1_2", CONTROLLER_PATH
    )
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError("E_CONTROLLER_IMPORT")
    controller = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = controller
    _SPEC.loader.exec_module(controller)


VERSION = "childlens-native-transfer-keychain-launcher-v1.2.0"
KEYCHAIN_SERVICE = "ChildLens-v1.2-Keeper-Repo-Token"
KEYCHAIN_ACCOUNT = "childlens-v1.2-read-only"
KEYCHAIN_TOOL = Path("/usr/bin/security")
TOKEN_ENVIRONMENT_VARIABLE = "CHILDLENS_SEAFILE_TOKEN"
FROZEN_CONSERVATIVE_BOUND_BYTES = 14_970_186_240
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~+/=-]{16,1024}$")


class LaunchError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise LaunchError(code)


@dataclass(frozen=True, repr=False)
class Bundle:
    quarantine_root: Path
    plan_path: Path
    config_path: Path
    restricted_receipt_path: Path
    aggregate_output_path: Path
    plan: dict[str, Any]
    config: dict[str, Any]


class Keychain(Protocol):
    def read(self) -> str: ...

    def delete(self) -> None: ...


class MacOSKeychain:
    def read(self) -> str:
        if not KEYCHAIN_TOOL.is_file():
            _fail("E_KEYCHAIN_TOOL")
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

    def delete(self) -> None:
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


def _private_regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o077 == 0


def _private_directory_chain(path: Path, root: Path) -> bool:
    current = path
    while True:
        try:
            info = current.lstat()
        except OSError:
            return False
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return False
        if stat.S_IMODE(info.st_mode) & 0o077:
            return False
        if current == root:
            return True
        if root not in current.parents:
            return False
        current = current.parent


def _candidate_document(path: Path) -> dict[str, Any] | None:
    if not _private_regular_file(path):
        return None
    try:
        if path.stat().st_size > MAX_CANDIDATE_BYTES:
            return None
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_BUNDLE_INVALID")
    return hashlib.sha256(encoded).hexdigest()


def _hidden_roots(search_root: Path, repository_root: Path) -> list[Path]:
    try:
        search = search_root.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
    except OSError:
        _fail("E_DISCOVERY_ROOT")
    if repository.parent != search:
        _fail("E_DISCOVERY_ROOT")
    roots: list[Path] = []
    try:
        for candidate in search.iterdir():
            name = candidate.name.lower()
            if (
                candidate == repository
                or not name.startswith(".")
                or "childlens" not in name
                or candidate.is_symlink()
                or not candidate.is_dir()
            ):
                continue
            if stat.S_IMODE(candidate.stat().st_mode) & 0o077:
                continue
            roots.append(candidate.resolve(strict=True))
    except OSError:
        _fail("E_DISCOVERY_ROOT")
    return sorted(roots, key=lambda path: path.name)


def discover_bundle(
    *,
    search_root: Path = SEARCH_ROOT,
    repository_root: Path = REPO_ROOT,
) -> Bundle:
    plan_candidates: list[tuple[Path, Path, dict[str, Any]]] = []
    config_candidates: list[tuple[Path, Path, dict[str, Any]]] = []
    for root in _hidden_roots(search_root, repository_root):
        try:
            walker = os.walk(root, followlinks=False)
            for directory, directories, files in walker:
                directory_path = Path(directory)
                if not _private_directory_chain(directory_path, root):
                    directories[:] = []
                    continue
                directories[:] = [
                    name
                    for name in directories
                    if not (directory_path / name).is_symlink()
                ]
                for name in files:
                    lowered = name.lower()
                    if not lowered.endswith(".json"):
                        continue
                    try:
                        relative_hint = "/".join(
                            part.lower() for part in (directory_path / name).relative_to(root).parts
                        )
                    except ValueError:
                        continue
                    transfer_context = "transfer" in relative_hint
                    is_plan_name = "plan" in lowered and (
                        "download" in relative_hint or transfer_context
                    )
                    is_config_name = "config" in lowered and transfer_context
                    if not (is_plan_name or is_config_name):
                        continue
                    path = directory_path / name
                    document = _candidate_document(path)
                    if document is None:
                        continue
                    schema = document.get("schema_version")
                    if is_plan_name and schema == controller.PLAN_SCHEMA:
                        plan_candidates.append((root, path, document))
                    if is_config_name and schema == controller.CONFIG_SCHEMA:
                        config_candidates.append((root, path, document))
        except OSError:
            _fail("E_DISCOVERY_SCAN")

    valid: list[Bundle] = []
    saw_schema_candidate = bool(plan_candidates or config_candidates)
    for plan_root, plan_path, raw_plan in plan_candidates:
        try:
            plan = controller.validate_plan(raw_plan)
        except controller.TransferError:
            continue
        plan_digest = _canonical_digest(plan)
        for config_root, config_path, raw_config in config_candidates:
            if config_root != plan_root:
                continue
            if raw_config.get("expected_download_plan_sha256") != plan_digest:
                continue
            try:
                config = controller.validate_config(raw_config, plan)
            except controller.TransferError:
                continue
            if (
                config["authentication"]["environment_variable"]
                != TOKEN_ENVIRONMENT_VARIABLE
                or config["authentication"]["scheme"] != "Bearer"
                or config["admission"]["mode"] != "CONSERVATIVE_ROUNDED"
                or not config["admission"]["amendment_frozen_before_media_open"]
                or sum(
                    int(row["upper_bound_bytes"])
                    for row in config["admission"]["conservative_bounds"]
                )
                != FROZEN_CONSERVATIVE_BOUND_BYTES
            ):
                continue
            if urllib_hostname(config["api_base_url"]) != "keeper.mpdl.mpg.de":
                continue
            if any(
                urllib_hostname(origin) != "keeper.mpdl.mpg.de"
                for origin in config["allowed_download_origins"]
            ):
                continue
            try:
                common_root = Path(
                    os.path.commonpath([plan_path.parent, config_path.parent])
                ).resolve(strict=True)
            except (OSError, ValueError):
                continue
            if (
                common_root != plan_root
                and plan_root not in common_root.parents
            ):
                continue
            if not _private_directory_chain(common_root, plan_root):
                continue
            valid.append(
                Bundle(
                    quarantine_root=common_root,
                    plan_path=plan_path,
                    config_path=config_path,
                    restricted_receipt_path=config_path.parent
                    / "native_transfer_receipt_v1_2.json",
                    aggregate_output_path=repository_root
                    / "output"
                    / "childlens_feasibility_v1_2"
                    / "native_transfer_aggregate.json",
                    plan=plan,
                    config=config,
                )
            )
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        _fail("E_BUNDLE_AMBIGUOUS")
    _fail("E_BUNDLE_INVALID" if saw_schema_candidate else "E_BUNDLE_NOT_FOUND")


def urllib_hostname(url: str) -> str | None:
    # Controller validation has already enforced a strict HTTPS origin.  This
    # local import avoids broadening the launcher's own accepted URL surface.
    import urllib.parse

    try:
        return urllib.parse.urlsplit(url).hostname
    except ValueError:
        return None


def _resume_state(bundle: Bundle) -> str | None:
    path = bundle.restricted_receipt_path
    if not path.exists():
        return None
    try:
        value = controller._read_json(path, "E_RECEIPT_READ")
        receipt = controller._validate_receipt(value, bundle.plan, bundle.config)
    except controller.TransferError:
        _fail("E_EXISTING_RECEIPT")
    return receipt["status"]


def execute(
    *,
    search_root: Path = SEARCH_ROOT,
    repository_root: Path = REPO_ROOT,
    keychain: Keychain | None = None,
    controller_run=None,
) -> dict[str, str]:
    bundle = discover_bundle(search_root=search_root, repository_root=repository_root)
    resume_state = _resume_state(bundle)
    output_parent = bundle.aggregate_output_path.parent
    try:
        output_parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    except OSError:
        _fail("E_AGGREGATE_OUTPUT")
    secrets = keychain or MacOSKeychain()
    token = secrets.read()
    if not TOKEN_PATTERN.fullmatch(token):
        _fail("E_KEYCHAIN_TOKEN")
    runner = controller_run or controller.run
    previous = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
    completed = False
    try:
        os.environ[TOKEN_ENVIRONMENT_VARIABLE] = token
        if resume_state is None:
            prepared = runner(
                phase="prepare",
                quarantine_root=bundle.quarantine_root,
                plan_path=bundle.plan_path,
                config_path=bundle.config_path,
                restricted_receipt_path=bundle.restricted_receipt_path,
                aggregate_output_path=bundle.aggregate_output_path,
            )
            if prepared.get("status") != "PREPARED":
                _fail("E_PREPARE_STATE")
        acquired = runner(
            phase="acquire",
            quarantine_root=bundle.quarantine_root,
            plan_path=bundle.plan_path,
            config_path=bundle.config_path,
            restricted_receipt_path=bundle.restricted_receipt_path,
            aggregate_output_path=bundle.aggregate_output_path,
        )
        transfer = acquired.get("transfer")
        if (
            acquired.get("status") != "COMPLETE"
            or not isinstance(transfer, dict)
            or transfer.get("complete") is not True
            or transfer.get("completed_count") != 15
        ):
            _fail("E_ACQUIRE_STATE")
        completed = True
    except controller.TransferError as exc:
        del exc
        _fail("E_CONTROLLER")
    finally:
        if previous is None:
            os.environ.pop(TOKEN_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[TOKEN_ENVIRONMENT_VARIABLE] = previous
        token = ""
    if not completed:
        _fail("E_ACQUIRE_STATE")
    secrets.delete()
    return {"status": "COMPLETE", "launcher_version": VERSION}


def main(argv: list[str] | None = None) -> int:
    old_umask = os.umask(0o077)
    try:
        arguments = sys.argv[1:] if argv is None else argv
        if arguments:
            _fail("E_ARGUMENTS")
        execute()
    except LaunchError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)
    print(json.dumps({"status": "ok", "state": "COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
