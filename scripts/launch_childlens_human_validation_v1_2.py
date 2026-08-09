#!/usr/bin/env python3
"""Launch the local human workflow without path arguments or environment state.

The launcher discovers one already initialized private workflow, keeps the
restricted root out of argv and child-process environments, and gives the
browser a quarantine-confined working directory so its relative profile, cache,
and download arguments cannot escape the restricted store.
"""

from __future__ import annotations

import os
import secrets
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlencode

import childlens_human_validation_v1_2 as workflow


BROWSER_ENV = "CHILDLENS_V12_BROWSER_BINARY"
CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
)


def find_browser() -> Path:
    explicit = os.environ.get(BROWSER_ENV)
    candidates = (Path(explicit),) if explicit else CHROME_CANDIDATES
    for candidate in candidates:
        try:
            metadata = candidate.stat()
        except OSError:
            continue
        if candidate.is_absolute() and stat.S_ISREG(metadata.st_mode) and os.access(candidate, os.X_OK):
            return candidate
    raise workflow.WorkflowError("E_RESTRICTED_BROWSER_UNAVAILABLE")


def confined_browser_command(root: Path, browser: Path, url: str) -> list[str]:
    root = root.resolve(strict=True)
    try:
        resolved_browser = browser.resolve(strict=True)
    except OSError:
        raise workflow.WorkflowError("E_RESTRICTED_BROWSER_UNAVAILABLE")
    if workflow._is_relative_to(resolved_browser, root):
        raise workflow.WorkflowError("E_BROWSER_BINARY_INSIDE_QUARANTINE")
    workflow_dir = root / workflow.WORKFLOW_DIR
    profile = workflow_dir / "browser_profile"
    cache = workflow_dir / "browser_cache"
    downloads = workflow_dir / "browser_downloads"
    for directory in (profile, cache, downloads):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        if not workflow._is_relative_to(directory.resolve(strict=True), root):
            raise workflow.WorkflowError("E_BROWSER_STORE_OUTSIDE_QUARANTINE")
    default_profile = profile / "Default"
    default_profile.mkdir(mode=0o700, exist_ok=True)
    os.chmod(default_profile, 0o700)
    preferences = default_profile / "Preferences"
    value: dict[str, object] = {}
    if preferences.exists():
        try:
            import json

            loaded = json.loads(preferences.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                value = loaded
        except (OSError, ValueError):
            raise workflow.WorkflowError("E_BROWSER_PREFERENCES_INVALID")
    value["download"] = {
        "default_directory": str(downloads),
        "directory_upgrade": True,
        "prompt_for_download": False,
    }
    value["savefile"] = {"default_directory": str(downloads)}
    workflow._atomic_write(preferences, workflow._canonical(value) + b"\n", mode=0o600)
    return [
        str(resolved_browser),
        "--user-data-dir=browser_profile",
        "--disk-cache-dir=browser_cache",
        "--download-default-directory=browser_downloads",
        "--disk-cache-size=1",
        "--media-cache-size=1",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-domain-reliability",
        "--disable-breakpad",
        "--disable-features=OptimizationHints,MediaRouter,AutofillServerCommunication",
        "--no-pings",
        f"--app={url}",
    ]


def browser_working_directory(root: Path) -> Path:
    """Return the private cwd against which all browser stores are resolved."""

    directory = (root.resolve(strict=True) / workflow.WORKFLOW_DIR).resolve(strict=True)
    if not workflow._is_relative_to(directory, root.resolve(strict=True)):
        raise workflow.WorkflowError("E_BROWSER_STORE_OUTSIDE_QUARANTINE")
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise workflow.WorkflowError("E_BROWSER_STORE_PERMISSIONS")
    return directory


def restricted_runtime_environment(
    root: Path,
    nonce: str,
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment containing neither the root nor a root alias."""

    source_values = os.environ if source is None else source
    root_text = str(root.resolve(strict=True))
    environment = {
        key: value
        for key, value in source_values.items()
        if key != workflow.ROOT_ENV and root_text not in value
    }
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    environment["CHILDLENS_V12_UI_NONCE"] = nonce
    if any(root_text in value for value in environment.values()):
        raise workflow.WorkflowError("E_RUNTIME_ENVIRONMENT_DISCLOSURE")
    return environment


def streamlit_command() -> list[str]:
    app = Path(__file__).with_name("childlens_human_validation_app_v1_2.py")
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=true",
        "--server.enableXsrfProtection=true",
        "--server.enableCORS=true",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
        "--browser.gatherUsageStats=false",
        "--logger.level=error",
    ]


def wait_for_local_server(server: subprocess.Popen[bytes], timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    health = "http://127.0.0.1:8501/_stcore/health"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise workflow.WorkflowError("E_LOCAL_SERVER_FAILED")
        try:
            with urllib.request.urlopen(health, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise workflow.WorkflowError("E_LOCAL_SERVER_TIMEOUT")


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied:
        print("E_ARGUMENTS", file=sys.stderr)
        return 2
    try:
        root = workflow.discover_runtime_root()
        browser = find_browser()
        command = confined_browser_command(root, browser, "http://127.0.0.1:8501/")
        browser_cwd = browser_working_directory(root)
    except workflow.WorkflowError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except OSError:
        print("E_LOCAL_RUNTIME", file=sys.stderr)
        return 2
    nonce = secrets.token_urlsafe(32)
    url = "http://127.0.0.1:8501/?" + urlencode({"launch_token": nonce})
    old_umask = os.umask(0o077)
    server: subprocess.Popen[bytes] | None = None
    try:
        environment = restricted_runtime_environment(root, nonce)
        server = subprocess.Popen(
            streamlit_command(),
            env=environment,
            cwd=workflow.REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_local_server(server)
        command[-1] = f"--app={url}"
        browser_process = subprocess.Popen(
            command,
            env=environment,
            cwd=browser_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return browser_process.wait()
    except workflow.WorkflowError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError):
        print("E_LOCAL_RUNTIME", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        os.umask(old_umask)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
