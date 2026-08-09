from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import PROTOCOL_ID, SCHEMA_VERSION


PURPOSES = (
    "construction_micro",
    "construction_qualification",
    "construction_mechanism",
    "excluded_rehearsal",
    "development",
)
ROLES = ("corpus", "model", "inference")
OPERATIONS = (
    "generate",
    "condition",
    "fit",
    "predict",
    "score",
    "control",
    "inference",
    "recompute",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lstat_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def require_lstat_absent(path: str | Path) -> None:
    value = Path(path)
    if _lstat_exists(value):
        raise FileExistsError(f"path must be pristine and lstat-absent: {value}")


def atomic_rename_directory_noreplace(
    source: str | Path,
    destination: str | Path,
) -> None:
    """Atomically publish one real directory without replacing any destination."""
    source_path = Path(source).absolute()
    destination_path = Path(destination).absolute()
    source_metadata = source_path.lstat()
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(
        source_metadata.st_mode
    ):
        raise RuntimeError("atomic directory source must be a real directory")
    _reject_symlink_ancestors(source_path.parent)
    _reject_symlink_ancestors(destination_path.parent)
    destination_parent = destination_path.parent
    parent_metadata = destination_parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise RuntimeError("atomic directory destination parent must be real")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_path)
    destination_bytes = os.fsencode(destination_path)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise RuntimeError("kernel no-replace directory rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"atomic directory destination already exists: {destination_path}"
            )
        raise OSError(error, os.strerror(error), str(destination_path))


def _reject_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"symlink ancestor forbidden: {current}")
        if current.parent == current:
            break
        current = current.parent


def _real_directory_root(root: str | Path) -> Path:
    supplied = Path(root)
    metadata = supplied.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"real directory root required: {supplied}")
    return supplied.resolve()


def atomic_write_bytes(
    path: str | Path,
    value: bytes,
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    _reject_symlink_ancestors(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination.parent)
    if not overwrite:
        require_lstat_absent(destination)
    temporary = destination.with_name(
        f".{destination.name}.atomic-{os.getpid()}-{hashlib.sha256(value).hexdigest()[:12]}"
    )
    require_lstat_absent(temporary)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if overwrite:
        os.replace(temporary, destination)
    else:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise
        finally:
            temporary.unlink(missing_ok=True)


def write_json(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> None:
    atomic_write_bytes(path, canonical_bytes(value), overwrite=overwrite)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_relative(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError(f"unsafe manifest path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe manifest path: {value!r}")
    normalized = pure.as_posix()
    if normalized != value:
        raise ValueError(f"non-canonical manifest path: {value!r}")
    return normalized


def ensure_confined_regular_file(root: str | Path, relative: str) -> Path:
    base = _real_directory_root(root)
    safe = _safe_relative(relative)
    candidate = base / safe
    current = base
    for part in PurePosixPath(safe).parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden: {current}")
    resolved_parent = candidate.parent.resolve(strict=True)
    if resolved_parent != base and base not in resolved_parent.parents:
        raise RuntimeError(f"path escapes root: {relative}")
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"regular file required: {candidate}")
    return candidate


def _file_row(root: Path, relative: str) -> dict[str, Any]:
    path = ensure_confined_regular_file(root, relative)
    metadata = path.lstat()
    return {
        "path": relative,
        "bytes": int(metadata.st_size),
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "sha256": sha256_file(path),
    }


def manifest_for_paths(root: str | Path, relatives: Iterable[str]) -> dict[str, Any]:
    base = _real_directory_root(root)
    values = list(map(str, relatives))
    if len(values) != len(set(values)):
        raise ValueError("duplicate manifest paths")
    paths = sorted(_safe_relative(value) for value in values)
    rows = [_file_row(base, relative) for relative in paths]
    return {
        "schema_version": "nursery-exact-file-manifest-v1",
        "file_count": len(rows),
        "files": rows,
        "digest_scheme": "canonical-json-of-sorted-file-rows",
        "digest": canonical_digest(rows),
    }


def tree_file_paths(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    base = _real_directory_root(root)
    excluded = set(map(str, exclude))
    output: list[str] = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden in manifested tree: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"special file forbidden in manifested tree: {relative}")
        if relative not in excluded:
            output.append(relative)
    return output


def tree_directory_paths(root: str | Path) -> list[str]:
    base = _real_directory_root(root)
    output: list[str] = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden in manifested tree: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            output.append(relative)
        elif not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"special file forbidden in manifested tree: {relative}")
    return output


def manifest_for_tree(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    return manifest_for_paths(root, tree_file_paths(root, exclude=exclude))


def verify_exact_manifest(
    root: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_filename: str | None = None,
    allowed_extra_files: Iterable[str] = (),
) -> dict[str, Any]:
    base = _real_directory_root(root)
    problems: list[str] = []
    if set(manifest) != {
        "schema_version",
        "file_count",
        "files",
        "digest_scheme",
        "digest",
    }:
        problems.append("top_level_schema")
    if manifest.get("schema_version") != "nursery-exact-file-manifest-v1":
        problems.append("schema_version")
    if manifest.get("digest_scheme") != "canonical-json-of-sorted-file-rows":
        problems.append("digest_scheme")
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list):
        return {"status": "FAIL", "problems": ["files_not_list"]}
    paths: list[str] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "mode", "sha256"}:
            problems.append(f"row_schema:{index}")
            continue
        try:
            paths.append(_safe_relative(str(row["path"])))
        except ValueError:
            problems.append(f"unsafe_path:{index}")
    if paths != sorted(paths):
        problems.append("rows_not_sorted")
    if len(paths) != len(set(paths)):
        problems.append("duplicate_paths")
    if int(manifest.get("file_count", -1)) != len(raw_rows):
        problems.append("file_count")
    if manifest.get("digest") != canonical_digest(raw_rows):
        problems.append("manifest_digest")
    observed_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping) or "path" not in row:
            continue
        try:
            observed = _file_row(base, _safe_relative(str(row["path"])))
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            problems.append(f"unreadable:{row.get('path')}:{type(error).__name__}")
            continue
        observed_rows.append(observed)
        if observed != dict(row):
            problems.append(f"row_mismatch:{row['path']}")
    excluded = set(map(str, allowed_extra_files))
    if manifest_filename is not None:
        excluded.add(manifest_filename)
    try:
        actual = set(tree_file_paths(base, exclude=excluded))
    except (RuntimeError, OSError) as error:
        problems.append(f"tree:{type(error).__name__}:{error}")
        actual = set()
    listed = set(paths)
    missing = sorted(listed - actual)
    unexpected = sorted(actual - listed)
    if missing:
        problems.append("missing_files")
    if unexpected:
        problems.append("unexpected_files")
    expected_directories = {
        PurePosixPath(path).parent.as_posix()
        for path in [*paths, *excluded]
        if PurePosixPath(path).parent.as_posix() != "."
    }
    expected_directories |= {
        parent.as_posix()
        for path in [*paths, *excluded]
        for parent in PurePosixPath(path).parents
        if parent.as_posix() not in {".", ""}
    }
    try:
        actual_directories = set(tree_directory_paths(base))
    except (RuntimeError, OSError) as error:
        problems.append(f"directories:{type(error).__name__}:{error}")
        actual_directories = set()
    unexpected_directories = sorted(actual_directories - expected_directories)
    if unexpected_directories:
        problems.append("unexpected_directories")
    return {
        "status": "PASS" if not problems else "FAIL",
        "file_count": len(raw_rows),
        "unique_path_count": len(set(paths)),
        "observed_row_count": len(observed_rows),
        "missing": missing,
        "unexpected": unexpected,
        "unexpected_directories": unexpected_directories,
        "problems": problems,
        "digest": manifest.get("digest"),
    }


def reject_forbidden_fields(
    value: Any,
    forbidden: set[str] | frozenset[str],
    *,
    path: str = "visible",
) -> None:
    if isinstance(value, Mapping):
        overlap = set(value) & set(forbidden)
        if overlap:
            raise ValueError(f"forbidden fields at {path}: {sorted(overlap)}")
        for key, child in value.items():
            reject_forbidden_fields(child, forbidden, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_forbidden_fields(child, forbidden, path=f"{path}[{index}]")


def _flatten_registry(registry: Mapping[str, Sequence[int]]) -> set[int]:
    return {int(value) for role in ROLES for value in registry[role]}


def _in_ranges(value: int, ranges: Sequence[Sequence[int]]) -> bool:
    return any(int(low) <= int(value) <= int(high) for low, high in ranges)


def load_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if value.get("protocol", {}).get("id") != PROTOCOL_ID:
        raise ValueError("wrong corrective protocol id")
    if value["protocol"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong corrective schema version")
    if value["protocol"].get("status") not in {"pre_freeze", "frozen"}:
        raise ValueError("status must be pre_freeze or frozen")
    protocol_false_flags = (
        "infant_learning_claim_authorized",
        "ecological_validity_claim_authorized",
        "development_execution_authorized",
        "confirmation_authorized",
        "prior_outcomes_authorized",
    )
    if any(value["protocol"].get(name) is not False for name in protocol_false_flags):
        raise ValueError("protocol authorization flags must remain false")
    design = value["design"]
    primitives = int(design["primitive_concepts"])
    manners = int(design["manner_concepts"])
    if (
        int(design["action_compositions"]) != primitives * manners
        or int(design["held_out_compositions_per_corpus"]) != primitives
        or list(design["primary_comparators"]) != ["absent", "shuffled"]
        or design["exact_window_comparator"] != "exact_window"
        or design["evaluation"].get("side_modality_withheld") is not True
    ):
        raise ValueError("frozen design-count/comparator contract mismatch")
    analysis = value["analysis"]
    if (
        analysis.get("comparator_definition")
        != "per_corpus_max_of_absent_and_shuffled"
        or analysis.get("all_identical_effects_fail") is not True
        or analysis.get("independent_unit") != "corpus_seed"
    ):
        raise ValueError("frozen analysis contract mismatch")
    parallel = value["parallel"]
    if any(
        parallel.get(name) is not True
        for name in (
            "one_inflight_model_per_corpus",
            "worker_isolated_shards",
            "worker_isolated_tmp",
            "atomic_final_publish",
            "require_jobs_1_jobs_n_byte_identity",
        )
    ):
        raise ValueError("frozen parallel safety flags must be true")
    firewall = value["firewalls"]
    if (
        firewall.get("confirmation_command_exists") is not False
        or firewall.get("confirmation_execution_supported") is not False
        or any(
            firewall.get(name) is not True
            for name in (
                "development_output_must_be_pristine",
                "authorization_consumption_required",
                "authorization_replay_forbidden",
                "exact_manifest_file_set_required",
                "symlinks_forbidden_in_package_snapshot_shards_and_outputs",
            )
        )
    ):
        raise ValueError("frozen firewall contract mismatch")
    if value["environment"].get("bytecode_writes_disabled") is not True:
        raise ValueError("bytecode writes must be disabled")
    preservation_hash_fields = (
        "preservation_baseline_sha256",
        "preservation_stat_sha256",
        "preservation_symlinks_sha256",
        "preservation_audit_manifest_sha256",
    )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value["paths"].get(name, "")))
        for name in preservation_hash_fields
    ):
        raise ValueError("preservation anchors must be lowercase SHA-256 digests")
    qualification = value["qualification_gates"]
    if (
        qualification.get("exact_present_null_effect_vector_equality_forbidden")
        is not True
        or float(qualification["presence_null_effect_equality_tolerance"]) <= 0.0
    ):
        raise ValueError("present/null dependence contract mismatch")
    registries = {
        name: {
            role: list(map(int, value["registries"][name][role]))
            for role in ROLES
        }
        for name in PURPOSES + ("confirmation_reserve",)
    }
    for name, registry in registries.items():
        for role, identifiers in registry.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate identifiers within {name}/{role}")
        flattened = [value for role in ROLES for value in registry[role]]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"identifier reused across roles within {name}")
    sets = {name: _flatten_registry(registry) for name, registry in registries.items()}
    names = list(sets)
    overlaps = {
        f"{left}|{right}": sorted(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if sets[left] & sets[right]
    }
    if overlaps:
        raise ValueError(f"registry overlap: {overlaps}")
    family_low, family_high = map(int, value["registries"]["namespace_family"])
    all_active = set().union(*sets.values())
    if any(not family_low <= seed <= family_high for seed in all_active):
        raise ValueError("active identifier outside corrective namespace")
    quarantine = value["registries"]["permanently_quarantined_ranges"]
    if any(_in_ranges(seed, quarantine) for seed in all_active):
        raise ValueError("quarantined identifier allocated")
    if len(registries["development"]["corpus"]) != int(
        value["analysis"]["development_corpus_count"]
    ):
        raise ValueError("development corpus count mismatch")
    if len(registries["development"]["model"]) != int(
        value["analysis"]["paired_model_replicates"]
    ):
        raise ValueError("development model count mismatch")
    if len(registries["construction_mechanism"]["model"]) != int(
        value["qualification_gates"]["disagreement_case_count"]
    ):
        raise ValueError("construction mechanism model count mismatch")
    if len(registries["confirmation_reserve"]["corpus"]) != int(
        value["analysis"]["development_corpus_count"]
    ):
        raise ValueError("confirmation corpus count mismatch")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.parents[1]
    )
    prior = root / str(value["registries"]["canonical_prior_registry"])
    if sha256_file(prior) != str(
        value["registries"]["canonical_prior_registry_sha256"]
    ):
        raise ValueError("canonical prior registry hash mismatch")
    prior_value = read_json(prior)
    prior_identifiers: set[int] = set()

    def collect_identifiers(child: Any) -> None:
        if isinstance(child, Mapping):
            for grandchild in child.values():
                collect_identifiers(grandchild)
        elif isinstance(child, list):
            for grandchild in child:
                if isinstance(grandchild, int) and not isinstance(grandchild, bool):
                    prior_identifiers.add(int(grandchild))
                else:
                    collect_identifiers(grandchild)

    collect_identifiers(prior_value.get("registries", {}))
    if all_active & prior_identifiers:
        raise ValueError("corrective identifiers overlap canonical prior registry")
    value["prior_identifier_count"] = len(prior_identifiers)
    value["prior_identifier_digest"] = canonical_digest(sorted(prior_identifiers))
    value["resolved_registries"] = registries
    value["repository_root"] = str(root)
    return value


def require_frozen(config: Mapping[str, Any]) -> None:
    if config["protocol"]["status"] != "frozen":
        raise RuntimeError("corrective package is not frozen")


_CAPABILITY_FACTORY_SEAL = object()
_PREFLIGHT_PROOF_FACTORY_SEAL = object()


class DevelopmentPreflightProof:
    __slots__ = (
        "authorization_digest",
        "authorization_path",
        "authorization_sha256",
        "snapshot_root",
        "snapshot_manifest_sha256",
        "parsed_contract_digest",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "claim_path",
        "issuance_receipt_path",
        "required_jobs",
        "argv_digest",
        "environment_digest",
        "_identity_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("development preflight proofs are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        authorization_digest: str,
        authorization_path: str | Path,
        authorization_sha256: str,
        snapshot_root: str | Path,
        snapshot_manifest_sha256: str,
        parsed_contract_digest: str,
        resolved_output_root: str | Path,
        resolved_staging_root: str | Path,
        resolved_inner_staging_root: str | Path,
        claim_path: str | Path,
        issuance_receipt_path: str | Path,
        required_jobs: int,
        argv_digest: str,
        environment_digest: str,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _PREFLIGHT_PROOF_FACTORY_SEAL:
            raise PermissionError("preflight proof requires strict verified issuance")
        values = {
            "authorization_digest": str(authorization_digest),
            "authorization_path": str(Path(authorization_path).resolve()),
            "authorization_sha256": str(authorization_sha256),
            "snapshot_root": str(Path(snapshot_root).resolve()),
            "snapshot_manifest_sha256": str(snapshot_manifest_sha256),
            "parsed_contract_digest": str(parsed_contract_digest),
            "resolved_output_root": str(Path(resolved_output_root).resolve()),
            "resolved_staging_root": str(Path(resolved_staging_root).resolve()),
            "resolved_inner_staging_root": str(
                Path(resolved_inner_staging_root).resolve()
            ),
            "claim_path": str(Path(claim_path).resolve()),
            "issuance_receipt_path": str(Path(issuance_receipt_path).resolve()),
            "required_jobs": int(required_jobs),
            "argv_digest": str(argv_digest),
            "environment_digest": str(environment_digest),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_identity_digest", canonical_digest(values))
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_frozen", True)


def _issue_development_preflight_proof(**values: Any) -> DevelopmentPreflightProof:
    return DevelopmentPreflightProof(
        **values,
        _seal=_PREFLIGHT_PROOF_FACTORY_SEAL,
    )


def _verify_development_preflight_proof(
    proof: DevelopmentPreflightProof,
) -> None:
    if (
        not isinstance(proof, DevelopmentPreflightProof)
        or proof._seal is not _PREFLIGHT_PROOF_FACTORY_SEAL
        or proof._frozen is not True
    ):
        raise PermissionError("development capability requires a strict preflight proof")
    values = {
        name: getattr(proof, name)
        for name in DevelopmentPreflightProof.__slots__
        if not name.startswith("_")
    }
    if canonical_digest(values) != proof._identity_digest:
        raise PermissionError("development preflight proof identity changed")
    if sha256_file(proof.authorization_path) != proof.authorization_sha256:
        raise PermissionError("preflight-bound authorization changed")
    snapshot_manifest = Path(proof.snapshot_root) / "snapshot_manifest.json"
    if sha256_file(snapshot_manifest) != proof.snapshot_manifest_sha256:
        raise PermissionError("preflight-bound snapshot manifest changed")


class AuthorizationCapability:
    __slots__ = (
        "purpose",
        "scope",
        "unit_id",
        "authorization_digest",
        "parsed_contract_digest",
        "claim_path",
        "claim_sha256",
        "issuance_receipt_path",
        "issuance_receipt_sha256",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "required_jobs",
        "_identity_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("authorization capabilities are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        purpose: str,
        scope: str,
        unit_id: str | None,
        authorization_digest: str,
        parsed_contract_digest: str,
        claim_path: str,
        claim_sha256: str,
        issuance_receipt_path: str,
        issuance_receipt_sha256: str,
        resolved_output_root: str,
        resolved_staging_root: str,
        resolved_inner_staging_root: str,
        required_jobs: int,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _CAPABILITY_FACTORY_SEAL:
            raise PermissionError(
                "development capability can only be issued from a consumed claim"
            )
        self.purpose = str(purpose)
        self.scope = str(scope)
        self.unit_id = None if unit_id is None else str(unit_id)
        self.authorization_digest = str(authorization_digest)
        self.parsed_contract_digest = str(parsed_contract_digest)
        self.claim_path = str(claim_path)
        self.claim_sha256 = str(claim_sha256)
        self.issuance_receipt_path = str(issuance_receipt_path)
        self.issuance_receipt_sha256 = str(issuance_receipt_sha256)
        self.resolved_output_root = str(resolved_output_root)
        self.resolved_staging_root = str(resolved_staging_root)
        self.resolved_inner_staging_root = str(resolved_inner_staging_root)
        self.required_jobs = int(required_jobs)
        identity = {
            name: getattr(self, name)
            for name in AuthorizationCapability.__slots__
            if not name.startswith("_")
        }
        self._identity_digest = canonical_digest(identity)
        self._seal = _seal
        self._frozen = True


def issue_development_capability(
    *,
    authorization_digest: str,
    parsed_contract_digest: str,
    claim_path: str | Path,
    claim_sha256: str,
    resolved_output_root: str | Path,
    resolved_staging_root: str | Path,
    resolved_inner_staging_root: str | Path,
    issuance_receipt_path: str | Path,
    required_jobs: int,
    preflight_proof: DevelopmentPreflightProof,
) -> AuthorizationCapability:
    _verify_development_preflight_proof(preflight_proof)
    path = Path(claim_path).resolve()
    if sha256_file(path) != str(claim_sha256):
        raise PermissionError("authorization-consumption claim digest mismatch")
    claim = read_json(path)
    expected_fields = {
        "schema_version",
        "status",
        "authorization_digest",
        "resolved_authorization_path",
        "authorization_sha256",
        "resolved_snapshot_root",
        "parsed_contract_digest",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_capability_receipt_path",
        "required_jobs",
        "non_replayable",
        "created_before_any_guarded_development_operation",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    if set(claim) != expected_fields:
        raise PermissionError("authorization-consumption claim schema mismatch")
    if (
        claim.get("schema_version")
        != "nursery-corrective-authorization-consumption-v1"
        or claim.get("status") != "ATTEMPT_CONSUMED"
        or claim.get("authorization_digest") != str(authorization_digest)
        or claim.get("parsed_contract_digest") != str(parsed_contract_digest)
        or Path(str(claim.get("resolved_output_root"))).resolve()
        != Path(resolved_output_root).resolve()
        or Path(str(claim.get("resolved_staging_root"))).resolve()
        != Path(resolved_staging_root).resolve()
        or Path(str(claim.get("resolved_inner_staging_root"))).resolve()
        != Path(resolved_inner_staging_root).resolve()
        or Path(str(claim.get("resolved_capability_receipt_path"))).resolve()
        != Path(issuance_receipt_path).resolve()
        or int(claim.get("required_jobs", -1)) != int(required_jobs)
        or claim.get("non_replayable") is not True
        or claim.get("created_before_any_guarded_development_operation") is not True
        or int(claim.get("development_outcome_count", -1)) != 0
        or int(claim.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("authorization-consumption claim content mismatch")
    if (
        preflight_proof.authorization_digest != str(authorization_digest)
        or preflight_proof.parsed_contract_digest != str(parsed_contract_digest)
        or Path(preflight_proof.claim_path) != path
        or Path(preflight_proof.issuance_receipt_path)
        != Path(issuance_receipt_path).resolve()
        or Path(preflight_proof.resolved_output_root)
        != Path(resolved_output_root).resolve()
        or Path(preflight_proof.resolved_staging_root)
        != Path(resolved_staging_root).resolve()
        or Path(preflight_proof.resolved_inner_staging_root)
        != Path(resolved_inner_staging_root).resolve()
        or preflight_proof.required_jobs != int(required_jobs)
    ):
        raise PermissionError("capability request differs from strict preflight proof")
    authorization_path = Path(str(claim["resolved_authorization_path"])).resolve()
    snapshot = Path(str(claim["resolved_snapshot_root"])).resolve()
    authorization = read_json(authorization_path)
    authorization_payload = {
        key: value for key, value in authorization.items() if key != "authorization_digest"
    }
    if (
        sha256_file(authorization_path) != str(claim["authorization_sha256"])
        or authorization.get("authorization_digest") != canonical_digest(authorization_payload)
        or authorization.get("authorization_digest") != str(authorization_digest)
        or authorization.get("status") != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY"
        or authorization.get("development_authorized") is not True
        or authorization.get("confirmation_authorized") is not False
        or authorization_path
        != Path(str(authorization.get("resolved_authorization_path"))).resolve()
        or path != Path(str(authorization.get("resolved_claim_path"))).resolve()
        or Path(str(authorization.get("resolved_snapshot_root"))).resolve() != snapshot
        or Path(str(authorization.get("resolved_output_root"))).resolve()
        != Path(resolved_output_root).resolve()
        or Path(str(authorization.get("resolved_staging_root"))).resolve()
        != Path(resolved_staging_root).resolve()
        or Path(str(authorization.get("resolved_inner_staging_root"))).resolve()
        != Path(resolved_inner_staging_root).resolve()
        or Path(str(authorization.get("resolved_capability_receipt_path"))).resolve()
        != Path(issuance_receipt_path).resolve()
        or int(authorization.get("required_jobs", -1)) != int(required_jobs)
    ):
        raise PermissionError("authorization artifact does not bind capability request")
    package = snapshot.parent
    if (
        snapshot.name != "frozen_source_snapshot"
        or package.name != "synthetic_corrective_development_launch_package_v1"
        or authorization_path.parent.name != "launch_seal"
        or authorization_path.parent.parent != package
    ):
        raise PermissionError("capability package/snapshot layout mismatch")
    config_path = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    config = load_config(config_path, repository_root=snapshot)
    if (
        config["protocol"]["status"] != "frozen"
        or sha256_file(config_path) != authorization.get("required_config_sha256")
        or config["resolved_registries"]["development"]
        != authorization.get("development_registry")
    ):
        raise PermissionError("capability frozen configuration mismatch")
    claim_relative = path.relative_to(package).as_posix()
    receipt = Path(issuance_receipt_path).resolve()
    receipt_relative = receipt.relative_to(package).as_posix()
    complete_manifest = read_json(package / "launch_seal/complete_file_manifest.json")
    complete_verification = verify_exact_manifest(
        package,
        complete_manifest,
        manifest_filename="launch_seal/complete_file_manifest.json",
        allowed_extra_files=[claim_relative, receipt_relative],
    )
    if complete_verification["status"] != "PASS":
        raise PermissionError(f"capability package manifest mismatch: {complete_verification}")
    require_lstat_absent(receipt)
    receipt_payload = {
        "schema_version": "nursery-corrective-capability-issuance-v1",
        "status": "PARENT_CAPABILITY_ISSUED",
        "authorization_digest": str(authorization_digest),
        "parsed_contract_digest": str(parsed_contract_digest),
        "claim_path": str(path),
        "claim_sha256": str(claim_sha256),
        "resolved_output_root": str(Path(resolved_output_root).resolve()),
        "resolved_staging_root": str(Path(resolved_staging_root).resolve()),
        "resolved_inner_staging_root": str(Path(resolved_inner_staging_root).resolve()),
        "required_jobs": int(required_jobs),
        "scope": "parent",
        "unit_id": None,
        "preflight_proof_identity_digest": preflight_proof._identity_digest,
        "parent_issuance_non_replayable": True,
    }
    receipt_value = {**receipt_payload, "receipt_digest": canonical_digest(receipt_payload)}
    write_json(receipt, receipt_value)
    receipt_sha256 = sha256_file(receipt)
    return AuthorizationCapability(
        purpose="development",
        scope="parent",
        unit_id=None,
        authorization_digest=str(authorization_digest),
        parsed_contract_digest=str(parsed_contract_digest),
        claim_path=str(path),
        claim_sha256=str(claim_sha256),
        issuance_receipt_path=str(receipt),
        issuance_receipt_sha256=receipt_sha256,
        resolved_output_root=str(Path(resolved_output_root).resolve()),
        resolved_staging_root=str(Path(resolved_staging_root).resolve()),
        resolved_inner_staging_root=str(Path(resolved_inner_staging_root).resolve()),
        required_jobs=int(required_jobs),
        _seal=_CAPABILITY_FACTORY_SEAL,
    )


def _issue_worker_development_capability(
    *,
    authorization_digest: str,
    parsed_contract_digest: str,
    claim_path: str | Path,
    claim_sha256: str,
    issuance_receipt_path: str | Path,
    issuance_receipt_sha256: str,
    resolved_output_root: str | Path,
    resolved_staging_root: str | Path,
    resolved_inner_staging_root: str | Path,
    required_jobs: int,
    unit_id: str,
) -> AuthorizationCapability:
    claim = Path(claim_path).resolve()
    receipt = Path(issuance_receipt_path).resolve()
    if sha256_file(claim) != str(claim_sha256) or sha256_file(receipt) != str(
        issuance_receipt_sha256
    ):
        raise PermissionError("worker capability parent evidence changed")
    value = read_json(receipt)
    payload = {key: child for key, child in value.items() if key != "receipt_digest"}
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "authorization_digest",
            "parsed_contract_digest",
            "claim_path",
            "claim_sha256",
            "resolved_output_root",
            "resolved_staging_root",
            "resolved_inner_staging_root",
            "required_jobs",
            "scope",
            "unit_id",
            "preflight_proof_identity_digest",
            "parent_issuance_non_replayable",
            "receipt_digest",
        }
        or value.get("receipt_digest") != canonical_digest(payload)
        or value.get("status") != "PARENT_CAPABILITY_ISSUED"
        or value.get("authorization_digest") != str(authorization_digest)
        or value.get("parsed_contract_digest") != str(parsed_contract_digest)
        or Path(str(value.get("claim_path"))).resolve() != claim
        or value.get("claim_sha256") != str(claim_sha256)
        or Path(str(value.get("resolved_output_root"))).resolve()
        != Path(resolved_output_root).resolve()
        or Path(str(value.get("resolved_staging_root"))).resolve()
        != Path(resolved_staging_root).resolve()
        or Path(str(value.get("resolved_inner_staging_root"))).resolve()
        != Path(resolved_inner_staging_root).resolve()
        or int(value.get("required_jobs", -1)) != int(required_jobs)
        or value.get("scope") != "parent"
        or value.get("unit_id") is not None
        or not isinstance(value.get("preflight_proof_identity_digest"), str)
        or len(str(value.get("preflight_proof_identity_digest"))) != 64
        or value.get("parent_issuance_non_replayable") is not True
    ):
        raise PermissionError("worker capability issuance receipt mismatch")
    parent = AuthorizationCapability(
        purpose="development",
        scope="parent",
        unit_id=None,
        authorization_digest=str(authorization_digest),
        parsed_contract_digest=str(parsed_contract_digest),
        claim_path=str(claim),
        claim_sha256=str(claim_sha256),
        issuance_receipt_path=str(receipt),
        issuance_receipt_sha256=str(issuance_receipt_sha256),
        resolved_output_root=str(Path(resolved_output_root).resolve()),
        resolved_staging_root=str(Path(resolved_staging_root).resolve()),
        resolved_inner_staging_root=str(Path(resolved_inner_staging_root).resolve()),
        required_jobs=int(required_jobs),
        _seal=_CAPABILITY_FACTORY_SEAL,
    )
    verify_development_capability(
        parent,
        required_scope="parent",
        resolved_output_root=resolved_output_root,
        resolved_staging_root=resolved_staging_root,
        resolved_inner_staging_root=resolved_inner_staging_root,
        required_jobs=required_jobs,
    )
    return AuthorizationCapability(
        purpose="development",
        scope="worker",
        unit_id=str(unit_id),
        authorization_digest=str(authorization_digest),
        parsed_contract_digest=str(parsed_contract_digest),
        claim_path=str(claim),
        claim_sha256=str(claim_sha256),
        issuance_receipt_path=str(receipt),
        issuance_receipt_sha256=str(issuance_receipt_sha256),
        resolved_output_root=str(Path(resolved_output_root).resolve()),
        resolved_staging_root=str(Path(resolved_staging_root).resolve()),
        resolved_inner_staging_root=str(Path(resolved_inner_staging_root).resolve()),
        required_jobs=int(required_jobs),
        _seal=_CAPABILITY_FACTORY_SEAL,
    )


def verify_development_capability(
    capability: AuthorizationCapability,
    *,
    resolved_output_root: str | Path | None = None,
    resolved_staging_root: str | Path | None = None,
    resolved_inner_staging_root: str | Path | None = None,
    required_jobs: int | None = None,
    required_scope: str | None = None,
    required_unit_id: str | None = None,
) -> None:
    if (
        not isinstance(capability, AuthorizationCapability)
        or capability._seal is not _CAPABILITY_FACTORY_SEAL
        or capability._frozen is not True
        or capability.purpose != "development"
    ):
        raise PermissionError("development requires an issued capability")
    identity = {
        name: getattr(capability, name)
        for name in AuthorizationCapability.__slots__
        if not name.startswith("_")
    }
    if canonical_digest(identity) != capability._identity_digest:
        raise PermissionError("development capability identity changed")
    if sha256_file(capability.claim_path) != capability.claim_sha256:
        raise PermissionError("consumed claim changed after capability issuance")
    if (
        sha256_file(capability.issuance_receipt_path)
        != capability.issuance_receipt_sha256
    ):
        raise PermissionError("capability issuance receipt changed")
    claim = read_json(capability.claim_path)
    expected_claim_fields = {
        "schema_version",
        "status",
        "authorization_digest",
        "resolved_authorization_path",
        "authorization_sha256",
        "resolved_snapshot_root",
        "parsed_contract_digest",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_capability_receipt_path",
        "required_jobs",
        "non_replayable",
        "created_before_any_guarded_development_operation",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    if (
        set(claim) != expected_claim_fields
        or claim.get("schema_version")
        != "nursery-corrective-authorization-consumption-v1"
        or claim.get("status") != "ATTEMPT_CONSUMED"
        or claim.get("authorization_digest") != capability.authorization_digest
        or claim.get("parsed_contract_digest") != capability.parsed_contract_digest
        or Path(str(claim.get("resolved_output_root"))).resolve()
        != Path(capability.resolved_output_root)
        or Path(str(claim.get("resolved_staging_root"))).resolve()
        != Path(capability.resolved_staging_root)
        or Path(str(claim.get("resolved_inner_staging_root"))).resolve()
        != Path(capability.resolved_inner_staging_root)
        or Path(str(claim.get("resolved_capability_receipt_path"))).resolve()
        != Path(capability.issuance_receipt_path)
        or int(claim.get("required_jobs", -1)) != capability.required_jobs
        or claim.get("non_replayable") is not True
        or claim.get("created_before_any_guarded_development_operation") is not True
        or int(claim.get("development_outcome_count", -1)) != 0
        or int(claim.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("capability differs from consumed claim")
    authorization_path = Path(str(claim["resolved_authorization_path"])).resolve()
    snapshot = Path(str(claim["resolved_snapshot_root"])).resolve()
    if (
        sha256_file(authorization_path) != str(claim["authorization_sha256"])
        or snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name
        != "synthetic_corrective_development_launch_package_v1"
    ):
        raise PermissionError("capability authority-chain path or digest mismatch")
    authorization = read_json(authorization_path)
    authorization_payload = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_digest"
    }
    if (
        authorization.get("authorization_digest")
        != canonical_digest(authorization_payload)
        or authorization.get("authorization_digest")
        != capability.authorization_digest
        or authorization.get("status")
        != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY"
        or authorization.get("development_authorized") is not True
        or authorization.get("confirmation_authorized") is not False
        or Path(str(authorization.get("resolved_authorization_path"))).resolve()
        != authorization_path
        or Path(str(authorization.get("resolved_snapshot_root"))).resolve()
        != snapshot
        or Path(str(authorization.get("resolved_claim_path"))).resolve()
        != Path(capability.claim_path)
        or Path(str(authorization.get("resolved_capability_receipt_path"))).resolve()
        != Path(capability.issuance_receipt_path)
        or Path(str(authorization.get("resolved_output_root"))).resolve()
        != Path(capability.resolved_output_root)
        or Path(str(authorization.get("resolved_staging_root"))).resolve()
        != Path(capability.resolved_staging_root)
        or Path(str(authorization.get("resolved_inner_staging_root"))).resolve()
        != Path(capability.resolved_inner_staging_root)
        or int(authorization.get("required_jobs", -1))
        != capability.required_jobs
    ):
        raise PermissionError("capability authorization chain mismatch")
    config_path = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    config = load_config(config_path, repository_root=snapshot)
    if (
        config["protocol"]["status"] != "frozen"
        or sha256_file(config_path)
        != str(authorization.get("required_config_sha256"))
        or config["resolved_registries"]["development"]
        != authorization.get("development_registry")
        or sha256_file(snapshot / "snapshot_manifest.json")
        != str(authorization.get("required_snapshot_manifest_sha256"))
    ):
        raise PermissionError("capability frozen snapshot chain mismatch")
    snapshot_manifest = read_json(snapshot / "snapshot_manifest.json")
    if verify_exact_manifest(
        snapshot,
        snapshot_manifest,
        manifest_filename="snapshot_manifest.json",
    )["status"] != "PASS":
        raise PermissionError("capability snapshot manifest mismatch")
    receipt = read_json(capability.issuance_receipt_path)
    expected_receipt_fields = {
        "schema_version",
        "status",
        "authorization_digest",
        "parsed_contract_digest",
        "claim_path",
        "claim_sha256",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "required_jobs",
        "scope",
        "unit_id",
        "preflight_proof_identity_digest",
        "parent_issuance_non_replayable",
        "receipt_digest",
    }
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema_version")
        != "nursery-corrective-capability-issuance-v1"
        or receipt.get("status") != "PARENT_CAPABILITY_ISSUED"
        or receipt.get("receipt_digest") != canonical_digest(receipt_payload)
        or receipt.get("authorization_digest") != capability.authorization_digest
        or receipt.get("parsed_contract_digest") != capability.parsed_contract_digest
        or Path(str(receipt.get("claim_path"))).resolve()
        != Path(capability.claim_path)
        or receipt.get("claim_sha256") != capability.claim_sha256
        or Path(str(receipt.get("resolved_output_root"))).resolve()
        != Path(capability.resolved_output_root)
        or Path(str(receipt.get("resolved_staging_root"))).resolve()
        != Path(capability.resolved_staging_root)
        or Path(str(receipt.get("resolved_inner_staging_root"))).resolve()
        != Path(capability.resolved_inner_staging_root)
        or int(receipt.get("required_jobs", -1)) != capability.required_jobs
        or receipt.get("scope") != "parent"
        or receipt.get("unit_id") is not None
        or not isinstance(receipt.get("preflight_proof_identity_digest"), str)
        or len(str(receipt.get("preflight_proof_identity_digest"))) != 64
        or receipt.get("parent_issuance_non_replayable") is not True
    ):
        raise PermissionError("capability differs from issuance receipt")
    package = snapshot.parent
    claim_relative = Path(capability.claim_path).relative_to(package).as_posix()
    receipt_relative = Path(capability.issuance_receipt_path).relative_to(
        package
    ).as_posix()
    complete_manifest = read_json(
        package / "launch_seal/complete_file_manifest.json"
    )
    if verify_exact_manifest(
        package,
        complete_manifest,
        manifest_filename="launch_seal/complete_file_manifest.json",
        allowed_extra_files=[claim_relative, receipt_relative],
    )["status"] != "PASS":
        raise PermissionError("capability package manifest changed")
    if (
        capability.scope not in {"parent", "worker"}
        or (capability.scope == "parent" and capability.unit_id is not None)
        or (capability.scope == "worker" and not capability.unit_id)
    ):
        raise PermissionError("capability scope/unit invariant failed")
    if resolved_output_root is not None and Path(capability.resolved_output_root) != Path(
        resolved_output_root
    ).resolve():
        raise PermissionError("development capability output root mismatch")
    if resolved_staging_root is not None and Path(
        capability.resolved_staging_root
    ) != Path(resolved_staging_root).resolve():
        raise PermissionError("development capability staging root mismatch")
    if resolved_inner_staging_root is not None and Path(
        capability.resolved_inner_staging_root
    ) != Path(resolved_inner_staging_root).resolve():
        raise PermissionError("development capability inner staging root mismatch")
    if required_jobs is not None and capability.required_jobs != int(required_jobs):
        raise PermissionError("development capability job count mismatch")
    if required_scope is not None and capability.scope != str(required_scope):
        raise PermissionError("development capability scope mismatch")
    if required_unit_id is not None and capability.unit_id != str(required_unit_id):
        raise PermissionError("development capability unit mismatch")


_EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL = object()


class ExcludedRehearsalCapability:
    __slots__ = (
        "purpose",
        "scope",
        "unit_id",
        "snapshot_root",
        "output_root",
        "persisted_input_root",
        "parallel_output_root",
        "attempt_receipt_path",
        "attempt_receipt_sha256",
        "required_jobs",
        "input_manifest_sha256",
        "scientific_contract_digest",
        "authorization_digest",
        "parsed_contract_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("excluded-rehearsal capabilities are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        scope: str,
        unit_id: str | None,
        snapshot_root: str | Path,
        output_root: str | Path,
        persisted_input_root: str | Path,
        parallel_output_root: str | Path,
        attempt_receipt_path: str | Path,
        attempt_receipt_sha256: str,
        required_jobs: int,
        input_manifest_sha256: str | None,
        scientific_contract_digest: str | None,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL:
            raise PermissionError(
                "excluded-rehearsal capability requires the consumed receipt"
            )
        self.purpose = "excluded_rehearsal"
        self.scope = str(scope)
        self.unit_id = None if unit_id is None else str(unit_id)
        self.snapshot_root = str(Path(snapshot_root).resolve())
        self.output_root = str(Path(output_root).resolve())
        self.persisted_input_root = str(Path(persisted_input_root).resolve())
        self.parallel_output_root = str(Path(parallel_output_root).resolve())
        self.attempt_receipt_path = str(Path(attempt_receipt_path).resolve())
        self.attempt_receipt_sha256 = str(attempt_receipt_sha256)
        self.required_jobs = int(required_jobs)
        self.input_manifest_sha256 = (
            None if input_manifest_sha256 is None else str(input_manifest_sha256)
        )
        self.scientific_contract_digest = (
            None
            if scientific_contract_digest is None
            else str(scientific_contract_digest)
        )
        self.authorization_digest = None
        self.parsed_contract_digest = None
        self._seal = _seal
        self._frozen = True


def _issue_excluded_rehearsal_capability(
    *,
    config: Mapping[str, Any],
    scope: str,
    unit_id: str | None,
    snapshot_root: str | Path,
    output_root: str | Path,
    persisted_input_root: str | Path,
    parallel_output_root: str | Path,
    attempt_receipt_path: str | Path,
    attempt_receipt_sha256: str,
    required_jobs: int,
    input_manifest_sha256: str | None = None,
    scientific_contract_digest: str | None = None,
) -> ExcludedRehearsalCapability:
    snapshot = Path(snapshot_root).resolve()
    repository = snapshot.parents[2]
    expected_output = (
        repository / str(config["paths"]["excluded_rehearsal"])
    ).resolve()
    expected_receipt = (
        repository
        / str(config["paths"]["excluded_rehearsal_attempt_receipt"])
    ).resolve()
    if (
        config["protocol"]["status"] != "frozen"
        or snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name
        != "synthetic_corrective_development_launch_package_v1"
        or Path(output_root).resolve() != expected_output
        or Path(attempt_receipt_path).resolve() != expected_receipt
        or int(required_jobs) != int(config["parallel"]["frozen_jobs"])
    ):
        raise PermissionError(
            "excluded-rehearsal capability differs from frozen configured paths"
        )
    capability = ExcludedRehearsalCapability(
        scope=scope,
        unit_id=unit_id,
        snapshot_root=snapshot_root,
        output_root=output_root,
        persisted_input_root=persisted_input_root,
        parallel_output_root=parallel_output_root,
        attempt_receipt_path=attempt_receipt_path,
        attempt_receipt_sha256=attempt_receipt_sha256,
        required_jobs=required_jobs,
        input_manifest_sha256=input_manifest_sha256,
        scientific_contract_digest=scientific_contract_digest,
        _seal=_EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL,
    )
    verify_excluded_rehearsal_capability(capability, required_scope=scope)
    return capability


def verify_excluded_rehearsal_capability(
    capability: ExcludedRehearsalCapability,
    *,
    required_scope: str | None = None,
    required_unit_id: str | None = None,
    persisted_input_root: str | Path | None = None,
    parallel_output_root: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    scientific_contract_digest: str | None = None,
) -> None:
    if (
        not isinstance(capability, ExcludedRehearsalCapability)
        or capability._seal is not _EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL
        or capability.purpose != "excluded_rehearsal"
    ):
        raise PermissionError(
            "excluded rehearsal requires a consumed attempt capability"
        )
    receipt_path = Path(capability.attempt_receipt_path)
    if sha256_file(receipt_path) != capability.attempt_receipt_sha256:
        raise PermissionError("excluded rehearsal attempt receipt changed")
    snapshot = Path(capability.snapshot_root)
    receipt = read_json(receipt_path)
    expected_receipt = {
        "schema_version": "nursery-corrective-excluded-rehearsal-attempt-v1",
        "status": "ONE_EXCLUDED_ATTEMPT_CONSUMED",
        "snapshot_root": str(snapshot),
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "freeze_receipt_sha256": sha256_file(snapshot.parent / "freeze_receipt.json"),
        "output_root": capability.output_root,
        "jobs": capability.required_jobs,
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    inner = Path(capability.output_root).with_name(
        f".{Path(capability.output_root).name}.cohort-staging"
    )
    if (
        receipt != expected_receipt
        or Path(capability.persisted_input_root) != (inner / "persisted_inputs").resolve()
        or Path(capability.parallel_output_root)
        != (inner / "parallel_compute").resolve()
        or capability.scope not in {"parent", "worker"}
        or (capability.scope == "parent" and capability.unit_id is not None)
        or (capability.scope == "worker" and not capability.unit_id)
        or (
            capability.scope == "parent"
            and (
                capability.input_manifest_sha256 is not None
                or capability.scientific_contract_digest is not None
            )
        )
        or (
            capability.scope == "worker"
            and (
                capability.input_manifest_sha256 is None
                or capability.scientific_contract_digest is None
            )
        )
    ):
        raise PermissionError("excluded rehearsal capability content mismatch")
    if required_scope is not None and capability.scope != str(required_scope):
        raise PermissionError("excluded rehearsal capability scope mismatch")
    if required_unit_id is not None and capability.unit_id != str(required_unit_id):
        raise PermissionError("excluded rehearsal capability unit mismatch")
    if persisted_input_root is not None and Path(
        capability.persisted_input_root
    ) != Path(persisted_input_root).resolve():
        raise PermissionError("excluded rehearsal persisted-input path mismatch")
    if parallel_output_root is not None and Path(
        capability.parallel_output_root
    ) != Path(parallel_output_root).resolve():
        raise PermissionError("excluded rehearsal parallel-output path mismatch")
    if (
        input_manifest_sha256 is not None
        and capability.input_manifest_sha256 != str(input_manifest_sha256)
    ):
        raise PermissionError("excluded rehearsal input commitment mismatch")
    if (
        scientific_contract_digest is not None
        and capability.scientific_contract_digest
        != str(scientific_contract_digest)
    ):
        raise PermissionError("excluded rehearsal scientific contract mismatch")


@dataclass(frozen=True, slots=True)
class IdentifierReference:
    role: str
    value: int


class IdentifierFirewall:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        purpose: str,
        capability: AuthorizationCapability | ExcludedRehearsalCapability | None = None,
        operation_contract_digest: str | None = None,
    ):
        if purpose not in PURPOSES:
            raise PermissionError(f"unsupported execution purpose: {purpose}")
        if purpose == "development":
            if not isinstance(capability, AuthorizationCapability):
                raise PermissionError("development requires verified capability")
            verify_development_capability(capability)
        elif purpose == "excluded_rehearsal":
            if not isinstance(capability, ExcludedRehearsalCapability):
                raise PermissionError(
                    "excluded rehearsal requires verified capability"
                )
            verify_excluded_rehearsal_capability(capability)
        elif capability is not None:
            raise PermissionError(
                "capability only valid for development or excluded rehearsal"
            )
        self.config = config
        self.purpose = purpose
        self.registry = config["resolved_registries"][purpose]
        self.capability = capability
        self.operation_contract_digest = operation_contract_digest
        self.operations: list[dict[str, Any]] = []
        self._previous_digest = "0" * 64

    def authorize(
        self,
        operation: str,
        references: Sequence[IdentifierReference],
        *,
        unit_id: str,
        local_sequence: int,
    ) -> str:
        if operation not in OPERATIONS:
            raise ValueError(f"unknown operation: {operation}")
        if (
            self.purpose in {"development", "excluded_rehearsal"}
            and self.capability is not None
            and self.capability.scope == "worker"
            and str(unit_id) != self.capability.unit_id
        ):
            raise PermissionError("worker capability cannot authorize another unit")
        if local_sequence != len(self.operations):
            raise ValueError("operation sequence must be contiguous")
        normalized = []
        quarantine = self.config["registries"]["permanently_quarantined_ranges"]
        for reference in references:
            if reference.role not in ROLES:
                raise ValueError(f"unknown identifier role: {reference.role}")
            value = int(reference.value)
            if _in_ranges(value, quarantine):
                raise PermissionError("permanently quarantined identifier")
            if value not in set(map(int, self.registry[reference.role])):
                raise PermissionError(
                    f"identifier {value} is not registered for {self.purpose}/{reference.role}"
                )
            normalized.append({"role": reference.role, "value": value})
        payload = {
            "purpose": self.purpose,
            "unit_id": str(unit_id),
            "local_sequence": int(local_sequence),
            "operation": operation,
            "references": normalized,
            "previous_digest": self._previous_digest,
        }
        entry_digest = canonical_digest(payload)
        self.operations.append({**payload, "entry_digest": entry_digest})
        self._previous_digest = entry_digest
        return entry_digest

    def ledger(self) -> dict[str, Any]:
        return {
            "schema_version": "nursery-operation-ledger-v1",
            "purpose": self.purpose,
            "authorization_digest": (
                self.capability.authorization_digest if self.capability else None
            ),
            "parsed_contract_digest": (
                self.capability.parsed_contract_digest if self.capability else None
            ),
            "operation_contract_digest": self.operation_contract_digest,
            "entry_count": len(self.operations),
            "terminal_digest": self._previous_digest,
            "entries": list(self.operations),
        }


def registry_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "nursery-corrective-registry-v1",
        "protocol_id": PROTOCOL_ID,
        "registries": config["resolved_registries"],
        "namespace_family": list(config["registries"]["namespace_family"]),
        "deliberately_unused_package_ranges": list(
            config["registries"]["deliberately_unused_package_ranges"]
        ),
        "permanently_quarantined_ranges": list(
            config["registries"]["permanently_quarantined_ranges"]
        ),
        "canonical_prior_registry": config["registries"]["canonical_prior_registry"],
        "canonical_prior_registry_sha256": config["registries"][
            "canonical_prior_registry_sha256"
        ],
    }


def verify_ledger(
    ledger: Mapping[str, Any],
    *,
    purpose: str,
    unit_id: str,
    expected_operations: Sequence[str],
    corpus_seed: int,
    model_seed: int,
    authorization_digest: str | None = None,
    parsed_contract_digest: str | None = None,
    operation_contract_digest: str | None = None,
) -> dict[str, Any]:
    problems: list[str] = []
    if set(ledger) != {
        "schema_version",
        "purpose",
        "authorization_digest",
        "parsed_contract_digest",
        "operation_contract_digest",
        "entry_count",
        "terminal_digest",
        "entries",
    }:
        problems.append("top_level_schema")
    entries = ledger.get("entries", [])
    if ledger.get("schema_version") != "nursery-operation-ledger-v1":
        problems.append("schema")
    if ledger.get("purpose") != purpose:
        problems.append("purpose")
    if ledger.get("authorization_digest") != authorization_digest:
        problems.append("authorization_digest")
    if ledger.get("parsed_contract_digest") != parsed_contract_digest:
        problems.append("parsed_contract_digest")
    if ledger.get("operation_contract_digest") != operation_contract_digest:
        problems.append("operation_contract_digest")
    if int(ledger.get("entry_count", -1)) != len(entries):
        problems.append("entry_count")
    if len(entries) != len(expected_operations):
        problems.append("expected_count")
    previous = "0" * 64
    for index, (entry, expected) in enumerate(zip(entries, expected_operations)):
        if set(entry) != {
            "purpose",
            "unit_id",
            "local_sequence",
            "operation",
            "references",
            "previous_digest",
            "entry_digest",
        }:
            problems.append(f"entry_schema:{index}")
        payload = {key: value for key, value in entry.items() if key != "entry_digest"}
        if entry.get("purpose") != purpose:
            problems.append(f"entry_purpose:{index}")
        if entry.get("unit_id") != unit_id:
            problems.append(f"unit:{index}")
        if int(entry.get("local_sequence", -1)) != index:
            problems.append(f"sequence:{index}")
        if entry.get("operation") != expected:
            problems.append(f"operation:{index}")
        if entry.get("previous_digest") != previous:
            problems.append(f"chain:{index}")
        if entry.get("entry_digest") != canonical_digest(payload):
            problems.append(f"digest:{index}")
        raw_references = list(entry.get("references", []))
        references = {
            (str(row.get("role")), int(row.get("value", -1)))
            for row in entry.get("references", [])
        }
        if references != {
            ("corpus", int(corpus_seed)),
            ("model", int(model_seed)),
        } or len(raw_references) != 2:
            problems.append(f"references:{index}")
        previous = str(entry.get("entry_digest", ""))
    if ledger.get("terminal_digest") != previous:
        problems.append("terminal_digest")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}
