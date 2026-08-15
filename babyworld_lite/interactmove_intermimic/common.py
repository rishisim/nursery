"""Shared deterministic contract helpers for the InteractMove protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """Raised when a Stage 1--3 artifact violates its frozen contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number is not permitted: {value}")


def read_json(path: Path) -> dict[str, Any]:
    """Read strict UTF-8 JSON while rejecting duplicate keys and NaN/Inf."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"JSON is not valid UTF-8: {path}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"top-level JSON value must be an object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the protocol's semantic JSON encoding."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical-JSON encodable: {exc}") from exc
    return text.encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any, *, overwrite: bool = False) -> None:
    """Pretty-print JSON atomically without silently replacing an artifact."""

    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    where: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing:
        raise ContractError(f"{where} is missing keys: {missing}")
    if extra:
        raise ContractError(f"{where} has unknown keys: {extra}")


def require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return value


def require_nonempty_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def require_integer(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where} must be an integer >= {minimum}")
    return value


def require_finite_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where} must be a finite number")
    return result


def require_vector(
    value: Any,
    *,
    length: int,
    where: str,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ContractError(f"{where} must be a length-{length} array")
    result = [
        require_finite_number(item, where=f"{where}[{index}]")
        for index, item in enumerate(value)
    ]
    if positive and any(item <= 0.0 for item in result):
        raise ContractError(f"{where} entries must be > 0")
    return result


def require_unit_quaternion_xyzw(
    value: Any,
    *,
    where: str,
    tolerance: float = 1e-5,
) -> list[float]:
    quaternion = require_vector(value, length=4, where=where)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm == 0.0 or abs(norm - 1.0) > tolerance:
        raise ContractError(
            f"{where} must be unit length within {tolerance}; got norm {norm}"
        )
    return quaternion


def canonicalize_quaternion_xyzw(
    value: Any,
    *,
    where: str,
    maximum_norm_error: float = 1e-3,
) -> tuple[list[float], bool]:
    quaternion = require_vector(value, length=4, where=where)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm == 0.0 or abs(norm - 1.0) > maximum_norm_error:
        raise ContractError(
            f"{where} has invalid quaternion norm {norm}; "
            f"maximum error is {maximum_norm_error}"
        )
    normalized = [item / norm for item in quaternion]
    changed = abs(norm - 1.0) > 1e-12
    sign_anchor = next(
        (item for item in (normalized[3], *normalized[:3]) if item != 0.0),
        0.0,
    )
    if sign_anchor < 0.0:
        normalized = [-item for item in normalized]
        changed = True
    unsigned_zero = [0.0 if item == 0.0 else item for item in normalized]
    if unsigned_zero != normalized or any(
        item == 0.0 and math.copysign(1.0, item) < 0.0 for item in normalized
    ):
        changed = True
    normalized = unsigned_zero
    return normalized, changed


def quaternion_xyzw_from_rpy(rpy_rad: Sequence[float]) -> list[float]:
    roll, pitch, yaw = rpy_rad
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    quaternion = [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
    canonical, _ = canonicalize_quaternion_xyzw(
        quaternion, where="derived RPY quaternion", maximum_norm_error=1e-9
    )
    return canonical


def transform_matrix(
    translation: Sequence[float], quaternion_xyzw: Sequence[float]
) -> list[list[float]]:
    x, y, z, w = quaternion_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), translation[0]],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), translation[1]],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_transform_matrices(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Multiply two explicit 4x4 transforms without a numeric dependency."""

    if len(left) != 4 or len(right) != 4 or any(
        len(row) != 4 for row in [*left, *right]
    ):
        raise ContractError("transform matrices must both be 4x4")
    return [
        [
            sum(float(left[row][inner]) * float(right[inner][column]) for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def safe_relative_file(root: Path, relative: Any, *, where: str) -> Path:
    """Resolve one declared POSIX file path beneath a non-symlink scene root."""

    text = require_nonempty_string(relative, where=where)
    if "\x00" in text or "\\" in text or ":" in text:
        raise ContractError(f"{where} must be a relative POSIX path")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ContractError(f"{where} must be a traversal-free relative POSIX path")

    if root.is_symlink():
        raise ContractError(f"scene root must not be a symlink: {root}")
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{where} traverses a symlink: {text}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ContractError(f"{where} does not exist: {text}") from exc
    if not resolved.is_relative_to(root_resolved):
        raise ContractError(f"{where} escapes the scene root: {text}")
    if not resolved.is_file():
        raise ContractError(f"{where} is not a regular file: {text}")
    return resolved


def posix_relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root.resolve(strict=True)).as_posix()
