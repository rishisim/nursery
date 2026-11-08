"""Label-free gaze windows aligned to the existing FramePairs visual windows."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def collate_frame_indices(dataset, indices):
    """Gather one bounded visual batch before casting on the accelerator.

    FramePairs' metadata and sampler remain authoritative. Cached float16 values
    retain their exact bits; the caller casts to float32 after device transfer.
    Frame order, repeated-first-frame padding and text tensors match FramePairs.
    """
    shape = dataset.caches[0].shape[1:]
    dtype = np.result_type(*(cache.dtype for cache in dataset.caches))
    locations = []
    for index in indices:
        here = dataset.locs[index]
        locations.append(here + [here[0]] * (dataset.Fmax - len(here)))
    locations = np.asarray(locations, dtype=np.int64).reshape(-1, 2)
    values = np.empty((len(locations), *shape), dtype=dtype)
    for cache_index in np.unique(locations[:, 0]):
        selected = locations[:, 0] == cache_index
        values[selected] = dataset.caches[cache_index][locations[selected, 1]]
    values = values.reshape(len(indices), dataset.Fmax, *shape)
    if dataset.cls_only:
        values = values[:, :, :1]
    values = values.reshape(len(indices), -1, values.shape[-1])
    text = torch.zeros((len(indices), dataset.max_len), dtype=torch.long)
    for row, index in enumerate(indices):
        text[row, :len(dataset.ids[index])] = torch.tensor(dataset.ids[index])
    return (torch.tensor(indices, dtype=torch.long), torch.from_numpy(values), text,
            torch.tensor([len(dataset.ids[index]) for index in indices], dtype=torch.long))


def dino_crop_points(points, width=320, height=240):
    """Map normalized image coordinates through the pinned DINO 256/224 crop.

    The processor truncates the resized long edge and floor-divides crop offsets.
    Points outside the crop remain outside [0, 1]; never clamp them onto a patch.
    """
    if min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive")
    short, long = sorted((width, height))
    resized_short, resized_long = 256, int(256 * long / short)
    rw, rh = ((resized_long, resized_short) if width >= height
              else (resized_short, resized_long))
    return (np.asarray(points, dtype=np.float32) * [rw, rh]
            - [(rw - 224) // 2, (rh - 224) // 2]) / 224


def _keys(table):
    if table[["video_id", "frame_idx"]].isna().any().any():
        raise ValueError("Missing frame key")
    frames = table.frame_idx.to_numpy()
    if not np.equal(frames, frames.astype(int)).all() or (frames < 0).any():
        raise ValueError("Frame indices must be nonnegative integers")
    return [f"{v}|{int(f)}" for v, f in zip(table.video_id, frames)]


def _closeups(cache, needed):
    """Read only requested rows, retaining one float32 patch mean per frame."""
    cache = Path(cache)
    index = pd.read_parquet(cache / "index.parquet")
    keys = _keys(index)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate close-up frame keys")
    values = np.load(cache / "emb.f16.npy", mmap_mode="r", allow_pickle=False)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[1] != 257):
        raise ValueError("Close-up cache must have compact vectors or 257 DINO tokens")
    rows = index.row.to_numpy()
    if (not np.equal(rows, rows.astype(int)).all() or len(set(rows)) != len(rows)
            or (rows < 0).any() or (rows >= len(values)).any()):
        raise ValueError("Invalid close-up cache row mapping")
    lookup = dict(zip(keys, rows.astype(int)))
    missing = set(needed) - lookup.keys()
    if missing:
        raise ValueError(f"Missing close-up features for {len(missing)} gaze frames")
    result = {}
    # Bound temporary memory independently of corpus/cache size.
    for key in needed:
        value = np.asarray(values[lookup[key]], dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite close-up feature: {key}")
        result[key] = value[1:].mean(0) if value.ndim == 2 else value.copy()
    return result, int(values.shape[-1])


class GazeData:
    """Build model contexts from keys, gaze coordinates and frozen image features.

    ``partitions`` maps frame keys to source-clip partitions; it must cover every
    accessible visual-window frame. ``allowed_keys`` may further restrict the
    selected training experience. Construct separate adapters per evaluation
    panel. No words, categories, object IDs or held/target labels are read.

    The supplied manifest must already match FramePairs with no dropped rows.
    Shuffle donors are other center moments within this manifest and partition;
    one deterministic donor window supplies both point and close-up together.
    """

    def __init__(self, manifest, lut, gaze, *, partitions, window=2,
                 closeup_cache=None, point_only=False, allowed_keys=None, seed=0):
        if window < 0 or int(window) != window:
            raise ValueError("Window must be a nonnegative integer")
        if len(manifest) == 0:
            raise ValueError("Gaze adapter needs a nonempty manifest")
        self.centers = _keys(manifest)
        self.point_only = point_only
        self.partitions = []
        self.frame_keys = []
        allowed = None if allowed_keys is None else set(allowed_keys)
        for center, (video, frame) in zip(self.centers, zip(manifest.video_id, manifest.frame_idx)):
            if center not in partitions:
                raise ValueError(f"Missing center partition: {center}")
            part = partitions[center]
            here = [f"{video}|{f}" for f in range(int(frame) - window, int(frame) + window + 1)
                    if f"{video}|{f}" in lut]
            if not here:
                raise ValueError(f"Missing visual window: {center}")
            if any(k not in partitions or partitions[k] != part for k in here):
                raise ValueError("Gaze window crosses or lacks a source-clip partition")
            if allowed is not None and any(k not in allowed for k in here):
                raise ValueError("Gaze window exceeds selected experience")
            self.frame_keys.append(here)
            self.partitions.append(part)
        table = pd.read_parquet(gaze) if isinstance(gaze, (str, Path)) else gaze
        keys = _keys(table)
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate gaze frame keys")
        coordinates = table[["gaze_x", "gaze_y"]].to_numpy(dtype=np.float32)
        null = np.isnan(coordinates)
        if (null[:, 0] != null[:, 1]).any() or np.isinf(coordinates).any():
            raise ValueError("Gaze must be finite coordinates or a wholly null point")
        valid = ~null.any(1)
        if ((coordinates[valid] < 0) | (coordinates[valid] >= 1)).any():
            raise ValueError("Raw gaze coordinates must lie in [0, 1)")
        lookup = dict(zip(keys, range(len(keys))))
        needed = set(k for here in self.frame_keys for k in here)
        if needed - lookup.keys():
            raise ValueError("Gaze table does not cover every accessible visual frame")
        valid_keys = sorted(k for k in needed if valid[lookup[k]])
        if not point_only and closeup_cache is None:
            raise ValueError("Close-up cache is required unless point_only")
        features, dim = ({}, 0) if point_only else _closeups(closeup_cache, valid_keys)
        shape = (len(manifest), 2 * window + 1)
        self.points = np.zeros((*shape, 2), np.float32)
        self.valid = np.zeros(shape, bool)
        self.frame_valid = np.zeros(shape, bool)
        self.closeup = None if point_only else np.zeros((*shape, dim), np.float32)
        cropped = dino_crop_points(coordinates)
        for i, here in enumerate(self.frame_keys):
            for j, key in enumerate(here):
                self.frame_valid[i, j] = True
                row = lookup[key]
                if valid[row]:
                    self.points[i, j] = cropped[row]
                    self.valid[i, j] = True
                    if self.closeup is not None:
                        self.closeup[i, j] = features[key]
        self.donor_indices = np.full(len(manifest), -1, np.int64)
        rng = np.random.default_rng(seed)
        groups = {}
        for i, (center, part) in enumerate(zip(self.centers, self.partitions)):
            groups.setdefault(part, {}).setdefault(center, []).append(i)
        moments = {part: list(group) for part, group in groups.items()}
        positions = {part: {key: i for i, key in enumerate(keys)} for part, keys in moments.items()}
        for i, (center, part) in enumerate(zip(self.centers, self.partitions)):
            if len(moments[part]) > 1:
                choice = int(rng.integers(len(moments[part]) - 1))
                choice += choice >= positions[part][center]
                other = moments[part][choice]
                self.donor_indices[i] = rng.choice(groups[part][other])

    def batch(self, indices, generator=None, *, mode="real", dropout=0.0, device="cpu"):
        """Return with-gaze or exact removed context; padding contributes no gaze."""
        if mode not in ("real", "shuffled", "removed"):
            raise ValueError("Gaze mode must be real, shuffled or removed")
        if not 0 <= dropout <= 1:
            raise ValueError("Gaze dropout must lie in [0, 1]")
        if mode == "removed":
            return None
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.centers)).any():
            raise ValueError("Invalid gaze batch indices")
        donors = self.donor_indices[indices] if mode == "shuffled" else indices
        if (donors < 0).any():
            raise ValueError("Shuffled gaze needs another moment in the same partition")
        keep = torch.ones(len(indices), dtype=torch.bool)
        if dropout:
            keep = torch.rand(len(indices), generator=generator) >= dropout
        ctx = {"gaze_point": torch.as_tensor(self.points[donors], device=device),
               "gaze_valid": torch.as_tensor(self.valid[donors] & self.frame_valid[indices], device=device),
               "keep": keep.to(device)}
        if self.closeup is not None:
            ctx["closeup"] = torch.as_tensor(self.closeup[donors], device=device)
        return ctx
