"""Resample in the existing metre, +Z world frame without floor correction."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .schema import FPS, validate_source


def continuous_quaternions(matrices):
    shape = matrices.shape[:-2]
    q = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(*shape, 4)
    q[0] *= np.where(q[0, ..., 3:4] < 0, -1, 1)
    for frame in range(1, len(q)):
        q[frame] *= np.where((q[frame] * q[frame - 1]).sum(-1, keepdims=True) < 0, -1, 1)
    return q


def resample_motion(arrays, source_fps):
    frames = validate_source(arrays)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source fps must be positive and finite")
    source_times = np.arange(frames) / source_fps
    # Frames represent intervals [t, t + 1/fps). Hold the last sample over
    # its remaining interval; never extrapolate a body or object trajectory.
    times = np.arange(int(np.ceil(frames * FPS / source_fps - 1e-9))) / FPS
    sample_times = np.minimum(times, source_times[-1])
    result = {}
    for name in ("trans", "joints", "trans_obj"):
        values = arrays[name].reshape(frames, -1)
        result[name] = np.stack([np.interp(sample_times, source_times, v) for v in values.T], axis=1)
        result[name] = result[name].reshape(len(times), *arrays[name].shape[1:])
    for name in ("poses", "poses_obj"):
        values = arrays[name].reshape(frames, -1, 3)
        rotations = [Slerp(source_times, Rotation.from_rotvec(values[:, joint]))(sample_times).as_matrix()
                     for joint in range(values.shape[1])]
        result[name] = np.stack(rotations, axis=1)
    nearest = np.minimum(np.floor(times * source_fps + 0.5).astype(int), frames - 1)
    result["contact"] = arrays["contact"][nearest].copy()
    return result, times
