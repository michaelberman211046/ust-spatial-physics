"""One source of truth for the publication experimental and synthetic sensor geometry.

Coordinates come from the RF MAT file, never from a fitted ideal ring.  The
common angular ordering is a permutation of the original channels; no waveform
or travel time is interpolated.  Coordinates are centered only to express them
in the synthetic image's centered physical frame.
"""
from functools import lru_cache
import hashlib
import os
import numpy as np

ENV_NAME = "UST_GEOMETRY_MAT"


@lru_cache(maxsize=2)
def measured_geometry(path):
    import h5py
    with h5py.File(path, "r") as handle:
        if "transducerPositionsXY" not in handle:
            raise ValueError("RF MAT file lacks transducerPositionsXY")
        xy = np.squeeze(np.asarray(handle["transducerPositionsXY"], dtype=np.float64))
    if xy.ndim != 2:
        raise ValueError(f"Invalid sensor coordinate shape: {xy.shape}")
    if xy.shape[0] == 2 and xy.shape[1] != 2:
        xy = xy.T
    if xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"Invalid sensor coordinates: {xy.shape}")
    center = xy.mean(axis=0)
    centered = xy - center
    angles = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2*np.pi)
    order = np.argsort(angles, kind="stable")
    points = centered[order]
    radii = np.linalg.norm(points, axis=1)
    if not 0.09 < float(np.median(radii)) < 0.13:
        raise ValueError("Sensor coordinates are not a ~22 cm diameter ring in metres")
    digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    return points, order.astype(np.int64), center, digest


def require_geometry(n_emitters, n_receivers, path=None):
    source = path or os.environ.get(ENV_NAME)
    if not source:
        raise RuntimeError(f"publication requires {ENV_NAME} pointing to the experimental RF MAT file")
    points, order, center, digest = measured_geometry(os.path.abspath(source))
    if int(n_emitters) != len(points) or int(n_receivers) != len(points):
        raise ValueError(f"publication measured ring has {len(points)} channels, requested {n_emitters}x{n_receivers}")
    return points, order, center, digest


def pixel_positions(nx, ny, dx, dy, n_emitters, n_receivers):
    points, _, _, _ = require_geometry(n_emitters, n_receivers)
    pixels = np.empty_like(points, dtype=np.float32)
    pixels[:, 0] = (nx-1)/2 + points[:, 0]/float(dx)
    pixels[:, 1] = (ny-1)/2 + points[:, 1]/float(dy)
    return pixels.copy(), pixels.copy()


def angular_mask(n_emitters, n_receivers, exclude_frac):
    points, _, _, _ = require_geometry(n_emitters, n_receivers)
    angle = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2*np.pi)
    diff = np.abs(angle[:, None]-angle[None, :])
    diff = np.minimum(diff, 2*np.pi-diff)
    return (diff > np.pi*float(exclude_frac)).astype(np.float32)










