# FILE: extract_measured_tof.py
"""Extract a single measured time-of-flight matrix from an RF acquisition.

Geometry, grid, and normalization parameters are read from the metadata of the
reference synthetic dataset supplied with ``--data_path``. Terminal, HTML, and
figure artifacts are written below ``--output_dir``.
"""

from __future__ import annotations
import config
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # allow mixed OpenMP (unsafe but unblocks)

import sys
from pathlib import Path
import argparse
import time as elapsed

import h5py
import numpy as np
import torch
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from logger import log_message, log_image
from settings import set_output_folder

import pprint
import runtime_context as _G
# ----------------------------
# Robust torch.load (silent)
# ----------------------------
import warnings

def robust_torch_load(path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            return torch.load(path, map_location='cpu', weights_only=False)


def resolve_path_maybe_project_relative(p: str) -> str:
    # If p exists as-is, return it. Otherwise, try <project_root>/<p> where
    # project_root is the parent of src/.
    pp = Path(p)
    if pp.exists():
        return str(pp)
    proj_root = Path(__file__).resolve().parent.parent
    cand = (proj_root / pp).resolve()
    if cand.exists():
        return str(cand)
    return str(pp)


# ----------------------------
# Make imports deterministic
# ----------------------------
# Ensure the directory containing this file (src/) is on sys.path,
# so imports like "import settings" resolve consistently.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ----------------------------
# Constants / metadata style
# ----------------------------
GRID_SPACING_VERSION = 2
GRID_CONVENTION = "node"   # dx = phys/(n-1)
# The geometry search already determines the angular origin from the recorded
# coordinates.  No empirical post-search channel shift is justified.
GEOMETRY_AUTO_SYNTHETIC_PHASE_SHIFT = 0


# ----------------------------
# Utilities
# ----------------------------
def _dx_dy_from_phys(nx: int, ny: int, phys_x: float, phys_y: float) -> tuple[float, float]:
    nx = int(nx)
    ny = int(ny)
    if nx < 2 or ny < 2:
        raise ValueError(f"nx, ny must be >= 2. Got nx={nx}, ny={ny}")
    dx = float(phys_x) / float(nx - 1)
    dy = float(phys_y) / float(ny - 1)
    return dx, dy


def _axis_array_from_metadata(meta: dict, keys: tuple[str, ...], expected_size: int) -> np.ndarray | None:
    for key in keys:
        if key not in meta:
            continue
        try:
            value = meta[key]
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            arr = arr.astype(np.float64).squeeze()
            if arr.ndim == 1 and arr.size == int(expected_size):
                return arr
            log_message(
                f"[tof-extraction] WARNING: metadata axis {key!r} has shape {arr.shape}, "
                f"expected ({int(expected_size)},); ignoring it."
            )
        except Exception as exc:
            log_message(f"[tof-extraction] WARNING: could not parse metadata axis {key!r}: {exc}")
    return None


def _build_physical_axes_for_saved_metadata(
    meta: dict,
    *,
    nx: int,
    ny: int,
    phys_x: float,
    phys_y: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return physical axes in meters for the saved experimental .pt file.

    The synthetic code stores spatial arrays in (nx, ny) and uses physical
    extents as sizes rather than centered coordinate bounds.  When the
    reference cache does not contain explicit axes, use the same convention:
    x in [0, phys_x], y in [0, phys_y].
    attach_evaluation_reference.py will use these axes when resampling Ali-style MATLAB GT.
    """
    if not isinstance(meta, dict):
        meta = {}

    x_axis = _axis_array_from_metadata(meta, ("x_axis_m", "x_axis", "x", "xs", "x_coords"), nx)
    y_axis = _axis_array_from_metadata(meta, ("y_axis_m", "y_axis", "y", "ys", "y_coords"), ny)
    if x_axis is not None and y_axis is not None:
        return x_axis, y_axis, "reference_metadata_axes"

    for a, b, c, d in [
        ("x_min", "x_max", "y_min", "y_max"),
        ("xmin", "xmax", "ymin", "ymax"),
        ("phys_x_min", "phys_x_max", "phys_y_min", "phys_y_max"),
    ]:
        if all(k in meta for k in (a, b, c, d)):
            x_axis = np.linspace(float(meta[a]), float(meta[b]), int(nx), dtype=np.float64)
            y_axis = np.linspace(float(meta[c]), float(meta[d]), int(ny), dtype=np.float64)
            return x_axis, y_axis, f"reference_metadata_bounds:{a},{b},{c},{d}"

    x_axis = np.linspace(0.0, float(phys_x), int(nx), dtype=np.float64)
    y_axis = np.linspace(0.0, float(phys_y), int(ny), dtype=np.float64)
    return x_axis, y_axis, "zero_to_phys_extent"


def _finite_stats(x: torch.Tensor) -> dict:
    """torch-version-safe stats over finite values only (no torch.nanmin)."""
    x = x.detach()
    m = torch.isfinite(x)
    if not torch.any(m):
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    xf = x[m]
    return {
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()) if xf.numel() > 1 else 0.0,
    }


def inspect_map_file(mat_path: str) -> None:
    """Inspect HDF5 MAT file contents (datasets + shapes)."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)

    log_message(f"\n=== Inspecting file: {mat_path} ===\n")
    with h5py.File(mat_path, "r") as f:
        log_message("Format: HDF5 / MATLAB v7.3\n")
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                log_message(f"[DATASET] {k}")
                log_message(f"  shape: {obj.shape}")
                log_message(f"  dtype: {obj.dtype}")
                if obj.attrs:
                    log_message("  attributes:")
                    for ak, av in obj.attrs.items():
                        log_message(f"    {ak}: {av}")
                log_message(".")


def _load_transducer_xy_from_mat(mat_path: str) -> np.ndarray | None:
    """Return transducer coordinates as an array of shape (N,2), in meters, if present."""
    try:
        with h5py.File(mat_path, "r") as f:
            if "transducerPositionsXY" not in f:
                return None
            xy = np.asarray(f["transducerPositionsXY"], dtype=np.float64)
    except Exception:
        return None

    xy = np.squeeze(xy)
    if xy.ndim != 2:
        return None
    if xy.shape[1] == 2:
        return xy.astype(np.float64, copy=False)
    if xy.shape[0] == 2:
        return xy.T.astype(np.float64, copy=False)
    return None


def estimate_physical_size_from_mat(mat_path: str, margin: float = 0.0) -> tuple[float | None, float | None]:
    """Estimate transducer-position span from transducerPositionsXY if present."""
    xy = _load_transducer_xy_from_mat(mat_path)
    if xy is None:
        return None, None
    phys_x = float(np.max(xy[:, 0]) - np.min(xy[:, 0])) + 2.0 * float(margin)
    phys_y = float(np.max(xy[:, 1]) - np.min(xy[:, 1])) + 2.0 * float(margin)
    return phys_x, phys_y


def estimate_ring_center_radius_from_mat(mat_path: str) -> dict:
    """Estimate center/radius statistics from the MATLAB transducer ring."""
    xy = _load_transducer_xy_from_mat(mat_path)
    if xy is None or xy.shape[0] < 3:
        return {}
    cx = float(np.mean(xy[:, 0]))
    cy = float(np.mean(xy[:, 1]))
    rr = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    return {
        "center_x": cx,
        "center_y": cy,
        "radius_mean": float(np.mean(rr)),
        "radius_median": float(np.median(rr)),
        "radius_min": float(np.min(rr)),
        "radius_max": float(np.max(rr)),
        "radius_std": float(np.std(rr)),
        "n_transducers": int(xy.shape[0]),
    }


def compute_geometric_tof_for_selection(mat_path: str, sample_spec: dict, c_geom: float = 1480.0) -> torch.Tensor | None:
    """Compute geometric source-receiver ToF from MATLAB transducer positions for selected tx/rx ids."""
    xy = _load_transducer_xy_from_mat(mat_path)
    if xy is None:
        return None
    tx_ids = np.asarray(sample_spec.get("tx_ids", []), dtype=np.int64)
    rx_ids = np.asarray(sample_spec.get("rx_ids", []), dtype=np.int64)
    if tx_ids.size == 0 or rx_ids.size == 0:
        return None
    if int(tx_ids.max(initial=0)) >= xy.shape[0] or int(rx_ids.max(initial=0)) >= xy.shape[0]:
        return None
    tx = xy[tx_ids, :]
    rx = xy[rx_ids, :]
    diff = tx[:, None, :] - rx[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return torch.tensor(dist / float(c_geom), dtype=torch.float32)


def _normalize_tof_for_diagnostics(T: torch.Tensor, tof_norm: dict | None) -> torch.Tensor:
    """Return a normalized copy for diagnostics only. Saved tensor remains in physical seconds."""
    if not isinstance(tof_norm, dict):
        return T.clone()
    ttype = str(tof_norm.get("type", "none")).lower()
    eps = float(tof_norm.get("eps", 1e-6))
    out = T.clone()
    finite = torch.isfinite(out)
    if not finite.any():
        return out
    if ttype == "zscore":
        mu = float(tof_norm.get("mean", 0.0))
        sig = float(tof_norm.get("std", 1.0))
        out[finite] = (out[finite] - mu) / (sig + eps)
    elif ttype == "max":
        mx = float(tof_norm.get("max", 1.0))
        out[finite] = out[finite] / (mx + eps)
    return out


def _sorted_index_and_inverse(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ids and inverse permutation to recover original order."""
    ids = np.asarray(ids, dtype=np.int64)
    perm = np.argsort(ids)
    ids_sorted = ids[perm]
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size, dtype=np.int64)
    return ids_sorted, inv_perm


def _evenly_spaced_ids(n_total: int, n_select: int) -> np.ndarray:
    n_total = int(n_total)
    n_select = int(n_select)
    if n_select >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.linspace(0, n_total - 1, n_select, dtype=np.int64)


def _alternating_tx_rx_ids(
    n_total: int,
    n_emitters: int,
    n_receivers: int,
    *,
    emitter_starts_at: int = 0,
    rotate_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select experimental transducers to match the synthetic geometry.

    Synthetic data use uniformly spaced emitters and uniformly spaced receivers
    shifted by half a receiver angular step.  For a 512-element experimental
    ring and a 64x64 synthetic geometry this gives

        tx_ids = 0, 8, 16, ..., 504
        rx_ids = 4, 12, 20, ..., 508

    before any optional integer ``rotate_offset``. This preserves uniform angular
    spacing and the half-step receiver offset used by the synthetic geometry.
    """
    n_total = int(n_total)
    n_emitters = int(n_emitters)
    n_receivers = int(n_receivers)
    if n_total < 2:
        raise ValueError(f"n_total must be >= 2, got {n_total}")
    if n_emitters <= 0 or n_receivers <= 0:
        raise ValueError(f"n_emitters and n_receivers must be positive, got {n_emitters}, {n_receivers}")
    from measured_geometry import require_geometry
    _, measured_order, _, _ = require_geometry(n_emitters, n_receivers)
    if n_total != measured_order.size:
        raise ValueError("RF channel count differs from publication measured geometry")
    return measured_order.copy(), measured_order.copy()

    base = int(emitter_starts_at)
    rot = int(rotate_offset)

    if n_emitters >= n_total:
        if n_emitters != n_total:
            raise ValueError(f"Cannot select n_emitters={n_emitters} from n_total={n_total}")
        tx_ids = np.arange(n_total, dtype=np.int64)
    else:
        tx_float = base + rot + (np.arange(n_emitters, dtype=np.float64) * float(n_total) / float(n_emitters))
        tx_ids = np.mod(np.rint(tx_float).astype(np.int64), n_total)

    if n_receivers >= n_total:
        if n_receivers != n_total:
            raise ValueError(f"Cannot select n_receivers={n_receivers} from n_total={n_total}")
        rx_ids = np.arange(n_total, dtype=np.int64)
    else:
        rx_float = base + rot + ((np.arange(n_receivers, dtype=np.float64) + 0.5) * float(n_total) / float(n_receivers))
        rx_ids = np.mod(np.rint(rx_float).astype(np.int64), n_total)

    if np.unique(tx_ids).size != tx_ids.size:
        raise RuntimeError(f"Emitter selection contains duplicates: {tx_ids.tolist()}")
    if np.unique(rx_ids).size != rx_ids.size:
        raise RuntimeError(f"Receiver selection contains duplicates: {rx_ids.tolist()}")
    # For downsampled synthetic-compatible selections (e.g. 64 from a 512-element
    # ring), transmit and receive pools are separated by a half synthetic step.
    # For full 512x512 experimental use, the same physical elements are used as
    # both transmitters and receivers, so overlap is expected and valid.
    if n_emitters < n_total and n_receivers < n_total and np.intersect1d(tx_ids, rx_ids).size != 0:
        raise RuntimeError("Emitter/receiver sets overlap; selection failed.")

    return tx_ids, rx_ids


def _apply_ring_order_correction(
    tx_ids: np.ndarray,
    rx_ids: np.ndarray,
    *,
    ring_order_mode: str,
    ring_joint_shift: int,
    xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Reorder selected experimental transducer ids to match the synthetic angular convention.

    The synthetic cache orders emitters and receivers by increasing angle:
    angle=0 at positive lateral direction and positive rotation toward positive
    axial direction.  Some experimental RF files store the physical ring in the
    opposite angular direction and with a different start element.  In that case
    the ToF matrix still has shape (emitter, receiver), but the learned inverse
    model sees a rotated/mirrored acquisition.

    This correction changes only the ordering of the selected raw RF channels.
    It does not use the experimental GT.
    """
    tx = np.asarray(tx_ids, dtype=np.int64).copy()
    rx = np.asarray(rx_ids, dtype=np.int64).copy()
    mode = str(ring_order_mode or "synthetic_index").lower().strip()
    shift = int(ring_joint_shift)
    if mode == "measured_geometry":
        if shift:
            raise ValueError("publication measured geometry forbids manual ring shifts")
        if not np.array_equal(tx, rx):
            raise ValueError("publication expects shared transmit/receive elements")
        return tx, rx, {"ring_order_mode": mode, "ring_joint_shift": 0,
                        "ring_order_correction_applied": True,
                        "measured_shared_tx_rx": True}

    derived = None
    if mode in ("geometry_auto", "auto_geometry", "geometry"):
        if xy is None:
            raise RuntimeError(
                "--ring_order_mode geometry_auto requires transducerPositionsXY in the MATLAB RF file."
            )
        derived = _derive_ring_order_from_geometry(tx, rx, xy)
        mode = str(derived["ring_order_mode"])
        shift = (
            int(derived["ring_joint_shift"])
            + int(GEOMETRY_AUTO_SYNTHETIC_PHASE_SHIFT)
            + int(ring_joint_shift)
        )
        log_message(
            "[ToF picker] geometry_auto synthetic phase convention: "
            f"derived_shift={int(derived['ring_joint_shift'])}, "
            f"phase_shift={int(GEOMETRY_AUTO_SYNTHETIC_PHASE_SHIFT)}, "
            f"manual_offset={int(ring_joint_shift)}, applied_shift={int(shift)}"
        )

    if mode in ("synthetic_index", "none", "legacy"):
        applied_mode = "synthetic_index"
    elif mode in ("reverse_both", "reverse"):
        tx = tx[::-1].copy()
        rx = rx[::-1].copy()
        applied_mode = "reverse_both"
    else:
        raise ValueError(
            f"Unknown --ring_order_mode={ring_order_mode!r}. "
            "Expected 'synthetic_index' or 'reverse_both'."
        )

    if shift != 0:
        tx = np.roll(tx, shift)
        rx = np.roll(rx, shift)

    if np.unique(tx).size != tx.size:
        raise RuntimeError(f"Corrected emitter selection contains duplicates: {tx.tolist()}")
    if np.unique(rx).size != rx.size:
        raise RuntimeError(f"Corrected receiver selection contains duplicates: {rx.tolist()}")
    overlap = np.intersect1d(tx, rx)
    full_shared_ring = (
        overlap.size == tx.size
        and overlap.size == rx.size
        and np.array_equal(np.sort(tx), np.sort(rx))
    )
    if overlap.size != 0 and not full_shared_ring:
        raise RuntimeError("Corrected emitter/receiver sets overlap; selection failed.")

    info = {
        "ring_order_mode": applied_mode,
        "ring_joint_shift": int(shift),
        "ring_joint_shift_geometry_auto_phase": int(GEOMETRY_AUTO_SYNTHETIC_PHASE_SHIFT) if derived is not None else 0,
        "ring_joint_shift_manual_offset": int(ring_joint_shift) if derived is not None else 0,
        "ring_order_correction_applied": bool(applied_mode != "synthetic_index" or shift != 0),
        "ring_order_note": (
            "Selected raw RF channel ids were reordered before ToF picking to match the "
            "synthetic angular convention consumed by the trained inverse model."
        ),
    }
    if derived is not None:
        info["ring_order_geometry_auto"] = dict(derived)
    return tx, rx, info


def _wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 2.0 * np.pi)


def _angle_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def _ring_angles_from_xy(xy: np.ndarray, ids: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"transducerPositionsXY must have shape (N,2), got {pts.shape}")
    if ids.size == 0:
        raise ValueError("Cannot derive ring order from an empty id list.")
    if int(ids.min()) < 0 or int(ids.max()) >= pts.shape[0]:
        raise ValueError(f"Selected ids out of range for transducerPositionsXY with N={pts.shape[0]}")
    center = np.mean(pts, axis=0)
    rel = pts[ids, :] - center[None, :]
    return _wrap_angle_rad(np.arctan2(rel[:, 1], rel[:, 0]))


def _derive_ring_order_from_geometry(tx_ids: np.ndarray, rx_ids: np.ndarray, xy: np.ndarray) -> dict:
    """
    Determine ring direction/start from transducerPositionsXY only.

    The target convention is the synthetic one from anatomy.generate_sensor_positions:
    emitters at angles 2*pi*i/Ne and receivers at 2*pi*(i+0.5)/Nr.
    """
    tx0 = np.asarray(tx_ids, dtype=np.int64)
    rx0 = np.asarray(rx_ids, dtype=np.int64)
    if tx0.size != rx0.size:
        raise ValueError("geometry_auto currently expects the selected emitter and receiver lists to have the same length.")
    n = int(tx0.size)
    if n <= 0:
        raise ValueError("geometry_auto requires at least one emitter/receiver.")

    tx_target = _wrap_angle_rad(2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n))
    rx_target = _wrap_angle_rad(2.0 * np.pi * (np.arange(n, dtype=np.float64) + 0.5) / float(n))

    best = None
    for mode in ("synthetic_index", "reverse_both"):
        tx_base = tx0[::-1].copy() if mode == "reverse_both" else tx0.copy()
        rx_base = rx0[::-1].copy() if mode == "reverse_both" else rx0.copy()
        for shift in range(n):
            tx = np.roll(tx_base, shift)
            rx = np.roll(rx_base, shift)
            tx_ang = _ring_angles_from_xy(xy, tx)
            rx_ang = _ring_angles_from_xy(xy, rx)
            tx_err = _angle_diff_rad(tx_ang, tx_target)
            rx_err = _angle_diff_rad(rx_ang, rx_target)
            rms = float(np.sqrt(0.5 * (np.mean(tx_err ** 2) + np.mean(rx_err ** 2))))
            mean_abs = float(0.5 * (np.mean(np.abs(tx_err)) + np.mean(np.abs(rx_err))))
            row = {
                "ring_order_mode": mode,
                "ring_joint_shift": int(shift),
                "angular_rms_rad": rms,
                "angular_rms_deg": float(np.degrees(rms)),
                "angular_mean_abs_rad": mean_abs,
                "angular_mean_abs_deg": float(np.degrees(mean_abs)),
                "tx_first_id_after_order": int(tx[0]),
                "rx_first_id_after_order": int(rx[0]),
                "tx_first_angle_deg": float(np.degrees(tx_ang[0])),
                "rx_first_angle_deg": float(np.degrees(rx_ang[0])),
            }
            if best is None or row["angular_rms_rad"] < best["angular_rms_rad"]:
                best = row

    if best is None:
        raise RuntimeError("geometry_auto could not evaluate any ring-order candidates.")
    log_message(
        "[ToF picker] geometry_auto ring order selected: "
        f"mode={best['ring_order_mode']}, shift={best['ring_joint_shift']}, "
        f"angular_rms={best['angular_rms_deg']:.4f} deg, "
        f"mean_abs={best['angular_mean_abs_deg']:.4f} deg"
    )
    return best


def _load_reference_metadata(data_path: str) -> dict:
    """
    Load metadata from a reference dataset .pt saved by your pipeline (train/test).
    Expected structure: torch.load(path)['metadata'].
    """
    data = robust_torch_load(resolve_path_maybe_project_relative(data_path))
    meta = data.get("metadata", {})
    if not isinstance(meta, dict):
        log_message(f"[tof-extraction] WARNING: metadata in {data_path} is not a dict; ignoring.")
        return {}
    keys = [
        "nx", "ny", "phys_x", "phys_y", "dx", "dy",
        "radius", "n_emitters", "n_receivers", "sos_min", "sos_max",
        "grid_spacing_version", "grid_convention", "tof_norm", "exclude_frac",
    ]
    log_message("[tof-extraction] Reference metadata (selected keys):")
    for k in keys:
        if k in meta:
            log_message(f"  {k} = {meta[k]!r}")
    return meta


def _meta_get(meta: dict, key: str, default):
    v = meta.get(key, default)
    return default if v is None else v


# ----------------------------
# Plotting helpers
# ----------------------------
def _set_sensor_angle_ticks(ax, n_sensors: int, axis: str = "x") -> None:
    n = int(n_sensors)
    if n <= 0:
        return
    tick_pos = np.array([0, n / 4, n / 2, 3 * n / 4], dtype=float)
    tick_lab = ["0", "90", "180", "270"]
    if axis.lower() == "x":
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
    elif axis.lower() == "y":
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)


def plot_tof_matrix(T_meas: np.ndarray, title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    T = np.asarray(T_meas)
    if T.ndim != 2:
        raise ValueError(f"T_meas must be 2D. Got shape={T.shape}")
    Ne, Nr = T.shape

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    _set_sensor_angle_ticks(ax, Nr, axis="x")
    _set_sensor_angle_ticks(ax, Ne, axis="y")
    ax.set_xlabel("Receiver angle (deg)")
    ax.set_ylabel("Emitter angle (deg)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ToF (s)")

    fig.tight_layout()
    log_image(fig)
    plt.close(fig)


def plot_single_emitter_tof(T_meas: np.ndarray, emitter_index: int, title: str) -> None:
    T = np.asarray(T_meas)
    if T.ndim != 2:
        raise ValueError(f"T_meas must be 2D. Got shape={T.shape}")
    Ne, Nr = T.shape
    e = int(emitter_index)
    if not (0 <= e < Ne):
        raise ValueError(f"emitter_index out of range: {e} (Ne={Ne})")

    y = T[e, :].astype(float, copy=False)
    x = np.arange(Nr, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, linewidth=1.5)
    _set_sensor_angle_ticks(ax, Nr, axis="x")
    ax.set_xlabel("Receiver angle (deg)")
    ax.set_ylabel("ToF (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    log_image(fig)
    plt.close(fig)


# ----------------------------
# ToF extraction (SINGLE sample only)
# ----------------------------
def _make_exclusion_mask_for_selection(tx_ids: np.ndarray, rx_ids: np.ndarray, nring: int, exclude_frac: float) -> np.ndarray:
    """Return True for channels that should be excluded by the limited-view rule."""
    from measured_geometry import angular_mask, require_geometry
    _, order, _, _ = require_geometry(len(tx_ids), len(rx_ids))
    if not np.array_equal(np.asarray(tx_ids), order) or not np.array_equal(np.asarray(rx_ids), order):
        raise ValueError("publication RF channels do not match measured geometry order")
    return angular_mask(len(tx_ids), len(rx_ids), exclude_frac) < 0.5
    tx_ids = np.asarray(tx_ids, dtype=np.int64)
    rx_ids = np.asarray(rx_ids, dtype=np.int64)
    out = np.zeros((tx_ids.size, rx_ids.size), dtype=bool)
    if float(exclude_frac) <= 0.0 or int(nring) <= 0:
        return out
    n_excl = int(np.floor(float(nring) * float(exclude_frac) / 2.0))
    for itx, tx in enumerate(tx_ids):
        dist_cw = (rx_ids - int(tx)) % int(nring)
        dist_ccw = (int(tx) - rx_ids) % int(nring)
        dist = np.minimum(dist_cw, dist_ccw)
        out[itx, :] = dist <= n_excl
    return out


def _compute_selected_geometric_tof_from_xy(xy: np.ndarray | None, tx_ids: np.ndarray, rx_ids: np.ndarray, c_geom: float) -> np.ndarray | None:
    """Compute ToF for the selected tx/rx ids using the MATLAB transducer coordinates."""
    if xy is None:
        return None
    tx_ids = np.asarray(tx_ids, dtype=np.int64)
    rx_ids = np.asarray(rx_ids, dtype=np.int64)
    if tx_ids.size == 0 or rx_ids.size == 0:
        return None
    if int(tx_ids.max(initial=0)) >= xy.shape[0] or int(rx_ids.max(initial=0)) >= xy.shape[0]:
        return None
    tx = xy[tx_ids, :]
    rx = xy[rx_ids, :]
    diff = tx[:, None, :] - rx[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return (dist / float(c_geom)).astype(np.float64)


def _selected_chord_lengths_from_xy(xy: np.ndarray | None, tx_ids: np.ndarray, rx_ids: np.ndarray) -> np.ndarray | None:
    """Return full selected source-receiver chord lengths from MATLAB transducer coordinates."""
    if xy is None:
        return None
    tx_ids = np.asarray(tx_ids, dtype=np.int64)
    rx_ids = np.asarray(rx_ids, dtype=np.int64)
    if tx_ids.size == 0 or rx_ids.size == 0:
        return None
    if int(tx_ids.max(initial=0)) >= xy.shape[0] or int(rx_ids.max(initial=0)) >= xy.shape[0]:
        return None
    tx = xy[tx_ids, :]
    rx = xy[rx_ids, :]
    diff = tx[:, None, :] - rx[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float64)


def _segment_circle_intersection_lengths(
    xy: np.ndarray | None,
    tx_ids: np.ndarray,
    rx_ids: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return full chord length and length inside a circular ROI for each selected tx/rx segment."""
    if xy is None:
        return None, None
    tx_ids = np.asarray(tx_ids, dtype=np.int64)
    rx_ids = np.asarray(rx_ids, dtype=np.int64)
    if tx_ids.size == 0 or rx_ids.size == 0:
        return None, None
    if int(tx_ids.max(initial=0)) >= xy.shape[0] or int(rx_ids.max(initial=0)) >= xy.shape[0]:
        return None, None

    p0 = xy[tx_ids, :].astype(np.float64, copy=False)[:, None, :]
    p1 = xy[rx_ids, :].astype(np.float64, copy=False)[None, :, :]
    d = p1 - p0
    seg_len = np.sqrt(np.sum(d * d, axis=-1))

    c = np.asarray(center_xy, dtype=np.float64).reshape(1, 1, 2)
    f = p0 - c
    a = np.sum(d * d, axis=-1)
    b = 2.0 * np.sum(f * d, axis=-1)
    cc = np.sum(f * f, axis=-1) - float(radius) ** 2
    disc = b * b - 4.0 * a * cc

    roi_len = np.zeros_like(seg_len, dtype=np.float64)
    valid = (a > 0.0) & (disc > 0.0)
    if np.any(valid):
        sqrt_disc = np.zeros_like(disc, dtype=np.float64)
        sqrt_disc[valid] = np.sqrt(disc[valid])
        denom = np.where(a > 0.0, 2.0 * a, np.nan)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        lo = np.maximum(np.minimum(t1, t2), 0.0)
        hi = np.minimum(np.maximum(t1, t2), 1.0)
        frac = np.maximum(hi - lo, 0.0)
        roi_len = frac * seg_len

    return seg_len.astype(np.float64), roi_len.astype(np.float64)


def _finite_np_stats(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    xf = arr[finite]
    return {
        "n": int(xf.size),
        "min": float(np.min(xf)),
        "max": float(np.max(xf)),
        "mean": float(np.mean(xf)),
        "std": float(np.std(xf)),
    }


def _ideal_synthetic_ring_total_lengths(
    n_emitters: int,
    n_receivers: int,
    radius_m: float,
) -> np.ndarray:
    """Return ideal-ring chord lengths in the synthetic channel convention."""
    ne = int(n_emitters)
    nr = int(n_receivers)
    radius_m = float(radius_m)
    if ne <= 0 or nr <= 0 or radius_m <= 0.0:
        raise ValueError(
            "Ideal synthetic ring requires positive dimensions and radius; "
            f"got ne={ne}, nr={nr}, radius_m={radius_m}."
        )
    theta_e = 2.0 * np.pi * np.arange(ne, dtype=np.float64) / float(ne)
    theta_r = (
        2.0 * np.pi * np.arange(nr, dtype=np.float64) / float(nr)
        + np.pi / float(nr)
    )
    emitter_xy = radius_m * np.stack([np.cos(theta_e), np.sin(theta_e)], axis=1)
    receiver_xy = radius_m * np.stack([np.cos(theta_r), np.sin(theta_r)], axis=1)
    return np.linalg.norm(
        emitter_xy[:, None, :] - receiver_xy[None, :, :], axis=-1
    ).astype(np.float64)


def _segment_circle_lengths_from_points(
    emitter_xy: np.ndarray,
    receiver_xy: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full/inside-ROI lengths for explicitly supplied emitter/receiver points."""
    p0 = np.asarray(emitter_xy, dtype=np.float64)[:, None, :]
    p1 = np.asarray(receiver_xy, dtype=np.float64)[None, :, :]
    d = p1 - p0
    seg_len = np.sqrt(np.sum(d * d, axis=-1))
    c = np.asarray(center_xy, dtype=np.float64).reshape(1, 1, 2)
    f = p0 - c
    a = np.sum(d * d, axis=-1)
    b = 2.0 * np.sum(f * d, axis=-1)
    cc = np.sum(f * f, axis=-1) - float(radius) ** 2
    disc = b * b - 4.0 * a * cc
    roi_len = np.zeros_like(seg_len)
    valid = (a > 0.0) & (disc > 0.0)
    if np.any(valid):
        sqrt_disc = np.zeros_like(disc)
        sqrt_disc[valid] = np.sqrt(disc[valid])
        denom = np.where(a > 0.0, 2.0 * a, np.nan)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        lo = np.maximum(np.minimum(t1, t2), 0.0)
        hi = np.minimum(np.maximum(t1, t2), 1.0)
        roi_len = np.maximum(hi - lo, 0.0) * seg_len
    return seg_len, roi_len


def _periodic_interp_receiver_axis(
    values: np.ndarray,
    valid: np.ndarray,
    source_angles: np.ndarray,
    target_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validity-aware periodic interpolation of (Ne,Nr) data along receiver angle."""
    val = np.asarray(values, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    src = _wrap_angle_rad(np.asarray(source_angles, dtype=np.float64).reshape(-1))
    dst = _wrap_angle_rad(np.asarray(target_angles, dtype=np.float64).reshape(-1))
    if val.ndim != 2 or ok.shape != val.shape or val.shape[1] != src.size:
        raise ValueError(
            f"Receiver interpolation shape mismatch: values={val.shape}, valid={ok.shape}, "
            f"source_angles={src.shape}."
        )
    order = np.argsort(src)
    src = src[order]
    val = val[:, order]
    ok = ok[:, order]
    src_ext = np.concatenate((src[-1:] - 2.0 * np.pi, src, src[:1] + 2.0 * np.pi))
    val_ext = np.concatenate((val[:, -1:], val, val[:, :1]), axis=1)
    ok_ext = np.concatenate((ok[:, -1:], ok, ok[:, :1]), axis=1)
    right = np.searchsorted(src_ext, dst, side="right")
    right = np.clip(right, 1, src_ext.size - 1)
    left = right - 1
    denom = np.maximum(src_ext[right] - src_ext[left], 1e-12)
    weight = (dst - src_ext[left]) / denom
    out = (1.0 - weight[None, :]) * val_ext[:, left] + weight[None, :] * val_ext[:, right]
    out_valid = ok_ext[:, left] & ok_ext[:, right]
    out[~out_valid] = np.nan
    return out, out_valid


def _robust_delay_slowness_fit(length_m: np.ndarray, tof_s: np.ndarray, iterations: int = 12):
    """Robustly fit tof = delay + slowness * length using Huber IRLS."""
    x = np.asarray(length_m, dtype=np.float64).reshape(-1)
    y = np.asarray(tof_s, dtype=np.float64).reshape(-1)
    use = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x, y = x[use], y[use]
    if x.size < 32 or float(np.std(x)) < 1e-5:
        raise RuntimeError("Insufficient chord-length diversity for joint delay/speed calibration.")
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    weights = np.ones_like(y)
    for _ in range(int(max(1, iterations))):
        sw = np.sqrt(np.maximum(weights, 1e-12))
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        err = y - X @ beta
        center = float(np.median(err))
        mad = float(np.median(np.abs(err - center)))
        scale = max(1.4826 * mad, 1e-10)
        cutoff = 1.5 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(err - center), 1e-30))
    err = y - X @ beta
    return float(beta[0]), float(beta[1]), {
        "count": int(x.size),
        "length_min_m": float(np.min(x)),
        "length_max_m": float(np.max(x)),
        "length_std_m": float(np.std(x)),
        "residual_median_us": float(np.median(err) * 1e6),
        "residual_mad_us": float(np.median(np.abs(err - np.median(err))) * 1e6),
        "residual_rms_us": float(np.sqrt(np.mean(err ** 2)) * 1e6),
    }


def apply_full_domain_tof_adjustment(
    T_meas: torch.Tensor,
    sample_spec: dict,
    *,
    mat_path: str,
    experimental_background_c: float,
    synthetic_reference_c: float,
    hardware_delay_us: float,
    synthetic_ring_radius: float,
    canonicalize_receiver_half_pitch: bool = True,
    calibration_mode: str = "explicit",
    calibration_support_radius: float = 0.060,
    calibration_max_chord_fraction: float = 0.01,
    calibration_min_channels: int = 4096,
    calibration_speed_min: float = 1400.0,
    calibration_speed_max: float = 1600.0,
) -> tuple[torch.Tensor, dict]:
    """Map experimental ToF to the synthetic full-domain geometry.

    This transformation never replaces a measured path with a reference value
    and never clips its residual. It
    subtracts the reference travel time for the actual MATLAB transducer chord,
    removes an explicitly supplied acquisition delay, and adds the reference
    travel time for the ideal synthetic chord.
    """
    if float(experimental_background_c) <= 0.0 or float(synthetic_reference_c) <= 0.0:
        raise ValueError("Full-domain reference sound speeds must be positive.")
    xy = _load_transducer_xy_from_mat(mat_path)
    tx_ids = np.asarray(sample_spec.get("tx_ids", []), dtype=np.int64)
    rx_ids = np.asarray(sample_spec.get("rx_ids", []), dtype=np.int64)
    if xy is None or tx_ids.size == 0 or rx_ids.size == 0:
        raise RuntimeError("Full-domain adjustment requires transducerPositionsXY and ordered tx/rx ids.")

    tx_xy = np.asarray(xy[tx_ids], dtype=np.float64)
    rx_xy = np.asarray(xy[rx_ids], dtype=np.float64)
    actual_length = np.linalg.norm(tx_xy[:, None, :] - rx_xy[None, :, :], axis=-1)
    picked = T_meas.detach().cpu().numpy().astype(np.float64, copy=True)
    valid = np.isfinite(picked)
    delay_s = float(hardware_delay_us) * 1e-6
    fitted_c = float(experimental_background_c)
    calibration_info = {"mode": str(calibration_mode), "applied": False}
    mode = str(calibration_mode).lower()
    if mode == "joint_delay_speed":
        _total, calibration_inside = _segment_circle_lengths_from_points(
            tx_xy, rx_xy, center_xy=(0.0, 0.0), radius=float(calibration_support_radius)
        )
        calibration_fraction = np.divide(calibration_inside, np.maximum(actual_length, 1e-12))
        calibration_use = valid & (calibration_fraction <= float(calibration_max_chord_fraction))
        ncal = int(np.sum(calibration_use))
        if ncal < int(calibration_min_channels):
            raise RuntimeError(
                f"Only {ncal} calibration channels remain outside the anatomy support; "
                f"at least {int(calibration_min_channels)} are required."
            )
        delay_s, slowness_s_per_m, fit_stats = _robust_delay_slowness_fit(
            actual_length[calibration_use], picked[calibration_use]
        )
        if slowness_s_per_m <= 0.0:
            raise RuntimeError(f"Joint calibration produced nonpositive slowness {slowness_s_per_m}.")
        fitted_c = 1.0 / slowness_s_per_m
        if not (float(calibration_speed_min) <= fitted_c <= float(calibration_speed_max)):
            raise RuntimeError(
                f"Joint calibration produced c={fitted_c:.3f} m/s outside the allowed "
                f"[{float(calibration_speed_min):.3f}, {float(calibration_speed_max):.3f}] range."
            )
        calibration_info = {
            "mode": mode,
            "applied": True,
            "support_radius_m": float(calibration_support_radius),
            "max_chord_fraction": float(calibration_max_chord_fraction),
            "candidate_count": ncal,
            "fitted_delay_us": float(delay_s * 1e6),
            "fitted_background_c_mps": float(fitted_c),
            "fit_stats": fit_stats,
        }
    elif mode != "explicit":
        raise ValueError("full-domain calibration mode must be 'explicit' or 'joint_delay_speed'.")

    residual = picked - actual_length / float(fitted_c) - delay_s

    ne, nr = picked.shape
    ideal_length = _ideal_synthetic_ring_total_lengths(ne, nr, float(synthetic_ring_radius))
    receiver_info = {"enabled": False}
    if bool(canonicalize_receiver_half_pitch):
        ring_center = np.mean(xy, axis=0)
        source_rx_angles = _wrap_angle_rad(np.arctan2(
            xy[rx_ids, 1] - ring_center[1], xy[rx_ids, 0] - ring_center[0]
        ))
        target_rx_angles = _wrap_angle_rad(
            2.0 * np.pi * (np.arange(nr, dtype=np.float64) + 0.5) / float(nr)
        )
        residual, valid = _periodic_interp_receiver_axis(
            residual, valid, source_rx_angles, target_rx_angles
        )
        receiver_info = {
            "enabled": True,
            "interpolation": "periodic_linear_validity_aware_residual_seconds",
            "target": "synthetic_half_pitch_receiver_grid",
        }

    adjusted = ideal_length / float(synthetic_reference_c) + residual
    adjusted[~valid] = np.nan
    info = {
        "full_domain_tof_adjustment_enabled": True,
        "experimental_background_c_requested": float(experimental_background_c),
        "experimental_background_c_used": float(fitted_c),
        "synthetic_reference_c": float(synthetic_reference_c),
        "hardware_delay_us_requested": float(hardware_delay_us),
        "hardware_delay_us_used": float(delay_s * 1e6),
        "physical_calibration": calibration_info,
        "synthetic_ring_radius_m": float(synthetic_ring_radius),
        "receiver_grid_canonicalization": receiver_info,
        "measured_channel_count": int(np.sum(valid)),
        "measured_channel_fraction": float(np.mean(valid)),
        "actual_chord_length_stats_m": _finite_np_stats(actual_length),
        "ideal_chord_length_stats_m": _finite_np_stats(ideal_length),
        "physical_residual_stats_s": _finite_np_stats(residual),
        "adjusted_tof_stats_s": _finite_np_stats(adjusted),
        "residual_clipping": False,
        "nonintersecting_channel_replacement": False,
    }
    return torch.from_numpy(adjusted.astype(np.float32)), info


def _fit_structured_timing_calibration(
    residual: np.ndarray,
    selection: np.ndarray,
    *,
    harmonics: int = 3,
    ridge: float = 1e-3,
    robust_iterations: int = 5,
) -> tuple[np.ndarray, dict]:
    """Fit a smooth acquisition-delay map without using the anatomical GT.

    The fitted map contains a global term and low-order circular Fourier terms
    of emitter angle, receiver angle, and their angular separation.  It is fit
    only on finite weak/non-ROI chords. Iteratively reweighted least squares
    prevents isolated picker errors from controlling the calibration.
    """
    r = np.asarray(residual, dtype=np.float64)
    use = np.asarray(selection, dtype=bool) & np.isfinite(r)
    ne, nr = r.shape
    nobs = int(np.sum(use))
    h = int(max(0, harmonics))
    ncoef = 1 + 6 * h
    if nobs < max(ncoef + 1, 16):
        raise RuntimeError(
            f"Structured timing calibration has only {nobs} usable chords; "
            f"at least {max(ncoef + 1, 16)} are required."
        )

    ae = 2.0 * np.pi * np.arange(ne, dtype=np.float64) / float(ne)
    ar = 2.0 * np.pi * (np.arange(nr, dtype=np.float64) + 0.5) / float(nr)
    ee = np.broadcast_to(ae[:, None], (ne, nr))
    rr = np.broadcast_to(ar[None, :], (ne, nr))
    dd = np.angle(np.exp(1j * (rr - ee)))

    cols = [np.ones((ne, nr), dtype=np.float64)]
    for k in range(1, h + 1):
        for angle in (ee, rr, dd):
            cols.extend((np.sin(k * angle), np.cos(k * angle)))
    design_full = np.stack(cols, axis=-1)
    X = design_full[use]
    y = r[use]

    penalty = np.eye(X.shape[1], dtype=np.float64) * float(max(0.0, ridge))
    penalty[0, 0] = 0.0
    weights = np.ones_like(y)
    beta = np.zeros((X.shape[1],), dtype=np.float64)
    for _ in range(int(max(1, robust_iterations))):
        sw = np.sqrt(np.maximum(weights, 1e-12))
        Xw = X * sw[:, None]
        yw = y * sw
        beta = np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)
        err = y - X @ beta
        center = float(np.median(err))
        mad = float(np.median(np.abs(err - center)))
        scale = max(1.4826 * mad, 1e-9)
        huber = 1.5 * scale
        weights = np.minimum(1.0, huber / np.maximum(np.abs(err), 1e-30))

    calibration = np.tensordot(design_full, beta, axes=([-1], [0]))
    before = y
    after = y - calibration[use]
    info = {
        "model": "robust_circular_fourier_emitter_receiver_separation",
        "harmonics": int(h),
        "ridge": float(max(0.0, ridge)),
        "robust_iterations": int(max(1, robust_iterations)),
        "candidate_count": int(nobs),
        "coefficient_count": int(beta.size),
        "global_delay_s": float(beta[0]),
        "global_delay_us": float(beta[0] * 1e6),
        "calibration_stats_s": _finite_np_stats(calibration),
        "weak_residual_before_stats_s": _finite_np_stats(before),
        "weak_residual_after_stats_s": _finite_np_stats(after),
        "weak_rms_before_us": float(np.sqrt(np.mean(before ** 2)) * 1e6),
        "weak_rms_after_us": float(np.sqrt(np.mean(after ** 2)) * 1e6),
    }
    return calibration, info


def apply_roi_tof_adjustment(
    T_meas: torch.Tensor,
    sample_spec: dict,
    *,
    mat_path: str,
    phys_x: float,
    phys_y: float,
    roi_center_x: float,
    roi_center_y: float,
    roi_radius: float,
    roi_coord_frame: str = "target",
    roi_outer_c: float = 1480.0,
    roi_reference_c: float = 1500.0,
    roi_synthetic_c: float = 1500.0,
    roi_min_chord_frac: float = 0.05,
    roi_residual_clip_us: float = 8.0,
    roi_nonintersect_mode: str = "reference",
    roi_timing_bias_mode: str = "none",
    roi_timing_bias_chord_frac_max: float = 0.12,
    roi_timing_bias_min_channels: int = 64,
    roi_timing_bias_harmonics: int = 3,
    roi_timing_bias_ridge: float = 1e-3,
    roi_timing_bias_robust_iterations: int = 5,
    synthetic_ring_radius: float | None = None,
    canonicalize_receiver_half_pitch: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Re-reference measured ToF so residuals emphasize a selected internal circular ROI.

    The MATLAB transducer coordinates are centered around the physical ring.  By default,
    ROI coordinates are supplied in the saved pipeline target frame, x/y in [0, phys].
    """
    xy = _load_transducer_xy_from_mat(mat_path)
    tx_ids = np.asarray(sample_spec.get("tx_ids", []), dtype=np.int64)
    rx_ids = np.asarray(sample_spec.get("rx_ids", []), dtype=np.int64)
    if xy is None or tx_ids.size == 0 or rx_ids.size == 0:
        raise RuntimeError("[tof-extraction] ROI ToF adjustment requires transducerPositionsXY and selected tx/rx ids.")

    frame = str(roi_coord_frame).lower()
    if frame == "target":
        center_xy = (float(roi_center_x) - 0.5 * float(phys_x), float(roi_center_y) - 0.5 * float(phys_y))
    elif frame == "matlab":
        center_xy = (float(roi_center_x), float(roi_center_y))
    else:
        raise ValueError(f"Unsupported roi_coord_frame={roi_coord_frame!r}; use 'target' or 'matlab'.")

    L_total, L_roi = _segment_circle_intersection_lengths(
        xy, tx_ids, rx_ids, center_xy=center_xy, radius=float(roi_radius)
    )
    if L_total is None or L_roi is None:
        raise RuntimeError("[tof-extraction] Could not compute ROI chord lengths from MATLAB transducer geometry.")

    L_out = np.maximum(L_total - L_roi, 0.0)
    frac = np.divide(L_roi, np.maximum(L_total, 1e-12))
    roi_channel = frac >= float(max(0.0, roi_min_chord_frac))

    T_np = T_meas.detach().cpu().numpy().astype(np.float64, copy=True)
    finite = np.isfinite(T_np)
    model_background = L_out / float(roi_outer_c) + L_roi / float(roi_reference_c)
    actual_full_reference = L_total / float(roi_synthetic_c)
    if synthetic_ring_radius is None:
        ring_center = np.mean(xy, axis=0)
        synthetic_ring_radius = float(
            np.mean(np.linalg.norm(xy - ring_center[None, :], axis=1))
        )
        synthetic_radius_source = "fallback_matlab_ring_mean"
    else:
        synthetic_ring_radius = float(synthetic_ring_radius)
        synthetic_radius_source = "reference_training_metadata"
    ideal_total = _ideal_synthetic_ring_total_lengths(
        T_np.shape[0], T_np.shape[1], synthetic_ring_radius
    )
    target_full_reference = ideal_total / float(roi_synthetic_c)
    residual = T_np - model_background

    # A threshold-based RF picker can add a nearly common positive delay that is
    # absent from the synthetic first-arrival ToF.  Estimate it only from valid,
    # weakly ROI-intersecting chords: these retain the acquisition timing while
    # minimizing contamination by the internal anatomy being reconstructed.
    timing_mode = str(roi_timing_bias_mode or "none").lower()
    timing_bias_s = 0.0
    timing_bias_count = 0
    timing_bias_mad_s = float("nan")
    timing_bias_applied = False
    structured_calibration_info = None
    weak_max = max(float(roi_min_chord_frac), float(roi_timing_bias_chord_frac_max))
    timing_selection = finite & (frac <= weak_max) & np.isfinite(residual)
    if timing_mode == "weak_chord_median":
        timing_values = residual[timing_selection]
        timing_bias_count = int(timing_values.size)
        if timing_bias_count >= int(max(1, roi_timing_bias_min_channels)):
            timing_bias_s = float(np.median(timing_values))
            timing_bias_mad_s = float(np.median(np.abs(timing_values - timing_bias_s)))
            residual = residual - timing_bias_s
            timing_bias_applied = True
    elif timing_mode == "structured_weak_chord":
        timing_bias_count = int(np.sum(timing_selection))
        if timing_bias_count >= int(max(1, roi_timing_bias_min_channels)):
            calibration, structured_calibration_info = _fit_structured_timing_calibration(
                residual,
                timing_selection,
                harmonics=int(roi_timing_bias_harmonics),
                ridge=float(roi_timing_bias_ridge),
                robust_iterations=int(roi_timing_bias_robust_iterations),
            )
            residual = residual - calibration
            timing_bias_s = float(structured_calibration_info["global_delay_s"])
            timing_values = residual[timing_selection]
            timing_bias_mad_s = float(
                np.median(np.abs(timing_values - np.median(timing_values)))
            )
            timing_bias_applied = True
    elif timing_mode != "none":
        raise ValueError(
            f"Unsupported roi_timing_bias_mode={roi_timing_bias_mode!r}; "
            "use 'none', 'weak_chord_median', or 'structured_weak_chord'."
        )

    receiver_grid_info = {
        "enabled": False,
        "source": "experimental_receiver_angles",
        "target": "synthetic_half_pitch_receiver_angles",
    }
    picked_target = T_np
    if bool(canonicalize_receiver_half_pitch):
        ring_center = np.mean(xy, axis=0)
        source_rx_angles = _wrap_angle_rad(
            np.arctan2(
                xy[rx_ids, 1] - ring_center[1],
                xy[rx_ids, 0] - ring_center[0],
            )
        )
        target_rx_angles = _wrap_angle_rad(
            2.0 * np.pi * (np.arange(T_np.shape[1], dtype=np.float64) + 0.5)
            / float(T_np.shape[1])
        )
        residual, finite = _periodic_interp_receiver_axis(
            residual, finite, source_rx_angles, target_rx_angles
        )
        picked_target, picked_valid = _periodic_interp_receiver_axis(
            T_np, np.isfinite(T_np), source_rx_angles, target_rx_angles
        )
        finite &= picked_valid

        ne, nr = T_np.shape
        theta_e = 2.0 * np.pi * np.arange(ne, dtype=np.float64) / float(ne)
        theta_r = 2.0 * np.pi * (np.arange(nr, dtype=np.float64) + 0.5) / float(nr)
        emitter_target_xy = synthetic_ring_radius * np.stack(
            [np.cos(theta_e), np.sin(theta_e)], axis=1
        )
        receiver_target_xy = synthetic_ring_radius * np.stack(
            [np.cos(theta_r), np.sin(theta_r)], axis=1
        )
        ideal_total, ideal_roi = _segment_circle_lengths_from_points(
            emitter_target_xy,
            receiver_target_xy,
            center_xy=(0.0, 0.0),
            radius=float(roi_radius),
        )
        target_full_reference = ideal_total / float(roi_synthetic_c)
        frac = np.divide(ideal_roi, np.maximum(ideal_total, 1e-12))
        roi_channel = frac >= float(max(0.0, roi_min_chord_frac))
        receiver_grid_info.update({
            "enabled": True,
            "interpolation": "periodic_linear_validity_aware_physical_residual_seconds",
            "source_angle_min_rad": float(np.min(source_rx_angles)),
            "source_angle_max_rad": float(np.max(source_rx_angles)),
            "target_half_pitch_rad": float(np.pi / float(nr)),
        })

    clip_s = float(max(0.0, roi_residual_clip_us)) * 1e-6
    clipped_count = 0
    if clip_s > 0.0:
        before = residual.copy()
        residual = np.clip(residual, -clip_s, clip_s)
        clipped_count = int(np.sum(np.isfinite(before) & (np.abs(before) > clip_s)))

    adjusted = target_full_reference + residual
    nonintersect = ~roi_channel
    mode = str(roi_nonintersect_mode).lower()
    if mode == "reference":
        adjusted[nonintersect] = target_full_reference[nonintersect]
    elif mode == "nan":
        adjusted[nonintersect] = np.nan
    elif mode == "keep":
        adjusted[nonintersect] = picked_target[nonintersect]
    else:
        raise ValueError(f"Unsupported roi_nonintersect_mode={roi_nonintersect_mode!r}; use reference, nan, or keep.")
    adjusted[~finite] = np.nan

    aux = {
        "roi_tof_adjustment_enabled": True,
        "roi_coord_frame": frame,
        "roi_center_target_m": [float(roi_center_x), float(roi_center_y)] if frame == "target" else None,
        "roi_center_matlab_m": [float(center_xy[0]), float(center_xy[1])],
        "roi_radius_m": float(roi_radius),
        "roi_outer_c": float(roi_outer_c),
        "roi_reference_c": float(roi_reference_c),
        "roi_synthetic_c": float(roi_synthetic_c),
        "roi_synthetic_ring_radius_m": float(synthetic_ring_radius),
        "roi_synthetic_ring_radius_source": synthetic_radius_source,
        "roi_geometry_canonicalization": (
            "Residual computed with actual MATLAB geometry; model-facing baseline "
            "uses the ideal synthetic ring geometry expected by training."
        ),
        "receiver_grid_canonicalization": receiver_grid_info,
        "roi_actual_minus_ideal_full_reference_stats_s": _finite_np_stats(
            actual_full_reference - target_full_reference
        ),
        "roi_min_chord_frac": float(roi_min_chord_frac),
        "roi_residual_clip_us": float(roi_residual_clip_us),
        "roi_residual_clip_mode": "hard_training_matched",
        "roi_residual_clipped_count": int(clipped_count),
        "roi_nonintersect_mode": mode,
        "roi_timing_bias_mode": timing_mode,
        "roi_timing_bias_chord_frac_max": float(weak_max),
        "roi_timing_bias_min_channels": int(roi_timing_bias_min_channels),
        "roi_timing_bias_candidate_count": int(timing_bias_count),
        "roi_timing_bias_applied": bool(timing_bias_applied),
        "roi_timing_bias_s": float(timing_bias_s),
        "roi_timing_bias_us": float(timing_bias_s * 1e6),
        "roi_timing_bias_mad_us": float(timing_bias_mad_s * 1e6),
        "roi_timing_bias_harmonics": int(roi_timing_bias_harmonics),
        "roi_timing_bias_ridge": float(roi_timing_bias_ridge),
        "roi_timing_bias_robust_iterations": int(roi_timing_bias_robust_iterations),
        "roi_structured_timing_calibration": structured_calibration_info,
        "roi_channel_count": int(np.sum(roi_channel)),
        "roi_channel_fraction": float(np.mean(roi_channel)),
        "roi_chord_fraction_stats": _finite_np_stats(frac),
        "roi_length_total_stats_m": _finite_np_stats(L_total),
        "roi_length_inside_stats_m": _finite_np_stats(L_roi),
        "roi_residual_before_clip_stats_s": _finite_np_stats(T_np - model_background),
        "roi_residual_after_clip_stats_s": _finite_np_stats(residual),
        "roi_adjusted_minus_full_reference_stats_s": _finite_np_stats(
            adjusted - target_full_reference
        ),
    }
    return torch.tensor(adjusted.astype(np.float32), dtype=torch.float32), aux


def _matlab_style_time_window(time: np.ndarray, center_tof: np.ndarray, *, twinpre: float, twinpost: float | None) -> np.ndarray:
    """MATLAB-style asymmetric Gaussian window centered at geometric ToF."""
    t = np.asarray(time, dtype=np.float64)[None, :]
    c = np.asarray(center_tof, dtype=np.float64)[:, None]
    pre = max(float(twinpre), 1e-30)
    before = np.maximum(c - t, 0.0) / pre
    if twinpost is None or (not np.isfinite(float(twinpost))):
        after = 0.0
    else:
        post = max(float(twinpost), 1e-30)
        after = np.maximum(t - c, 0.0) / post
    return np.exp(-0.5 * (before + after) ** 2).astype(np.float32)


def _first_threshold_pick(env: np.ndarray, i0: int, thresh_rel: float) -> np.ndarray:
    """Current first-arrival rule: first envelope crossing after i0."""
    nr, nt = env.shape
    first = np.full((nr,), -1, dtype=np.int64)
    if i0 >= nt:
        return first
    env_max = np.max(env[:, i0:], axis=1)
    thr = env_max * float(thresh_rel)
    mask = env[:, i0:] >= thr[:, None]
    has = mask.any(axis=1)
    if np.any(has):
        first[has] = np.argmax(mask[has], axis=1) + int(i0)
    return first


def _windowed_peak_pick(env: np.ndarray, time: np.ndarray, center_tof: np.ndarray, *, twinpre: float, twinpost: float) -> np.ndarray:
    """Peak envelope index inside a finite window around center_tof."""
    nr, nt = env.shape
    first = np.full((nr,), -1, dtype=np.int64)
    t = np.asarray(time, dtype=np.float64)
    center_tof = np.asarray(center_tof, dtype=np.float64)
    for j in range(nr):
        lo = center_tof[j] - float(twinpre)
        hi = center_tof[j] + float(twinpost)
        m = (t >= lo) & (t <= hi)
        if not np.any(m):
            continue
        idxs = np.nonzero(m)[0]
        jj = int(np.argmax(env[j, idxs]))
        first[j] = int(idxs[jj])
    return first


def _windowed_first_threshold_pick(
    env: np.ndarray,
    time: np.ndarray,
    center_tof: np.ndarray,
    *,
    twinpre: float,
    twinpost: float,
    thresh_rel: float,
) -> np.ndarray:
    """
    First envelope threshold crossing inside a finite geometric window.

    The threshold is computed from the maximum envelope value inside the same
    finite window, not from the entire post-skip trace.  This prevents late,
    high-energy reflections from setting the threshold for the first-arrival
    pick.
    """
    nr, nt = env.shape
    first = np.full((nr,), -1, dtype=np.int64)
    t = np.asarray(time, dtype=np.float64)
    center_tof = np.asarray(center_tof, dtype=np.float64)

    for j in range(nr):
        lo = center_tof[j] - float(twinpre)
        hi = center_tof[j] + float(twinpost)
        m = (t >= lo) & (t <= hi)
        if not np.any(m):
            continue
        idxs = np.nonzero(m)[0]
        vals = env[j, idxs]
        vmax = float(np.max(vals)) if vals.size else 0.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            continue
        thr = float(thresh_rel) * vmax
        above = vals >= thr
        if np.any(above):
            first[j] = int(idxs[int(np.argmax(above))])

    return first


def load_qt_malignancy_mat_and_pick_tof_single(
    mat_path: str,
    *,
    n_emitters: int = 32,
    n_receivers: int = 32,
    emitter_starts_at: int = 0,
    rotate_offset: int = 0,
    ring_order_mode: str = "synthetic_index",
    ring_joint_shift: int = 0,
    t_skip: float = 0.0,
    env_smooth: int = 9,
    thresh_rel: float = 0.15,
    subtract_min_per_emitter: bool = True,
    exclude_frac: float = 0.25,
    apply_gaussian_window: bool = True,
    gaussian_scale: float = 0.5,
    bandpass_low: float | None = None,
    bandpass_high: float | None = None,
    c_geom: float = 1480.0,
    geom_window_pre_frac: float = 0.05,
    geom_window_post_frac: float = float("inf"),
    peak_window_pre_frac: float = 0.05,
    peak_window_post_frac: float = 0.15,
    matched_template_count: int = 2048,
    matched_template_pre_samples: int = 48,
    matched_template_post_samples: int = 96,
    matched_template_min_correlation: float = 0.25,
    matched_template_max_lag_samples: int = 12,
) -> tuple[torch.Tensor, dict]:
    """
    Extract five ToF variants from the same RF data.

    Variants:
      1. current_threshold: the existing Hilbert-envelope first-threshold picker.
      2. geom_window_threshold: MATLAB-centered geometric-window threshold picker
         with the possibly one-sided MATLAB-style Gaussian window.
      3. geom_window_peak: envelope peak inside a finite geometric window.
      4. geom_window_first_threshold_finite: first threshold crossing inside a
         finite geometric window to reduce sensitivity to late waveform energy.
      5. geom_window_matched_template: robust self-template normalized
         cross-correlation inside the same finite geometric window.

    The first return value is the current-threshold tensor. The full
    variant dictionary is stored in sample_spec['tof_variants'] for the caller.
    """
    with h5py.File(mat_path, "r") as f:
        if "full_dataset" not in f or "time" not in f:
            raise KeyError(f"Expected keys 'full_dataset' and 'time' in {mat_path}. Found: {list(f.keys())}")

        full_dataset = f["full_dataset"]
        time = np.asarray(f["time"]).reshape(-1).astype(np.float64)
        Ntx, Nrx, Nt = full_dataset.shape
        log_message(f"[ToF picker] full_dataset shape: Ntx={Ntx}, Nrx={Nrx}, Nt={Nt}")

        tx_ids, rx_ids = _alternating_tx_rx_ids(
            Ntx, n_emitters, n_receivers,
            emitter_starts_at=int(emitter_starts_at), rotate_offset=int(rotate_offset)
        )
        xy = _load_transducer_xy_from_mat(mat_path)
        tx_ids_raw = tx_ids.copy()
        rx_ids_raw = rx_ids.copy()
        tx_ids, rx_ids, ring_order_info = _apply_ring_order_correction(
            tx_ids,
            rx_ids,
            ring_order_mode=str(ring_order_mode),
            ring_joint_shift=int(ring_joint_shift),
            xy=xy,
        )
        log_message(f"[ToF picker] emitter_starts_at={int(emitter_starts_at)}, rotate_offset={int(rotate_offset)}")
        log_message(
            f"[ToF picker] ring_order_mode={ring_order_info['ring_order_mode']}, "
            f"ring_joint_shift={ring_order_info['ring_joint_shift']}, "
            f"applied={ring_order_info['ring_order_correction_applied']}"
        )
        log_message(f"[ToF picker] tx_ids_raw (len={tx_ids_raw.size}): {tx_ids_raw.tolist()}")
        log_message(f"[ToF picker] rx_ids_raw (len={rx_ids_raw.size}): {rx_ids_raw.tolist()}")
        log_message(f"[ToF picker] tx_ids_ordered (len={tx_ids.size}): {tx_ids.tolist()}")
        log_message(f"[ToF picker] rx_ids_ordered (len={rx_ids.size}): {rx_ids.tolist()}")
        if tx_ids.size > 1 and rx_ids.size > 1:
            log_message(
                f"[ToF picker] synthetic-matched selection check: "
                f"tx_step={float(np.median(np.diff(np.sort(tx_ids)))):.3f} raw elements, "
                f"rx_step={float(np.median(np.diff(np.sort(rx_ids)))):.3f} raw elements, "
            )
        if Nrx != Ntx:
            log_message(f"[ToF picker] WARNING: Ntx != Nrx ({Ntx} vs {Nrx}); selection assumes a ring.")

        T_geom_np = _compute_selected_geometric_tof_from_xy(xy, tx_ids, rx_ids, c_geom=float(c_geom))
        max_geom_all = float(np.nanmax(T_geom_np)) if T_geom_np is not None else 0.0
        if T_geom_np is None:
            log_message("[ToF picker] WARNING: transducerPositionsXY unavailable; geometric-window variants fall back to current picker.")
        else:
            log_message(
                f"[ToF picker] geometric ToF selected: min={np.nanmin(T_geom_np):.6e}, "
                f"max={np.nanmax(T_geom_np):.6e}, c_geom={float(c_geom):.2f} m/s"
            )

        rx_sorted, inv_rx = _sorted_index_and_inverse(rx_ids)
        i0 = int(np.searchsorted(time, float(t_skip), side="left"))
        i0 = max(0, min(i0, Nt - 1))

        env_smooth = int(max(1, env_smooth))
        if env_smooth > 1:
            kernel = np.ones(env_smooth, dtype=np.float32) / float(env_smooth)
            pad = env_smooth // 2
        else:
            kernel = None
            pad = 0

        bp_b = None; bp_a = None
        if (bandpass_low is not None) and (bandpass_high is not None):
            try:
                import scipy.signal
                lo = float(bandpass_low); hi = float(bandpass_high)
                if Nt > 1:
                    dt_est = float(np.median(np.diff(time)))
                    fs = 1.0 / dt_est
                    if lo < 1e3: lo *= 1e6
                    if hi < 1e3: hi *= 1e6
                    nyq = fs / 2.0
                    if hi >= nyq: hi = 0.99 * nyq
                    if lo <= 0.0: lo = 0.01 * nyq
                    bp_b, bp_a = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype="bandpass")
                    log_message(f"[ToF picker] Applying bandpass filter: low={lo:.2e} Hz, high={hi:.2e} Hz, fs={fs:.2e} Hz")
            except Exception as e:
                log_message(f"[ToF picker] WARNING: could not design bandpass filter: {e}")
                bp_b = None; bp_a = None

        variant_names = [
            "current_threshold", "geom_window_threshold", "geom_window_peak",
            "geom_window_first_threshold_finite", "geom_window_matched_template",
        ]
        T_variants = {name: np.full((tx_ids.size, rx_ids.size), np.nan, dtype=np.float32) for name in variant_names}
        excl = _make_exclusion_mask_for_selection(tx_ids, rx_ids, int(Nrx), float(exclude_frac))
        log_message(
            f"[ToF picker] Limited-view mask: kept={int((~excl).sum())}/{excl.size} "
            f"({float((~excl).mean()):.4f}), per emitter kept={int(np.round((~excl).sum(axis=1).mean()))}, "
            f"excluded={int(np.round(excl.sum(axis=1).mean()))}"
        )

        template_pre = int(max(4, matched_template_pre_samples))
        template_post = int(max(4, matched_template_post_samples))
        template_len = template_pre + template_post + 1
        template_target = int(max(32, matched_template_count))
        template_segments = []

        geom_twinpre = float(geom_window_pre_frac) * max_geom_all
        geom_twinpost = None if (not np.isfinite(float(geom_window_post_frac))) else float(geom_window_post_frac) * max_geom_all
        peak_twinpre = float(peak_window_pre_frac) * max_geom_all
        peak_twinpost = float(peak_window_post_frac) * max_geom_all
        log_message(
            f"[ToF picker] variant windows: geom_threshold twinpre={geom_twinpre:.6e}, "
            f"twinpost={'Inf' if geom_twinpost is None else f'{geom_twinpost:.6e}'}, "
            f"finite_first_threshold_window=[-{peak_twinpre:.6e}, +{peak_twinpost:.6e}], "
            f"peak_window=[-{peak_twinpre:.6e}, +{peak_twinpost:.6e}] around geom ToF"
        )

        for itx, tx in enumerate(tx_ids):
            Xs = np.asarray(full_dataset[tx, rx_sorted, :], dtype=np.float32)
            X = Xs[inv_rx, :]

            if apply_gaussian_window and i0 > 0:
                skip_dur = float(time[i0] - time[0]) if (i0 > 0) else 0.0
                sigma = float(skip_dur) * float(gaussian_scale)
                window = np.ones((Nt,), dtype=np.float32)
                if sigma > 0.0 and skip_dur > 0.0:
                    t_pre = time[:i0]
                    window[:i0] = np.exp(-0.5 * ((t_pre - time[i0]) ** 2) / (sigma ** 2)).astype(np.float32)
                X = X * window[np.newaxis, :]

            if (bp_b is not None) and (bp_a is not None):
                try:
                    import scipy.signal
                    X = scipy.signal.filtfilt(bp_b, bp_a, X, axis=-1)
                except Exception as e:
                    log_message(f"[ToF picker] WARNING: bandpass filtering failed: {e}")

            env = np.abs(hilbert(X, axis=-1)).astype(np.float32)
            if kernel is not None:
                envp = np.pad(env, ((0, 0), (pad, pad)), mode="edge")
                env = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, envp)

            idx_current = _first_threshold_pick(env, i0, thresh_rel)
            if T_geom_np is not None and max_geom_all > 0:
                W = _matlab_style_time_window(time, T_geom_np[itx, :], twinpre=geom_twinpre, twinpost=geom_twinpost)
                idx_geom_thr = _first_threshold_pick(env * W, i0, thresh_rel)
                idx_peak = _windowed_peak_pick(env, time, T_geom_np[itx, :], twinpre=peak_twinpre, twinpost=peak_twinpost)
                idx_geom_first_finite = _windowed_first_threshold_pick(
                    env,
                    time,
                    T_geom_np[itx, :],
                    twinpre=peak_twinpre,
                    twinpost=peak_twinpost,
                    thresh_rel=thresh_rel,
                )
            else:
                idx_geom_thr = idx_current.copy()
                idx_peak = idx_current.copy()
                idx_geom_first_finite = idx_current.copy()

            # Construct a pulse-shape ensemble aligned by the preliminary
            # finite-window onset. This onset is used only for template
            # construction; final arrival times are obtained in a second pass.
            for j in range(rx_ids.size):
                ip = int(idx_geom_first_finite[j])
                if excl[itx, j] or ip < template_pre or ip + template_post >= Nt:
                    continue
                seg = np.asarray(X[j, ip - template_pre:ip + template_post + 1], dtype=np.float64)
                seg = seg - float(np.mean(seg))
                norm = float(np.linalg.norm(seg))
                if not np.isfinite(norm) or norm <= 1e-12:
                    continue
                seg = seg / norm
                # Remove arbitrary polarity before taking a robust median.
                if seg[int(np.argmax(np.abs(seg)))] < 0.0:
                    seg = -seg
                noise_lo = max(i0, ip - 4 * template_pre)
                noise_hi = max(noise_lo + 1, ip - template_pre)
                noise = float(np.median(env[j, noise_lo:noise_hi])) if noise_hi > noise_lo else 0.0
                signal = float(np.max(env[j, ip:ip + template_post + 1]))
                quality = signal / max(noise, 1e-12)
                template_segments.append((quality, seg.astype(np.float32)))

            for name, idx in [
                ("current_threshold", idx_current),
                ("geom_window_threshold", idx_geom_thr),
                ("geom_window_peak", idx_peak),
                ("geom_window_first_threshold_finite", idx_geom_first_finite),
            ]:
                good = idx >= 0
                if np.any(good):
                    T_variants[name][itx, good] = time[idx[good]].astype(np.float32)

            j = np.where(rx_ids == tx)[0]
            if j.size == 1:
                for name in variant_names:
                    T_variants[name][itx, j[0]] = np.nan
            if np.any(excl[itx, :]):
                for name in variant_names:
                    T_variants[name][itx, excl[itx, :]] = np.nan

            if subtract_min_per_emitter:
                for name in variant_names:
                    row = T_variants[name][itx, :]
                    if np.any(np.isfinite(row)):
                        T_variants[name][itx, :] = row - np.nanmin(row)

        if len(template_segments) < 32:
            raise RuntimeError(
                f"Matched-template picker found only {len(template_segments)} usable template traces."
            )
        template_segments.sort(key=lambda item: item[0], reverse=True)
        selected_template_segments = template_segments[:template_target]
        template_stack = np.stack([item[1] for item in selected_template_segments], axis=0).astype(np.float64)
        matched_template = np.median(template_stack, axis=0)
        matched_template -= float(np.mean(matched_template))
        template_norm = float(np.linalg.norm(matched_template))
        if not np.isfinite(template_norm) or template_norm <= 1e-12:
            raise RuntimeError("Matched-template pulse collapsed to zero after robust aggregation.")
        matched_template /= template_norm
        log_message(
            f"[ToF picker] Matched template built from {template_stack.shape[0]} traces; "
            f"length={template_len}, pre={template_pre}, post={template_post}."
        )

        matched_corr = np.full((tx_ids.size, rx_ids.size), np.nan, dtype=np.float32)
        matched_correction_samples = np.full((tx_ids.size, rx_ids.size), np.nan, dtype=np.float32)
        matched_refined_count = 0
        matched_low_confidence_fallback_count = 0
        matched_boundary_fallback_count = 0
        max_lag = int(max(1, matched_template_max_lag_samples))
        for itx, tx in enumerate(tx_ids):
            Xs = np.asarray(full_dataset[tx, rx_sorted, :], dtype=np.float32)
            X = Xs[inv_rx, :]
            if apply_gaussian_window and i0 > 0:
                skip_dur = float(time[i0] - time[0])
                sigma = float(skip_dur) * float(gaussian_scale)
                window = np.ones((Nt,), dtype=np.float32)
                if sigma > 0.0 and skip_dur > 0.0:
                    t_pre = time[:i0]
                    window[:i0] = np.exp(-0.5 * ((t_pre - time[i0]) ** 2) / (sigma ** 2)).astype(np.float32)
                X = X * window[np.newaxis, :]
            if (bp_b is not None) and (bp_a is not None):
                import scipy.signal
                X = scipy.signal.filtfilt(bp_b, bp_a, X, axis=-1)

            for j in range(rx_ids.size):
                preliminary_time = float(T_variants["geom_window_first_threshold_finite"][itx, j])
                if excl[itx, j] or not np.isfinite(preliminary_time):
                    continue
                preliminary_index = int(np.argmin(np.abs(time - preliminary_time)))
                center_start = preliminary_index - template_pre
                first_start = max(0, center_start - max_lag)
                last_start = min(Nt - template_len, center_start + max_lag)
                if last_start < first_start:
                    T_variants["geom_window_matched_template"][itx, j] = np.float32(preliminary_time)
                    matched_boundary_fallback_count += 1
                    continue
                trace_stop = last_start + template_len
                trace = np.asarray(X[j, first_start:trace_stop], dtype=np.float64)
                if trace.size < template_len:
                    T_variants["geom_window_matched_template"][itx, j] = np.float32(preliminary_time)
                    continue
                trace = trace - float(np.mean(trace))
                corr = np.correlate(trace, matched_template, mode="valid")
                energy = np.sqrt(np.convolve(trace * trace, np.ones(template_len), mode="valid"))
                corr_norm = corr / np.maximum(energy, 1e-12)
                score = np.abs(corr_norm)
                q = int(np.argmax(score))
                best = float(score[q])
                matched_corr[itx, j] = best
                if not np.isfinite(best) or best < float(matched_template_min_correlation):
                    T_variants["geom_window_matched_template"][itx, j] = np.float32(preliminary_time)
                    matched_low_confidence_fallback_count += 1
                    continue
                if q == 0 or q == score.size - 1:
                    T_variants["geom_window_matched_template"][itx, j] = np.float32(preliminary_time)
                    matched_boundary_fallback_count += 1
                    continue
                delta = 0.0
                if 0 < q < score.size - 1:
                    ym, y0, yp = float(score[q - 1]), float(score[q]), float(score[q + 1])
                    denom = ym - 2.0 * y0 + yp
                    if abs(denom) > 1e-15:
                        delta = float(np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5))
                pick_index = float(first_start + q + template_pre) + delta
                pick_time = float(np.interp(pick_index, np.arange(Nt, dtype=np.float64), time))
                correction = float(pick_index - preliminary_index)
                if abs(correction) <= float(max_lag) + 0.5:
                    T_variants["geom_window_matched_template"][itx, j] = np.float32(pick_time)
                    matched_correction_samples[itx, j] = np.float32(correction)
                    matched_refined_count += 1
                else:
                    T_variants["geom_window_matched_template"][itx, j] = np.float32(preliminary_time)
                    matched_boundary_fallback_count += 1

        valid_corr = matched_corr[np.isfinite(matched_corr)]
        accepted = np.isfinite(T_variants["geom_window_matched_template"])
        log_message(
            f"[ToF picker] Matched-template valid={int(accepted.sum())}/{int((~excl).sum())}; "
            f"refined={matched_refined_count}, low_confidence_fallback={matched_low_confidence_fallback_count}, "
            f"boundary_fallback={matched_boundary_fallback_count}, max_lag_samples={max_lag}; "
            f"correlation min/median/max="
            f"{float(np.min(valid_corr)):.4f}/{float(np.median(valid_corr)):.4f}/{float(np.max(valid_corr)):.4f}."
        )
        log_message(
            "[ToF picker] Matched-minus-onset correction [samples]: "
            f"{_finite_np_stats(matched_correction_samples)}"
        )

    for name in variant_names:
        T_variants[name] = np.where(np.isinf(T_variants[name]), np.nan, T_variants[name]).astype(np.float32)

    sample_spec = {
        "sample_k": 0,
        "emitter_starts_at": int(emitter_starts_at),
        "rotate_offset": int(rotate_offset),
        "tx_ids_raw": [int(x) for x in tx_ids_raw.tolist()],
        "rx_ids_raw": [int(x) for x in rx_ids_raw.tolist()],
        "tx_ids": [int(x) for x in tx_ids.tolist()],
        "rx_ids": [int(x) for x in rx_ids.tolist()],
        "ring_order": dict(ring_order_info),
        "t_skip": float(t_skip),
        "env_smooth": int(env_smooth),
        "thresh_rel": float(thresh_rel),
        "subtract_min_per_emitter": bool(subtract_min_per_emitter),
        "exclude_frac": float(exclude_frac),
        "apply_gaussian_window": bool(apply_gaussian_window),
        "gaussian_scale": float(gaussian_scale),
        "bandpass_low": None if bandpass_low is None else float(bandpass_low),
        "bandpass_high": None if bandpass_high is None else float(bandpass_high),
        "c_geom": float(c_geom),
        "geom_window_pre_frac": float(geom_window_pre_frac),
        "geom_window_post_frac": float(geom_window_post_frac),
        "peak_window_pre_frac": float(peak_window_pre_frac),
        "peak_window_post_frac": float(peak_window_post_frac),
        "matched_template_count_requested": int(matched_template_count),
        "matched_template_count_used": int(len(selected_template_segments)),
        "matched_template_pre_samples": int(template_pre),
        "matched_template_post_samples": int(template_post),
        "matched_template_min_correlation": float(matched_template_min_correlation),
        "matched_template_max_lag_samples": int(max_lag),
        "matched_template_accepted_count": int(np.isfinite(T_variants["geom_window_matched_template"]).sum()),
        "matched_template_refined_count": int(matched_refined_count),
        "matched_template_low_confidence_fallback_count": int(matched_low_confidence_fallback_count),
        "matched_template_boundary_fallback_count": int(matched_boundary_fallback_count),
        "matched_template_correction_samples_stats": _finite_np_stats(matched_correction_samples),
        "matched_template_correlation_stats": _finite_np_stats(matched_corr),
    }
    sample_spec["tof_variants"] = {name: torch.tensor(T_variants[name], dtype=torch.float32) for name in variant_names}
    return sample_spec["tof_variants"]["current_threshold"], sample_spec

# ----------------------------
# Save payload + splits
# ----------------------------
    # PROJECT SPATIAL CONVENTION:
    #   Any 2D spatial maps are stored as (nx, ny).
    #   Any 3D spatial tensors are stored as (1, nx, ny).
    #   Transpose to (ny, nx) is allowed ONLY for plotting/display.

def save_single_tof_dataset(
    T_meas_32x32: torch.Tensor,
    *,
    out_path: str,
    nx: int,
    ny: int,
    phys_x: float,
    phys_y: float,
    sos_min: float,
    sos_max: float,
    radius: float,
    n_emitters: int,
    n_receivers: int,
    sample_spec: dict,
    tof_mask: torch.Tensor | None = None,
    tof_norm: dict | None = None,
    reference_metadata: dict | None = None,
    phys_x_est: float | None = None,
    phys_y_est: float | None = None,
    ring_stats: dict | None = None,
    x_axis_m: np.ndarray | None = None,
    y_axis_m: np.ndarray | None = None,
    axis_source: str | None = None,
) -> str:
    """
    Save a single-sample dataset containing ToF ONLY.

    Pipeline contract:
      - extract_measured_tof.py exports measured ToF + metadata only.
      - GT SoS is inserted later by attach_evaluation_reference.py.
      - Stored spatial arrays across the project follow (nx, ny). (Here we store none.)

    Saved dict schema:
      {"tof": (1,Ne,Nr), "metadata": {..., "has_gt": False}}
    """
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if T_meas_32x32.ndim != 2:
        raise ValueError(f"Expected (Ne,Nr) ToF, got shape={tuple(T_meas_32x32.shape)}")

    tof = T_meas_32x32.unsqueeze(0).cpu()

    dx, dy = _dx_dy_from_phys(nx, ny, phys_x, phys_y)
    if x_axis_m is None or y_axis_m is None:
        x_axis_m, y_axis_m, axis_source = _build_physical_axes_for_saved_metadata(
            reference_metadata or {},
            nx=nx,
            ny=ny,
            phys_x=phys_x,
            phys_y=phys_y,
        )
    x_axis_m = np.asarray(x_axis_m, dtype=np.float64).squeeze()
    y_axis_m = np.asarray(y_axis_m, dtype=np.float64).squeeze()
    if x_axis_m.shape != (int(nx),) or y_axis_m.shape != (int(ny),):
        raise ValueError(
            f"Physical axis shape mismatch: x_axis_m={x_axis_m.shape}, y_axis_m={y_axis_m.shape}, "
            f"expected ({int(nx)},), ({int(ny)},)."
        )

    metadata = {
        "nx": int(nx),
        "ny": int(ny),
        "phys_x": float(phys_x),
        "phys_y": float(phys_y),
        "dx": float(dx),
        "dy": float(dy),
        "grid_spacing": (float(dx), float(dy)),
        "grid_spacing_version": int(GRID_SPACING_VERSION),
        "grid_convention": str(GRID_CONVENTION),
        "x_min": float(np.min(x_axis_m)),
        "x_max": float(np.max(x_axis_m)),
        "y_min": float(np.min(y_axis_m)),
        "y_max": float(np.max(y_axis_m)),
        "x_axis_m": torch.tensor(x_axis_m.astype(np.float32)),
        "y_axis_m": torch.tensor(y_axis_m.astype(np.float32)),
        "physical_axis_source": str(axis_source or "unspecified"),

        "radius": float(radius),
        "n_emitters": int(n_emitters),
        "n_receivers": int(n_receivers),
        "sos_min": float(sos_min),
        "sos_max": float(sos_max),

        "source": "QT experimental ToF (Malignancy.mat)",
        "note": "Measured ToF extracted from experimental RF channel signals. No SoS/GT is stored in this file.",
        "N": 1,

        "sample_spec": dict(sample_spec),
        "has_gt": False,
        "tof_units": "seconds",
        "tof_saved_state": "physical_seconds_with_tof_norm_metadata_for_dataset_loader",
    }

    if isinstance(tof_norm, dict):
        metadata["tof_norm"] = dict(tof_norm)
        metadata["note_tof_norm"] = (
            "The saved ToF tensor remains in physical seconds. The copied tof_norm metadata "
            "matches the reference synthetic training cache; PairsCacheDataset applies it at load time."
        )
    else:
        metadata["tof_norm"] = {"type": "none", "eps": 1e-6}
        metadata["note_tof_norm"] = "No reference tof_norm metadata was found; ToF will be loaded without normalization."

    if isinstance(reference_metadata, dict):
        for k in (
            "tof_norm", "exclude_frac", "phys_x", "phys_y", "radius",
            "nx", "ny", "n_emitters", "n_receivers", "sos_water",
        ):
            if k in reference_metadata:
                metadata[f"reference_{k}"] = reference_metadata[k]

    if ring_stats:
        metadata["mat_transducer_ring"] = dict(ring_stats)

    if phys_x_est is not None:
        metadata["phys_x_est_from_mat"] = float(phys_x_est)
    if phys_y_est is not None:
        metadata["phys_y_est_from_mat"] = float(phys_y_est)

    payload = {"tof": tof, "metadata": metadata}
    if tof_mask is not None:
        m = tof_mask.detach().cpu().float() if isinstance(tof_mask, torch.Tensor) else torch.tensor(tof_mask, dtype=torch.float32)
        if m.ndim != 2:
            raise ValueError(f"tof_mask must be 2D (Ne,Nr), got shape={tuple(m.shape)}")
        payload["tof_mask"] = m.numpy().astype(np.float32)

    torch.save(payload, out_path)
    return str(Path(out_path).resolve())


def save_single_splits(out_pt_path: str) -> str:
    p = Path(str(out_pt_path))
    stem = p.with_suffix("") if p.suffix else p
    splits_path = stem.with_name(stem.name + "_splits.pt")
    splits = {"train": [], "val": [], "test": [0], "N": 1}
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(splits, str(splits_path))
    return str(splits_path.resolve())


# ----------------------------
# Main
# ----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract a single experimental ToF matrix and save it as a pipeline-compatible dataset .pt.")

    p.add_argument("--data_path", type=str, required=True,
                   help="Reference synthetic dataset package. Geometry and normalization metadata are read from this file.")
    p.add_argument("--mat_path", type=str, default="documents/Literature/Malignancy.mat",
                   help="Path to Malignancy.mat (MATLAB v7.3 HDF5).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output folder for terminal/HTML/figures. If omitted, uses <parent of --out_pt>/waveform_log.")
    p.add_argument("--out_pt", type=str, default=None,
                   help="Output .pt to save. Default: alongside --data_path as qt_malignancy_T_meas_single.pt")

    p.add_argument("--emitter_starts_at", type=int, default=0, choices=[0, 1],
                   help="0: emitters even / receivers odd. 1: emitters odd / receivers even.")
    p.add_argument("--rotate_offset", type=int, default=0,
                   help="Rotate chosen indices around ring by this offset (mod Ntx).")
    p.add_argument("--ring_order_mode", type=str, default="synthetic_index",
                   choices=["synthetic_index", "reverse_both", "geometry_auto", "measured_geometry"],
                   help=(
                       "Angular ordering of the selected experimental ring ids before ToF picking. "
                       "'synthetic_index' keeps the synthetic index order. 'reverse_both' reverses both "
                       "emitter and receiver selected ids. 'geometry_auto' derives direction and cyclic "
                       "start from transducerPositionsXY by matching the synthetic angular convention."
                   ))
    p.add_argument("--ring_joint_shift", type=int, default=0,
                   help=(
                       "Common cyclic shift applied to both selected emitter and receiver id lists after "
                       "--ring_order_mode. Advanced diagnostic override only; normal experimental runs "
                       "should use --ring_order_mode geometry_auto without this option."
                   ))
    p.add_argument(
        "--canonicalize_receiver_half_pitch",
        action="store_true",
        help=(
            "Interpolate physical experimental ToF residuals from the actual receiver angles "
            "onto the synthetic half-pitch receiver grid before rebuilding model-facing ToF. "
            "Use this when the experiment shares physical Tx/Rx elements but synthetic training "
            "placed receivers at angular indices (i+0.5)/N."
        ),
    )
    p.add_argument("--full_domain_tof_enable", action="store_true",
                   help="Use full-domain geometry re-referencing without ROI erasure or residual clipping.")
    p.add_argument("--full_domain_background_c", type=float, default=1480.0,
                   help="Reference sound speed [m/s] for actual experimental chord geometry.")
    p.add_argument("--full_domain_synthetic_c", type=float, default=1500.0,
                   help="Reference sound speed [m/s] for the ideal synthetic chord geometry.")
    p.add_argument("--full_domain_hardware_delay_us", type=float, default=0.0,
                   help="Explicit acquisition/picker delay [microseconds]; never inferred from anatomy in this mode.")
    p.add_argument("--full_domain_calibration_mode", choices=["explicit", "joint_delay_speed"], default="explicit",
                   help="Use an explicit speed/delay or robustly fit both on paths outside the anatomy support.")
    p.add_argument("--full_domain_calibration_support_radius", type=float, default=0.060,
                   help="Central anatomy radius [m] excluded from physical delay/speed calibration.")
    p.add_argument("--full_domain_calibration_max_chord_fraction", type=float, default=0.01,
                   help="Maximum fraction of a calibration chord allowed inside the anatomy support.")
    p.add_argument("--full_domain_calibration_min_channels", type=int, default=4096)
    p.add_argument("--full_domain_calibration_speed_min", type=float, default=1400.0)
    p.add_argument("--full_domain_calibration_speed_max", type=float, default=1600.0)

    p.add_argument("--t_skip", type=float, default=0.0, help="Seconds to ignore at start of each trace.")
    p.add_argument("--env_smooth", type=int, default=9, help="Envelope moving-average window (>=1).")
    p.add_argument("--thresh_rel", type=float, default=0.15, help="Threshold fraction relative to per-trace max envelope.")
    p.add_argument("--no_subtract_min_per_emitter", action="store_true",
                   help="Disable per-emitter minimum subtraction (otherwise enabled).")
    p.add_argument("--plot_emitter_index", type=int, default=5, help="Emitter index to plot as 1D curve.")

    # Ali et al.-inspired preprocessing options
    p.add_argument("--exclude_frac", type=float, default=0.25,
                   help="Fraction of the ring around each transmitter to exclude from ToF picks (0.25 = exclude nearest 25% of ring).")
    p.add_argument("--no_gaussian_window", action="store_true",
                   help="Disable the single-sided Gaussian window applied before envelope detection.")
    p.add_argument("--gaussian_scale", type=float, default=0.5,
                   help="Scale factor controlling the width of the Gaussian window relative to the t_skip region (higher = wider window).")
    p.add_argument("--bandpass_low", type=float, default=None,
                   help="Lower cutoff frequency of bandpass filter in MHz. Leave unset to disable bandpass filtering.")
    p.add_argument("--bandpass_high", type=float, default=None,
                   help="Upper cutoff frequency of bandpass filter in MHz. Leave unset to disable bandpass filtering.")
    p.add_argument("--nan_fill", type=str, default="max",
                   choices=["max", "median", "row_max", "row_median", "none"],
                   help=("How to fill NaN/masked ToF entries before saving. "
                         "Use 'max' to keep the previous conservative behavior; "
                         "use 'row_max' to preserve row-dependent scale; "
                         "use 'none' only for diagnostic plots, not for reconstruction."))
    p.add_argument("--c_geom", type=float, default=1480.0,
                   help=("Homogeneous sound speed [m/s] used for geometric ToF windowing/diagnostics from "
                         "transducerPositionsXY. Default 1480 follows the Ali et al. experimental initialization."))
    p.add_argument("--allow_missing_tof_norm", action="store_true",
                   help="Allow saving an experimental ToF file when the reference training cache has no tof_norm metadata.")
    p.add_argument("--tof_variant_to_save", type=str, default="current_threshold",
                   choices=["current_threshold", "geom_window_threshold", "geom_window_peak",
                            "geom_window_first_threshold_finite", "geom_window_matched_template"],
                   help="Which extracted ToF variant to save to --out_pt for downstream reconstruction.")
    p.add_argument("--save_all_tof_variants", action="store_true",
                   help="Also save sidecar .pt files for all ToF variants next to --out_pt.")
    p.add_argument("--geom_window_pre_frac", type=float, default=0.05,
                   help="MATLAB-style pre-arrival window width as fraction of max geometric ToF.")
    p.add_argument("--geom_window_post_frac", type=float, default=float('inf'),
                   help="MATLAB-style post-arrival window width as fraction of max geometric ToF. Use inf to match MATLAB.")
    p.add_argument("--peak_window_pre_frac", type=float, default=0.05,
                   help="Peak-picker search window before geometric ToF, as fraction of max geometric ToF.")
    p.add_argument("--peak_window_post_frac", type=float, default=0.15,
                   help="Peak-picker search window after geometric ToF, as fraction of max geometric ToF.")
    p.add_argument("--matched_template_count", type=int, default=2048,
                   help="Maximum number of onset-aligned RF traces used to form the robust pulse template.")
    p.add_argument("--matched_template_pre_samples", type=int, default=48,
                   help="Template samples retained before the preliminary onset.")
    p.add_argument("--matched_template_post_samples", type=int, default=96,
                   help="Template samples retained after the preliminary onset.")
    p.add_argument("--matched_template_min_correlation", type=float, default=0.25,
                   help="Minimum normalized absolute template correlation required for a valid arrival pick.")
    p.add_argument("--matched_template_max_lag_samples", type=int, default=12,
                   help=("Maximum local matched-template correction on either side of the preliminary "
                         "finite-window onset. Boundary solutions fall back to that onset."))
    p.add_argument("--roi_tof_enable", action="store_true",
                   help=(
                       "Apply ROI-aware ToF re-referencing after waveform picking. This removes an estimated "
                       "outer/background propagation contribution and keeps residual ToF mainly associated with "
                       "an internal circular anatomy region."
                   ))
    p.add_argument("--roi_center_x", type=float, default=0.12,
                   help="ROI center x-coordinate [m]. By default interpreted in target/pipeline coordinates.")
    p.add_argument("--roi_center_y", type=float, default=0.12,
                   help="ROI center y-coordinate [m]. By default interpreted in target/pipeline coordinates.")
    p.add_argument("--roi_radius", type=float, default=0.060,
                   help="Internal circular ROI radius [m].")
    p.add_argument("--roi_coord_frame", choices=["target", "matlab"], default="target",
                   help="'target': coordinates are in saved x/y axes [0,phys]. 'matlab': coordinates are in raw MATLAB transducer coordinates.")
    p.add_argument("--roi_outer_c", type=float, default=1480.0,
                   help="Assumed sound speed [m/s] outside the internal ROI for ToF re-referencing.")
    p.add_argument("--roi_reference_c", type=float, default=1500.0,
                   help="Reference sound speed [m/s] inside the ROI before adding residual ToF.")
    p.add_argument("--roi_synthetic_c", type=float, default=1500.0,
                   help="Reference full-chord speed [m/s] used to rebuild the ToF matrix presented to the trained network.")
    p.add_argument("--roi_min_chord_frac", type=float, default=0.05,
                   help="Minimum fraction of a transmitter-receiver chord that must cross the ROI to keep the measured ROI residual.")
    p.add_argument("--roi_residual_clip_us", type=float, default=8.0,
                   help="Clip ROI residuals to +/- this many microseconds. Use 0 to disable clipping.")
    p.add_argument("--roi_nonintersect_mode", choices=["reference", "nan", "keep"], default="reference",
                   help=(
                       "How to handle channels whose chord barely intersects the ROI: "
                       "'reference' sets residual to zero, 'nan' masks them before fill, 'keep' leaves the picked ToF."
                   ))
    p.add_argument("--roi_timing_bias_mode",
                   choices=["none", "weak_chord_median", "structured_weak_chord"],
                   default="none",
                   help=(
                       "Optional experimental RF-picker delay correction applied after ROI background subtraction "
                       "and before the training-matched hard residual clip. 'structured_weak_chord' robustly fits "
                       "smooth emitter, receiver, and angular-separation terms using only weak/non-ROI chords. "
                       "'weak_chord_median' fits only one global delay; 'none' reproduces the uncalibrated baseline."
                   ))
    p.add_argument("--roi_timing_bias_chord_frac_max", type=float, default=0.12,
                   help=(
                       "Maximum ROI chord fraction used to estimate the weak-chord median timing bias. "
                       "The lower bound is --roi_min_chord_frac."
                   ))
    p.add_argument("--roi_timing_bias_min_channels", type=int, default=64,
                   help="Minimum number of weak ROI chords required before applying the estimated timing bias.")
    p.add_argument("--roi_timing_bias_harmonics", type=int, default=3,
                   help="Circular Fourier harmonics for structured timing calibration.")
    p.add_argument("--roi_timing_bias_ridge", type=float, default=1e-3,
                   help="Ridge stabilization for structured timing calibration.")
    p.add_argument("--roi_timing_bias_robust_iterations", type=int, default=5,
                   help="Robust reweighting iterations for structured timing calibration.")
    return p



def _log_tof_quality(tag: str, T: torch.Tensor, eps_seconds: float = 1e-12) -> None:
    finite = torch.isfinite(T)
    n_total = int(T.numel())
    n_fin = int(finite.sum().item())
    n_nan = n_total - n_fin
    if n_fin > 0:
        Tf = T[finite]
        n_zero = int((torch.abs(Tf) <= float(eps_seconds)).sum().item())
        log_message(
            f"[{tag}] finite={n_fin}/{n_total} ({n_fin/max(n_total,1):.4f}), "
            f"nan_or_inf={n_nan}, near_zero(|T|<={eps_seconds:.1e})={n_zero} "
            f"({n_zero/max(n_fin,1):.4f} of finite)"
        )
        log_message(
            f"[{tag}] finite stats: min={Tf.min().item():.6e}, max={Tf.max().item():.6e}, "
            f"mean={Tf.mean().item():.6e}, std={Tf.std(unbiased=False).item() if Tf.numel()>1 else 0.0:.6e}"
        )
    else:
        log_message(f"[{tag}] WARNING: no finite ToF entries found. total={n_total}")


def _fill_missing_tof(T: torch.Tensor, method: str) -> torch.Tensor:
    """
    Fill NaN/inf entries created by limited-view masking or failed picks.

    The network/test pipeline expects finite tensors. The missing entries are
    not physical measurements, so this fill is a numerical compatibility step.
    """
    method = str(method).lower()
    if method == "none":
        return T

    out = T.clone()
    finite = torch.isfinite(out)
    if finite.all():
        return out
    if not finite.any():
        log_message("[tof-extraction] WARNING: cannot fill ToF; no finite entries exist.")
        return out

    if method == "max":
        value = out[finite].max()
        out[~finite] = value
        log_message(f"[tof-extraction] Filled {(~finite).sum().item()} missing ToF values with global max {value.item():.6e}.")
        return out

    if method == "median":
        value = out[finite].median()
        out[~finite] = value
        log_message(f"[tof-extraction] Filled {(~finite).sum().item()} missing ToF values with global median {value.item():.6e}.")
        return out

    if method in ("row_max", "row_median"):
        global_value = out[finite].max() if method == "row_max" else out[finite].median()
        total_filled = 0
        for i in range(out.shape[0]):
            row = out[i]
            row_finite = torch.isfinite(row)
            row_missing = ~row_finite
            if row_missing.any():
                if row_finite.any():
                    value = row[row_finite].max() if method == "row_max" else row[row_finite].median()
                else:
                    value = global_value
                row[row_missing] = value
                total_filled += int(row_missing.sum().item())
        log_message(f"[tof-extraction] Filled {total_filled} missing ToF values using method='{method}'.")
        return out

    raise ValueError(f"Unknown nan_fill method: {method}")


def _log_runtime_context():
    keys = [
        k for k in sorted(_G.__dict__.keys())
        if k.isupper() and not k.startswith("_")
    ]
    log_message("[evaluation] runtime context values:")
    for k in keys:
        v = getattr(_G, k)
        log_message(f"  {k} = {v!r}")

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Resolve output defaults before configuring logs.
    if args.out_pt is None:
        args.out_pt = str(Path(args.data_path).with_name("qt_malignancy_T_meas_single.pt"))
    if args.output_dir is None:
        args.output_dir = str(Path(args.out_pt).parent / "waveform_log")

    os.makedirs(args.output_dir, exist_ok=True)
    set_output_folder(args.output_dir)

    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message("...................")

    log_message("============== runtime context ==============")
    _log_runtime_context()
    log_message("============== End runtime context ==============")
    log_message(".")
    
    log_message(f"[tof-extraction] data_path={args.data_path}")

    args.data_path = resolve_path_maybe_project_relative(args.data_path)
    args.mat_path = resolve_path_maybe_project_relative(args.mat_path)

    log_message(f"[tof-extraction] mat_path={args.mat_path}")
    log_message(f"[tof-extraction] out_pt={args.out_pt}")
    log_message(f"[tof-extraction] output_dir={args.output_dir}")

    t0 = elapsed.time()

    meta = _load_reference_metadata(args.data_path)
    reference_tof_norm = meta.get("tof_norm", None) if isinstance(meta, dict) else None
    if isinstance(reference_tof_norm, dict):
        log_message(f"[tof-extraction] Reference ToF normalization copied from training cache: {reference_tof_norm}")
    else:
        msg = (
            "[tof-extraction] reference dataset has no tof_norm metadata. "
            "Experimental ToF must use the same normalization as the synthetic training cache; "
            "rerun cache creation with tof_norm metadata or pass --allow_missing_tof_norm for diagnostics only."
        )
        if not bool(args.allow_missing_tof_norm):
            raise RuntimeError(msg)
        log_message(f"WARNING: {msg}")

    # Pull key parameters from metadata (with safe fallbacks)
    nx = int(_meta_get(meta, "nx", 128))
    ny = int(_meta_get(meta, "ny", 96))
    phys_x = float(_meta_get(meta, "phys_x", 1.0))
    phys_y = float(_meta_get(meta, "phys_y", 1.0))
    sos_min = float(_meta_get(meta, "sos_min", 1350.0))
    sos_max = float(_meta_get(meta, "sos_max", 1650.0))
    radius = float(_meta_get(meta, "radius", 0.0))
    n_emitters = int(_meta_get(meta, "n_emitters", 32))
    n_receivers = int(_meta_get(meta, "n_receivers", 32))

    dx_calc, dy_calc = _dx_dy_from_phys(nx, ny, phys_x, phys_y)
    x_axis_m, y_axis_m, axis_source = _build_physical_axes_for_saved_metadata(
        meta,
        nx=nx,
        ny=ny,
        phys_x=phys_x,
        phys_y=phys_y,
    )
    dx_meta = meta.get("dx", None)
    dy_meta = meta.get("dy", None)
    if dx_meta is not None and dy_meta is not None:
        try:
            dx_meta = float(dx_meta); dy_meta = float(dy_meta)
            rel_dx = abs(dx_meta - dx_calc) / max(abs(dx_calc), 1e-12)
            rel_dy = abs(dy_meta - dy_calc) / max(abs(dy_calc), 1e-12)
            if (rel_dx > 1e-3) or (rel_dy > 1e-3):
                log_message(
                    f"[tof-extraction] WARNING: dx/dy mismatch vs metadata: "
                    f"dx_meta={dx_meta:.6e}, dx_calc={dx_calc:.6e}, "
                    f"dy_meta={dy_meta:.6e}, dy_calc={dy_calc:.6e}"
                )
        except Exception:
            pass

    log_message(f"[tof-extraction] Using nx={nx}, ny={ny}, phys_x={phys_x}, phys_y={phys_y}")
    log_message(
        f"[tof-extraction] Physical axes for GT alignment: source={axis_source}, "
        f"x=[{float(x_axis_m.min()):.6e}, {float(x_axis_m.max()):.6e}], "
        f"y=[{float(y_axis_m.min()):.6e}, {float(y_axis_m.max()):.6e}]"
    )
    log_message(f"[tof-extraction] Using n_emitters={n_emitters}, n_receivers={n_receivers}, radius={radius}")
    log_message(f"[tof-extraction] Using sos_min={sos_min}, sos_max={sos_max}")
    log_message(
        f"[tof-extraction] Picker settings: t_skip={args.t_skip}, env_smooth={args.env_smooth}, "
        f"thresh_rel={args.thresh_rel}, subtract_min_per_emitter={not args.no_subtract_min_per_emitter}, "
        f"exclude_frac={args.exclude_frac}, gaussian_window={not args.no_gaussian_window}, "
        f"gaussian_scale={args.gaussian_scale}, bandpass_low={args.bandpass_low}, "
        f"bandpass_high={args.bandpass_high}, nan_fill={args.nan_fill}, "
        f"tof_variant_to_save={args.tof_variant_to_save}"
    )
    log_message(
        f"[tof-extraction] Variant settings: geom_window_pre_frac={args.geom_window_pre_frac}, "
        f"geom_window_post_frac={args.geom_window_post_frac}, peak_window_pre_frac={args.peak_window_pre_frac}, "
        f"peak_window_post_frac={args.peak_window_post_frac}, c_geom={args.c_geom}"
    )
    log_message(".")

    inspect_map_file(args.mat_path)

    T_meas, sample_spec = load_qt_malignancy_mat_and_pick_tof_single(
        args.mat_path,
        n_emitters=n_emitters,
        n_receivers=n_receivers,
        emitter_starts_at=int(args.emitter_starts_at),
        rotate_offset=int(args.rotate_offset),
        ring_order_mode=str(args.ring_order_mode),
        ring_joint_shift=int(args.ring_joint_shift),
        t_skip=float(args.t_skip),
        env_smooth=int(args.env_smooth),
        thresh_rel=float(args.thresh_rel),
        subtract_min_per_emitter=not bool(args.no_subtract_min_per_emitter),
        exclude_frac=float(args.exclude_frac),
        apply_gaussian_window=(not args.no_gaussian_window),
        gaussian_scale=float(args.gaussian_scale),
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        c_geom=float(args.c_geom),
        geom_window_pre_frac=float(args.geom_window_pre_frac),
        geom_window_post_frac=float(args.geom_window_post_frac),
        peak_window_pre_frac=float(args.peak_window_pre_frac),
        peak_window_post_frac=float(args.peak_window_post_frac),
        matched_template_count=int(args.matched_template_count),
        matched_template_pre_samples=int(args.matched_template_pre_samples),
        matched_template_post_samples=int(args.matched_template_post_samples),
        matched_template_min_correlation=float(args.matched_template_min_correlation),
        matched_template_max_lag_samples=int(args.matched_template_max_lag_samples),
    )

    tof_variants = sample_spec.get("tof_variants", {})
    if isinstance(tof_variants, dict) and len(tof_variants) > 0:
        log_message("[tof-extraction] Extracted ToF variants from the same RF data:")
        for _name, _T in tof_variants.items():
            _log_tof_quality(f"tof_extraction/variant/{_name}/before_fill", _T)
            try:
                plot_tof_matrix(_T.cpu().numpy(), title=f"ToF variant before fill: {_name}")
            except Exception as e:
                log_message(f"[tof-extraction] WARNING: could not plot ToF variant {_name}: {e}")
        _chosen = str(args.tof_variant_to_save)
        if _chosen not in tof_variants:
            raise ValueError(f"Requested --tof_variant_to_save={_chosen}, but available variants are {list(tof_variants.keys())}")
        T_meas = tof_variants[_chosen].clone()
        sample_spec["tof_variant_saved"] = _chosen
        log_message(f"[tof-extraction] Selected ToF variant for --out_pt: {_chosen}")
        sample_spec.pop("tof_variants", None)
    else:
        log_message("[tof-extraction] WARNING: no ToF variants returned; using legacy T_meas.")

    phys_x_est, phys_y_est = estimate_physical_size_from_mat(args.mat_path, margin=0.005)
    if (phys_x_est is not None) and (phys_y_est is not None):
        log_message(f"Estimated phys_x, phys_y from Malignancy.mat transducer span + margin: {(phys_x_est, phys_y_est)}")
    else:
        log_message("Could not estimate phys_x/phys_y from mat (transducerPositionsXY missing).")

    ring_stats = estimate_ring_center_radius_from_mat(args.mat_path)
    if ring_stats:
        log_message(f"[tof-extraction] MATLAB transducer ring stats: {ring_stats}")
        if radius > 0.0:
            rel_r = abs(float(ring_stats.get("radius_mean", radius)) - float(radius)) / max(float(radius), 1e-12)
            if rel_r > 0.02:
                log_message(
                    f"[tof-extraction] WARNING: metadata radius={radius:.6e} differs from MATLAB mean ring radius "
                    f"{ring_stats.get('radius_mean', float('nan')):.6e} by {100.0*rel_r:.2f}%."
                )
    else:
        log_message("[tof-extraction] Could not estimate MATLAB ring radius from transducerPositionsXY.")

    log_message(f"T_meas shape: {tuple(T_meas.shape)}, dtype: {T_meas.dtype}")
    _log_tof_quality("tof_extraction/raw_picked_before_fill", T_meas)

    if bool(getattr(args, "full_domain_tof_enable", False)) and bool(getattr(args, "roi_tof_enable", False)):
        raise ValueError("--full_domain_tof_enable and --roi_tof_enable are mutually exclusive.")

    if bool(getattr(args, "full_domain_tof_enable", False)):
        log_message(
            "[tof-extraction] Full-domain ToF adjustment enabled: "
            f"experimental_c={float(args.full_domain_background_c):.6g}, "
            f"synthetic_c={float(args.full_domain_synthetic_c):.6g}, "
            f"hardware_delay_us={float(args.full_domain_hardware_delay_us):.6g}, "
            f"half_pitch={bool(args.canonicalize_receiver_half_pitch)}"
        )
        T_meas, full_spec = apply_full_domain_tof_adjustment(
            T_meas,
            sample_spec,
            mat_path=args.mat_path,
            experimental_background_c=float(args.full_domain_background_c),
            synthetic_reference_c=float(args.full_domain_synthetic_c),
            hardware_delay_us=float(args.full_domain_hardware_delay_us),
            synthetic_ring_radius=float(radius),
            canonicalize_receiver_half_pitch=bool(args.canonicalize_receiver_half_pitch),
            calibration_mode=str(args.full_domain_calibration_mode),
            calibration_support_radius=float(args.full_domain_calibration_support_radius),
            calibration_max_chord_fraction=float(args.full_domain_calibration_max_chord_fraction),
            calibration_min_channels=int(args.full_domain_calibration_min_channels),
            calibration_speed_min=float(args.full_domain_calibration_speed_min),
            calibration_speed_max=float(args.full_domain_calibration_speed_max),
        )
        sample_spec["full_domain_tof_adjustment"] = dict(full_spec)
        sample_spec["tof_variant_saved_raw_picker"] = str(sample_spec.get("tof_variant_saved", "unknown"))
        sample_spec["tof_variant_saved"] = str(sample_spec.get("tof_variant_saved", "unknown")) + "_full_domain_adjusted"
        log_message("[tof-extraction] Full-domain adjustment summary:")
        for k, v in full_spec.items():
            log_message(f"  {k}: {v}")
        _log_tof_quality("tof_extraction/full_domain_adjusted_before_fill", T_meas)

    if bool(getattr(args, "roi_tof_enable", False)):
        log_message(
            "[tof-extraction] ROI-aware ToF adjustment enabled: "
            f"center=({float(args.roi_center_x):.6g},{float(args.roi_center_y):.6g}) m "
            f"frame={args.roi_coord_frame}, radius={float(args.roi_radius):.6g} m, "
            f"outer_c={float(args.roi_outer_c):.3g}, roi_ref_c={float(args.roi_reference_c):.3g}, "
            f"synthetic_c={float(args.roi_synthetic_c):.3g}, nonintersect={args.roi_nonintersect_mode}, "
            f"timing_bias={args.roi_timing_bias_mode}"
        )
        T_meas, roi_spec = apply_roi_tof_adjustment(
            T_meas,
            sample_spec,
            mat_path=args.mat_path,
            phys_x=phys_x,
            phys_y=phys_y,
            roi_center_x=float(args.roi_center_x),
            roi_center_y=float(args.roi_center_y),
            roi_radius=float(args.roi_radius),
            roi_coord_frame=str(args.roi_coord_frame),
            roi_outer_c=float(args.roi_outer_c),
            roi_reference_c=float(args.roi_reference_c),
            roi_synthetic_c=float(args.roi_synthetic_c),
            roi_min_chord_frac=float(args.roi_min_chord_frac),
            roi_residual_clip_us=float(args.roi_residual_clip_us),
            roi_nonintersect_mode=str(args.roi_nonintersect_mode),
            roi_timing_bias_mode=str(args.roi_timing_bias_mode),
            roi_timing_bias_chord_frac_max=float(args.roi_timing_bias_chord_frac_max),
            roi_timing_bias_min_channels=int(args.roi_timing_bias_min_channels),
            roi_timing_bias_harmonics=int(args.roi_timing_bias_harmonics),
            roi_timing_bias_ridge=float(args.roi_timing_bias_ridge),
            roi_timing_bias_robust_iterations=int(args.roi_timing_bias_robust_iterations),
            synthetic_ring_radius=float(radius),
            canonicalize_receiver_half_pitch=bool(args.canonicalize_receiver_half_pitch),
        )
        sample_spec["roi_tof_adjustment"] = dict(roi_spec)
        sample_spec["tof_variant_saved_raw_picker"] = str(sample_spec.get("tof_variant_saved", "unknown"))
        sample_spec["tof_variant_saved"] = str(sample_spec.get("tof_variant_saved", "unknown")) + "_roi_adjusted"
        log_message("[tof-extraction] ROI adjustment summary:")
        for k in [
            "roi_channel_count",
            "roi_channel_fraction",
            "roi_timing_bias_mode",
            "roi_timing_bias_candidate_count",
            "roi_timing_bias_applied",
            "roi_timing_bias_us",
            "roi_timing_bias_mad_us",
            "roi_structured_timing_calibration",
            "roi_residual_clipped_count",
            "roi_chord_fraction_stats",
            "roi_residual_before_clip_stats_s",
            "roi_residual_after_clip_stats_s",
            "roi_adjusted_minus_full_reference_stats_s",
        ]:
            log_message(f"  {k}: {roi_spec.get(k)}")
        _log_tof_quality("tof_extraction/roi_adjusted_before_fill", T_meas)

    # Preserve true measurement availability before fill. This is critical:
    # synthetic limited-view caches are zeroed after normalization through tof_mask.
    # The experimental file therefore stores finite values for compatibility, but also stores tof_mask.
    tof_mask = torch.isfinite(T_meas).float()
    log_message(
        f"[tof-extraction] ToF availability mask before fill: kept={int(tof_mask.sum().item())}/{tof_mask.numel()} "
        f"({float(tof_mask.mean().item()):.4f})"
    )
    try:
        row_counts = tof_mask.sum(dim=1)
        col_counts = tof_mask.sum(dim=0)
        log_message(
            "[tof-extraction] ToF valid-count diagnostic after exclusion/picking: "
            f"per-emitter min/median/max={int(row_counts.min().item())}/"
            f"{float(row_counts.median().item()):.1f}/{int(row_counts.max().item())}; "
            f"per-receiver min/median/max={int(col_counts.min().item())}/"
            f"{float(col_counts.median().item()):.1f}/{int(col_counts.max().item())}"
        )
    except Exception as exc:
        log_message(f"[tof-extraction] WARNING: could not compute ToF valid-count diagnostic: {exc}")

    T_geom = compute_geometric_tof_for_selection(args.mat_path, sample_spec, c_geom=float(args.c_geom))
    if T_geom is not None:
        finite = torch.isfinite(T_meas)
        if finite.any():
            diff = T_meas[finite] - T_geom[finite]
            log_message(
                f"[tof-extraction] Geometric ToF diagnostic using c_geom={float(args.c_geom):.2f} m/s: "
                f"geom_min={T_geom[finite].min().item():.6e}, geom_max={T_geom[finite].max().item():.6e}, "
                f"picked_minus_geom_mean={diff.mean().item():.6e}, "
                f"picked_minus_geom_std={diff.std(unbiased=False).item() if diff.numel()>1 else 0.0:.6e}"
            )
            sample_spec["geom_tof_c_geom"] = float(args.c_geom)
            sample_spec["geom_tof_picked_minus_geom_mean"] = float(diff.mean().item())
            sample_spec["geom_tof_picked_minus_geom_std"] = float(diff.std(unbiased=False).item()) if diff.numel() > 1 else 0.0
    else:
        log_message("[tof-extraction] Geometric ToF diagnostic skipped: transducerPositionsXY unavailable or incompatible.")

    # Plot the raw picked matrix before fill so excluded/missing values are visible.
    try:
        plot_tof_matrix(T_meas.cpu().numpy(), title="Measured ToF matrix before fill (NaN = masked/missing)")
    except Exception as e:
        log_message(f"[tof-extraction] WARNING: could not plot raw ToF matrix before fill: {e}")

    T_meas = _fill_missing_tof(T_meas, args.nan_fill)
    _log_tof_quality("tof_extraction/final_saved_physical_seconds", T_meas)
    if isinstance(reference_tof_norm, dict):
        T_diag_norm = _normalize_tof_for_diagnostics(T_meas, reference_tof_norm) * tof_mask
        _log_tof_quality("tof_extraction/final_loaded_equivalent_after_reference_norm_and_mask", T_diag_norm, eps_seconds=1e-9)

    plot_tof_matrix(T_meas.cpu().numpy(), title=f"Measured ToF matrix saved (nan_fill={args.nan_fill})")
    if args.plot_emitter_index is not None:
        try:
            plot_single_emitter_tof(
                T_meas.cpu().numpy(),
                emitter_index=int(args.plot_emitter_index),
                title=f"Measured ToF (single sample) emitter {int(args.plot_emitter_index)}",
            )
        except Exception as e:
            log_message(f"[tof-extraction] WARNING: could not plot emitter curve: {e}")

    if bool(args.save_all_tof_variants) and isinstance(locals().get("tof_variants", None), dict):
        base = Path(str(args.out_pt))
        stem = base.with_suffix("") if base.suffix else base
        suffix = base.suffix if base.suffix else ".pt"
        for _name, _Traw in tof_variants.items():
            _Tfilled = _fill_missing_tof(_Traw.clone(), args.nan_fill)
            _spec = dict(sample_spec)
            _spec["tof_variant_saved"] = _name
            _side = stem.with_name(stem.name + f"_{_name}").with_suffix(suffix)
            save_single_tof_dataset(
                _Tfilled,
                out_path=str(_side),
                nx=nx,
                ny=ny,
                phys_x=phys_x,
                phys_y=phys_y,
                sos_min=sos_min,
                sos_max=sos_max,
                radius=radius,
                n_emitters=n_emitters,
                n_receivers=n_receivers,
                sample_spec=_spec,
                tof_mask=torch.isfinite(_Traw).float(),
                tof_norm=reference_tof_norm,
                reference_metadata=meta,
                phys_x_est=phys_x_est,
                phys_y_est=phys_y_est,
                ring_stats=ring_stats,
                x_axis_m=x_axis_m,
                y_axis_m=y_axis_m,
                axis_source=axis_source,
            )
            log_message(f"[tof-extraction] Saved sidecar ToF variant {_name}: {_side}")

    out_pt = save_single_tof_dataset(
        T_meas,
        out_path=args.out_pt,
        nx=nx,
        ny=ny,
        phys_x=phys_x,
        phys_y=phys_y,
        sos_min=sos_min,
        sos_max=sos_max,
        radius=radius,
        n_emitters=n_emitters,
        n_receivers=n_receivers,
        sample_spec=sample_spec,
        tof_mask=tof_mask,
        tof_norm=reference_tof_norm,
        reference_metadata=meta,
        phys_x_est=phys_x_est,
        phys_y_est=phys_y_est,
        ring_stats=ring_stats,
        x_axis_m=x_axis_m,
        y_axis_m=y_axis_m,
        axis_source=axis_source,
    )

    out_splits = save_single_splits(out_pt)

    log_message("Saved single-sample dataset:")
    log_message(f"  {out_pt}")
    log_message("Saved splits:")
    log_message(f"  {out_splits}")

    log_message("Sample identification (traceable to Malignancy.mat indices):")
    log_message(
        f"  emitter_starts_at={sample_spec['emitter_starts_at']}, rotate_offset={sample_spec['rotate_offset']}"
    )
    log_message(f"  tx_ids (len={len(sample_spec['tx_ids'])}): {sample_spec['tx_ids']}")
    log_message(f"  rx_ids (len={len(sample_spec['rx_ids'])}): {sample_spec['rx_ids']}")

    t1 = elapsed.time()
    log_message(f"Elapsed Time: {t1 - t0:.2f}s")










