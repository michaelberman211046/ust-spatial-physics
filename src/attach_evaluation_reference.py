# attach_evaluation_reference.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import config
from settings import set_output_folder
from logger import log_message, log_image


from pathlib import Path
import argparse
import warnings
import numpy as np
import torch
import h5py

import matplotlib.pyplot as plt
import runtime_context as _G
import pprint
import sys
from sos_display import set_sos_axes, sos_extent, sos_to_display
from dataset import (
    PairsCacheDataset,
    load_pairs_cache_metadata,
)
# --------------------------------------------------------------------------------------
# Robust torch.load (silent) + path resolution
# --------------------------------------------------------------------------------------

def robust_torch_load(path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            return torch.load(path, map_location='cpu', weights_only=False)


def resolve_path_maybe_project_relative(p: str) -> str:
    pp = Path(p)
    if pp.exists():
        return str(pp)
    proj_root = Path(__file__).resolve().parent.parent
    cand = (proj_root / pp).resolve()
    if cand.exists():
        return str(cand)
    return str(pp)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).squeeze()
    return x.reshape(-1)


def _metadata_array(md: dict, key: str) -> np.ndarray | None:
    if key not in md:
        return None
    value = md[key]
    try:
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        return arr.astype(np.float64).squeeze()
    except Exception as exc:
        log_message(f"[axes] WARNING: could not parse metadata key {key!r}: {exc}")
        return None


def load_mat_vars(mat_path: str) -> dict:
    """
    Load required variables from a MATLAB -v7.3 (.mat = HDF5).
    """
    wanted = ["VEL_ESTIM_ITER", "niterSoSPerFreq", "niterAttenPerFreq", "xi", "yi"]
    out = {}
    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        for k in wanted:
            if k in f:
                out[k] = np.array(f[k])

        if "VEL_ESTIM_ITER" not in out:
            raise KeyError(f"VEL_ESTIM_ITER not found in {mat_path}. Keys: {keys}")

    log_message(f"VEL_ESTIM_ITER.shape = {out['VEL_ESTIM_ITER'].shape}")
    if "xi" in out:
        log_message(f"xi.shape = {out['xi'].shape}")
    if "yi" in out:
        log_message(f"yi.shape = {out['yi'].shape}")
    if "niterSoSPerFreq" in out:
        log_message(f"niterSoSPerFreq.shape = {out['niterSoSPerFreq'].shape}")
    if "niterAttenPerFreq" in out:
        log_message(f"niterAttenPerFreq.shape = {out['niterAttenPerFreq'].shape}")

    return out


def infer_niter_from_matvars(mat: dict) -> int | None:
    if "niterSoSPerFreq" in mat and "niterAttenPerFreq" in mat:
        n_sos = int(_as_1d(mat["niterSoSPerFreq"]).sum())
        n_att = int(_as_1d(mat["niterAttenPerFreq"]).sum())
        return n_sos + n_att
    return None


def pick_sos_iteration(
    mat: dict,
    iter_index: int = -1,
    vel_key: str = "VEL_ESTIM_ITER",
) -> np.ndarray:
    """
    Pick the SoS (velocity) map for a requested iteration.

    MATLAB stores:
        VEL_ESTIM_ITER(:,:,iter) = VEL_ESTIM;   % iter is 1-based in MATLAB

    iter_index:
      - negative: Python indexing from the end
      - non-negative: Python 0-based index (MATLAB iter k => iter_index=k-1)

    Returns:
      2D array (as stored by the MAT reader; orientation handled later by this script).
    """
    vel = np.asarray(mat[vel_key])
    if vel.ndim == 2:
        return vel
    if vel.ndim != 3:
        raise ValueError(f"{vel_key} must be 3D, got shape={vel.shape}")

    niter_expected = infer_niter_from_matvars(mat)
    shape = vel.shape

    iter_axes = []
    if niter_expected is not None:
        iter_axes = [ax for ax, s in enumerate(shape) if s == niter_expected]

    if not iter_axes:
        # Conservative fallback for files without recognized waveform-inversion fields.
        # detect (niter, ny, nx) vs (ny, nx, niter) when ny==nx
        if shape[0] != shape[1] and shape[1] == shape[2]:
            iter_axes = [0]
        elif shape[2] != shape[0] and shape[0] == shape[1]:
            iter_axes = [2]
        else:
            raise ValueError(f"Cannot determine iteration axis for {vel_key} shape={shape}")

    if len(iter_axes) != 1:
        raise ValueError(f"Ambiguous iteration axis candidates {iter_axes} for shape={shape}")

    iter_ax = iter_axes[0]
    niter_available = shape[iter_ax]

    idx = (niter_available + iter_index) if iter_index < 0 else iter_index
    if not (0 <= idx < niter_available):
        raise IndexError(
            f"Requested iter_index={iter_index} -> idx={idx} out of range: "
            f"0..{niter_available - 1} (niter_available={niter_available})"
        )

    sos2 = np.take(vel, indices=idx, axis=iter_ax)
    if sos2.ndim != 2:
        raise ValueError(f"After slicing iteration idx={idx}, expected 2D, got shape={sos2.shape}")
    return sos2


def to_m_per_s_if_needed(v: np.ndarray) -> np.ndarray:
    """
    If median is small (<20), assume mm/us and convert to m/s by *1000.
    """
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return v
    med = float(np.median(vv))
    if med < 20.0:
        return v * 1000.0
    return v


def build_target_axes_from_metadata(md: dict, xi_src: np.ndarray, yi_src: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Build target (x,y) axes (meters) for the simulation grid.
    Preference:
      1) explicit axes in metadata
      2) explicit bounds in metadata
      3) zero-to-physical extent from phys_x/phys_y
      4) fallback to MATLAB xi/yi extent
    """
    nx = int(md["nx"])
    ny = int(md["ny"])

    # explicit axes
    for kx, ky in [
        ("x_axis_m", "y_axis_m"),
        ("x_axis", "y_axis"),
        ("x", "y"),
        ("xs", "ys"),
        ("x_coords", "y_coords"),
    ]:
        if kx in md and ky in md:
            x = _metadata_array(md, kx)
            y = _metadata_array(md, ky)
            if x is not None and y is not None and x.ndim == 1 and y.ndim == 1 and x.size == nx and y.size == ny:
                return x, y, f"metadata_axes:{kx},{ky}"
            log_message(
                f"[axes] WARNING: metadata axes {kx!r}/{ky!r} have sizes "
                f"{None if x is None else x.shape}/{None if y is None else y.shape}; "
                f"expected ({nx},)/({ny},)."
            )

    # bounds
    bound_keys = [
        ("x_min", "x_max", "y_min", "y_max"),
        ("xmin", "xmax", "ymin", "ymax"),
        ("phys_x_min", "phys_x_max", "phys_y_min", "phys_y_max"),
    ]
    for a, b, c, d in bound_keys:
        if a in md and b in md and c in md and d in md:
            x_min = float(md[a]); x_max = float(md[b])
            y_min = float(md[c]); y_max = float(md[d])
            x = np.linspace(x_min, x_max, nx, dtype=np.float64)
            y = np.linspace(y_min, y_max, ny, dtype=np.float64)
            return x, y, f"metadata_bounds:{a},{b},{c},{d}"

    if "phys_x" in md and "phys_y" in md:
        phys_x = float(md["phys_x"])
        phys_y = float(md["phys_y"])
        if phys_x > 0.0 and phys_y > 0.0:
            x = np.linspace(0.0, phys_x, nx, dtype=np.float64)
            y = np.linspace(0.0, phys_y, ny, dtype=np.float64)
            log_message(
                "[axes] WARNING: metadata has no explicit axes/bounds; "
                "using zero-to-physical-extent axes from phys_x/phys_y."
            )
            return x, y, "zero_to_phys_extent"

    log_message(".")
    log_message("Simulation metadata has no physical axes/bounds.")
    log_message("Resampling GT using internal MATLAB layout (ny,nx) for interpolation; saved tensors are converted to (nx,ny).")
    log_message(".")
    
    x = np.linspace(float(xi_src.min()), float(xi_src.max()), nx, dtype=np.float64)
    y = np.linspace(float(yi_src.min()), float(yi_src.max()), ny, dtype=np.float64)
    return x, y, "fallback_matlab_extent"


def validate_axis_overlap(axis: np.ndarray, src_axis: np.ndarray, name: str) -> None:
    axis = np.asarray(axis, dtype=np.float64).squeeze()
    src_axis = np.asarray(src_axis, dtype=np.float64).squeeze()
    lo = float(np.min(axis))
    hi = float(np.max(axis))
    src_lo = float(np.min(src_axis))
    src_hi = float(np.max(src_axis))
    tol = 1e-9 + 0.01 * max(abs(src_hi - src_lo), 1e-12)
    if lo < src_lo - tol or hi > src_hi + tol:
        raise RuntimeError(
            f"Target {name}-axis [{lo:.6e}, {hi:.6e}] lies outside MATLAB {name}-axis "
            f"[{src_lo:.6e}, {src_hi:.6e}]. Check experimental geometry metadata before resampling."
        )


def align_source_axes_to_target_origin(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_tgt: np.ndarray,
    y_tgt: np.ndarray,
    *,
    rel_span_tol: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Keep the synthetic target convention unchanged, but allow MATLAB GT axes
    that describe the same physical field-of-view with a different origin.

    Common case:
      synthetic metadata: x=[0, 0.24], y=[0, 0.24]
      MATLAB xi/yi:       x=[-0.12, 0.12], y=[-0.12, 0.12]

    In that case, shift the MATLAB source axes by +0.12 before interpolation.
    This preserves the synthetic coordinate system while avoiding a false
    out-of-bounds failure.
    """
    x_src = np.asarray(x_src, dtype=np.float64).squeeze()
    y_src = np.asarray(y_src, dtype=np.float64).squeeze()
    x_tgt = np.asarray(x_tgt, dtype=np.float64).squeeze()
    y_tgt = np.asarray(y_tgt, dtype=np.float64).squeeze()

    def _axis_shift(src: np.ndarray, tgt: np.ndarray, axis_name: str) -> tuple[np.ndarray, float, bool, dict]:
        src_min = float(np.min(src))
        src_max = float(np.max(src))
        tgt_min = float(np.min(tgt))
        tgt_max = float(np.max(tgt))
        src_span = src_max - src_min
        tgt_span = tgt_max - tgt_min
        span_rel = abs(src_span - tgt_span) / max(abs(tgt_span), 1e-12)
        src_center = 0.5 * (src_min + src_max)
        tgt_center = 0.5 * (tgt_min + tgt_max)
        shift = tgt_center - src_center
        should_shift = span_rel <= float(rel_span_tol) and abs(shift) > 1e-12
        info = {
            f"{axis_name}_src_min_original": src_min,
            f"{axis_name}_src_max_original": src_max,
            f"{axis_name}_target_min": tgt_min,
            f"{axis_name}_target_max": tgt_max,
            f"{axis_name}_span_relative_difference": float(span_rel),
            f"{axis_name}_source_center_shift_applied": float(shift if should_shift else 0.0),
        }
        if should_shift:
            return src + shift, float(shift), True, info
        return src, 0.0, False, info

    x_aligned, x_shift, x_changed, x_info = _axis_shift(x_src, x_tgt, "x")
    y_aligned, y_shift, y_changed, y_info = _axis_shift(y_src, y_tgt, "y")
    info = {}
    info.update(x_info)
    info.update(y_info)
    info["source_axes_shifted_to_target_origin"] = bool(x_changed or y_changed)

    if x_changed or y_changed:
        log_message(
            "[axes] MATLAB source axes shifted to synthetic target origin before resampling: "
            f"x_shift={x_shift:.6e}, y_shift={y_shift:.6e}. "
            "The saved target axes remain unchanged."
        )
    return x_aligned, y_aligned, info


def log_axis_direction(axis: np.ndarray, name: str) -> dict:
    axis = np.asarray(axis, dtype=np.float64).squeeze()
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name}-axis must be 1D with at least two entries, got shape={axis.shape}")
    diffs = np.diff(axis)
    increasing = bool(np.all(diffs > 0))
    decreasing = bool(np.all(diffs < 0))
    monotone = bool(increasing or decreasing)
    info = {
        f"{name}_first": float(axis[0]),
        f"{name}_last": float(axis[-1]),
        f"{name}_min": float(np.min(axis)),
        f"{name}_max": float(np.max(axis)),
        f"{name}_increasing": increasing,
        f"{name}_decreasing": decreasing,
        f"{name}_strictly_monotone": monotone,
    }
    log_message(
        f"[axes] {name}: first={info[f'{name}_first']:.6e}, last={info[f'{name}_last']:.6e}, "
        f"min={info[f'{name}_min']:.6e}, max={info[f'{name}_max']:.6e}, "
        f"increasing={increasing}, decreasing={decreasing}"
    )
    if not monotone:
        raise RuntimeError(
            f"MATLAB {name}-axis is not strictly monotone. Refusing to resample because orientation is ambiguous."
        )
    return info


def enforce_increasing_source_axes(
    src_yx: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Make interpolation axes increasing and flip the source image consistently.

    This is not a display operation.  It changes only the interpolation layout
    so that searchsorted-based interpolation samples the intended physical
    coordinates.
    """
    src = np.asarray(src_yx, dtype=np.float32)
    x = np.asarray(x_src, dtype=np.float64).squeeze()
    y = np.asarray(y_src, dtype=np.float64).squeeze()
    info = {}
    info.update(log_axis_direction(x, "x_source_before_resample"))
    info.update(log_axis_direction(y, "y_source_before_resample"))

    flipped_x = False
    flipped_y = False
    if x[0] > x[-1]:
        x = x[::-1].copy()
        src = src[:, ::-1].copy()
        flipped_x = True
    if y[0] > y[-1]:
        y = y[::-1].copy()
        src = src[::-1, :].copy()
        flipped_y = True

    info["x_source_reversed_for_resampling"] = bool(flipped_x)
    info["y_source_reversed_for_resampling"] = bool(flipped_y)
    if flipped_x or flipped_y:
        log_message(
            "[axes] Source map was reversed before interpolation so that axes are increasing: "
            f"x_reversed={flipped_x}, y_reversed={flipped_y}."
        )
    info.update(log_axis_direction(x, "x_source_after_resample"))
    info.update(log_axis_direction(y, "y_source_after_resample"))
    return src, x, y, info


def bilinear_resample_on_rect_grid(
    src_yx: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_tgt: np.ndarray,
    y_tgt: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of src(y,x) on (y_src,x_src) to (y_tgt,x_tgt).
    """
    src = np.asarray(src_yx, dtype=np.float64)
    x_src = np.asarray(x_src, dtype=np.float64).squeeze()
    y_src = np.asarray(y_src, dtype=np.float64).squeeze()
    x_tgt = np.asarray(x_tgt, dtype=np.float64).squeeze()
    y_tgt = np.asarray(y_tgt, dtype=np.float64).squeeze()

    if src.ndim != 2:
        raise ValueError(f"src must be 2D, got {src.shape}")

    ny_src, nx_src = src.shape
    if nx_src != x_src.size or ny_src != y_src.size:
        raise ValueError(f"Axis sizes mismatch: src={src.shape}, x_src={x_src.size}, y_src={y_src.size}")

    # Clip targets to source bounds
    x = np.clip(x_tgt, x_src[0], x_src[-1])
    y = np.clip(y_tgt, y_src[0], y_src[-1])

    ix = np.searchsorted(x_src, x, side="right") - 1
    iy = np.searchsorted(y_src, y, side="right") - 1
    ix = np.clip(ix, 0, nx_src - 2)
    iy = np.clip(iy, 0, ny_src - 2)

    x0 = x_src[ix]; x1 = x_src[ix + 1]
    y0 = y_src[iy]; y1 = y_src[iy + 1]
    tx = (x - x0) / (x1 - x0 + 1e-30)
    ty = (y - y0) / (y1 - y0 + 1e-30)

    tx2 = tx[None, :]
    ty2 = ty[:, None]

    IX = ix[None, :].repeat(y.size, axis=0)
    IY = iy[:, None].repeat(x.size, axis=1)

    v00 = src[IY, IX]
    v10 = src[IY, IX + 1]
    v01 = src[IY + 1, IX]
    v11 = src[IY + 1, IX + 1]

    out = (1 - tx2) * (1 - ty2) * v00 + tx2 * (1 - ty2) * v10 + (1 - tx2) * ty2 * v01 + tx2 * ty2 * v11
    return out.astype(np.float32)


def orient_matlab_hdf5_sos_slice_to_yx(sos_raw_2d: np.ndarray, layout: str) -> np.ndarray:
    """
    Convert the selected HDF5-loaded MATLAB velocity slice to image coordinates
    ``(y,x)`` before physical-coordinate resampling.

    MATLAB images are naturally indexed as (row, column) = (y, x).  For v7.3
    MAT files, h5py exposes array dimensions in HDF5 order, and different files
    or loaders may present the selected 2-D slice as either:

      - ``hdf5_yx``: already (y,x), so no transpose is applied.
      - ``hdf5_xy``: (x,y), so a transpose is required to obtain (y,x).

    version 20 keeps the Ali/MATLAB reference convention explicit. The default
    is ``hdf5_xy``, which reproduces the earlier transpose-before-resampling
    behavior that matches the MATLAB reconstruction view. The alternative
    ``hdf5_yx`` remains available for diagnostic comparisons.
    """
    arr = np.asarray(sos_raw_2d, dtype=np.float32)
    key = str(layout).lower()
    if key == "hdf5_yx":
        return arr
    if key == "hdf5_xy":
        return arr.T
    raise ValueError(f"Unknown MATLAB HDF5 slice layout {layout!r}. Expected 'hdf5_yx' or 'hdf5_xy'.")



def build_sos_orientation_variants_xy(sos_xy: np.ndarray, nx: int, ny: int) -> dict[str, torch.Tensor]:
    """
    Build diagnostic GT orientation variants in the pipeline convention (1,nx,ny).

    These variants are saved only for evaluation/diagnostics. They do not change
    the primary saved GT under key ``sos``.  For non-square grids, transpose-
    based variants whose shape no longer matches (nx,ny) are skipped.
    """
    arr = np.asarray(sos_xy, dtype=np.float32)
    expected = (int(nx), int(ny))
    candidates = {
        "as_saved": arr,
        "transpose": arr.T,
        "flip_x": np.flip(arr, axis=0),
        "flip_y": np.flip(arr, axis=1),
        "flip_xy": np.flip(np.flip(arr, axis=0), axis=1),
        "transpose_flip_x": np.flip(arr.T, axis=0),
        "transpose_flip_y": np.flip(arr.T, axis=1),
        "transpose_flip_xy": np.flip(np.flip(arr.T, axis=0), axis=1),
    }
    out: dict[str, torch.Tensor] = {}
    for name, cand in candidates.items():
        cand = np.ascontiguousarray(cand, dtype=np.float32)
        if cand.shape == expected:
            out[name] = torch.tensor(cand, dtype=torch.float32).unsqueeze(0)
    return out


def plot_orientation_diagnostic_xy(
    sos_xy: np.ndarray,
    *,
    nx: int,
    ny: int,
    phys_x: float,
    phys_y: float,
    title: str,
) -> None:
    """
    Plot possible display orientations for manual inspection.

    The primary saved tensor remains key 'sos'.  This diagnostic is deliberately
    visual only; it does not select an orientation by comparing with a model
    prediction or by minimizing an error.
    """
    variants = build_sos_orientation_variants_xy(sos_xy, nx, ny)
    order = [
        "as_saved",
        "transpose",
        "flip_x",
        "flip_y",
        "flip_xy",
        "transpose_flip_x",
        "transpose_flip_y",
        "transpose_flip_xy",
    ]
    names = [name for name in order if name in variants]
    if not names:
        log_message("[gt-orientation] WARNING: no orientation diagnostic variants matched target grid shape.")
        return

    vmin = float(np.nanmin(sos_xy))
    vmax = float(np.nanmax(sos_xy))
    extent = sos_extent(phys_x, phys_y)
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        img_xy = variants[name][0]
        im = ax.imshow(
            sos_to_display(img_xy),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=extent,
            aspect="equal",
        )
        ax.axis("on")
        ax.set_title(name)
        set_sos_axes(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("SoS [m/s]")
    fig.suptitle(title)
    fig.tight_layout()
    log_image(fig)
    plt.close(fig)


def plot_side_by_side(
    left_yx: np.ndarray,
    right_yx: np.ndarray,
    x_left: np.ndarray,
    y_left: np.ndarray,
    x_right: np.ndarray,
    y_right: np.ndarray,
    title_left: str,
    title_right: str,
    suptitle: str,
    out_png: str | None,
):
    """
    Side-by-side debug plot with a *centered* shared colorbar that does not overlap panels.
    Uses physical axes in meters and square aspect.
    """

    left = np.asarray(left_yx)
    right = np.asarray(right_yx)

    vmin = float(np.nanmin([np.nanmin(left), np.nanmin(right)]))
    vmax = float(np.nanmax([np.nanmax(left), np.nanmax(right)]))

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)

    # 3 columns: left | colorbar | right
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.05, 1.0], wspace=0.10)

    axL = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[0, 2])

    imL = axL.imshow(
        left,
        extent=[float(x_left.min()), float(x_left.max()), float(y_left.max()), float(y_left.min())],
        cmap="gray",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    axL.set_title(title_left)
    axL.set_xlabel("Lateral [m]")
    axL.set_ylabel("Axial [m]")

    imR = axR.imshow(
        right,
        extent=[float(x_right.min()), float(x_right.max()), float(y_right.max()), float(y_right.min())],
        cmap="gray",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    axR.set_title(title_right)
    axR.set_xlabel("Lateral [m]")
    axR.set_ylabel("")  # avoid overlap with center colorbar

    # Center colorbar, fit to panels
    cbar = fig.colorbar(imR, cax=cax)
    cbar.set_label("m/s")

    fig.suptitle(suptitle)

    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        log_message(f"[debug] saved: {out_png}")

    log_image(fig)
    plt.close(fig)


def _log_runtime_context():
    keys = [
        k for k in sorted(_G.__dict__.keys())
        if k.isupper() and not k.startswith("_")
    ]
    log_message("[evaluation] runtime context values:")
    for k in keys:
        v = getattr(_G, k)
        log_message(f"  {k} = {v!r}")




# --------------------------------------------------------------------------------------
# Experimental ToF diagnostics against synthetic-loader convention
# --------------------------------------------------------------------------------------

def _to_numpy_float32(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _squeeze_first_tof_matrix(tof_like) -> np.ndarray:
    arr = _to_numpy_float32(tof_like)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            log_message(f"[tof-diagnostic] WARNING: ToF has {arr.shape[0]} samples; using sample 0 for diagnostic.")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected ToF shape (Ne,Nr) or (1,Ne,Nr), got {arr.shape}")
    return arr.astype(np.float32)


def _extract_tof_mask(raw: dict, tof_2d: np.ndarray) -> np.ndarray:
    mask = None
    for key in ("tof_mask", "valid_mask", "tof_valid_mask", "measurement_mask"):
        if key in raw and raw[key] is not None:
            mask = _to_numpy_float32(raw[key])
            break
    if mask is None:
        # Fallback only. Current extract_measured_tof.py should save tof_mask before missing-value fill.
        mask = np.isfinite(tof_2d).astype(np.float32)
        log_message("[tof-diagnostic] WARNING: no saved tof_mask found; falling back to finite(ToF).")
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != tof_2d.shape:
        raise ValueError(f"ToF mask shape {mask.shape} does not match ToF shape {tof_2d.shape}")
    return (mask > 0.5).astype(np.float32)


def _get_tof_norm_from_metadata(md: dict) -> dict:
    tn = md.get("tof_norm", None) if isinstance(md, dict) else None
    if not isinstance(tn, dict):
        tn = md.get("reference_tof_norm", None) if isinstance(md, dict) else None
    if not isinstance(tn, dict):
        tn = {"type": "none", "eps": 1e-6}
    return dict(tn)


def _normalize_tof_like_dataset_loader(tof_2d: np.ndarray, tof_norm: dict) -> np.ndarray:
    tof = np.asarray(tof_2d, dtype=np.float32).copy()
    ttype = str(tof_norm.get("type", "none")).lower()
    eps = float(tof_norm.get("eps", 1e-6))
    if ttype == "zscore":
        mu = float(tof_norm.get("mean", 0.0))
        sig = float(tof_norm.get("std", 1.0))
        return (tof - mu) / (sig + eps)
    if ttype == "max":
        mx = float(tof_norm.get("max", 1.0))
        return tof / (mx + eps)
    return tof


def _safe_stats(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _format_stats(d: dict) -> str:
    return (
        f"n={d['n']} min={d['min']:.6e} max={d['max']:.6e} "
        f"mean={d['mean']:.6e} std={d['std']:.6e}"
    )


def _mean_over_valid(mat: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    m = mask.astype(np.float32)
    num = np.sum(mat * m, axis=axis)
    den = np.sum(m, axis=axis)
    out = np.full_like(num, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=(den > 0))
    return out


def log_experimental_tof_diagnostic(raw: dict, reference_data_path: str | None = None) -> None:
    """
    Diagnostic only. It does not modify the saved ToF or GT.

    The computation mirrors PairsCacheDataset:
      1. ToF is stored in physical seconds.
      2. The stored/reference tof_norm is applied.
      3. The validity mask is applied after normalization.
    """
    if not isinstance(raw, dict) or "tof" not in raw:
        log_message("[tof-diagnostic] skipped: input .pt has no 'tof' key.")
        return

    md = raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), dict) else {}
    tof = _squeeze_first_tof_matrix(raw["tof"])
    mask = _extract_tof_mask(raw, tof)
    tof_norm = _get_tof_norm_from_metadata(md)
    tof_normed = _normalize_tof_like_dataset_loader(tof, tof_norm)
    tof_loaded_equiv = tof_normed * mask
    valid = mask > 0.5
    valid_count = int(valid.sum())
    total_count = int(valid.size)
    variant = "unknown"
    sample_spec = md.get("sample_spec", {}) if isinstance(md.get("sample_spec", {}), dict) else {}
    if "tof_variant_saved" in sample_spec:
        variant = str(sample_spec.get("tof_variant_saved"))

    log_message(".")
    log_message("[tof-diagnostic] Experimental ToF distribution after dataset-loader-equivalent normalization")
    log_message(f"[tof-diagnostic] variant={variant}")
    log_message(f"[tof-diagnostic] validity mask: kept={valid_count}/{total_count} ({valid_count / max(total_count,1):.4f})")
    log_message(f"[tof-diagnostic] tof_norm={tof_norm}")
    log_message(f"[tof-diagnostic] raw seconds valid: {_format_stats(_safe_stats(tof[valid]))}")
    log_message(f"[tof-diagnostic] normalized valid: {_format_stats(_safe_stats(tof_normed[valid]))}")
    log_message(f"[tof-diagnostic] loaded-equivalent all entries: {_format_stats(_safe_stats(tof_loaded_equiv))}")
    if valid_count > 0:
        z = tof_normed[valid]
        log_message(
            "[tof-diagnostic] normalized valid fractions: "
            f"|z|>1={float(np.mean(np.abs(z) > 1.0)):.4f}, "
            f"|z|>2={float(np.mean(np.abs(z) > 2.0)):.4f}, "
            f"|z|>3={float(np.mean(np.abs(z) > 3.0)):.4f}"
        )

    em_mean = _mean_over_valid(tof_normed, mask, axis=1)
    rx_mean = _mean_over_valid(tof_normed, mask, axis=0)
    log_message(f"[tof-diagnostic] emitter mean normalized z: {_format_stats(_safe_stats(em_mean))}")
    log_message(f"[tof-diagnostic] receiver mean normalized z: {_format_stats(_safe_stats(rx_mean))}")

    if reference_data_path is None or str(reference_data_path).strip() == "":
        log_message("[tof-diagnostic] reference_data_path not supplied; synthetic channel-wise comparison skipped.")
        log_message(".")
        return

    ref_path = resolve_path_maybe_project_relative(str(reference_data_path))
    if not Path(ref_path).exists():
        log_message(f"[tof-diagnostic] WARNING: reference_data_path not found: {reference_data_path}")
        log_message(".")
        return
    try:
        ref_md = load_pairs_cache_metadata(
            ref_path
        )

        if (
            not isinstance(ref_md, dict)
            or not ref_md
        ):
            raise ValueError(
                "reference pairs cache "
                "has no metadata"
            )

        ref_ds = PairsCacheDataset(
            ref_path,
            sos_min=float(
                ref_md.get(
                    "sos_min",
                    1400.0,
                )
            ),
            sos_max=float(
                ref_md.get(
                    "sos_max",
                    1650.0,
                )
            ),
            return_tof_mask=True,
        )

    except Exception as exc:
        log_message(
            "[tof-diagnostic] WARNING: "
            "failed to open reference "
            f"pairs cache "
            f"{reference_data_path}: "
            f"{exc}"
        )
        log_message(".")
        return

    ref_mask = (
        np.asarray(
            ref_ds.tof_mask,
            dtype=np.float32,
        )
        if getattr(
            ref_ds,
            "tof_mask",
            None,
        )
        is not None
        else np.ones(
            tof.shape,
            dtype=np.float32,
        )
    )

    if ref_mask.ndim == 3:
        ref_mask = ref_mask[0]

    if ref_mask.shape != tof.shape:
        log_message(
            "[tof-diagnostic] WARNING: "
            f"reference mask shape "
            f"{ref_mask.shape} does not "
            f"match experimental ToF "
            f"shape {tof.shape}; "
            "channel-wise comparison skipped."
        )
        log_message(".")
        return

    ref_valid = ref_mask > 0.5

    ref_norm = getattr(
        ref_ds,
        "tof_norm",
        _get_tof_norm_from_metadata(
            ref_md
        ),
    )

    # Per-channel streaming statistics.
    sum_ch = np.zeros(
        tof.shape,
        dtype=np.float64,
    )

    sumsq_ch = np.zeros(
        tof.shape,
        dtype=np.float64,
    )

    count_ch = np.zeros(
        tof.shape,
        dtype=np.int64,
    )

    # Global statistics for the diagnostic log.
    global_count = 0
    global_sum = 0.0
    global_sumsq = 0.0
    global_min = np.inf
    global_max = -np.inf

    for sample_idx in range(
        len(ref_ds)
    ):
        sample = ref_ds[sample_idx]

        # sample[1] is already normalized and masked by
        # PairsCacheDataset, just like the model sees it.
        ref_tof_i = (
            sample[1]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        sample_valid = (
            ref_valid
            & np.isfinite(
                ref_tof_i
            )
        )

        values = ref_tof_i[
            sample_valid
        ]

        if values.size == 0:
            continue

        sum_ch[
            sample_valid
        ] += values

        sumsq_ch[
            sample_valid
        ] += values * values

        count_ch[
            sample_valid
        ] += 1

        global_count += int(
            values.size
        )

        global_sum += float(
            np.sum(
                values,
                dtype=np.float64,
            )
        )

        global_sumsq += float(
            np.sum(
                values * values,
                dtype=np.float64,
            )
        )

        global_min = min(
            global_min,
            float(
                np.min(values)
            ),
        )

        global_max = max(
            global_max,
            float(
                np.max(values)
            ),
        )

    if global_count <= 0:
        log_message(
            "[tof-diagnostic] WARNING: "
            "reference cache has no finite "
            "valid ToF values."
        )
        log_message(".")
        return

    global_mean = (
        global_sum
        / float(global_count)
    )

    global_variance = max(
        0.0,
        global_sumsq
        / float(global_count)
        - global_mean
        * global_mean,
    )

    global_stats = {
        "n":
            int(global_count),
        "min":
            float(global_min),
        "max":
            float(global_max),
        "mean":
            float(global_mean),
        "std":
            float(
                np.sqrt(
                    global_variance
                )
            ),
    }

    ref_ch_mean = np.full(
        tof.shape,
        np.nan,
        dtype=np.float64,
    )

    ref_ch_std = np.full(
        tof.shape,
        np.nan,
        dtype=np.float64,
    )

    has_samples = count_ch > 0

    ref_ch_mean[
        has_samples
    ] = (
        sum_ch[has_samples]
        / count_ch[has_samples]
    )

    ref_ch_variance = np.zeros(
        tof.shape,
        dtype=np.float64,
    )

    ref_ch_variance[
        has_samples
    ] = np.maximum(
        0.0,
        sumsq_ch[has_samples]
        / count_ch[has_samples]
        - ref_ch_mean[
            has_samples
        ] ** 2,
    )

    ref_ch_std[
        has_samples
    ] = np.sqrt(
        ref_ch_variance[
            has_samples
        ]
    )

    log_message(
        "[tof-diagnostic] synthetic "
        f"reference path: {ref_path}"
    )

    log_message(
        "[tof-diagnostic] synthetic "
        f"reference tof_norm={ref_norm}"
    )

    log_message(
        "[tof-diagnostic] synthetic "
        "normalized valid: "
        f"{_format_stats(global_stats)}"
    )
    finite_std = ref_ch_std[np.isfinite(ref_ch_std) & (ref_ch_std > 0.0)]
    if finite_std.size > 0:
        std_floor = max(1e-3, 0.01 * float(np.median(finite_std)))
    else:
        std_floor = 1e-3

    common_valid = valid & ref_valid & np.isfinite(ref_ch_mean) & np.isfinite(ref_ch_std)
    ch_valid = common_valid & (ref_ch_std > std_floor)
    skipped_low_std = common_valid & ~ch_valid

    log_message(
        f"[tof-diagnostic] experimental vs synthetic channel-z usable channels: "
        f"{int(ch_valid.sum())}/{int(common_valid.sum())}; std_floor={std_floor:.6e}"
    )
    log_message(
        f"[tof-diagnostic] experimental vs synthetic channel-z skipped channels due to "
        f"near-zero/nonfinite synthetic std: {int(skipped_low_std.sum())}/{int(common_valid.sum())}"
    )

    if int(ch_valid.sum()) == 0:
        log_message(
            "[tof-diagnostic] channel-wise comparison skipped: "
            "no common valid channels with sufficiently nonzero synthetic std."
        )
        log_message(".")
        return

    ch_z = (tof_loaded_equiv[ch_valid] - ref_ch_mean[ch_valid]) / ref_ch_std[ch_valid]
    log_message(f"[tof-diagnostic] experimental vs synthetic channel-z: {_format_stats(_safe_stats(ch_z))}")
    log_message(
        "[tof-diagnostic] experimental vs synthetic channel-z fractions: "
        f"|z|>1={float(np.mean(np.abs(ch_z) > 1.0)):.4f}, "
        f"|z|>2={float(np.mean(np.abs(ch_z) > 2.0)):.4f}, "
        f"|z|>3={float(np.mean(np.abs(ch_z) > 3.0)):.4f}"
    )

    # Mean-difference diagnostics should not be normalized by per-channel std,
    # but they should use the same robust valid-channel set for consistency.
    diff = tof_loaded_equiv - ref_ch_mean
    em_diff = _mean_over_valid(diff, ch_valid.astype(np.float32), axis=1)
    rx_diff = _mean_over_valid(diff, ch_valid.astype(np.float32), axis=0)
    log_message(f"[tof-diagnostic] emitter mean exp-minus-synthetic-normalized: {_format_stats(_safe_stats(em_diff))}")
    log_message(f"[tof-diagnostic] receiver mean exp-minus-synthetic-normalized: {_format_stats(_safe_stats(rx_diff))}")
    log_message(".")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_path", required=True, help="*_WaveformInversionResults.mat (v7.3)")
    ap.add_argument("--tof_pt", required=True, help="ToF .pt (dict with 'tof' and 'metadata')")
    ap.add_argument("--out_pt", required=True, help="Output .pt with GT SoS inserted (resampled if needed)")

    ap.add_argument(
        "--gt_iter",
        type=int,
        default=30,
        help="Which stored iteration to use for GT SoS. Python indexing: -1=last, 0=first. "
             "For MATLAB iter k (1-based), pass --gt_iter (k-1).",
    )

    ap.add_argument("--debug_plot", action="store_true", help="Plot side-by-side GT full-res and resampled.")
    ap.add_argument("--debug_png", default=None, help="If set, save side-by-side plot PNG to this path.")
    ap.add_argument(
        "--skip_orientation_diagnostic",
        action="store_true",
        help="Do not embed the GT orientation diagnostic panel in the HTML output.",
    )
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--reference_data_path", type=str, default=None, help="Optional synthetic training cache .pt for ToF-distribution diagnostics. Diagnostic only.")
    ap.add_argument("--skip_tof_diagnostic", action="store_true",
                    help="Skip optional ToF distribution diagnostics; GT conversion is unchanged.")
    ap.add_argument(
        "--matlab_hdf5_slice_layout",
        type=str,
        choices=["hdf5_yx", "hdf5_xy"],
        default="hdf5_xy",
        help=(
            "Orientation of the selected 2-D VEL_ESTIM_ITER slice after h5py loading. "
            "'hdf5_yx' means the slice is already image (row,column)=(y,x). "
            "'hdf5_xy' reproduces the earlier transpose-before-resampling behavior."
        ),
    )
    ap.add_argument(
        "--gt_primary_orientation",
        type=str,
        default="as_saved",
        choices=["as_saved", "transpose", "flip_x", "flip_y", "flip_xy", "transpose_flip_x", "transpose_flip_y", "transpose_flip_xy"],
        help=(
            "Resolved GT orientation to write into key 'sos'. For the experimental pipeline this "
            "must remain 'as_saved', corresponding to the MATLAB/Ali display convention. "
            "Other choices are retained only for explicit manual debugging, not for automatic MSE selection."
        ),
    )

    args = ap.parse_args()
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

    mat_path = resolve_path_maybe_project_relative(str(args.mat_path))
    tof_pt = resolve_path_maybe_project_relative(str(args.tof_pt))
    out_pt = str(args.out_pt)

    # Load target .pt
    raw = robust_torch_load(resolve_path_maybe_project_relative(tof_pt))
    if not (isinstance(raw, dict) and ("tof" in raw) and ("metadata" in raw)):
        raise KeyError("tof_pt must be a dict with keys at least: 'tof', 'metadata'")

    # Sanitize the incoming ToF tensor.  In practice the waveform
    # picker initializes missing picks to NaN, but prior versions of the
    # pipeline could emit ֲ±Inf entries.  Such values break downstream
    # normalization and reconstruction.  Replace any infinite values
    # with NaN here so that subsequent loaders (e.g. load_tof_with_gt.py)
    # see NaN rather than inf.
    try:
        tof = raw.get("tof")
        if tof is not None:
            # Support both torch.Tensor and numpy-like arrays
            if isinstance(tof, torch.Tensor):
                if torch.isinf(tof).any():
                    nan_scalar = torch.tensor(float('nan'), dtype=tof.dtype, device=tof.device)
                    raw["tof"] = torch.where(torch.isinf(tof), nan_scalar, tof)
            else:
                # Convert to numpy array, replace inf with nan, then back to tensor
                arr = np.asarray(tof).astype(np.float32)
                if np.isinf(arr).any():
                    arr[np.isinf(arr)] = np.nan
                    raw["tof"] = torch.tensor(arr)
    except Exception:
        # If sanitization fails, proceed without modification
        pass

    md = dict(raw["metadata"])
    nx_sim = int(md["nx"])
    ny_sim = int(md["ny"])
    log_message(f"[pt] target grid from metadata: (nx,ny)=({nx_sim},{ny_sim})")
    tof_shape = tuple(np.asarray(_to_numpy_float32(raw["tof"])).shape)
    expected_ne = int(md.get("n_emitters", -1))
    expected_nr = int(md.get("n_receivers", -1))
    if len(tof_shape) == 3:
        actual_ne, actual_nr = int(tof_shape[-2]), int(tof_shape[-1])
    elif len(tof_shape) == 2:
        actual_ne, actual_nr = int(tof_shape[0]), int(tof_shape[1])
    else:
        raise ValueError(f"ToF tensor has unsupported shape {tof_shape}; expected (1,Ne,Nr) or (Ne,Nr).")
    if expected_ne > 0 and expected_nr > 0 and (actual_ne != expected_ne or actual_nr != expected_nr):
        raise ValueError(
            f"ToF shape ({actual_ne},{actual_nr}) does not match metadata "
            f"n_emitters/n_receivers=({expected_ne},{expected_nr})."
        )
    tof_norm = md.get("tof_norm", md.get("reference_tof_norm", None))
    if not isinstance(tof_norm, dict) or str(tof_norm.get("type", "none")).lower() == "none":
        log_message(
            "[pt] WARNING: ToF metadata has no effective training normalization. "
            "Experimental evaluation may be inconsistent with the synthetic-trained model."
        )

    if not args.skip_tof_diagnostic:
        log_experimental_tof_diagnostic(raw, reference_data_path=args.reference_data_path)

    # Load MATLAB results
    mat = load_mat_vars(mat_path)
    if "xi" not in mat or "yi" not in mat:
        raise KeyError("xi/yi not found in .mat results; cannot resample on physical coordinates.")
    xi = np.asarray(mat["xi"]).squeeze()
    yi = np.asarray(mat["yi"]).squeeze()

    # Pick requested iteration and convert units if needed
    sos_raw_2d = pick_sos_iteration(mat, iter_index=args.gt_iter)
    sos_raw_2d = to_m_per_s_if_needed(sos_raw_2d)
    log_message(f"[mat] selected gt_iter={args.gt_iter}, raw slice shape={sos_raw_2d.shape}")

    # MATLAB-consistent y,x map for physical coords (used for resampling/side-by-side debug).
    # version 20 makes this convention explicit instead of silently transposing.
    sos_full_yx = orient_matlab_hdf5_sos_slice_to_yx(
        sos_raw_2d,
        layout=args.matlab_hdf5_slice_layout,
    )
    log_message(
        "[gt-orientation] MATLAB HDF5 selected slice interpreted as "
        f"{args.matlab_hdf5_slice_layout}; source map for resampling has shape "
        f"{sos_full_yx.shape}=(ny_src,nx_src)."
    )

    # Target axes (meters)
    x_tgt, y_tgt, axis_source = build_target_axes_from_metadata(md, xi_src=xi, yi_src=yi)
    xi_resample, yi_resample, source_axis_alignment = align_source_axes_to_target_origin(
        xi,
        yi,
        x_tgt,
        y_tgt,
    )
    sos_full_yx, xi_resample, yi_resample, source_axis_order = enforce_increasing_source_axes(
        sos_full_yx,
        xi_resample,
        yi_resample,
    )
    source_axis_alignment.update(source_axis_order)
    validate_axis_overlap(x_tgt, xi_resample, "x")
    validate_axis_overlap(y_tgt, yi_resample, "y")
    log_message(
        f"[axes] target source={axis_source}; "
        f"x=[{float(x_tgt.min()):.6e}, {float(x_tgt.max()):.6e}] "
        f"within MATLAB/resampled-source x=[{float(np.min(xi_resample)):.6e}, {float(np.max(xi_resample)):.6e}]; "
        f"y=[{float(y_tgt.min()):.6e}, {float(y_tgt.max()):.6e}] "
        f"within MATLAB/resampled-source y=[{float(np.min(yi_resample)):.6e}, {float(np.max(yi_resample)):.6e}]"
    )

    # Resample to simulation grid
    sos_sim_yx = bilinear_resample_on_rect_grid(
        src_yx=sos_full_yx,
        x_src=xi_resample,
        y_src=yi_resample,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
    )
    log_message(f"[resample] sos_sim_yx.shape = {sos_sim_yx.shape} (expected (ny,nx)=({ny_sim},{nx_sim}))")
    if sos_sim_yx.shape != (ny_sim, nx_sim):
        raise RuntimeError(
            f"Resampling produced {sos_sim_yx.shape}, expected {(ny_sim, nx_sim)}. "
            f"Check metadata axes/bounds and xi/yi."
        )

    # Side-by-side debug plot (full-res vs resampled) with centered shared colorbar
    if args.debug_plot:
        plot_side_by_side(
            left_yx=sos_full_yx,
            right_yx=sos_sim_yx,
            x_left=xi_resample,
            y_left=yi_resample,
            x_right=x_tgt,
            y_right=y_tgt,
            title_left=f"Full-res GT (801x801)\nVEL_ESTIM_ITER iter={args.gt_iter}",
            title_right=f"Resampled GT ({ny_sim}x{nx_sim})\nSaved into .pt",
            suptitle=f"GT SoS: full-res vs resampled (gt_iter={args.gt_iter})",
            out_png=args.debug_png,
        )

    # Store into output .pt in project convention: (1, nx, ny). Only plotting uses transpose.
    sos_sim_xy = sos_sim_yx.T  # (nx, ny) pipeline convention
    if not bool(args.skip_orientation_diagnostic):
        plot_orientation_diagnostic_xy(
            sos_sim_xy,
            nx=nx_sim,
            ny=ny_sim,
            phys_x=float(np.max(x_tgt) - np.min(x_tgt)),
            phys_y=float(np.max(y_tgt) - np.min(y_tgt)),
            title=(
                "Experimental GT orientation diagnostic\n"
                "Primary saved orientation is 'as_saved'; alternatives are visual checks only"
            ),
        )
    # Code-resolved GT convention.  The Ali/MATLAB display orientation is the
    # authoritative experimental ground truth. Exactly one GT tensor is written
    # to key 'sos'; MSE-selectable flipped or transposed variants are not saved.
    primary_orientation = str(args.gt_primary_orientation)
    if primary_orientation != "as_saved":
        raise ValueError(
            "Experimental GT orientation must remain 'as_saved' for the normal pipeline. "
            f"Got --gt_primary_orientation={primary_orientation!r}."
        )
    sos_tensor = torch.tensor(sos_sim_xy[None, :, :].astype(np.float32))

    out = dict(raw)
    out["sos"] = sos_tensor
    out.pop("sos_orientation_variants", None)
    log_message("[gt-orientation] Primary GT written to key 'sos': as_saved (MATLAB/Ali display convention).")

    out_md = dict(md)
    out_md["note_gt"] = (
        f"GT SoS inserted from {Path(mat_path).name}, VEL_ESTIM_ITER, gt_iter={args.gt_iter}. "
        f"Resampled using (xi,yi) to target (ny,nx); saved as (1,nx,ny)=({nx_sim},{ny_sim}). "
        f"Primary orientation resolved with matlab_hdf5_slice_layout={args.matlab_hdf5_slice_layout}; "
        "gt_primary_orientation=as_saved (MATLAB/Ali display convention)."
    )
    
    out_md.update({"x_min": float(x_tgt.min()), "x_max": float(x_tgt.max()), "y_min": float(y_tgt.min()), "y_max": float(y_tgt.max())})
    out_md["gt_axis_source"] = str(axis_source)
    out_md["gt_source_axis_alignment"] = dict(source_axis_alignment)
    out_md["has_gt"] = True
    out_md["gt_orientation_primary"] = primary_orientation
    out_md["gt_matlab_hdf5_slice_layout"] = str(args.matlab_hdf5_slice_layout)
    out_md["gt_orientation_resolution"] = (
        "The only valid metric target is key 'sos', stored as as_saved in the MATLAB/Ali display convention. "
        "No alternate GT orientations are saved or selected by MSE."
    )
    out_md["gt_orientation_variants"] = []
    out_md["gt_orientation_note"] = (
        "GT orientation is fixed by code convention, not by prediction-dependent diagnostic MSE."
    )
    out_md["x_axis_m"] = torch.tensor(x_tgt.astype(np.float32))
    out_md["y_axis_m"] = torch.tensor(y_tgt.astype(np.float32))

    out["metadata"] = out_md

    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)
    log_message(f"Saved: {out_pt}")


if __name__ == "__main__":
    main()











