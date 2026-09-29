# FILE: dataset.py
"""
Dataset utilities for Ultrasound Tomography.

Implements the requested layered workflow:

1) Persistent SoS bank (SoS only) + persistent train/val/test split indices
2) Experiment-dependent *pairs cache* (.pt) computed from the SoS bank:
   - contains SoS/ToF and optional PDE representation
   - enables fast training and reliable --resume across epochs (no ToF recomputation)

Backward compatibility:
- The original TomographyDataset remains available with the same API.
"""

from __future__ import annotations

import os
import random
import pprint
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from anatomy import create_random_sos_map, generate_sensor_positions
from msfm import msfm
from logger import log_message
from concurrent.futures import ProcessPoolExecutor
# -----------------------------
# Helpers
# -----------------------------


def _format_seconds(seconds: float) -> str:
    """Format elapsed/remaining seconds for logger messages."""
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}min"
    hours = minutes / 60.0
    return f"{hours:.2f}h"

def compute_limited_view_mask(n_emitters: int, n_receivers: int, exclude_frac: float) -> np.ndarray:
    """Use the measured ring's angular exclusion rule for synthetic ToFs."""
    from measured_geometry import angular_mask
    return angular_mask(n_emitters, n_receivers, exclude_frac)



def make_cache_paths(data_path: str | Path) -> dict[str, str]:
    """Create consistent cache paths from a dataset file path.

    If data_path ends with ".pt", the stem is taken without that suffix, so:
        ".../ultrasound_data.pt" -> ".../ultrasound_data"

    Returned keys:
        - stem
        - sos_bank_path
        - splits_path
        - pairs_cache_path
    """
    p = Path(str(data_path))
    stem = str(p.with_suffix(""))
    return {
        "stem": stem,
        "sos_bank_path": stem + "_sos_bank.pt",
        "splits_path": stem + "_splits.pt",
        "pairs_cache_path": stem + "_pairs_cache.pt",
    }

def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
def _pairs_chunk_root(cache_path: str | Path) -> Path:
    """
    Return the directory that stores the progressive pairs-cache chunks.

    Example:
        ultrasound_data_20000_pairs.pt
        ->
        ultrasound_data_20000_pairs_chunks/
    """
    p = Path(str(cache_path))
    return p.with_suffix("").with_name(p.stem + "_chunks")


def _pairs_chunk_dir(cache_path: str | Path, chunk_id: int) -> Path:
    """Return the directory of a specific pairs-cache chunk."""
    return _pairs_chunk_root(cache_path) / f"chunk_{int(chunk_id):04d}"


def _atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    """
    Save a NumPy array atomically.

    The temporary file is replaced only after the complete array was written.
    Therefore an interrupted write is never treated as a completed chunk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "wb") as f:
        np.save(f, np.asarray(array), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def _atomic_torch_save(payload: Dict, path: str | Path) -> None:
    """
    Save a small PyTorch dictionary atomically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    torch.save(
        payload,
        str(tmp),
        pickle_protocol=4,
    )

    os.replace(tmp, path)


class _RunningStats:
    """
    Streaming population statistics.

    Produces mean/std/max values equivalent to NumPy population statistics,
    without retaining all samples in memory.
    """

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64).reshape(-1)

        if x.size == 0:
            return

        x = x[np.isfinite(x)]

        if x.size == 0:
            return

        self.count += int(x.size)
        self.sum += float(np.sum(x, dtype=np.float64))
        self.sumsq += float(np.sum(x * x, dtype=np.float64))
        self.max = max(self.max, float(np.max(x)))

    def merge(self, state: Optional[Dict]) -> None:
        """
        Merge statistics stored in a completed chunk manifest.
        """
        if not isinstance(state, dict):
            return

        self.count += int(state.get("count", 0))
        self.sum += float(state.get("sum", 0.0))
        self.sumsq += float(state.get("sumsq", 0.0))

        maximum = float(state.get("max", -np.inf))

        if np.isfinite(maximum):
            self.max = max(self.max, maximum)

    def state_dict(self) -> Dict:
        return {
            "count": int(self.count),
            "sum": float(self.sum),
            "sumsq": float(self.sumsq),
            "max": float(self.max),
        }

    def final(self) -> Dict[str, float]:
        if self.count <= 0:
            return {
                "mean": 0.0,
                "std": 1.0,
                "max": 1.0,
            }

        mean = self.sum / float(self.count)

        variance = max(
            0.0,
            self.sumsq / float(self.count) - mean * mean,
        )

        maximum = self.max if np.isfinite(self.max) else 1.0

        return {
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "max": float(maximum),
        }
def _pairs_worker_initializer() -> None:
    """
    Limit each worker to one internal thread.

    Parallelism is provided by multiple Python processes, so allowing every
    process to create many additional internal threads can overload the CPU.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _compute_pairs_sample_worker(task: Dict) -> Dict:
    """
    Compute all emitter MSFM maps for one SoS sample.

    One worker receives one complete sample. This avoids creating a separate
    multiprocessing task for every emitter.
    """
    local_i = int(task["local_i"])
    sample_index = int(task["sample_index"])

    sos_map = np.asarray(
        task["sos_map"],
        dtype=np.float32,
    )

    emitters = np.asarray(task["emitters"])

    receiver_x = np.asarray(
        task["receiver_x"],
        dtype=np.intp,
    )

    receiver_y = np.asarray(
        task["receiver_y"],
        dtype=np.intp,
    )

    dx = float(task["dx"])
    dy = float(task["dy"])

    include_pde = bool(task["include_pde"])
    pde_emitters_k = int(task["pde_emitters_k"])

    colloc_emitters_i = task.get(
        "colloc_emitters_i",
        None,
    )

    colloc_xy = task.get(
        "colloc_xy",
        None,
    )

    raw_tof_matrix = np.empty(
        (
            int(emitters.shape[0]),
            int(receiver_x.size),
        ),
        dtype=np.float32,
    )

    resid_acc = None
    gradmag_acc = None

    colloc_T_i = None
    colloc_slot_for_emitter: Dict[int, int] = {}

    if colloc_emitters_i is not None:
        if colloc_xy is None:
            raise RuntimeError(
                "colloc_emitters_i was supplied without colloc_xy."
            )

        colloc_emitters_i = np.asarray(
            colloc_emitters_i,
            dtype=np.int64,
        )

        colloc_xy = np.asarray(
            colloc_xy,
            dtype=np.int64,
        )

        colloc_slot_for_emitter = {
            int(emitter_index): int(slot)
            for slot, emitter_index
            in enumerate(colloc_emitters_i.tolist())
        }

        colloc_T_i = np.empty(
            (
                int(colloc_emitters_i.size),
                int(colloc_xy.shape[0]),
            ),
            dtype=np.float32,
        )

        colloc_x = colloc_xy[:, 0].astype(
            np.intp,
            copy=False,
        )

        colloc_y = colloc_xy[:, 1].astype(
            np.intp,
            copy=False,
        )

    else:
        colloc_x = None
        colloc_y = None

    sample_t0 = time.perf_counter()

    for e_idx, emitter_pos in enumerate(emitters):
        t_map = msfm(
            sos_map,
            emitter_pos,
            dx=dx,
            dy=dy,
        )

        # Vectorized receiver extraction.
        raw_tof_matrix[e_idx, :] = np.asarray(
            t_map[receiver_x, receiver_y],
            dtype=np.float32,
        )

        colloc_slot = colloc_slot_for_emitter.get(
            int(e_idx),
            None,
        )

        if (
            colloc_slot is not None
            and colloc_T_i is not None
            and colloc_x is not None
            and colloc_y is not None
        ):
            colloc_T_i[colloc_slot, :] = np.asarray(
                t_map[colloc_x, colloc_y],
                dtype=np.float32,
            )

        if include_pde and e_idx < pde_emitters_k:
            dTdx, dTdy = np.gradient(
                t_map.astype(np.float32),
                dx,
                dy,
                edge_order=1,
            )

            grad_mag = np.sqrt(
                dTdx**2 + dTdy**2
            )

            resid = (
                sos_map.astype(
                    np.float32,
                    copy=False,
                )
                * grad_mag
                - 1.0
            )

            if resid_acc is None:
                resid_acc = resid
                gradmag_acc = grad_mag
            else:
                resid_acc += resid
                gradmag_acc += grad_mag

    pde_i = None

    if include_pde:
        if (
            pde_emitters_k > 0
            and resid_acc is not None
            and gradmag_acc is not None
        ):
            resid_mean = (
                resid_acc
                / float(pde_emitters_k)
            )

            gradmag_mean = (
                gradmag_acc
                / float(pde_emitters_k)
            )

        else:
            resid_mean = np.zeros(
                sos_map.shape,
                dtype=np.float32,
            )

            gradmag_mean = np.zeros(
                sos_map.shape,
                dtype=np.float32,
            )

        pde_i = np.stack(
            [
                gradmag_mean,
                resid_mean,
            ],
            axis=0,
        ).astype(np.float32)

    return {
        "local_i": local_i,
        "sample_index": sample_index,
        "raw_tof_matrix": raw_tof_matrix,
        "pde": pde_i,
        "colloc_emitters_i": colloc_emitters_i,
        "colloc_T": colloc_T_i,
        "roi_radius": task.get("roi_radius", None),
        "roi_outer_c": task.get("roi_outer_c", None),
        "sample_elapsed": (
            time.perf_counter()
            - sample_t0
        ),
    }


def _iter_pairs_sample_results(
    tasks: Sequence[Dict],
    pairs_workers: int,
):
    """
    Run sample computations serially or with multiple CPU processes.

    executor.map preserves input order, so output samples and streaming
    statistics remain deterministic.
    """
    pairs_workers = int(max(1, pairs_workers))

    if pairs_workers == 1:
        for task in tasks:
            yield _compute_pairs_sample_worker(task)

        return

    with ProcessPoolExecutor(
        max_workers=pairs_workers,
        initializer=_pairs_worker_initializer,
    ) as executor:
        yield from executor.map(
            _compute_pairs_sample_worker,
            tasks,
            chunksize=1,
        )
def load_pairs_cache_metadata(cache_path: str | Path) -> Dict:
    """
    Load only the metadata from either:

    1. The original classic single-file pairs cache.
    2. The new small chunked-cache index file.
    """
    payload = torch.load(
        str(cache_path),
        weights_only=False,
    )

    metadata = (
        payload.get("metadata", {})
        if isinstance(payload, dict)
        else {}
    )

    return metadata if isinstance(metadata, dict) else {}
def _segment_circle_lengths(points_a: np.ndarray, points_b: np.ndarray, center_xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Length of each line segment that lies inside a circle.

    points_a and points_b are arrays with shape (..., 2) in physical metres.
    The return shape is points_a.shape[:-1].
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    c = np.asarray(center_xy, dtype=np.float64).reshape((1,) * (a.ndim - 1) + (2,))
    r = float(radius_m)
    if r <= 0.0:
        return np.zeros(a.shape[:-1], dtype=np.float32)

    d = b - a
    f = a - c
    aa = np.sum(d * d, axis=-1)
    bb = 2.0 * np.sum(f * d, axis=-1)
    cc = np.sum(f * f, axis=-1) - r * r

    length_total = np.sqrt(np.maximum(aa, 0.0))
    length_inside = np.zeros_like(length_total, dtype=np.float64)
    valid = aa > 1e-24
    disc = bb * bb - 4.0 * aa * cc
    hit = valid & (disc >= 0.0)
    if not np.any(hit):
        return length_inside.astype(np.float32)

    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[hit] = np.sqrt(np.maximum(disc[hit], 0.0))
    t0 = (-bb - sqrt_disc) / (2.0 * np.where(valid, aa, 1.0))
    t1 = (-bb + sqrt_disc) / (2.0 * np.where(valid, aa, 1.0))
    lo = np.maximum(0.0, np.minimum(t0, t1))
    hi = np.minimum(1.0, np.maximum(t0, t1))
    frac = np.maximum(0.0, hi - lo)
    length_inside[hit] = frac[hit] * length_total[hit]
    return length_inside.astype(np.float32)


def _build_roi_chord_geometry(
    emitters_px: np.ndarray,
    receivers_px: np.ndarray,
    dx: float,
    dy: float,
    roi_center_x: float,
    roi_center_y: float,
    roi_radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full, in-ROI, and outside-ROI path lengths for all Tx/Rx chords."""
    emit_m = np.stack([emitters_px[:, 0] * float(dx), emitters_px[:, 1] * float(dy)], axis=1).astype(np.float64)
    recv_m = np.stack([receivers_px[:, 0] * float(dx), receivers_px[:, 1] * float(dy)], axis=1).astype(np.float64)
    a = emit_m[:, None, :]
    b = recv_m[None, :, :]
    full = np.linalg.norm(b - a, axis=-1).astype(np.float32)
    inside = _segment_circle_lengths(a, b, np.array([float(roi_center_x), float(roi_center_y)]), float(roi_radius))
    outside = np.maximum(full - inside, 0.0).astype(np.float32)
    return full, inside, outside


def _apply_synthetic_roi_tof_reference(
    tof_matrix: np.ndarray,
    full_len: np.ndarray,
    roi_len: np.ndarray,
    outer_len: np.ndarray,
    *,
    outer_c: float,
    reference_c: float,
    synthetic_c: float,
    min_chord_frac: float,
    nonintersect_mode: str,
    residual_clip_us: float,
) -> np.ndarray:
    """Apply the same ROI-referenced timing model used for experimental ToF extraction.

    The raw Eikonal ToF is decomposed relative to a two-region background model:
        model_background = outer_length / outer_c + roi_length / reference_c
        residual = raw_eikonal_tof - model_background
        adjusted_tof = full_length / synthetic_c + residual

    Chords that barely intersect the ROI are assigned the homogeneous synthetic
    reference by default, matching extract_measured_tof.py's experimental behavior.
    """
    T = np.asarray(tof_matrix, dtype=np.float32)
    full = np.asarray(full_len, dtype=np.float32)
    inside = np.asarray(roi_len, dtype=np.float32)
    outside = np.asarray(outer_len, dtype=np.float32)
    eps = np.float32(1e-12)

    model_background = outside / np.float32(max(float(outer_c), 1e-6)) + inside / np.float32(max(float(reference_c), 1e-6))
    full_reference = full / np.float32(max(float(synthetic_c), 1e-6))
    residual = T - model_background
    clip_s = float(residual_clip_us) * 1e-6
    if clip_s > 0.0:
        residual = np.clip(residual, -clip_s, clip_s)
    adjusted = full_reference + residual

    chord_frac = np.divide(inside, np.maximum(full, eps), out=np.zeros_like(inside), where=full > eps)
    weak = chord_frac < float(min_chord_frac)
    mode = str(nonintersect_mode or "reference").lower()
    if mode == "reference":
        adjusted[weak] = full_reference[weak]
    elif mode == "keep":
        adjusted[weak] = T[weak]
    elif mode == "nan":
        adjusted[weak] = np.nan
    else:
        adjusted[weak] = full_reference[weak]
    return adjusted.astype(np.float32)


# -----------------------------
# SoS bank (persistent)
# -----------------------------

@dataclass
class SoSBankMetadata:
    num_samples: int
    nx: int
    ny: int
    phys_x: float
    phys_y: float
    dx: float
    dy: float
    sos_water: float
    sos_min: float
    sos_max: float
    max_shapes: int
    min_shapes: int
    radius: float
    n_emitters: int
    n_receivers: int
    data_seed: int
    use_random_support_envelope: bool = False
    support_radius_min_frac: float = 0.65
    support_radius_max_frac: float = 0.95
    support_center_jitter_frac: float = 0.05
    support_ellipse_prob: float = 0.35
    support_ellipse_axis_jitter_frac: float = 0.15
    shape_mode: str = "ellipses"
    p_ellipse: float = 0.65
    p_triangle: float = 0.20
    p_polygon: float = 0.15
    p_rod: float = 0.0
    shape_size_min_px: int = 4
    shape_size_max_px: int = 24
    polygon_vertices_min: int = 4
    polygon_vertices_max: int = 8

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


def save_sos_bank(path: str | Path, sos: np.ndarray, metadata: SoSBankMetadata) -> None:
    _ensure_parent_dir(path)
    torch.save({"sos": sos.astype(np.float32), "metadata": metadata.as_dict()}, str(path))
    log_message(f"[dataset.py] Saved SoS bank: {path}  (N={sos.shape[0]})")


def load_sos_bank(path: str | Path) -> Dict:
    payload = torch.load(str(path), weights_only=False)
    if "sos" not in payload or "metadata" not in payload:
        raise ValueError(f"SoS bank missing required keys: {path}")
    return payload


def create_sos_bank(
    *,
    num_samples: int,
    nx: int,
    ny: int,
    phys_x: float,
    phys_y: float,
    sos_water: float,
    sos_min: float,
    sos_max: float,
    max_shapes: int,
    min_shapes: int = 0,
    radius: float = 0.04,
    n_emitters: int = 32,
    n_receivers: int = 32,
    data_seed: int = 0,
    use_random_support_envelope: bool = False,
    support_radius_min_frac: float = 0.65,
    support_radius_max_frac: float = 0.95,
    support_center_jitter_frac: float = 0.05,
    support_ellipse_prob: float = 0.35,
    support_ellipse_axis_jitter_frac: float = 0.15,
    shape_mode: str = "ellipses",
    p_ellipse: float = 0.65,
    p_triangle: float = 0.20,
    p_polygon: float = 0.15,
    p_rod: float = 0.0,
    shape_size_min_px: int = 4,
    shape_size_max_px: int = 24,
    polygon_vertices_min: int = 4,
    polygon_vertices_max: int = 8,
) -> Tuple[np.ndarray, SoSBankMetadata]:
    dx, dy = phys_x / (nx - 1), phys_y / (ny - 1)
    sos_list: List[np.ndarray] = []
    base_seed = int(data_seed)

    for i in range(int(num_samples)):
        if base_seed != 0:
            s_i = base_seed + i
            np.random.seed(s_i)
            random.seed(s_i)

        sos_map = create_random_sos_map(
            nx, ny, dx, dy, radius,
            sos_water, sos_min, sos_max, max_shapes, min_shapes=min_shapes,
            use_random_support_envelope=bool(use_random_support_envelope),
            support_radius_min_frac=float(support_radius_min_frac),
            support_radius_max_frac=float(support_radius_max_frac),
            support_center_jitter_frac=float(support_center_jitter_frac),
            support_ellipse_prob=float(support_ellipse_prob),
            support_ellipse_axis_jitter_frac=float(support_ellipse_axis_jitter_frac),
            shape_mode=str(shape_mode),
            p_ellipse=float(p_ellipse),
            p_triangle=float(p_triangle),
            p_polygon=float(p_polygon),
            p_rod=float(p_rod),
            shape_size_min_px=int(shape_size_min_px),
            shape_size_max_px=int(shape_size_max_px),
            polygon_vertices_min=int(polygon_vertices_min),
            polygon_vertices_max=int(polygon_vertices_max),
        )
        sos_list.append(sos_map.astype(np.float32))

    sos = np.array(sos_list, dtype=np.float32)  # same convention as existing code

    md = SoSBankMetadata(
        num_samples=int(num_samples),
        nx=int(nx), ny=int(ny),
        phys_x=float(phys_x), phys_y=float(phys_y),
        dx=float(dx), dy=float(dy),
        sos_water=float(sos_water),
        sos_min=float(sos_min), sos_max=float(sos_max),
        max_shapes=int(max_shapes),
        min_shapes=int(max(1, min(int(min_shapes) if int(min_shapes) > 0 else max(1, int(max_shapes) // 2), int(max_shapes)))),
        radius=float(radius),
        n_emitters=int(n_emitters),
        n_receivers=int(n_receivers),
        data_seed=int(data_seed),
        use_random_support_envelope=bool(use_random_support_envelope),
        support_radius_min_frac=float(support_radius_min_frac),
        support_radius_max_frac=float(support_radius_max_frac),
        support_center_jitter_frac=float(support_center_jitter_frac),
        support_ellipse_prob=float(support_ellipse_prob),
        support_ellipse_axis_jitter_frac=float(support_ellipse_axis_jitter_frac),
        shape_mode=str(shape_mode),
        p_ellipse=float(p_ellipse),
        p_triangle=float(p_triangle),
        p_polygon=float(p_polygon),
        p_rod=float(p_rod),
        shape_size_min_px=int(shape_size_min_px),
        shape_size_max_px=int(shape_size_max_px),
        polygon_vertices_min=int(polygon_vertices_min),
        polygon_vertices_max=int(polygon_vertices_max),
    )
    return sos, md


def get_or_make_sos_bank(
    *,
    sos_bank_path: str | Path,
    make_if_missing: bool,
    **kwargs,
) -> Dict:
    sos_bank_path = str(sos_bank_path)
    if os.path.exists(sos_bank_path):
        log_message(f"[dataset.py] Loading SoS bank: {sos_bank_path}")
        return load_sos_bank(sos_bank_path)

    if not make_if_missing:
        raise FileNotFoundError(f"SoS bank does not exist: {sos_bank_path}")

    log_message(f"[dataset.py] Creating SoS bank: {sos_bank_path}")
    sos, md = create_sos_bank(**kwargs)
    save_sos_bank(sos_bank_path, sos, md)
    return {"sos": sos, "metadata": md.as_dict()}


# -----------------------------
# Splits (persistent)
# -----------------------------

def make_splits(N: int, seed: int = 0, frac_train: float = 0.8, frac_val: float = 0.1) -> Dict:
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")
    if frac_train <= 0 or frac_val < 0 or (frac_train + frac_val) >= 1.0:
        raise ValueError("Invalid split fractions.")

    gen = torch.Generator().manual_seed(int(seed) if int(seed) != 0 else 42)
    perm = torch.randperm(N, generator=gen).tolist()

    n_train = int(round(frac_train * N))
    n_val = int(round(frac_val * N))
    n_train = min(max(n_train, 1), N - 2)
    n_val = min(max(n_val, 1), N - n_train - 1)

    return {
        "N": N,
        "seed": int(seed),
        "fractions": [float(frac_train), float(frac_val), float(1.0 - frac_train - frac_val)],
        "train_idx": perm[:n_train],
        "val_idx": perm[n_train:n_train + n_val],
        "test_idx": perm[n_train + n_val:],
    }


def save_splits(path: str | Path, splits: Dict) -> None:
    _ensure_parent_dir(path)
    torch.save(splits, str(path))
    log_message(f"[dataset.py] Saved splits: {path}")


def load_splits(path: str | Path) -> Dict:
    splits = torch.load(str(path), weights_only=False)
    required = {"train_idx", "val_idx", "test_idx", "N"}
    if not required.issubset(set(splits.keys())):
        raise ValueError(f"Splits file missing required keys {required}: {path}")
    return splits


def get_or_make_splits(
    *,
    splits_path: str | Path,
    N: int,
    seed: int,
    make_if_missing: bool,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    extra: Optional[Dict] = None,
) -> Dict:
    splits_path = str(splits_path)
    if os.path.exists(splits_path):
        log_message(f"[dataset.py] Loading splits: {splits_path}")
        splits = load_splits(splits_path)
        if int(splits.get("N", -1)) != int(N):
            raise ValueError(f"Split N mismatch. splits N={splits.get('N')} but SoS bank N={N}")
        return splits

    if not make_if_missing:
        raise FileNotFoundError(f"Splits file does not exist: {splits_path}")

    splits = make_splits(N=N, seed=seed, frac_train=frac_train, frac_val=frac_val)
    if extra:
        splits.update(extra)
    save_splits(splits_path, splits)
    return splits


# -----------------------------
# Pairs cache (experiment-dependent) built from SoS bank
# -----------------------------

def build_pairs_cache_from_sos_bank(
    *,
    sos_bank: Dict,
    cache_path: str | Path,
    include_pde: bool = False,
    pde_emitters_k: int = 1,
    exclude_frac: float = 0.0,
    tof_norm: str = "zscore",
    tof_norm_eps: float = 1e-6,
    store_collocation_T: bool = False,
    n_collocation: int = 512,
    colloc_emitters_k: int = 4,
    colloc_seed: int = 0,
    synthetic_roi_tof_enable: bool = False,
    synthetic_roi_center_x: Optional[float] = None,
    synthetic_roi_center_y: Optional[float] = None,
    synthetic_roi_radius_min: float = 0.055,
    synthetic_roi_radius_max: float = 0.070,
    synthetic_roi_outer_c_min: float = 1460.0,
    synthetic_roi_outer_c_max: float = 1500.0,
    synthetic_roi_reference_c: float = 1500.0,
    synthetic_roi_synthetic_c: float = 1500.0,
    synthetic_roi_min_chord_frac: float = 0.05,
    synthetic_roi_residual_clip_us: float = 8.0,
    synthetic_roi_nonintersect_mode: str = "reference",
    synthetic_roi_seed: int = 0,
    chunk_size: int = 2000,
    resume_chunks: bool = True,
    pairs_workers: int = 1,
) -> Dict:
    """
    Build the pairs cache progressively.

    The main cache_path file becomes a small index file.

    The actual arrays are written into:

        <cache_stem>_chunks/chunk_0000/
        <cache_stem>_chunks/chunk_0001/
        ...

    Only one chunk is held in RAM at a time.
    """

    _ensure_parent_dir(cache_path)

    cache_path = Path(str(cache_path))
    chunk_root = _pairs_chunk_root(cache_path)
    chunk_root.mkdir(parents=True, exist_ok=True)

    # Avoid an unnecessary copy when the SoS bank is already float32.
    sos_data = np.asarray(
        sos_bank["sos"],
        dtype=np.float32,
    )

    md = dict(sos_bank["metadata"])

    nx = int(md["nx"])
    ny = int(md["ny"])

    dx = float(md["dx"])
    dy = float(md["dy"])

    phys_x = float(
        md.get(
            "phys_x",
            dx * max(nx - 1, 1),
        )
    )

    phys_y = float(
        md.get(
            "phys_y",
            dy * max(ny - 1, 1),
        )
    )

    radius = float(md.get("radius", 0.04))

    n_emitters = int(md.get("n_emitters", 32))
    n_receivers = int(md.get("n_receivers", 32))

    total_samples = int(sos_data.shape[0])

    chunk_size = int(max(1, chunk_size))
    pairs_workers = int(max(1, pairs_workers))

    pairs_workers = min(
        pairs_workers,
        int(os.cpu_count() or 1),
        total_samples,
    )
    num_chunks = int(
        np.ceil(
            total_samples / float(chunk_size)
        )
    )

    emitters, receivers = generate_sensor_positions(
        nx,
        ny,
        dx,
        dy,
        radius,
        n_emitters,
        n_receivers,
    )
    emitters_array = np.asarray(emitters)

    receiver_x = np.asarray(
        [
            int(receiver_pos[0])
            for receiver_pos in receivers
        ],
        dtype=np.intp,
    )

    receiver_y = np.asarray(
        [
            int(receiver_pos[1])
            for receiver_pos in receivers
        ],
        dtype=np.intp,
    )
    include_pde = bool(include_pde)

    k = (
        int(
            max(
                0,
                min(
                    int(pde_emitters_k),
                    n_emitters,
                ),
            )
        )
        if include_pde
        else 0
    )

    store_collocation_T = bool(store_collocation_T)

    n_collocation = int(
        max(
            1,
            n_collocation,
        )
    )

    colloc_emitters_k = int(
        max(
            1,
            min(
                int(colloc_emitters_k),
                n_emitters,
            ),
        )
    )

    synthetic_roi_tof_enable = bool(
        synthetic_roi_tof_enable
    )

    tof_norm = str(
        tof_norm or "none"
    ).lower()

    if tof_norm not in {"none", "zscore", "max"}:
        raise ValueError(
            f"Unsupported tof_norm={tof_norm}. "
            "Use none|zscore|max"
        )

    exclude_frac = float(
        exclude_frac or 0.0
    )

    tof_mask = None

    if exclude_frac > 0.0:
        tof_mask = compute_limited_view_mask(
            n_emitters=n_emitters,
            n_receivers=n_receivers,
            exclude_frac=exclude_frac,
        ).astype(np.float32)

        kept_per_row = (
            int(
                np.round(
                    float(
                        tof_mask.sum(axis=1).mean()
                    )
                )
            )
            if tof_mask.size
            else 0
        )

        excluded_per_row = int(
            n_receivers - kept_per_row
        )

        log_message(
            f"[dataset.py] Limited-view mask: "
            f"kept={int(tof_mask.sum())}/{tof_mask.size} "
            f"({float(tof_mask.mean()):.4f}), "
            f"per emitter kept~{kept_per_row}, "
            f"excluded~{excluded_per_row}"
        )

    # ---------------------------------------------------------
    # Global collocation locations and RNG
    # ---------------------------------------------------------

    colloc_xy: Optional[np.ndarray] = None
    colloc_rng: Optional[np.random.Generator] = None

    if store_collocation_T:
        # The same RNG sequence as the original implementation.
        colloc_rng = np.random.default_rng(
            int(colloc_seed)
            if int(colloc_seed) != 0
            else 123
        )

        xs = colloc_rng.integers(
            low=0,
            high=nx,
            size=n_collocation,
            endpoint=False,
        )

        ys = colloc_rng.integers(
            low=0,
            high=ny,
            size=n_collocation,
            endpoint=False,
        )

        colloc_xy = np.stack(
            [xs, ys],
            axis=1,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Synthetic ROI configuration
    # ---------------------------------------------------------

    roi_rng: Optional[np.random.Generator] = None

    if synthetic_roi_tof_enable:
        # The same RNG sequence as the original implementation.
        roi_rng = np.random.default_rng(
            int(synthetic_roi_seed)
            if int(synthetic_roi_seed) != 0
            else 99173
        )

        roi_cx = (
            float(phys_x) / 2.0
            if synthetic_roi_center_x is None
            else float(synthetic_roi_center_x)
        )

        roi_cy = (
            float(phys_y) / 2.0
            if synthetic_roi_center_y is None
            else float(synthetic_roi_center_y)
        )

        rlo = float(
            min(
                synthetic_roi_radius_min,
                synthetic_roi_radius_max,
            )
        )

        rhi = float(
            max(
                synthetic_roi_radius_min,
                synthetic_roi_radius_max,
            )
        )

        clo = float(
            min(
                synthetic_roi_outer_c_min,
                synthetic_roi_outer_c_max,
            )
        )

        chi = float(
            max(
                synthetic_roi_outer_c_min,
                synthetic_roi_outer_c_max,
            )
        )

        log_message(
            "[dataset.py] Synthetic ROI-referenced ToF enabled: "
            f"center=({roi_cx:.6g},{roi_cy:.6g}) m, "
            f"radius=[{rlo:.6g},{rhi:.6g}] m, "
            f"outer_c=[{clo:.6g},{chi:.6g}] m/s, "
            f"reference_c={float(synthetic_roi_reference_c):.6g}, "
            f"synthetic_c={float(synthetic_roi_synthetic_c):.6g}, "
            f"nonintersect={synthetic_roi_nonintersect_mode}, "
            f"clip_us={float(synthetic_roi_residual_clip_us):.6g}"
        )

    else:
        roi_cx = None
        roi_cy = None
        rlo = None
        rhi = None
        clo = None
        chi = None

    log_message(
        f"[dataset.py] Building chunked pairs cache: "
        f"{cache_path}"
    )

    log_message(
        f"[dataset.py] chunks root: "
        f"{chunk_root}"
    )

    log_message(
        f"[dataset.py] chunk_size={chunk_size}, "
        f"num_chunks={num_chunks}"
    )
    log_message(
        f"[dataset.py] pairs_workers={pairs_workers}"
    )
    log_message(
        f"[dataset.py] include_pde={include_pde}, "
        f"pde_emitters_k={k}"
    )

    log_message(
        f"[dataset.py] tof_norm={tof_norm}"
    )

    log_message(
        f"[dataset.py] "
        f"store_collocation_T={store_collocation_T}, "
        f"n_collocation={n_collocation}, "
        f"colloc_emitters_k={colloc_emitters_k}"
    )

    total_msfm = (
        total_samples * int(n_emitters)
    )

    log_message(
        f"[dataset.py] Pairs-cache MSFM workload: "
        f"samples={total_samples}, "
        f"emitters={n_emitters}, "
        f"receivers={n_receivers}, "
        f"total_msfm_solves={total_msfm}."
    )

    if store_collocation_T:
        log_message(
            "[dataset.py] Collocation T targets are sampled "
            "from the same MSFM maps used for ToF; "
            "they add storage and indexing cost, "
            "not extra MSFM solves."
        )

    # ---------------------------------------------------------
    # Global running statistics
    # ---------------------------------------------------------

    tof_stats = _RunningStats()
    raw_stats = _RunningStats()

    roi_count = 0
    roi_radius_sum = 0.0
    roi_outer_c_sum = 0.0

    # ---------------------------------------------------------
    # Resume completed chunks
    # ---------------------------------------------------------

    completed_chunks = 0

    if resume_chunks:
        for chunk_id in range(num_chunks):
            cdir = _pairs_chunk_dir(
                cache_path,
                chunk_id,
            )

            manifest_path = (
                cdir / "manifest.pt"
            )

            if not manifest_path.exists():
                break

            try:
                manifest = torch.load(
                    str(manifest_path),
                    weights_only=False,
                )

                start = int(
                    manifest.get("start", -1)
                )

                end = int(
                    manifest.get("end", -1)
                )

                expected_start = (
                    chunk_id * chunk_size
                )

                expected_end = min(
                    expected_start + chunk_size,
                    total_samples,
                )

                if (
                    not bool(
                        manifest.get(
                            "complete",
                            False,
                        )
                    )
                    or start != expected_start
                    or end != expected_end
                ):
                    break

                for field in manifest.get(
                    "fields",
                    [],
                ):
                    field_path = (
                        cdir / f"{field}.npy"
                    )

                    if not field_path.exists():
                        raise FileNotFoundError(
                            field_path
                        )

                tof_stats.merge(
                    manifest.get("tof_stats")
                )

                raw_stats.merge(
                    manifest.get("raw_stats")
                )

                roi_summary = manifest.get(
                    "roi_summary",
                    {},
                )

                roi_count += int(
                    roi_summary.get(
                        "count",
                        0,
                    )
                )

                roi_radius_sum += float(
                    roi_summary.get(
                        "radius_sum",
                        0.0,
                    )
                )

                roi_outer_c_sum += float(
                    roi_summary.get(
                        "outer_c_sum",
                        0.0,
                    )
                )

                completed_chunks += 1

            except Exception as exc:
                log_message(
                    f"[dataset.py] Existing chunk "
                    f"{chunk_id} is incomplete/invalid "
                    f"and will be rebuilt: {exc}"
                )

                break

    start_sample = min(
        completed_chunks * chunk_size,
        total_samples,
    )

    if completed_chunks > 0:
        log_message(
            f"[dataset.py] Resuming pairs cache after "
            f"{completed_chunks}/{num_chunks} complete chunks "
            f"({start_sample}/{total_samples} samples)."
        )

    # Advance the RNGs over all samples already stored in chunks.
    # This preserves the same random sequence as a clean run.
    if colloc_rng is not None:
        for _ in range(start_sample):
            colloc_rng.choice(
                n_emitters,
                size=colloc_emitters_k,
                replace=False,
            )

    if roi_rng is not None:
        for _ in range(start_sample):
            roi_rng.uniform(rlo, rhi)
            roi_rng.uniform(clo, chi)

    build_t0 = time.perf_counter()

    completed_msfm = (
        start_sample * n_emitters
    )

    progress_every_samples = max(
        1,
        min(
            100,
            total_samples // 20
            if total_samples >= 20
            else 1,
        ),
    )

    # ---------------------------------------------------------
    # Build one chunk at a time
    # ---------------------------------------------------------

    for chunk_id in range(
        completed_chunks,
        num_chunks,
    ):
        start = chunk_id * chunk_size

        end = min(
            start + chunk_size,
            total_samples,
        )

        count = end - start

        cdir = _pairs_chunk_dir(
            cache_path,
            chunk_id,
        )

        cdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Predictable memory consumption.
        chunk_tof = np.empty(
            (
                count,
                n_emitters,
                n_receivers,
            ),
            dtype=np.float32,
        )

        chunk_raw = (
            np.empty_like(chunk_tof)
            if synthetic_roi_tof_enable
            else None
        )

        chunk_pde = (
            np.empty(
                (
                    count,
                    2,
                    nx,
                    ny,
                ),
                dtype=np.float32,
            )
            if include_pde
            else None
        )

        chunk_colloc_emitters = (
            np.empty(
                (
                    count,
                    colloc_emitters_k,
                ),
                dtype=np.int64,
            )
            if store_collocation_T
            else None
        )

        chunk_colloc_T = (
            np.empty(
                (
                    count,
                    colloc_emitters_k,
                    n_collocation,
                ),
                dtype=np.float32,
            )
            if store_collocation_T
            else None
        )

        chunk_tof_stats = _RunningStats()
        chunk_raw_stats = _RunningStats()

        chunk_roi_count = 0
        chunk_roi_radius_sum = 0.0
        chunk_roi_outer_c_sum = 0.0

        chunk_roi_records: List[Dict] = []
        # -----------------------------------------------------
        # Prepare deterministic per-sample tasks
        # -----------------------------------------------------

        tasks: List[Dict] = []

        for local_i, i in enumerate(
            range(start, end)
        ):
            colloc_emitters_i = None

            if store_collocation_T:
                assert colloc_rng is not None

                colloc_emitters_i = (
                    colloc_rng.choice(
                        n_emitters,
                        size=colloc_emitters_k,
                        replace=False,
                    ).astype(np.int64)
                )

            rr = None
            oc = None

            if synthetic_roi_tof_enable:
                assert roi_rng is not None

                rr = float(
                    roi_rng.uniform(
                        rlo,
                        rhi,
                    )
                )

                oc = float(
                    roi_rng.uniform(
                        clo,
                        chi,
                    )
                )

            tasks.append(
                {
                    "local_i": int(local_i),
                    "sample_index": int(i),
                    "sos_map": sos_data[i],
                    "emitters": emitters_array,
                    "receiver_x": receiver_x,
                    "receiver_y": receiver_y,
                    "dx": float(dx),
                    "dy": float(dy),
                    "include_pde": bool(include_pde),
                    "pde_emitters_k": int(k),
                    "colloc_emitters_i":
                        colloc_emitters_i,
                    "colloc_xy": colloc_xy,
                    "roi_radius": rr,
                    "roi_outer_c": oc,
                }
            )

        # -----------------------------------------------------
        # Compute samples on multiple CPU processes
        # -----------------------------------------------------

        for result in _iter_pairs_sample_results(
            tasks,
            pairs_workers,
        ):
            local_i = int(result["local_i"])
            i = int(result["sample_index"])

            raw_tof_matrix = np.asarray(
                result["raw_tof_matrix"],
                dtype=np.float32,
            )

            pde_i = result.get("pde", None)

            colloc_emitters_i = result.get(
                "colloc_emitters_i",
                None,
            )

            colloc_T_i = result.get(
                "colloc_T",
                None,
            )

            rr = result.get(
                "roi_radius",
                None,
            )

            oc = result.get(
                "roi_outer_c",
                None,
            )

            sample_elapsed = float(
                result.get(
                    "sample_elapsed",
                    0.0,
                )
            )

            completed_msfm += n_emitters

            # -------------------------------------------------
            # ROI-referenced ToF
            # -------------------------------------------------

            if synthetic_roi_tof_enable:
                if rr is None or oc is None:
                    raise RuntimeError(
                        "Missing synthetic ROI parameters "
                        f"for sample {i}."
                    )

                (
                    full_len,
                    roi_len,
                    outer_len,
                ) = _build_roi_chord_geometry(
                    emitters,
                    receivers,
                    dx,
                    dy,
                    float(roi_cx),
                    float(roi_cy),
                    float(rr),
                )

                tof_matrix = (
                    _apply_synthetic_roi_tof_reference(
                        raw_tof_matrix,
                        full_len,
                        roi_len,
                        outer_len,
                        outer_c=float(oc),
                        reference_c=float(
                            synthetic_roi_reference_c
                        ),
                        synthetic_c=float(
                            synthetic_roi_synthetic_c
                        ),
                        min_chord_frac=float(
                            synthetic_roi_min_chord_frac
                        ),
                        nonintersect_mode=str(
                            synthetic_roi_nonintersect_mode
                        ),
                        residual_clip_us=float(
                            synthetic_roi_residual_clip_us
                        ),
                    )
                )

                assert chunk_raw is not None

                chunk_raw[
                    local_i
                ] = raw_tof_matrix

                full_safe = np.maximum(
                    full_len,
                    1e-12,
                )

                chord_frac = (
                    roi_len / full_safe
                )

                record = {
                    "sample_index": int(i),
                    "roi_radius_m": float(rr),
                    "roi_outer_c": float(oc),
                    "roi_channel_fraction": float(
                        np.mean(
                            chord_frac
                            >= float(
                                synthetic_roi_min_chord_frac
                            )
                        )
                    ),
                    "roi_chord_fraction_mean": float(
                        np.mean(chord_frac)
                    ),
                    "roi_chord_fraction_max": float(
                        np.max(chord_frac)
                    ),
                    "raw_tof_mean_s": float(
                        np.nanmean(
                            raw_tof_matrix
                        )
                    ),
                    "adjusted_tof_mean_s": float(
                        np.nanmean(
                            tof_matrix
                        )
                    ),
                    "adjusted_minus_raw_mean_s": float(
                        np.nanmean(
                            tof_matrix
                            - raw_tof_matrix
                        )
                    ),
                }

                chunk_roi_records.append(
                    record
                )

                chunk_roi_count += 1
                chunk_roi_radius_sum += float(rr)
                chunk_roi_outer_c_sum += float(oc)

            else:
                tof_matrix = raw_tof_matrix

            # -------------------------------------------------
            # Mask and streaming normalization statistics
            # -------------------------------------------------

            if tof_mask is not None:
                valid = tof_mask > 0.5

                chunk_tof_stats.update(
                    tof_matrix[valid]
                )

                if synthetic_roi_tof_enable:
                    chunk_raw_stats.update(
                        raw_tof_matrix[valid]
                    )

                tof_to_store = (
                    tof_matrix * tof_mask
                ).astype(np.float32)

            else:
                chunk_tof_stats.update(
                    tof_matrix
                )

                if synthetic_roi_tof_enable:
                    chunk_raw_stats.update(
                        raw_tof_matrix
                    )

                tof_to_store = (
                    tof_matrix.astype(
                        np.float32,
                        copy=False,
                    )
                )

            chunk_tof[
                local_i
            ] = tof_to_store

            # -------------------------------------------------
            # Optional PDE
            # -------------------------------------------------

            if (
                include_pde
                and chunk_pde is not None
            ):
                if pde_i is None:
                    raise RuntimeError(
                        f"Worker returned no PDE data "
                        f"for sample {i}."
                    )

                chunk_pde[
                    local_i
                ] = np.asarray(
                    pde_i,
                    dtype=np.float32,
                )

            # -------------------------------------------------
            # Optional collocation data
            # -------------------------------------------------

            if (
                store_collocation_T
                and chunk_colloc_emitters
                is not None
                and chunk_colloc_T
                is not None
            ):
                if (
                    colloc_emitters_i is None
                    or colloc_T_i is None
                ):
                    raise RuntimeError(
                        f"Worker returned incomplete "
                        f"collocation data for sample {i}."
                    )

                chunk_colloc_emitters[
                    local_i
                ] = np.asarray(
                    colloc_emitters_i,
                    dtype=np.int64,
                )

                chunk_colloc_T[
                    local_i
                ] = np.asarray(
                    colloc_T_i,
                    dtype=np.float32,
                )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                (
                    i + 1
                )
                % progress_every_samples
                == 0
                or (
                    i + 1
                )
                == total_samples
            ):
                elapsed = (
                    time.perf_counter()
                    - build_t0
                )

                done_this_run = max(
                    1,
                    completed_msfm
                    - start_sample
                    * n_emitters,
                )

                rate = (
                    done_this_run
                    / max(
                        elapsed,
                        1e-12,
                    )
                )

                remaining = (
                    total_msfm
                    - completed_msfm
                ) / max(
                    rate,
                    1e-12,
                )

                log_message(
                    f"[dataset.py] Cache progress: "
                    f"samples "
                    f"{i + 1}/{total_samples} "
                    f"("
                    f"{100.0 * (i + 1) / max(total_samples, 1):.1f}"
                    f"%), "
                    f"MSFM "
                    f"{completed_msfm}/{total_msfm}, "
                    f"workers={pairs_workers}, "
                    f"worker_sample_time="
                    f"{_format_seconds(sample_elapsed)}, "
                    f"elapsed="
                    f"{_format_seconds(elapsed)}, "
                    f"ETA="
                    f"{_format_seconds(remaining)}"
                )
        # -----------------------------------------------------
        # Save completed chunk
        # -----------------------------------------------------

        fields = [
            "sos",
            "tof",
        ]

        _atomic_save_npy(
            cdir / "sos.npy",
            sos_data[start:end],
        )

        _atomic_save_npy(
            cdir / "tof.npy",
            chunk_tof,
        )

        if chunk_raw is not None:
            _atomic_save_npy(
                cdir / "tof_raw_eikonal.npy",
                chunk_raw,
            )

            fields.append(
                "tof_raw_eikonal"
            )

        if chunk_pde is not None:
            _atomic_save_npy(
                cdir / "pde.npy",
                chunk_pde,
            )

            fields.append("pde")

        if (
            chunk_colloc_emitters
            is not None
            and chunk_colloc_T
            is not None
        ):
            _atomic_save_npy(
                cdir
                / "colloc_emitters_idx.npy",
                chunk_colloc_emitters,
            )

            _atomic_save_npy(
                cdir / "colloc_T.npy",
                chunk_colloc_T,
            )

            fields.extend(
                [
                    "colloc_emitters_idx",
                    "colloc_T",
                ]
            )

        if chunk_roi_records:
            _atomic_torch_save(
                {
                    "records":
                        chunk_roi_records
                },
                cdir
                / "synthetic_roi_records.pt",
            )

        manifest = {
            "complete": True,
            "format":
                "pairs_cache_chunk_npy_v1",
            "chunk_id": int(chunk_id),
            "start": int(start),
            "end": int(end),
            "count": int(count),
            "fields": fields,
            "tof_stats":
                chunk_tof_stats.state_dict(),
            "raw_stats": (
                chunk_raw_stats.state_dict()
                if synthetic_roi_tof_enable
                else None
            ),
            "roi_summary": {
                "count":
                    int(chunk_roi_count),
                "radius_sum":
                    float(
                        chunk_roi_radius_sum
                    ),
                "outer_c_sum":
                    float(
                        chunk_roi_outer_c_sum
                    ),
            },
        }

        # manifest.pt is written last, so it marks a complete chunk.
        _atomic_torch_save(
            manifest,
            cdir / "manifest.pt",
        )

        tof_stats.merge(
            chunk_tof_stats.state_dict()
        )

        if synthetic_roi_tof_enable:
            raw_stats.merge(
                chunk_raw_stats.state_dict()
            )

        roi_count += chunk_roi_count
        roi_radius_sum += (
            chunk_roi_radius_sum
        )
        roi_outer_c_sum += (
            chunk_roi_outer_c_sum
        )

        log_message(
            f"[dataset.py] Saved pairs chunk "
            f"{chunk_id + 1}/{num_chunks}: "
            f"{cdir} "
            f"samples={start}:{end}"
        )

        # Release the large arrays before the next chunk.
        del chunk_tof
        del chunk_raw
        del chunk_pde
        del chunk_colloc_emitters
        del chunk_colloc_T

    # ---------------------------------------------------------
    # Final metadata and small index file
    # ---------------------------------------------------------

    tof_final = tof_stats.final()

    md_pairs = dict(md)

    md_pairs.update(
        {
            "N": int(total_samples),
            "include_pde":
                bool(include_pde),
            "pde_emitters_k":
                int(k),
            "pde_type": (
                "eikonal_traveltime"
                if include_pde
                else None
            ),
            "pairs_cache_path":
                str(cache_path),
            "exclude_frac":
                float(exclude_frac),
            "tof_norm": {
                "type":
                    tof_norm,
                "eps":
                    float(tof_norm_eps),
                "mean":
                    float(
                        tof_final["mean"]
                    ),
                "std":
                    float(
                        tof_final["std"]
                    ),
                "max":
                    float(
                        tof_final["max"]
                    ),
            },
            "chunked": True,
            "chunk_format":
                "npy_mmap_v1",
            "chunk_size":
                int(chunk_size),
            "pairs_workers":
                int(pairs_workers),
            "num_chunks":
                int(num_chunks),
            "chunk_root_name":
                chunk_root.name,
            "chunk_fields": [
                "sos",
                "tof",
                *(
                    ["tof_raw_eikonal"]
                    if synthetic_roi_tof_enable
                    else []
                ),
                *(
                    ["pde"]
                    if include_pde
                    else []
                ),
                *(
                    [
                        "colloc_emitters_idx",
                        "colloc_T",
                    ]
                    if store_collocation_T
                    else []
                ),
            ],
        }
    )

    if synthetic_roi_tof_enable:
        raw_final = raw_stats.final()

        md_pairs[
            "raw_eikonal_tof_norm"
        ] = {
            "type":
                tof_norm,
            "eps":
                float(tof_norm_eps),
            "mean":
                float(
                    raw_final["mean"]
                ),
            "std":
                float(
                    raw_final["std"]
                ),
            "max":
                float(
                    raw_final["max"]
                ),
            "computed_on": (
                "limited_view_kept_channels"
                if tof_mask is not None
                else "all_channels"
            ),
        }

        md_pairs[
            "synthetic_roi_tof"
        ] = {
            "enabled": True,
            "description": (
                "Saved ToF has been "
                "re-referenced through "
                "a two-region ROI model: "
                "raw Eikonal residual "
                "relative to outer/ROI "
                "background plus homogeneous "
                "synthetic reference."
            ),
            "center_x_m":
                float(roi_cx),
            "center_y_m":
                float(roi_cy),
            "radius_min_m":
                float(rlo),
            "radius_max_m":
                float(rhi),
            "outer_c_min":
                float(clo),
            "outer_c_max":
                float(chi),
            "reference_c":
                float(
                    synthetic_roi_reference_c
                ),
            "synthetic_c":
                float(
                    synthetic_roi_synthetic_c
                ),
            "min_chord_frac":
                float(
                    synthetic_roi_min_chord_frac
                ),
            "residual_clip_us":
                float(
                    synthetic_roi_residual_clip_us
                ),
            "nonintersect_mode":
                str(
                    synthetic_roi_nonintersect_mode
                ),
            "seed":
                int(synthetic_roi_seed),
            "sample_radius_mean_m":
                float(
                    roi_radius_sum
                    / max(
                        roi_count,
                        1,
                    )
                ),
            "sample_outer_c_mean":
                float(
                    roi_outer_c_sum
                    / max(
                        roi_count,
                        1,
                    )
                ),
            "records_storage":
                "per_chunk:"
                "synthetic_roi_records.pt",
        }

    index_payload: Dict = {
        "format":
            "pairs_cache_chunked_npy_v1",
        "metadata":
            md_pairs,
        "N":
            int(total_samples),
        "chunk_root_name":
            chunk_root.name,
    }

    if tof_mask is not None:
        index_payload[
            "tof_mask"
        ] = tof_mask.astype(
            np.float32
        )

    if colloc_xy is not None:
        index_payload[
            "colloc_xy"
        ] = colloc_xy.astype(
            np.float32
        )

    # The index is written only after all chunks are complete.
    _atomic_torch_save(
        index_payload,
        cache_path,
    )

    log_message(
        f"[dataset.py] Saved chunked pairs cache index: "
        f"{cache_path} "
        f"(N={total_samples}, "
        f"chunks={num_chunks})"
    )

    return index_payload
class PairsCacheDataset(Dataset):
    """
    Load either:

    1. The original single-file pairs cache.
    2. The new progressive chunked mmap cache.

    The public __getitem__ format remains unchanged.
    """

    def __init__(
        self,
        cache_path: str,
        sos_min: float,
        sos_max: float,
        return_tof_mask: bool = False,
    ):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Pairs cache does not exist: "
                f"{cache_path}"
            )

        self.cache_path = str(cache_path)

        data = torch.load(
            self.cache_path,
            weights_only=False,
        )

        self.metadata = (
            data.get("metadata", {})
            if isinstance(data, dict)
            else {}
        )

        self._chunked = bool(
            isinstance(
                self.metadata,
                dict,
            )
            and self.metadata.get(
                "chunked",
                False,
            )
            and self.metadata.get(
                "chunk_format"
            )
            == "npy_mmap_v1"
        )

        self._chunk_handles: Dict[
            int,
            Dict[str, np.ndarray],
        ] = {}

        self.sos_min = float(sos_min)
        self.sos_max = float(sos_max)

        self.return_tof_mask = bool(
            return_tof_mask
        )

        if self._chunked:
            self._N = int(
                self.metadata.get(
                    "N",
                    data.get("N", 0),
                )
            )

            self.chunk_size = int(
                self.metadata["chunk_size"]
            )

            self.num_chunks = int(
                self.metadata["num_chunks"]
            )

            root_name = str(
                self.metadata.get(
                    "chunk_root_name",
                    data.get(
                        "chunk_root_name",
                        "",
                    ),
                )
            )

            self.chunk_root = (
                Path(self.cache_path).parent
                / root_name
            )

            if not self.chunk_root.exists():
                raise FileNotFoundError(
                    "Pairs chunk directory "
                    f"does not exist: "
                    f"{self.chunk_root}"
                )

            fields = set(
                map(
                    str,
                    self.metadata.get(
                        "chunk_fields",
                        [],
                    ),
                )
            )

            self.has_raw_eikonal = (
                "tof_raw_eikonal"
                in fields
            )

            self.has_pde = (
                "pde" in fields
            )

            # Compatibility attributes.
            # Full arrays intentionally remain unloaded.
            self.sos_data = None
            self.tof_data = None
            self.tof_raw_eikonal = None
            self.pde_data = None
            self.colloc_emitters_idx = None
            self.colloc_T = None

            self.tof_mask = (
                np.asarray(
                    data["tof_mask"],
                    dtype=np.float32,
                )
                if (
                    "tof_mask" in data
                    and data["tof_mask"]
                    is not None
                )
                else None
            )

            self.colloc_xy = (
                np.asarray(
                    data["colloc_xy"],
                    dtype=np.float32,
                )
                if (
                    "colloc_xy" in data
                    and data["colloc_xy"]
                    is not None
                )
                else None
            )

            self.has_collocation = bool(
                self.colloc_xy is not None
                and "colloc_emitters_idx"
                in fields
                and "colloc_T"
                in fields
            )

        else:
            # Backward-compatible classic cache.
            self.sos_data = np.asarray(
                data["sos"],
                dtype=np.float32,
            )

            self.tof_data = np.asarray(
                data["tof"],
                dtype=np.float32,
            )

            self.tof_raw_eikonal = (
                np.asarray(
                    data[
                        "tof_raw_eikonal"
                    ],
                    dtype=np.float32,
                )
                if (
                    "tof_raw_eikonal"
                    in data
                    and data[
                        "tof_raw_eikonal"
                    ]
                    is not None
                )
                else None
            )

            self.tof_mask = (
                np.asarray(
                    data["tof_mask"],
                    dtype=np.float32,
                )
                if (
                    "tof_mask" in data
                    and data["tof_mask"]
                    is not None
                )
                else None
            )

            self.pde_data = (
                np.asarray(
                    data["pde"],
                    dtype=np.float32,
                )
                if (
                    "pde" in data
                    and data["pde"]
                    is not None
                )
                else None
            )

            self.colloc_xy = (
                np.asarray(
                    data["colloc_xy"],
                    dtype=np.float32,
                )
                if (
                    "colloc_xy" in data
                    and data["colloc_xy"]
                    is not None
                )
                else None
            )

            self.colloc_emitters_idx = (
                np.asarray(
                    data[
                        "colloc_emitters_idx"
                    ],
                    dtype=np.int64,
                )
                if (
                    "colloc_emitters_idx"
                    in data
                    and data[
                        "colloc_emitters_idx"
                    ]
                    is not None
                )
                else None
            )

            self.colloc_T = (
                np.asarray(
                    data["colloc_T"],
                    dtype=np.float32,
                )
                if (
                    "colloc_T" in data
                    and data["colloc_T"]
                    is not None
                )
                else None
            )

            self._N = int(
                self.sos_data.shape[0]
            )

            self.has_raw_eikonal = (
                self.tof_raw_eikonal
                is not None
            )

            self.has_pde = (
                self.pde_data
                is not None
            )

            self.has_collocation = bool(
                self.colloc_xy
                is not None
                and self.colloc_emitters_idx
                is not None
                and self.colloc_T
                is not None
            )

        # -----------------------------------------------------
        # ToF normalization
        # -----------------------------------------------------

        tof_norm = (
            self.metadata.get(
                "tof_norm",
                None,
            )
            if isinstance(
                self.metadata,
                dict,
            )
            else None
        )

        if not isinstance(
            tof_norm,
            dict,
        ):
            if self._chunked:
                raise ValueError(
                    "Chunked pairs cache "
                    "is missing "
                    "metadata['tof_norm']."
                )

            tof_norm = {
                "type": "zscore",
                "eps": 1e-6,
                "mean": float(
                    np.mean(
                        self.tof_data
                    )
                ),
                "std": float(
                    np.std(
                        self.tof_data
                    )
                ),
                "max": float(
                    np.max(
                        self.tof_data
                    )
                ),
            }

            if isinstance(
                self.metadata,
                dict,
            ):
                self.metadata[
                    "tof_norm"
                ] = tof_norm

        self.tof_norm = tof_norm

        raw_tof_norm = (
            self.metadata.get(
                "raw_eikonal_tof_norm",
                None,
            )
            if isinstance(
                self.metadata,
                dict,
            )
            else None
        )

        self.raw_eikonal_tof_norm = (
            raw_tof_norm
            if isinstance(
                raw_tof_norm,
                dict,
            )
            else tof_norm
        )

    def __getstate__(self):
        """
        Do not pickle open mmap handles into
        Windows DataLoader worker processes.
        """
        state = dict(self.__dict__)
        state["_chunk_handles"] = {}
        return state

    def _open_chunk(
        self,
        chunk_id: int,
    ) -> Dict[str, np.ndarray]:
        chunk_id = int(chunk_id)

        if chunk_id in self._chunk_handles:
            return self._chunk_handles[
                chunk_id
            ]

        cdir = (
            self.chunk_root
            / f"chunk_{chunk_id:04d}"
        )

        manifest_path = (
            cdir / "manifest.pt"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                "Missing pairs chunk "
                f"manifest: "
                f"{manifest_path}"
            )

        manifest = torch.load(
            str(manifest_path),
            weights_only=False,
        )

        if not bool(
            manifest.get(
                "complete",
                False,
            )
        ):
            raise RuntimeError(
                f"Pairs chunk is not "
                f"complete: {cdir}"
            )

        handles: Dict[
            str,
            np.ndarray,
        ] = {}

        handles["_manifest"] = manifest

        for field in manifest.get(
            "fields",
            [],
        ):
            field = str(field)

            field_path = (
                cdir / f"{field}.npy"
            )

            if not field_path.exists():
                raise FileNotFoundError(
                    "Missing pairs chunk "
                    f"field: {field_path}"
                )

            handles[field] = np.load(
                str(field_path),
                mmap_mode="r",
                allow_pickle=False,
            )

        self._chunk_handles[
            chunk_id
        ] = handles

        return handles

    def _sample_arrays(
        self,
        idx: int,
    ) -> Dict[str, np.ndarray]:
        idx = int(idx)

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if not self._chunked:
            output: Dict[
                str,
                np.ndarray,
            ] = {
                "sos":
                    self.sos_data[idx],
                "tof":
                    self.tof_data[idx],
            }

            if (
                self.tof_raw_eikonal
                is not None
            ):
                output[
                    "tof_raw_eikonal"
                ] = self.tof_raw_eikonal[
                    idx
                ]

            if self.pde_data is not None:
                output["pde"] = (
                    self.pde_data[idx]
                )

            if (
                self.colloc_emitters_idx
                is not None
            ):
                emitters = (
                    self.colloc_emitters_idx
                )

                if emitters.ndim == 1:
                    output[
                        "colloc_emitters_idx"
                    ] = emitters

                elif (
                    emitters.ndim == 2
                    and emitters.shape[0]
                    == len(self)
                ):
                    output[
                        "colloc_emitters_idx"
                    ] = emitters[idx]

                elif (
                    emitters.ndim == 2
                    and emitters.shape[0]
                    == 1
                ):
                    output[
                        "colloc_emitters_idx"
                    ] = emitters[0]

            if self.colloc_T is not None:
                output[
                    "colloc_T"
                ] = self.colloc_T[idx]

            return output

        chunk_id = (
            idx // self.chunk_size
        )

        local_idx = (
            idx
            - chunk_id
            * self.chunk_size
        )

        handles = self._open_chunk(
            chunk_id
        )

        manifest = handles[
            "_manifest"
        ]

        count = int(
            manifest.get(
                "count",
                0,
            )
        )

        if (
            local_idx < 0
            or local_idx >= count
        ):
            raise IndexError(
                f"local index "
                f"{local_idx} outside "
                f"chunk {chunk_id} "
                f"count={count}"
            )

        return {
            field: array[local_idx]
            for field, array
            in handles.items()
            if field != "_manifest"
        }

    def __len__(self) -> int:
        return int(self._N)

    def __getitem__(self, idx: int):
        sample = self._sample_arrays(
            int(idx)
        )

        # -----------------------------------------------------
        # SoS normalization
        # -----------------------------------------------------

        sos = (
            np.asarray(
                sample["sos"],
                dtype=np.float32,
            )
            - self.sos_min
        ) / (
            self.sos_max
            - self.sos_min
            + 1e-12
        )

        sos_t = torch.tensor(
            sos,
            dtype=torch.float32,
        )

        # -----------------------------------------------------
        # ToF normalization
        # -----------------------------------------------------

        tof = np.asarray(
            sample["tof"],
            dtype=np.float32,
        ).copy()

        norm = (
            self.tof_norm
            or {"type": "none"}
        )

        norm_type = str(
            norm.get(
                "type",
                "none",
            )
        ).lower()

        eps = float(
            norm.get(
                "eps",
                1e-6,
            )
        )

        if norm_type == "zscore":
            mean = float(
                norm.get(
                    "mean",
                    0.0,
                )
            )

            std = float(
                norm.get(
                    "std",
                    1.0,
                )
            )

            tof = (
                tof - mean
            ) / (
                std + eps
            )

        elif norm_type == "max":
            maximum = float(
                norm.get(
                    "max",
                    1.0,
                )
            )

            tof = (
                tof
                / (
                    maximum
                    + eps
                )
            )

        elif norm_type == "none":
            pass

        else:
            pass

        # Reapply the limited-view mask after normalization.
        if self.tof_mask is not None:
            tof = (
                tof * self.tof_mask
            )

        tof_t = torch.tensor(
            tof,
            dtype=torch.float32,
        )

        mask_np = (
            self.tof_mask.astype(
                np.float32
            )
            if self.tof_mask
            is not None
            else np.ones_like(
                tof,
                dtype=np.float32,
            )
        )

        tof_mask_t = torch.tensor(
            mask_np,
            dtype=torch.float32,
        )

        # -----------------------------------------------------
        # Optional PDE
        # -----------------------------------------------------

        pde_t = None

        if "pde" in sample:
            pde_t = torch.tensor(
                np.asarray(
                    sample["pde"],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )

        # -----------------------------------------------------
        # Optional raw Eikonal ToF
        # -----------------------------------------------------

        raw_tof_item = None

        if "tof_raw_eikonal" in sample:
            raw_tof = np.asarray(
                sample[
                    "tof_raw_eikonal"
                ],
                dtype=np.float32,
            ).copy()

            if self.tof_mask is not None:
                raw_tof = (
                    raw_tof
                    * self.tof_mask
                )

            raw_tof_item = {
                "raw_eikonal_tof_phys":
                    torch.tensor(
                        raw_tof,
                        dtype=torch.float32,
                    ),
                "raw_eikonal_tof_norm":
                    self.raw_eikonal_tof_norm,
            }

        # -----------------------------------------------------
        # Optional collocation data
        # -----------------------------------------------------

        colloc = None

        if (
            self.colloc_xy is not None
            and "colloc_emitters_idx"
            in sample
            and "colloc_T"
            in sample
        ):
            colloc_T_phys = np.asarray(
                sample["colloc_T"],
                dtype=np.float32,
            )

            emitters_i = np.asarray(
                sample[
                    "colloc_emitters_idx"
                ],
                dtype=np.int64,
            )

            if self.colloc_xy.ndim == 2:
                colloc_xy_i = (
                    self.colloc_xy
                )

            elif (
                self.colloc_xy.ndim
                == 3
                and self.colloc_xy.shape[0]
                == len(self)
            ):
                colloc_xy_i = (
                    self.colloc_xy[
                        int(idx)
                    ]
                )

            else:
                raise ValueError(
                    "Unsupported colloc_xy "
                    f"shape "
                    f"{self.colloc_xy.shape}"
                )

            if colloc_T_phys.ndim != 2:
                raise ValueError(
                    "Unsupported colloc_T "
                    f"sample shape "
                    f"{colloc_T_phys.shape}; "
                    "expected (k,P)"
                )

            if (
                int(
                    colloc_T_phys.shape[0]
                )
                != int(
                    emitters_i.shape[0]
                )
            ):
                raise ValueError(
                    "colloc_T/emitters "
                    f"mismatch for sample "
                    f"{idx}: colloc_T has "
                    f"k="
                    f"{colloc_T_phys.shape[0]}, "
                    f"emitters has "
                    f"k="
                    f"{emitters_i.shape[0]}"
                )

            colloc = {
                "colloc_xy":
                    torch.tensor(
                        colloc_xy_i,
                        dtype=torch.float32,
                    ),
                "colloc_emitters_idx":
                    torch.tensor(
                        emitters_i,
                        dtype=torch.long,
                    ),
                "colloc_T_phys":
                    torch.tensor(
                        colloc_T_phys,
                        dtype=torch.float32,
                    ),
                "colloc_T":
                    torch.tensor(
                        colloc_T_phys,
                        dtype=torch.float32,
                    ),
            }

        # Preserve the original return order.
        items = [
            sos_t,
            tof_t,
        ]

        if self.return_tof_mask:
            items.append(
                tof_mask_t
            )

        if pde_t is not None:
            items.append(
                pde_t
            )

        if colloc is not None:
            items.append(
                colloc
            )

        if raw_tof_item is not None:
            items.append(
                raw_tof_item
            )

        return tuple(items)
# -----------------------------
# Original in-memory dataset retained for older internal modes.
# -----------------------------

class TomographyDataset(Dataset):
    def __init__(self, num_samples, nx, ny, phys_x, phys_y,
                 sos_water, sos_min, sos_max, max_shapes,
                 min_shapes=0,
                 radius=0.04, dll_path=None, cache_path=None,
                 n_emitters=32, n_receivers=32,
                 include_pde: bool = False,
                 pde_emitters_k: int = 1,
                 data_seed: int = 0):

        self.num_samples = num_samples
        self.nx, self.ny = nx, ny
        self.phys_x, self.phys_y = phys_x, phys_y
        self.sos_water = sos_water
        self.sos_min, self.sos_max = sos_min, sos_max
        self.max_shapes = max_shapes
        self.min_shapes = int(max(1, min(int(min_shapes) if int(min_shapes) > 0 else max(1, int(max_shapes) // 2), int(max_shapes))))
        self.radius = radius
        self.n_emitters = n_emitters
        self.n_receivers = n_receivers
        self.cache_path = cache_path
        self.dx, self.dy = phys_x / (nx - 1), phys_y / (ny - 1)

        self.include_pde = bool(include_pde)
        self.pde_emitters_k = int(max(0, min(pde_emitters_k, n_emitters)))
        self.data_seed = int(data_seed)

        self.metadata = {
            'num_samples': num_samples, 'nx': nx, 'ny': ny,
            'phys_x': phys_x, 'phys_y': phys_y, 'sos_water': sos_water,
            'sos_min': sos_min, 'sos_max': sos_max, 'max_shapes': max_shapes, 'min_shapes': self.min_shapes,
            'radius': radius, 'n_emitters': n_emitters, 'n_receivers': n_receivers,
            'grid_spacing': (self.dx, self.dy),
            'include_pde': self.include_pde,
            'pde_emitters_k': self.pde_emitters_k,
            'data_seed': int(data_seed),
            'pde_type': 'eikonal_traveltime' if self.include_pde else None,
        }

        self.emitters, self.receivers = generate_sensor_positions(
            nx, ny, self.dx, self.dy, radius, n_emitters, n_receivers
        )

        if cache_path and os.path.exists(cache_path):
            log_message(f"[dataset.py] Loading data from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=False)
            self.sos_data = data['sos']
            self.tof_data = data['tof']
            self.pde_data = data.get('pde', None)
            if 'metadata' in data:
                log_message("[dataset.py] Verified Cache Metadata:")
                log_message(pprint.pformat(data['metadata']))
                self.metadata = data['metadata']
        else:
            log_message(f"[dataset.py] Generating {num_samples} samples...")
            self.sos_data, self.tof_data, self.pde_data = self._generate_data()
            if cache_path:
                self._save_cache()

    def _generate_data(self):
        sos_list, tof_list, pde_list = [], [], []
        base_seed = int(self.data_seed)

        for i in range(self.num_samples):
            if base_seed != 0:
                s_i = base_seed + i
                np.random.seed(s_i)
                random.seed(s_i)

            sos_map = create_random_sos_map(
                self.nx, self.ny, self.dx, self.dy, self.radius,
                self.sos_water, self.sos_min, self.sos_max, self.max_shapes,
                min_shapes=self.min_shapes
            )

            tof_matrix = np.zeros((self.n_emitters, self.n_receivers), dtype=np.float32)
            resid_acc = None
            gradmag_acc = None
            k = self.pde_emitters_k if self.include_pde else 0

            for e_idx, emitter_pos in enumerate(self.emitters):
                t_map = msfm(sos_map, emitter_pos, dx=self.dx, dy=self.dy)

                for r_idx, receiver_pos in enumerate(self.receivers):
                    tof_matrix[e_idx, r_idx] = t_map[int(receiver_pos[0]), int(receiver_pos[1])]

                if self.include_pde and e_idx < k:
                    dTdx, dTdy = np.gradient(t_map.astype(np.float32), self.dx, self.dy, edge_order=1)
                    grad_mag = np.sqrt(dTdx**2 + dTdy**2)
                    resid = sos_map.astype(np.float32) * grad_mag - 1.0

                    if resid_acc is None:
                        resid_acc = resid
                        gradmag_acc = grad_mag
                    else:
                        resid_acc += resid
                        gradmag_acc += grad_mag

            sos_list.append(sos_map.astype(np.float32))
            tof_list.append(tof_matrix)

            if self.include_pde:
                if k > 0 and resid_acc is not None:
                    resid_mean = resid_acc / float(k)
                    gradmag_mean = gradmag_acc / float(k)
                else:
                    resid_mean = np.zeros((self.nx, self.ny), dtype=np.float32)
                    gradmag_mean = np.zeros((self.nx, self.ny), dtype=np.float32)

                pde_list.append(np.stack([gradmag_mean, resid_mean], axis=0).astype(np.float32))

        pde_arr = np.array(pde_list) if self.include_pde else None
        return np.array(sos_list), np.array(tof_list), pde_arr

    def _save_cache(self):
        payload = {'sos': self.sos_data, 'tof': self.tof_data, 'metadata': self.metadata}
        if self.include_pde and self.pde_data is not None:
            payload['pde'] = self.pde_data
        torch.save(payload, self.cache_path)

    def __len__(self):
        return len(self.sos_data)

    def __getitem__(self, idx):
        sos = (self.sos_data[idx] - self.sos_min) / (self.sos_max - self.sos_min + 1e-12)
        sos_t = torch.tensor(sos, dtype=torch.float32)
        tof_t = torch.tensor(self.tof_data[idx], dtype=torch.float32)

        if self.include_pde and self.pde_data is not None:
            pde_t = torch.tensor(self.pde_data[idx], dtype=torch.float32)
            return sos_t, tof_t, pde_t
        return sos_t, tof_t











