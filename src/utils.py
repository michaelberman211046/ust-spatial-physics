# FILE: utils.py
"""
utils.py

Plotting + small utility helpers.

NOTE (dx,dy):
- The pipeline uses the node-spacing convention consistently:
      dx = phys_x / (nx - 1),  dy = phys_y / (ny - 1).
- This file does not define the spacing convention; it receives dx,dy from the
  caller and uses them for plotting and diagnostic ray tracing.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from logger import log_image, log_message
from msfm import msfm


# ---------------------------------------------------------------------
# Colorbar helper
# ---------------------------------------------------------------------
def add_tight_cbar(ax, im, label: str = ""):
    """Attach a colorbar to the right of `ax`, keeping layout stable."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = plt.colorbar(im, cax=cax)
    if label:
        cb.set_label(label)
    return cb


# ---------------------------------------------------------------------
# Physics setup plot
# ---------------------------------------------------------------------
def plot_physics_setup(sos_map, emitters, receivers, dx, dy, phys_x, phys_y, sos_min, sos_max):
    """Plot SoS + a sparse set of traced rays + sensors (diagnostic)."""
    sos_map = np.asarray(sos_map, dtype=float)
    emitters = np.asarray(emitters)
    receivers = np.asarray(receivers)

    nx, ny = sos_map.shape
    fig, ax = plt.subplots(figsize=(12, 10))
    extent = [0, float(phys_x), 0, float(phys_y)]

    # Draw SoS Map (diagnostic overlay)
    im = ax.imshow(
        sos_map.T,
        cmap="jet",
        extent=extent,
        origin="lower",
        alpha=0.40,
        vmin=float(sos_min),
        vmax=float(sos_max),
    )
    add_tight_cbar(ax, im, "m/s")

    # Trace a subset of rays for clarity
    skip = 8
    for i in range(0, len(emitters), skip):
        t_map = msfm(sos_map, emitters[i], dx=dx, dy=dy)
        gx, gy = np.gradient(t_map, dx, dy)

        for j in range(0, len(receivers), skip):
            curr_x, curr_y = float(receivers[j][0]), float(receivers[j][1])
            path_x, path_y = [curr_x * dx], [curr_y * dy]

            for _ in range(800):
                ix, iy = int(round(curr_x)), int(round(curr_y))
                if not (0 <= ix < nx and 0 <= iy < ny):
                    break

                dist_px = float(np.hypot(curr_x - emitters[i][0], curr_y - emitters[i][1]))
                if dist_px < 1.0:
                    break

                vx, vy = float(gx[ix, iy]), float(gy[ix, iy])
                mag = float(np.hypot(vx, vy))
                if mag > 1e-9:
                    # step towards lower travel time
                    curr_x -= (vx / mag) * 0.7
                    curr_y -= (vy / mag) * 0.7

                path_x.append(curr_x * dx)
                path_y.append(curr_y * dy)

            ax.plot(path_x, path_y, "b-", linewidth=0.5, alpha=0.3)

    # Plot Sensors (assumed in index units here)
    ax.scatter(emitters[:, 0] * dx, emitters[:, 1] * dy, c="red", s=20, label="Emitters")
    ax.scatter(receivers[:, 0] * dx, receivers[:, 1] * dy, c="black", s=20, label="Receivers")

    ax.set_title("Ray Paths (msfm propagation)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best")

    log_image(fig)
    plt.close(fig)


# ---------------------------------------------------------------------
# Display-only conversions
# ---------------------------------------------------------------------
def _conditional_denormalize_sos_for_display(s, sos_min: float, sos_max: float):
    """
    If `s` looks like normalized NN output, map to [sos_min, sos_max].
    Otherwise assume already in physical units (m/s).
    """
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).any():
        return s

    vmin = float(np.nanmin(s))
    vmax = float(np.nanmax(s))

    # Heuristic: normalized-like values live in a small numeric range
    if abs(vmax) <= 10.0 and abs(vmin) <= 10.0:
        return s * (float(sos_max) - float(sos_min)) + float(sos_min)

    return s


def _denormalize_tof_for_display(t, tof_norm):
    """
    If `tof_norm` is a zscore dict and `t` looks z-scored, undo:
        t_phys = t * std + mean
    Else return as-is.
    """
    t = np.asarray(t, dtype=float)

    if tof_norm is None or not isinstance(tof_norm, dict):
        return t
    kind = str(tof_norm.get("type", "")).lower()

    if kind == "zscore":
        mean = float(tof_norm.get("mean", 0.0))
        std = float(tof_norm.get("std", 1.0))
        eps = float(tof_norm.get("eps", 1e-12))
        if abs(std) < eps:
            std = eps
        if np.nanmax(np.abs(t)) < 50.0:
            return t * std + mean
        return t

    if kind == "max":
        scale = float(tof_norm.get("max", 1.0)) + float(tof_norm.get("eps", 1e-12))
        if np.nanmax(np.abs(t)) <= 10.0:
            return t * scale
        return t

    return t


def plot_results(*args, **kwargs):
    """Plot GT/pred SoS (gray) and optionally ToF (viridis).

    Backwards-compatible calling patterns (supports current + exp9 style):
      1) Positional (current evaluation interface):
         plot_results(true_sos, pred_sos, true_tof, epoch, phys_x, phys_y, sos_min, sos_max, title_suffix="...")
      2) Keyword style (exp9-style):
         plot_results(true_sos=..., pred_sos=..., true_tof=..., epoch=..., phys_x=..., phys_y=...,
                     sos_min=..., sos_max=..., title_suffix=..., tof_norm=...)
      3) Alternate keyword names (older):
         sos_gt / sos_pred, tof
    """

    # ---- map positional args to canonical keyword names ----
    if len(args) >= 2:
        kwargs.setdefault("true_sos", args[0])
        kwargs.setdefault("pred_sos", args[1])
    if len(args) >= 3:
        kwargs.setdefault("true_tof", args[2])
    if len(args) >= 4:
        kwargs.setdefault("epoch", args[3])
    if len(args) >= 6:
        kwargs.setdefault("phys_x", args[4])
        kwargs.setdefault("phys_y", args[5])
    if len(args) >= 8:
        kwargs.setdefault("sos_min", args[6])
        kwargs.setdefault("sos_max", args[7])
    if len(args) >= 9:
        kwargs.setdefault("title_suffix", args[8])

    true_sos = kwargs.pop("true_sos", kwargs.pop("sos_gt", None))
    pred_sos = kwargs.pop("pred_sos", kwargs.pop("sos_pred", None))
    if true_sos is None or pred_sos is None:
        raise TypeError("plot_results requires true_sos/pred_sos (or sos_gt/sos_pred).")

    true_tof = kwargs.pop("true_tof", kwargs.pop("tof", None))
    epoch = kwargs.pop("epoch", 0)
    phys_x = kwargs.pop("phys_x", None)
    phys_y = kwargs.pop("phys_y", None)
    sos_min = float(kwargs.pop("sos_min", 1350.0))
    sos_max = float(kwargs.pop("sos_max", 1650.0))
    title_suffix = str(kwargs.pop("title_suffix", ""))
    tof_norm = kwargs.pop("tof_norm", None)
    tof_mask = kwargs.pop("tof_mask", None)
    water_tof_norm = kwargs.pop("water_tof_norm", None)

    # ignore any unknown extra kwargs (keeps compatibility)

    true_sos = np.squeeze(np.asarray(true_sos))
    pred_sos = np.squeeze(np.asarray(pred_sos))
    if true_sos.shape != pred_sos.shape:
        raise ValueError(f"SoS shape mismatch: {true_sos.shape} vs {pred_sos.shape}")

    true_sos_disp = _conditional_denormalize_sos_for_display(true_sos, sos_min, sos_max)
    pred_sos_disp = _conditional_denormalize_sos_for_display(pred_sos, sos_min, sos_max)
    err = np.abs(true_sos_disp - pred_sos_disp)

    if true_tof is not None:
        true_tof = np.squeeze(np.asarray(true_tof))
        if tof_mask is not None and water_tof_norm is not None:
            m = np.squeeze(np.asarray(tof_mask, dtype=float))
            w = np.squeeze(np.asarray(water_tof_norm, dtype=float))
            if m.shape == true_tof.shape and w.shape == true_tof.shape:
                true_tof = true_tof * m + w * (1.0 - m)
        true_tof_disp = _denormalize_tof_for_display(true_tof, tof_norm)
    else:
        true_tof_disp = None

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # --- extent / axes convention ---
    # Use the "upper" origin convention (axial increases downward), consistent with
    # current load_tof_with_gt.py and attach_evaluation_reference.py.
    extent = None
    if phys_x is not None and phys_y is not None:
        extent = [0.0, float(phys_x), float(phys_y), 0.0]  # origin='upper'

    im = axs[0, 0].imshow(
        true_sos_disp.T,
        cmap="gray",
        extent=extent,
        origin="upper",
        vmin=sos_min,
        vmax=sos_max,
        aspect="equal",
    )
    add_tight_cbar(axs[0, 0], im, "m/s")
    axs[0, 0].set_title("Ground Truth SoS")
    if extent is not None:
        axs[0, 0].set_xlabel("Lateral [m]")
        axs[0, 0].set_ylabel("Axial [m]")

    im = axs[0, 1].imshow(
        pred_sos_disp.T,
        cmap="gray",
        extent=extent,
        origin="upper",
        vmin=sos_min,
        vmax=sos_max,
        aspect="equal",
    )
    add_tight_cbar(axs[0, 1], im, "m/s")
    axs[0, 1].set_title("Predicted SoS")
    if extent is not None:
        axs[0, 1].set_xlabel("Lateral [m]")
        axs[0, 1].set_ylabel("Axial [m]")

    im = axs[1, 0].imshow(err.T, cmap="magma", extent=extent, origin="upper", aspect="equal")
    add_tight_cbar(axs[1, 0], im, "abs err")
    axs[1, 0].set_title("Absolute Error")
    if extent is not None:
        axs[1, 0].set_xlabel("Lateral [m]")
        axs[1, 0].set_ylabel("Axial [m]")

    if true_tof_disp is not None:
        im = axs[1, 1].imshow(true_tof_disp * 1e6, cmap="viridis", origin="lower", aspect="equal")
        add_tight_cbar(axs[1, 1], im, "ToF [microseconds]")
        axs[1, 1].set_title("ToF")
        axs[1, 1].set_xlabel("Receiver index")
        axs[1, 1].set_ylabel("Emitter index")
        if tof_mask is not None and water_tof_norm is not None:
            axs[1, 1].text(
                0.02, 0.03,
                "invalid channels displayed as homogeneous-water reference",
                transform=axs[1, 1].transAxes,
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
            )
    else:
        axs[1, 1].axis("off")

    fig.suptitle(f"Epoch {epoch} {title_suffix}".strip())
    fig.tight_layout()

    log_image(fig)
    plt.close(fig)


def plot_learning_curves(train_loss, val_loss):
    """Plot training/validation loss curves."""
    train_loss = np.asarray(train_loss, dtype=float)
    val_loss = np.asarray(val_loss, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label="train")
    if val_loss.size:
        ax.plot(val_loss, label="val")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curves")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    log_image(fig)
    plt.close(fig)










