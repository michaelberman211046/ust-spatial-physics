"""
Shared display convention for SoS tensors.

Project tensors store spatial SoS maps as (nx, ny), where the first axis is
lateral and the second axis is axial.  Matplotlib image display expects
(rows, columns) = (axial, lateral), so display must transpose spatial maps.
This module keeps that convention in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sos_extent(phys_x: float, phys_y: float) -> list[float]:
    """Physical image extent for origin='upper': lateral horizontal, axial vertical."""
    return [0.0, float(phys_x), float(phys_y), 0.0]


def sos_to_display(sos_xy: Any) -> np.ndarray:
    """
    Convert a project-convention SoS map (nx, ny) to display layout (ny, nx).

    Accepts NumPy arrays or CPU/GPU torch tensors.  The returned array is only
    for plotting; it must not be saved back as the canonical tensor.
    """
    if hasattr(sos_xy, "detach"):
        arr = sos_xy.detach().cpu().numpy()
    else:
        arr = np.asarray(sos_xy)
    if arr.ndim != 2:
        raise ValueError(f"sos_to_display expects a 2D (nx,ny) map, got shape={arr.shape}")
    return arr.T


def set_sos_axes(ax) -> None:
    ax.set_xlabel("Lateral [m]")
    ax.set_ylabel("Axial [m]")










