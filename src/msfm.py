"""
msfm.py
Robust Eikonal solver using scikit-fmm (skfmm).

Pipeline convention:
    sos_map.shape == (nx, ny)
    axis 0 == x
    axis 1 == y
    source_index is ordered as (x, y) in pixel/node coordinates

The solver uses the node-spacing convention supplied by the caller:
    dx = phys_x / (nx - 1)
    dy = phys_y / (ny - 1)

For isotropic grids, we deliberately pass scalar dx to skfmm.  This preserves
exactly the stable numerical path used by the earlier working pipeline, while
still using the corrected node spacing.  For genuinely anisotropic grids, we
pass the spacing tuple in array-axis order, (dx, dy).  Silent Euclidean-distance
fallbacks are not allowed, because they corrupt the ToF physics.
"""

from __future__ import annotations

import numpy as np
import skfmm


def msfm(sos_map, source_index, dx=1.0, dy=1.0, order=2):
    """
    Solve the Eikonal equation using scikit-fmm.

    Parameters
    ----------
    sos_map : np.ndarray
        2D speed-of-sound map in m/s, shape (nx, ny), with axis 0 = x
        and axis 1 = y.
    source_index : tuple[float, float]
        Source location in pixel/node coordinates, ordered as (x, y).
    dx, dy : float
        Physical node spacings in meters.
    order : int
        Kept for API compatibility with older calls.  It is not passed to
        skfmm, because some installed skfmm versions do not accept this keyword.

    Returns
    -------
    np.ndarray
        Travel-time map in seconds, shape (nx, ny).
    """
    sos_map = np.asarray(sos_map, dtype=np.float64)

    if sos_map.ndim != 2:
        raise ValueError(f"[msfm.py] sos_map must be 2D, got shape {sos_map.shape}")
    if not np.all(np.isfinite(sos_map)):
        raise ValueError("[msfm.py] sos_map contains non-finite values.")
    if np.any(sos_map <= 0.0):
        raise ValueError("[msfm.py] sos_map must contain strictly positive speeds.")

    dx = float(dx)
    dy = float(dy)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"[msfm.py] Invalid grid spacing: dx={dx}, dy={dy}")

    nx, ny = sos_map.shape
    ex = int(round(float(source_index[0])))
    ey = int(round(float(source_index[1])))
    ex = int(np.clip(ex, 0, nx - 1))
    ey = int(np.clip(ey, 0, ny - 1))

    phi = np.ones((nx, ny), dtype=np.float64)
    phi[ex, ey] = -1.0

    # Preserve the earlier working skfmm code path when the grid is isotropic.
    # For rectangular domains, use the anisotropic spacing tuple in the same
    # order as the array axes: axis 0 is x, axis 1 is y.
    spacing = dx if np.isclose(dx, dy, rtol=1e-12, atol=1e-15) else (dx, dy)

    try:
        t_map = skfmm.travel_time(phi, speed=sos_map, dx=spacing)
    except Exception as e:
        raise RuntimeError(
            "[msfm.py] skfmm.travel_time failed. "
            f"shape={sos_map.shape}, source=({ex},{ey}), dx={dx}, dy={dy}, "
            f"spacing_arg={spacing!r}. Original error: {e}"
        ) from e

    t_map = np.asarray(t_map, dtype=np.float64)
    if not np.all(np.isfinite(t_map)):
        raise RuntimeError(
            "[msfm.py] skfmm returned non-finite travel times. "
            f"shape={sos_map.shape}, source=({ex},{ey}), dx={dx}, dy={dy}"
        )

    return t_map










