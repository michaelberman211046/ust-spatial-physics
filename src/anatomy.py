"""
anatomy.py

Synthetic anatomy generation for the ultrasound tomography pipeline.

Default behavior is backward-compatible with the original smooth-shape generator.
Optional flags add a random support envelope and mixed sharp/smooth internal shapes.
"""

import numpy as np
from cv2 import ellipse as draw_ellipse
from cv2 import fillPoly
from logger import log_message


def generate_sensor_positions(nx, ny, dx, dy, radius, n_emitters, n_receivers):
    """Use measured shared-element ring coordinates in image pixel units.

    The radius argument is retained for API compatibility; the MAT coordinates
    are the geometry of record.  publication deliberately refuses an absent MAT file.
    """
    from measured_geometry import pixel_positions
    return pixel_positions(int(nx), int(ny), float(dx), float(dy), int(n_emitters), int(n_receivers))


def _pixel_spacing_for_shape(dx: float, dy: float) -> float:
    """Use an isotropic pixel scale for shape-size conversion."""
    dx = float(dx)
    dy = float(dy)
    if dx <= 0 or dy <= 0:
        return 1.0
    return 0.5 * (dx + dy)


def _inside_ellipse_support(cx, cy, center_x, center_y, ax, ay, angle_rad, margin_px=0.0) -> bool:
    """Return True if an axis-aligned bounding radius fits inside an ellipse-like support.

    This deliberately uses a conservative bounding-radius test for placement. The final
    support mask is applied after drawing all structures, so small boundary violations are
    still clipped to water.
    """
    dxp = float(cx) - float(center_x)
    dyp = float(cy) - float(center_y)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    xr = ca * dxp - sa * dyp
    yr = sa * dxp + ca * dyp
    ax_eff = max(float(ax) - float(margin_px), 1.0)
    ay_eff = max(float(ay) - float(margin_px), 1.0)
    return (xr / ax_eff) ** 2 + (yr / ay_eff) ** 2 <= 1.0


def _make_support_mask(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    radius: float,
    *,
    use_random_support_envelope: bool,
    support_radius_min_frac: float,
    support_radius_max_frac: float,
    support_center_jitter_frac: float,
    support_ellipse_prob: float,
    support_ellipse_axis_jitter_frac: float,
):
    """Create a per-sample random support envelope.

    The support is the region in which synthetic anatomy can differ from water. Outside it,
    the SoS is forced back to sos_water. When disabled, the support equals the sensor-circle
    support used by the original generator.
    """
    px = _pixel_spacing_for_shape(dx, dy)
    sensor_radius_px = float(radius) / px
    center_x = (float(nx) - 1.0) / 2.0
    center_y = (float(ny) - 1.0) / 2.0

    if use_random_support_envelope:
        rmin = float(np.clip(support_radius_min_frac, 0.05, 1.0))
        rmax = float(np.clip(support_radius_max_frac, rmin, 1.0))
        support_r_px = sensor_radius_px * float(np.random.uniform(rmin, rmax))
        jitter_px = float(max(0.0, support_center_jitter_frac)) * sensor_radius_px
        support_cx = center_x + float(np.random.uniform(-jitter_px, jitter_px))
        support_cy = center_y + float(np.random.uniform(-jitter_px, jitter_px))

        if np.random.rand() < float(np.clip(support_ellipse_prob, 0.0, 1.0)):
            axis_j = float(max(0.0, support_ellipse_axis_jitter_frac))
            ax = support_r_px * float(np.random.uniform(max(0.2, 1.0 - axis_j), 1.0 + axis_j))
            ay = support_r_px * float(np.random.uniform(max(0.2, 1.0 - axis_j), 1.0 + axis_j))
            angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        else:
            ax = support_r_px
            ay = support_r_px
            angle = 0.0
    else:
        support_cx = center_x
        support_cy = center_y
        ax = sensor_radius_px
        ay = sensor_radius_px
        angle = 0.0

    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32))
    ca = np.cos(-angle)
    sa = np.sin(-angle)
    xr = ca * (xx - support_cx) - sa * (yy - support_cy)
    yr = sa * (xx - support_cx) + ca * (yy - support_cy)
    mask = ((xr / max(ax, 1.0)) ** 2 + (yr / max(ay, 1.0)) ** 2 <= 1.0)

    return mask.astype(bool), (support_cx, support_cy, ax, ay, angle), sensor_radius_px


def _sample_shape_axes(shape_size_min_px: int, shape_size_max_px: int):
    lo = int(max(1, shape_size_min_px))
    hi = int(max(lo + 1, shape_size_max_px + 1))
    return int(np.random.randint(lo, hi)), int(np.random.randint(lo, hi))


def _draw_random_polygon(sos_map, cx, cy, radius_px, n_vertices, sos_val, angle0=None):
    if angle0 is None:
        angle0 = float(np.random.uniform(0.0, 2.0 * np.pi))
    n_vertices = int(max(3, n_vertices))
    base_angles = angle0 + 2.0 * np.pi * np.arange(n_vertices) / float(n_vertices)
    # Irregular but not degenerate.
    radial = radius_px * np.random.uniform(0.55, 1.0, size=n_vertices)
    xs = cx + radial * np.cos(base_angles)
    ys = cy + radial * np.sin(base_angles)
    pts = np.stack([ys, xs], axis=1).round().astype(np.int32)  # cv2 expects (col,row)=(y,x)
    fillPoly(sos_map, [pts], float(sos_val))


def _draw_random_rod(sos_map, cx, cy, length_px, width_px, sos_val, angle0=None):
    """Draw an elongated asymmetric quadrilateral.

    A rod is represented as a four-edged polygon with high aspect ratio and
    unequal side offsets. This gives line-like inclusions without making them
    perfectly rectangular, which better matches some elongated structures in
    the experimental FWI reference.
    """
    if angle0 is None:
        angle0 = float(np.random.uniform(0.0, 2.0 * np.pi))
    length_px = float(max(3.0, length_px))
    width_px = float(max(1.0, width_px))
    ca = float(np.cos(angle0))
    sa = float(np.sin(angle0))
    ux, uy = ca, sa
    vx, vy = -sa, ca

    half_len_left = 0.5 * length_px * float(np.random.uniform(0.75, 1.25))
    half_len_right = 0.5 * length_px * float(np.random.uniform(0.75, 1.25))
    half_w_left = 0.5 * width_px * float(np.random.uniform(0.45, 1.25))
    half_w_right = 0.5 * width_px * float(np.random.uniform(0.45, 1.25))
    skew = width_px * float(np.random.uniform(-0.75, 0.75))

    p0x = cx - half_len_left * ux - half_w_left * vx
    p0y = cy - half_len_left * uy - half_w_left * vy
    p1x = cx + half_len_right * ux + skew * vx - half_w_right * vx
    p1y = cy + half_len_right * uy + skew * vy - half_w_right * vy
    p2x = cx + half_len_right * ux + skew * vx + half_w_right * vx
    p2y = cy + half_len_right * uy + skew * vy + half_w_right * vy
    p3x = cx - half_len_left * ux + half_w_left * vx
    p3y = cy - half_len_left * uy + half_w_left * vy

    xs = np.array([p0x, p1x, p2x, p3x], dtype=np.float32)
    ys = np.array([p0y, p1y, p2y, p3y], dtype=np.float32)
    pts = np.stack([ys, xs], axis=1).round().astype(np.int32)  # cv2 expects (col,row)=(y,x)
    fillPoly(sos_map, [pts], float(sos_val))


def create_random_sos_map(
    nx,
    ny,
    dx,
    dy,
    radius,
    sos_water,
    sos_min,
    sos_max,
    max_shapes,
    min_shapes=None,
    *,
    use_random_support_envelope=False,
    support_radius_min_frac=0.65,
    support_radius_max_frac=0.95,
    support_center_jitter_frac=0.05,
    support_ellipse_prob=0.35,
    support_ellipse_axis_jitter_frac=0.15,
    shape_mode="ellipses",
    p_ellipse=0.65,
    p_triangle=0.20,
    p_polygon=0.15,
    p_rod=0.0,
    shape_size_min_px=4,
    shape_size_max_px=24,
    polygon_vertices_min=4,
    polygon_vertices_max=8,
):
    """Create a random SoS map.

    Backward-compatible default: ellipse-only inclusions placed inside the sensor ring.

    Optional experimental-domain extension:
      - Per-sample random support envelope. Outside the support, SoS is water.
      - Mixed internal shapes, including ellipses, triangles, polygons, and rods.
    """
    nx = int(nx)
    ny = int(ny)
    sos_map = np.full((nx, ny), float(sos_water), dtype=np.float32)

    support_mask, support_info, sensor_radius_px = _make_support_mask(
        nx, ny, dx, dy, radius,
        use_random_support_envelope=bool(use_random_support_envelope),
        support_radius_min_frac=float(support_radius_min_frac),
        support_radius_max_frac=float(support_radius_max_frac),
        support_center_jitter_frac=float(support_center_jitter_frac),
        support_ellipse_prob=float(support_ellipse_prob),
        support_ellipse_axis_jitter_frac=float(support_ellipse_axis_jitter_frac),
    )
    support_cx, support_cy, support_ax, support_ay, support_angle = support_info

    max_shapes = int(max(1, max_shapes))
    if min_shapes is None or int(min_shapes) <= 0:
        min_shapes = max(1, max_shapes // 2)
    min_shapes = int(max(1, min(min_shapes, max_shapes)))
    num_shapes = int(np.random.randint(min_shapes, max_shapes + 1))

    mode = str(shape_mode).lower().strip()
    if mode not in ("ellipses", "mixed"):
        mode = "ellipses"

    probs = np.array(
        [
            max(0.0, float(p_ellipse)),
            max(0.0, float(p_triangle)),
            max(0.0, float(p_polygon)),
            max(0.0, float(p_rod)),
        ],
        dtype=np.float64,
    )
    if probs.sum() <= 0.0:
        probs[:] = np.array([1.0, 0.0, 0.0, 0.0])
    probs = probs / probs.sum()

    for _ in range(num_shapes):
        valid_placement = False
        attempts = 0
        while not valid_placement and attempts < 100:
            cx = int(np.random.randint(0, nx))
            cy = int(np.random.randint(0, ny))

            ax1, ax2 = _sample_shape_axes(shape_size_min_px, shape_size_max_px)
            max_axis = max(ax1, ax2)

            if _inside_ellipse_support(cx, cy, support_cx, support_cy, support_ax, support_ay, support_angle, margin_px=max_axis + 2.0):
                sos_val = float(np.random.uniform(float(sos_min), float(sos_max)))

                if mode == "mixed":
                    shape_type = np.random.choice(["ellipse", "triangle", "polygon", "rod"], p=probs)
                else:
                    shape_type = "ellipse"

                if shape_type == "ellipse":
                    angle = int(np.random.randint(0, 360))
                    draw_ellipse(sos_map, (cy, cx), (ax1, ax2), angle, 0, 360, sos_val, -1)
                elif shape_type == "triangle":
                    _draw_random_polygon(sos_map, cx, cy, float(max_axis), 3, sos_val)
                elif shape_type == "rod":
                    length_px = float(max_axis) * float(np.random.uniform(2.5, 8.0))
                    width_px = float(max(1, min(ax1, ax2))) * float(np.random.uniform(0.45, 1.25))
                    _draw_random_rod(sos_map, cx, cy, length_px, width_px, sos_val)
                else:
                    vmin = int(max(4, polygon_vertices_min))
                    vmax = int(max(vmin, polygon_vertices_max))
                    n_vertices = int(np.random.randint(vmin, vmax + 1))
                    _draw_random_polygon(sos_map, cx, cy, float(max_axis), n_vertices, sos_val)

                valid_placement = True
            attempts += 1

    # Enforce the randomly sampled support envelope. This is the key mechanism
    # that generates many water annuli/support sizes across the synthetic bank.
    sos_map[~support_mask] = float(sos_water)

    return sos_map.astype(np.float32)










