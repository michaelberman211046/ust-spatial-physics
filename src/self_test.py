"""Fast deterministic checks for publication measured geometry and calibration."""

import numpy as np
import torch
import argparse
from pathlib import Path

from build_physical_ray_setup import geometry
from reconstruct_physical_ray import predict_residual
from anatomy import generate_sensor_positions
from dataset import compute_limited_view_mask
from extract_measured_tof import _make_exclusion_mask_for_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_stamp", required=True)
    args = parser.parse_args()
    grid = 5
    image = torch.arange(grid * grid, dtype=torch.float32).reshape(grid, grid)
    ix, iy = 1, 3
    index = ix * grid + iy  # project tensors are (lateral, axial)
    assert image.reshape(-1)[index] == image[ix, iy]
    ray = torch.tensor([[index, index, index]], dtype=torch.int32)
    length = torch.tensor([.12])
    estimate = predict_residual(image, ray, length, 32)
    assert torch.allclose(estimate, image[ix, iy].reshape(1) * length, atol=1e-6)

    n = 512
    _, _, chords, distance = geometry(n, .1091)
    points, order, _, digest = __import__("measured_geometry").require_geometry(n, n)
    assert np.array_equal(np.sort(order), np.arange(n))
    assert np.allclose(chords, np.linalg.norm(points[:, None]-points[None, :], axis=-1))
    assert np.allclose(np.diag(chords), 0)
    synthetic_tx, synthetic_rx = generate_sensor_positions(200, 200, .24/199, .24/199, .1091, n, n)
    synthetic_m = (synthetic_tx-99.5)*(.24/199)
    assert np.max(abs(synthetic_m-points)) < 2e-7
    assert np.array_equal(synthetic_tx, synthetic_rx)
    synthetic_mask = compute_limited_view_mask(n, n, .25) > .5
    experimental_mask = ~_make_exclusion_mask_for_selection(order, order, n, .25)
    assert np.array_equal(synthetic_mask, experimental_mask)
    stamp = Path(args.geometry_stamp)
    if stamp.exists() and stamp.read_text(encoding="ascii").strip() != digest:
        raise RuntimeError("Existing publication training directory was built from a different RF geometry")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(digest+"\n", encoding="ascii")
    print(f"[publication SELF TEST] measured shared-element geometry {digest[:12]}, mask, ray indexing and integral passed")


if __name__ == "__main__":
    main()










