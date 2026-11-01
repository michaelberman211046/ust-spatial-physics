#!/usr/bin/env python3
"""Create manuscript figures exclusively from saved numerical outputs.

This program does not load model checkpoints, run inference, train a model, or
modify pipeline results.  It reads existing NPZ, NPY, and PT files and writes
publication-quality PNG and PDF figures with larger text and equal-sized reconstruction panels.  Each reconstruction figure shows the
reference, the complete frozen-model reconstruction before post-inference
optimization, and the final Eikonal-refined reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import hashlib
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 120, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })

def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, text: str) -> None:
    ax.text(0.025, 0.975, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82})


def image_panel(ax, field, title: str, extent, vmin: float, vmax: float):
    value = np.asarray(field)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"Non-finite or non-2D SoS field: {value.shape}")
    image = ax.imshow(value.T, origin="upper", extent=extent, cmap="gray",
                      vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=8)
    ax.set_xlabel("Lateral position (m)")
    ax.set_ylabel("Axial position (m)")
    ax.set_aspect("equal")
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], prune="upper"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], prune="upper"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    return image

def colorbar(fig, ax, image, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    bar = fig.colorbar(image, cax=cax)
    bar.set_label(label)


def scalar(report, name: str) -> float:
    return float(np.asarray(report[name]).reshape(()))


def _one_dimensional_indices(report, path: Path) -> np.ndarray:
    for key in ("sample_indices", "evaluated_indices"):
        if key in report.files:
            values = np.asarray(report[key], dtype=np.int64).reshape(-1)
            if values.size:
                return values
    raise KeyError(f"{path} lacks sample_indices/evaluated_indices provenance")


def _sample_stack(report, key: str, path: Path) -> np.ndarray:
    if key not in report.files:
        raise KeyError(f"{path} lacks {key}")
    values = np.asarray(report[key])
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"{path}:{key} has unexpected shape {values.shape}")
    return values


def _assert_same_image(left, right, description: str, atol: float = 1.0e-4) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise RuntimeError(
            f"{description} have different shapes: {left.shape} and {right.shape}.")
    if not np.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True):
        maximum = float(np.nanmax(np.abs(left - right)))
        raise RuntimeError(f"{description} do not match (maximum difference {maximum:g}).")


def matched_synthetic(supervised_file: Path, refined_file: Path, requested_index=None):
    with np.load(supervised_file, allow_pickle=False) as supervised, \
            np.load(refined_file, allow_pickle=False) as refined:
        for key in ("target_mps", "prediction_mps", "phys_x", "phys_y",
                    "sos_min", "sos_max"):
            if key not in supervised.files:
                raise KeyError(f"{supervised_file} lacks {key}")
        for key in ("target_mps", "stage8_mps", "refined_mps"):
            if key not in refined.files:
                raise KeyError(f"{refined_file} lacks {key}")

        split = str(np.asarray(supervised.get("evaluation_split", "")).reshape(()))
        refined_split = str(np.asarray(refined.get("evaluation_split", "")).reshape(()))
        if split.lower() != "test" or refined_split.lower() != "test":
            raise RuntimeError(
                "Synthetic publication panels must originate from the held-out test split; "
                f"found {split!r} and {refined_split!r}.")

        supervised_indices = _one_dimensional_indices(supervised, supervised_file)
        refined_indices = _one_dimensional_indices(refined, refined_file)
        common = np.intersect1d(supervised_indices, refined_indices)
        if common.size == 0:
            raise RuntimeError("The synthetic reports contain no common recorded sample index.")
        if requested_index is None and common.size != 1:
            raise RuntimeError("Multiple matched samples: specify --sample-index explicitly.")
        sample_index = int(common[0]) if requested_index is None else int(requested_index)
        if sample_index not in common:
            raise RuntimeError(f"Requested sample {sample_index} is not present in both reports.")
        for key in ("phys_x", "phys_y", "sos_min", "sos_max"):
            if key not in refined.files:
                raise KeyError(f"{refined_file} lacks {key}")
            if not np.isclose(scalar(supervised, key), scalar(refined, key), rtol=0, atol=1e-7):
                raise RuntimeError(f"Synthetic report metadata mismatch: {key}")
        supervised_row = int(np.flatnonzero(supervised_indices == sample_index)[0])
        refined_row = int(np.flatnonzero(refined_indices == sample_index)[0])

        targets = _sample_stack(supervised, "target_mps", supervised_file)
        frozen = _sample_stack(supervised, "prediction_mps", supervised_file)
        refined_targets = _sample_stack(refined, "target_mps", refined_file)
        refined_frozen = _sample_stack(refined, "stage8_mps", refined_file)
        finals = _sample_stack(refined, "refined_mps", refined_file)
        if (len(supervised_indices) != len(targets)
                or len(frozen) != len(targets)
                or len(refined_indices) != len(refined_targets)
                or len(refined_frozen) != len(refined_targets)
                or len(finals) != len(refined_targets)):
            raise RuntimeError("Synthetic sample-index provenance does not match report array lengths.")
        target = targets[supervised_row]
        frozen_image = frozen[supervised_row]
        _assert_same_image(target, refined_targets[refined_row],
                           "Synthetic reference images for the recorded test sample")
        _assert_same_image(frozen_image, refined_frozen[refined_row],
                           "Frozen-model images for the recorded test sample")
        return {
            "reference": target.copy(),
            "frozen": frozen_image.copy(),
            "final": finals[refined_row].copy(),
            "sample_index": sample_index,
            "phys_x": scalar(supervised, "phys_x"),
            "phys_y": scalar(supervised, "phys_y"),
            "sos_min": scalar(supervised, "sos_min"),
            "sos_max": scalar(supervised, "sos_max"),
        }


def reconstruction_figure(fields, titles, extent, vmin, vmax, output, name):
    # A separate colorbar column leaves all three image axes equally sized.
    fig = plt.figure(figsize=(7.5, 2.95))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045],
                           left=0.085, right=0.91, bottom=0.20,
                           top=0.79, wspace=0.12)
    axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    image = None
    for number, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        image = image_panel(ax, field, title, extent, vmin, vmax)
        panel_label(ax, f"({chr(ord('a') + number)})")
        if number:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    cax = fig.add_subplot(grid[0, 3])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Speed of sound (m s$^{-1}$)", labelpad=7)
    save(fig, output, name)

def synthetic_figure(supervised_file: Path, refined_file: Path, output: Path, sample_index=None):
    item = matched_synthetic(supervised_file, refined_file, sample_index)
    reconstruction_figure(
        [item["reference"], item["frozen"], item["final"]],
        ["Synthetic SoS\nreference", "Physics-informed\nlearned reconstruction",
         "Final Eikonal-refined\nreconstruction"],
        [0, item["phys_x"], item["phys_y"], 0],
        item["sos_min"], item["sos_max"], output,
        "figure_synthetic_reconstruction",
    )
    return item


def measured_figure(base_file: Path, refined_file: Path, output: Path):
    with np.load(base_file, allow_pickle=False) as base, \
            np.load(refined_file, allow_pickle=False) as refined:
        for key in ("gt_mps", "prior_mps", "agreement_weighted_mps",
                    "phys_x", "phys_y",
                    "sos_min", "sos_max"):
            if key not in base.files:
                raise KeyError(f"{base_file} lacks {key}")
        for key in ("stage10_mps", "refined_mps"):
            if key not in refined.files:
                raise KeyError(f"{refined_file} lacks {key}")
        for key in ("phys_x", "phys_y", "sos_min", "sos_max"):
            if key not in refined.files or not np.isclose(
                    scalar(base, key), scalar(refined, key), rtol=0, atol=1e-7):
                raise RuntimeError(f"Measured report metadata mismatch: {key}")
        # The later Eikonal report must start from the saved output of the
        # preceding measured-data physical reconstruction.  This check guards
        # against accidentally mixing results from different runs.
        _assert_same_image(np.asarray(base["agreement_weighted_mps"]),
                           np.asarray(refined["stage10_mps"]),
                           "Measured refinement inputs")
        fields = [np.asarray(base["gt_mps"]).copy(),
                  np.asarray(base["prior_mps"]).copy(),
                  np.asarray(refined["refined_mps"]).copy()]
        extent = [0, scalar(base, "phys_x"), scalar(base, "phys_y"), 0]
        vmin, vmax = scalar(base, "sos_min"), scalar(base, "sos_max")
    reconstruction_figure(
        fields,
        ["Independent FWI\nreference",
         "Physics-informed\nlearned reconstruction",
         "Final Eikonal-refined\nreconstruction"],
        extent, vmin, vmax, output, "figure_measured_reconstruction",
    )


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validity_mask(tof, mask=None) -> np.ndarray:
    tof = np.asarray(tof)
    if mask is None:
        valid = np.isfinite(tof)
        if np.any(tof == 0):
            valid &= tof != 0
        return valid
    valid = np.asarray(mask)
    while valid.ndim > 2:
        valid = valid[0]
    if valid.shape != tof.shape:
        raise ValueError(
            f"ToF and validity-mask shapes differ: {tof.shape} and {valid.shape}"
        )
    return valid.astype(bool) & np.isfinite(tof)


def load_pt_tof(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Reading the saved PT data requires PyTorch.") from exc
    package = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or "tof" not in package:
        raise KeyError(f"{path} does not contain a tof array")
    tof = tof_matrix(_numpy(package["tof"]))
    mask = None
    for key in ("tof_mask", "valid_mask", "channel_mask", "mask"):
        if key in package:
            mask = _numpy(package[key])
            break
    return tof, _validity_mask(tof, mask)


def load_synthetic_test_tof(data_path: Path, sample_index: int,
                            sos_min: float, sos_max: float):
    try:
        # The script may be stored in src or in a separate review-package folder.
        dataset_dir = Path.cwd() / "src"
        if dataset_dir.is_dir():
            sys.path.insert(0, str(dataset_dir))
        from dataset import PairsCacheDataset
    except ImportError as exc:
        raise RuntimeError(
            "The synthetic dataset reader could not be imported. Run this script "
            "from the repository root with the script stored in src."
        ) from exc
    dataset = PairsCacheDataset(
        str(data_path),
        sos_min=float(sos_min),
        sos_max=float(sos_max),
        return_tof_mask=True,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"Recorded test sample {sample_index} is outside dataset length {len(dataset)}."
        )
    sample = dataset._sample_arrays(sample_index)
    for key in ("tof_raw_eikonal", "tof"):
        if key in sample:
            tof = tof_matrix(np.asarray(sample[key]))
            mask = None
            for mask_key in ("tof_mask", "valid_mask", "channel_mask", "mask"):
                if mask_key in sample:
                    mask = np.asarray(sample[mask_key])
                    break
            # In the progressive cache the limited-view mask is stored once in
            # the dataset index, not repeated in every chunk/sample.
            if mask is None and getattr(dataset, "tof_mask", None) is not None:
                mask = np.asarray(dataset.tof_mask)
            return tof, _validity_mask(tof, mask), sample_index
    raise KeyError(f"Synthetic dataset sample {sample_index} contains no saved ToF array.")


def tof_matrix(array) -> np.ndarray:
    value = np.asarray(array)
    while value.ndim > 2:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional ToF matrix; got {value.shape}")
    return value.astype(float, copy=False)


def tof_figure(synthetic_data: Path, synthetic_index: int,
               sos_min: float, sos_max: float,
               measured_file: Path, output: Path):
    synthetic_array, synthetic_mask, loaded_index = load_synthetic_test_tof(
        synthetic_data, synthetic_index, sos_min, sos_max
    )
    synthetic = tof_matrix(synthetic_array)
    measured, measured_mask = load_pt_tof(measured_file)
    # Saved travel times are in seconds. Display microseconds.
    synthetic = synthetic * 1.0e6
    measured = measured * 1.0e6
    synthetic_display = np.where(synthetic_mask, synthetic, np.nan)
    measured_display = np.where(measured_mask, measured, np.nan)
    values = np.concatenate([synthetic_display[np.isfinite(synthetic_display)],
                             measured_display[np.isfinite(measured_display)]])
    vmin, vmax = np.percentile(values, [1, 99])
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), constrained_layout=True)
    images = []
    for i, (ax, matrix, title) in enumerate(zip(
            axes, (synthetic_display, measured_display),
            ("Calculated synthetic ToF",
             "RF-derived measured ToF"))):
        images.append(ax.imshow(matrix, origin="lower", aspect="equal", cmap=cmap,
                                vmin=vmin, vmax=vmax, interpolation="nearest"))
        ax.set_title(title)
        ax.set_xlabel("Receiver index")
        ax.set_ylabel("Emitter index")
        panel_label(ax, f"({chr(ord('a') + i)})")
    colorbar(fig, axes[-1], images[-1], r"Time of flight ($\mu$s)")
    save(fig, output, "figure_synthetic_measured_tof")


def synthetic_difference_figure(item, output):
    errors = [item["frozen"] - item["reference"],
              item["final"] - item["reference"]]
    limit = max(float(np.max(np.abs(v))) for v in errors)
    limit = max(limit, 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), constrained_layout=True)
    for j, (ax, field, title) in enumerate(zip(axes, errors,
            ["Learned minus reference", "Refined minus reference"])):
        im = ax.imshow(field.T, origin="upper",
            extent=[0, item["phys_x"], item["phys_y"], 0],
            cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Lateral position (m)")
        if j == 0:
            ax.set_ylabel("Axial position (m)")
        panel_label(ax, f"({chr(ord('a') + j)})")
    fig.colorbar(im, ax=axes, label="Signed SoS error (m s$^{-1}$)", shrink=0.85)
    save(fig, output, "figure_synthetic_refinement_errors")


def manifest(args, item):
    sources = {}
    for name in ("synthetic_supervised_report", "synthetic_refined_report",
                 "measured_base_report", "measured_refined_report"):
        path = getattr(args, name)
        sources[name] = {"path": str(path.resolve()),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {"synthetic_evaluation_split": "test",
              "synthetic_sample_index": item["sample_index"],
              "reconstruction_panels": ["reference", "complete_frozen_learned", "final_refined"],
              "array_display": "SoS arrays transposed; axial coordinate increases downward",
              "interpolation": "nearest",
              "sources": sources,
              "tof_figure_generated": not args.reconstructions_only,
              "optional_error_figure_generated": args.include_error_maps,
              "purpose": "Saved-output extraction; no inference, training or refinement"}
    for name in ("synthetic_data_pt", "measured_tof"):
        path = getattr(args, name)
        if path is not None:
            result[name] = str(path.resolve())
    (args.output_dir / "figure_provenance.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-supervised-report", type=Path, required=True)
    parser.add_argument("--synthetic-refined-report", type=Path, required=True)
    parser.add_argument("--measured-base-report", type=Path, required=True)
    parser.add_argument("--measured-refined-report", type=Path, required=True)
    parser.add_argument("--synthetic-data-pt", type=Path)
    parser.add_argument("--measured-tof", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--reconstructions-only", action="store_true",
                        help="Generate only the two reconstruction figures; leave ToF untouched.")
    parser.add_argument("--include-error-maps", action="store_true",
                        help="Also save optional synthetic signed-error maps for inspection.")
    args = parser.parse_args()
    if not args.reconstructions_only and (args.synthetic_data_pt is None or args.measured_tof is None):
        parser.error("Both ToF input paths are required unless --reconstructions-only is used.")
    for path in (args.synthetic_supervised_report, args.synthetic_refined_report,
                 args.measured_base_report, args.measured_refined_report,
                 args.synthetic_data_pt, args.measured_tof):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    synthetic_item = synthetic_figure(
        args.synthetic_supervised_report,
        args.synthetic_refined_report,
        args.output_dir, args.sample_index,
    )
    measured_figure(args.measured_base_report,
                    args.measured_refined_report, args.output_dir)
    if not args.reconstructions_only:
        tof_figure(
            args.synthetic_data_pt,
            synthetic_item["sample_index"],
            synthetic_item["sos_min"],
            synthetic_item["sos_max"],
            args.measured_tof,
            args.output_dir,
        )
    if args.include_error_maps:
        synthetic_difference_figure(synthetic_item, args.output_dir)
    manifest(args, synthetic_item)
    print(f"Publication figures written to {args.output_dir}")


if __name__ == "__main__":
    main()

