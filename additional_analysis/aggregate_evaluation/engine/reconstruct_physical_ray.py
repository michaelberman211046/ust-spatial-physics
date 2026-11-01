import argparse
import pprint
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from logger import log_message
from settings import set_output_folder

VERSION = "measured-coordinate-heldout-gated-physical-ray-v1"


def robust(x, weight, beta):
    ax = x.abs()
    huber = torch.where(ax < beta, .5 * ax.square() / beta, ax - .5 * beta)
    return (huber * weight).sum() / weight.sum().clamp_min(1)


def tv(x, mask):
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs() * mask[..., 1:, :]
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs() * mask[..., :, 1:]
    return dy.mean() + dx.mean()


def predict_residual(slowness_delta, pixel_indices, ray_length, chunk):
    """Trapezoidal straight-ray integral through a lateral-first image tensor."""
    flat = slowness_delta.reshape(-1)
    n_samples = pixel_indices.shape[1]
    if n_samples < 2:
        raise ValueError("At least two ray samples are required")
    parts = []
    for start in range(0, pixel_indices.shape[0], chunk):
        ids = pixel_indices[start:start + chunk].long()
        values = flat[ids]
        integral = values.sum(1) - .5 * (values[:, 0] + values[:, -1])
        parts.append(integral * ray_length[start:start + chunk] / (n_samples - 1))
    return torch.cat(parts, 0)


def main():
    p = argparse.ArgumentParser()
    for key in ("setup_pt", "out_pt", "report_npz", "output_dir"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--holdout_sectors", default="0,2,4,6")
    p.add_argument("--angular_sectors", type=int, default=8)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=.035)
    p.add_argument("--max_correction_mps", type=float, default=120)
    p.add_argument("--huber_beta_us", type=float, default=.20)
    p.add_argument("--prior_weight", type=float, default=.035)
    p.add_argument("--tv_weight", type=float, default=.020)
    p.add_argument("--curvature_weight", type=float, default=.008)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--patience", type=int, default=22)
    p.add_argument("--ray_chunk", type=int, default=8192)
    p.add_argument("--min_holdout_improvement_us", type=float, default=.01)
    p.add_argument("--min_accepted_folds", type=int, default=2)
    a = p.parse_args()
    set_output_folder(a.output_dir)
    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(a))}")
    log_message(f"[physical_reconstruction] version={VERSION}")

    device = torch.device(a.device if not a.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    setup = torch.load(a.setup_pt, map_location=device, weights_only=False)
    if setup.get("version") != "measured-coordinate-single-calibration-ray-setup-v1":
        raise RuntimeError("publication requires a newly generated measured-coordinate ray cache")
    geometry, pg = setup["geometry"], setup["physical_geometry"]
    lo, hi, water = float(geometry["sos_min"]), float(geometry["sos_max"]), float(pg["water_sos"])
    pixels = setup["ray_pixel_indices"].to(device)
    lengths = setup["ray_length_m"].to(device)
    target = setup["target_residual_seconds"].to(device)
    weight = setup["ray_weight"].to(device)
    emitters = setup["ray_emitter_index"].to(device).long()
    full_prior = setup["prior_mps"].to(device)
    full_support = setup["soft_support"].to(device).clamp(0, 1)
    grid = int(pg["grid_size"])
    prior = F.interpolate(full_prior, size=(grid, grid), mode="bilinear", align_corners=False)
    support = F.interpolate(full_support, size=(grid, grid), mode="bilinear", align_corners=False)
    initial = water + support * (prior - water)
    initial_full = water + full_support * (full_prior - water)

    def ray_predict(image):
        return predict_residual(1 / image[0, 0] - 1 / water, pixels, lengths, a.ray_chunk)

    sectors = [int(s.strip()) for s in a.holdout_sectors.split(",") if s.strip()]
    if not sectors or len(set(sectors)) != len(sectors) or any(s < 0 or s >= a.angular_sectors for s in sectors):
        raise ValueError("Invalid holdout sector list")
    fold_images, records = [], []
    for sector in sectors:
        validation = ((emitters * a.angular_sectors) // 512 == sector).float() * weight
        fit = (weight - validation).clamp_min(0)
        if float(validation.sum()) < 1 or float(fit.sum()) < 1:
            raise RuntimeError(f"Empty fit or validation rays for sector {sector}")
        raw = torch.nn.Parameter(torch.zeros_like(prior))
        optimizer = torch.optim.Adam([raw], lr=a.lr)
        with torch.no_grad():
            initial_validation = float(robust((ray_predict(initial) - target) * 1e6, validation, a.huber_beta_us).cpu())
        best = initial.detach().cpu().clone()
        best_validation, best_step, stale, history = initial_validation, 0, 0, []
        for step in range(a.steps + 1):
            # One support application; the forward model uses precisely this image.
            field = water + support * ((prior + a.max_correction_mps * torch.tanh(raw)).clamp(lo, hi) - water)
            residual_us = (ray_predict(field) - target) * 1e6
            fit_loss = robust(residual_us, fit, a.huber_beta_us)
            val_loss = robust(residual_us, validation, a.huber_beta_us)
            anchor = ((field - initial).abs() * support).sum() / support.sum().clamp_min(1) / 100
            smooth = tv(field / 100, support)
            curvature = (F.avg_pool2d(field, 3, 1, 1) - field).abs().mean() / 100
            total = fit_loss + a.prior_weight * anchor + a.tv_weight * smooth + a.curvature_weight * curvature
            row = [float(v.detach().cpu()) for v in (total, fit_loss, val_loss, anchor, smooth)]
            history.append(row)
            if step >= a.warmup and row[2] < best_validation - 1e-6:
                best_validation, best, best_step, stale = row[2], field.detach().cpu().clone(), step, 0
            elif step >= a.warmup:
                stale += 1
            if step % 15 == 0:
                log_message(f"[physical_reconstruction] sector={sector} step={step} fit_us={row[1]:.6g} holdout_us={row[2]:.6g} initial_us={initial_validation:.6g}")
            if stale >= a.patience:
                break
            if step < a.steps:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_([raw], 1)
                optimizer.step()
        improvement = initial_validation - best_validation
        accepted = best_step > 0 and improvement >= a.min_holdout_improvement_us
        image = F.interpolate(best, size=full_prior.shape[-2:], mode="bilinear", align_corners=False)
        fold_images.append(image)
        records.append({"sector": sector, "initial_validation_us": initial_validation,
                        "best_step": best_step, "best_validation_us": best_validation,
                        "holdout_improvement_us": improvement, "accepted": accepted,
                        "history": history})
        log_message(f"[physical_reconstruction] sector={sector} initial={initial_validation:.6g} selected={best_validation:.6g} improvement={improvement:.6g} accepted={accepted}")

    all_folds = torch.cat(fold_images, 0)
    accepted_indices = [i for i, item in enumerate(records) if item["accepted"]]
    passed = len(accepted_indices) >= a.min_accepted_folds
    initial_cpu = initial_full.detach().cpu()
    support_cpu = full_support.detach().cpu()
    if passed:
        accepted_stack = all_folds[accepted_indices]
        primary = torch.quantile(accepted_stack, .5, dim=0, keepdim=True)
        spread = torch.quantile(accepted_stack, .75, dim=0, keepdim=True) - torch.quantile(accepted_stack, .25, dim=0, keepdim=True)
    else:
        primary, spread = initial_cpu.clone(), torch.zeros_like(initial_cpu)
        log_message(f"[physical_reconstruction] GATE FAILED: {len(accepted_indices)}/{len(sectors)} improved; primary output is frozen Stage 7 prior")
    inside = spread[support_cpu > .5]
    agreement_scale = torch.quantile(inside, .75).clamp_min(5) if inside.numel() else torch.as_tensor(5.)
    agreement = torch.exp(-torch.square(spread / agreement_scale)) * support_cpu
    weighted = initial_cpu + agreement * (primary - initial_cpu)
    output = {"kind": "experimental_physical_ray", "version": VERSION,
              "prior_mps": initial_cpu, "fold_images_mps": all_folds,
              "reconstruction_mps": primary, "agreement_weighted_reconstruction_mps": weighted,
              "agreement_weight": agreement, "agreement_scale_mps": float(agreement_scale),
              "interquartile_spread_mps": spread, "fold_records": records,
              "accepted_fold_count": len(accepted_indices), "min_accepted_folds": a.min_accepted_folds,
              "physical_refinement_accepted": passed, "soft_support": support_cpu,
              "reference_diagnostic_normalized": setup.get("reference_diagnostic_normalized"),
              "hardware_calibration": setup.get("hardware_calibration"), "setup_pt": a.setup_pt}
    Path(a.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, a.out_pt)
    gt = setup.get("reference_diagnostic_normalized")
    gt_mps = (gt[0, 0] * (hi - lo) + lo).detach().cpu().numpy() if gt is not None else np.full(initial_cpu.shape[-2:], np.nan)
    histories = np.empty(len(records), dtype=object)
    for i, item in enumerate(records):
        histories[i] = np.asarray(item["history"], dtype=np.float32)
    np.savez_compressed(a.report_npz, gt_mps=gt_mps,
        prior_mps=initial_cpu[0, 0].numpy(), reconstruction_mps=primary[0, 0].numpy(),
        agreement_weighted_mps=weighted[0, 0].numpy(), agreement_weight=agreement[0, 0].numpy(),
        agreement_scale_mps=np.asarray(float(agreement_scale)), spread_mps=spread[0, 0].numpy(),
        fold_images_mps=all_folds[:, 0].numpy(), soft_support=support_cpu[0, 0].numpy(),
        histories=histories, sectors=np.asarray(sectors),
        initial_validation_us=np.asarray([r["initial_validation_us"] for r in records]),
        best_steps=np.asarray([r["best_step"] for r in records]),
        best_validation_us=np.asarray([r["best_validation_us"] for r in records]),
        holdout_improvement_us=np.asarray([r["holdout_improvement_us"] for r in records]),
        accepted=np.asarray([r["accepted"] for r in records]),
        accepted_fold_count=np.asarray(len(accepted_indices)),
        physical_refinement_accepted=np.asarray(passed),
        phys_x=np.asarray(float(geometry["phys_x"])), phys_y=np.asarray(float(geometry["phys_y"])),
        sos_min=np.asarray(lo), sos_max=np.asarray(hi))
    log_message(f"[physical_reconstruction] saved={a.out_pt}; report={a.report_npz}; accepted_folds={len(accepted_indices)}/{len(sectors)}")


if __name__ == "__main__":
    main()










