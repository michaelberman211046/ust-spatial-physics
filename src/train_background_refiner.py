import argparse
import copy
import os
import pprint
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from anatomy import generate_sensor_positions
from dataset import PairsCacheDataset, load_splits, make_cache_paths
from inverse_model_io import load_inverse_model
from latent_operator_models import (
    SCRIPT_FAMILY,
    ConditionalLatentEncoder,
    batch_progress,
    blur,
    denormalize_tof_seconds,
    normalize_tof_seconds,
    format_duration,
    grad_l1,
    highpass,
    load_ae_checkpoint,
    load_matched_operator_checkpoint,
    masked_l1,
    masked_mse,
    support_apply,
    tv_l1,
)
from logger import log_image as _logger_log_image, log_message as _logger_log_message
from settings import set_output_folder
from reconstruction_utils import make_circular_support_mask, make_homogeneous_water_tof_norm, make_model_input


SCRIPT_VERSION = "background-refiner-v1"


def _normalize_output_dir(path):
    path = str(path)
    if path and not path.endswith(("/", "\\")):
        path += "/"
    return path


def _ensure_missing_logger_html(exc):
    """Recover from a missing HTML log file without modifying the shared logger."""
    missing = getattr(exc, "filename", None)
    if not missing or not str(missing).lower().endswith(".html"):
        return False
    folder = os.path.dirname(missing)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(missing):
        with open(missing, "w", encoding="utf-8") as f:
            f.write("<html>\n<body>\n</body>\n</html>\n")
    return True


def log_message(message):
    try:
        _logger_log_message(message)
    except FileNotFoundError as exc:
        if not _ensure_missing_logger_html(exc):
            raise
        _logger_log_message(message)


def log_image(fig):
    try:
        _logger_log_image(fig)
    except FileNotFoundError as exc:
        if not _ensure_missing_logger_html(exc):
            raise
        _logger_log_image(fig)


def cuda_memory_text(device):
    if device.type != "cuda":
        return "cuda_memory=not_applicable"
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f"cuda_allocated={allocated:.3f}GiB cuda_reserved={reserved:.3f}GiB cuda_peak={peak:.3f}GiB"


class IndexedSubset(Subset):
    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        item = self.dataset[real_idx]
        return (*item, torch.tensor(real_idx, dtype=torch.long))

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


class StraightRayAdjoint(nn.Module):
    """Straight-ray linearized ToF adjoint around the initial SoS image.

    For a small SoS perturbation dc, the straight-ray approximation is
        dT_er ~= - integral_gamma dc(x) / c0(x)^2 ds.
    Its adjoint maps a ToF residual r_er to an image-space correction evidence
    q(x) ~= - sum_er r_er / c0(x)^2 along rays.

    This is used as a fast physics feature, not as a final inverse solver.
    """
    def __init__(
        self,
        nx,
        ny,
        phys_x,
        phys_y,
        radius,
        n_emitters,
        n_receivers,
        ray_samples=128,
        ray_chunk=4096,
        device=None,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_emitters = int(n_emitters)
        self.n_receivers = int(n_receivers)
        self.ray_samples = int(ray_samples)
        self.ray_chunk = max(1, int(ray_chunk))
        self.phys_x = float(phys_x)
        self.phys_y = float(phys_y)

        dx = float(phys_x) / max(1, int(nx) - 1)
        dy = float(phys_y) / max(1, int(ny) - 1)
        emitters, receivers = generate_sensor_positions(
            int(nx), int(ny), dx, dy, float(radius), int(n_emitters), int(n_receivers)
        )
        # Build geometry on the CPU in bounded chunks to limit GPU memory use.
        e = torch.tensor(np.asarray(emitters, dtype=np.float32), device="cpu")
        r = torch.tensor(np.asarray(receivers, dtype=np.float32), device="cpu")

        e_xy = torch.stack([e[:, 0] * dx - float(phys_x) / 2.0, e[:, 1] * dy - float(phys_y) / 2.0], dim=1)
        r_xy = torch.stack([r[:, 0] * dx - float(phys_x) / 2.0, r[:, 1] * dy - float(phys_y) / 2.0], dim=1)

        ee = e_xy[:, None, :].expand(int(n_emitters), int(n_receivers), 2).reshape(-1, 2)
        rr = r_xy[None, :, :].expand(int(n_emitters), int(n_receivers), 2).reshape(-1, 2)
        t = torch.linspace(0.0, 1.0, int(ray_samples), device="cpu").view(1, -1, 1)
        pix_parts, valid_parts = [], []
        geometry_chunk = max(1024, min(self.ray_chunk, 8192))
        for start in range(0, int(ee.shape[0]), geometry_chunk):
            end = min(int(ee.shape[0]), start + geometry_chunk)
            e_part, r_part = ee[start:end], rr[start:end]
            pts = e_part[:, None, :] * (1.0 - t) + r_part[:, None, :] * t
            ix = torch.round((pts[..., 0] + float(phys_x) / 2.0) / dx).to(torch.int32)
            iy = torch.round((pts[..., 1] + float(phys_y) / 2.0) / dy).to(torch.int32)
            valid = (ix >= 0) & (ix < int(nx)) & (iy >= 0) & (iy < int(ny))
            ix.clamp_(0, int(nx) - 1)
            iy.clamp_(0, int(ny) - 1)
            pix_parts.append((ix * int(ny) + iy).contiguous())
            valid_parts.append(valid.contiguous())
        pix = torch.cat(pix_parts, dim=0)
        valid = torch.cat(valid_parts, dim=0)
        ds_per_ray = torch.linalg.norm(rr - ee, dim=1).clamp_min(1e-9) / max(1, int(ray_samples) - 1)

        self.register_buffer("pix_idx", pix.to(device=device, dtype=torch.int32), persistent=False)
        self.register_buffer("valid", valid.to(device=device, dtype=torch.bool), persistent=False)
        self.register_buffer("ds_per_ray", ds_per_ray.to(device=device, dtype=torch.float32), persistent=False)

    def forward(self, residual_seconds, base_sos_mps, tof_mask=None, support_mask=None):
        b = int(residual_seconds.shape[0])
        n_pix = self.nx * self.ny
        residual = residual_seconds.reshape(b, -1).float()
        if tof_mask is not None:
            residual = residual * tof_mask.reshape(b, -1).float()

        c_flat = base_sos_mps[:, 0].reshape(b, -1).float().clamp_min(1200.0)
        q_flat = torch.zeros((b, n_pix), device=residual.device, dtype=residual.dtype)
        norm_flat = torch.zeros((b, n_pix), device=residual.device, dtype=residual.dtype)
        n_rays = int(self.pix_idx.shape[0])
        for start in range(0, n_rays, self.ray_chunk):
            end = min(n_rays, start + self.ray_chunk)
            pix_part = self.pix_idx[start:end].to(dtype=torch.long)
            samples = int(pix_part.shape[1])
            pix_batch = pix_part.reshape(1, -1).expand(b, -1)
            c_ray = torch.gather(c_flat, 1, pix_batch).reshape(b, end - start, samples)
            valid = self.valid[start:end].to(dtype=residual.dtype).unsqueeze(0)
            ds = self.ds_per_ray[start:end].view(1, -1, 1)
            coeff = valid * ds / c_ray.square().clamp_min(1.0)
            contrib = -residual[:, start:end].unsqueeze(-1) * coeff
            q_flat.scatter_add_(1, pix_batch, contrib.reshape(b, -1))
            norm_flat.scatter_add_(1, pix_batch, coeff.reshape(b, -1))
        q = q_flat / norm_flat.clamp_min(1e-12)
        q = q.view(b, 1, self.nx, self.ny)
        if support_mask is not None:
            q = q * support_mask.to(device=q.device, dtype=q.dtype).unsqueeze(0)
        q = q - q.flatten(1).mean(dim=1).view(b, 1, 1, 1)
        scale = q.flatten(1).abs().quantile(0.95, dim=1).view(b, 1, 1, 1).clamp_min(1e-9)
        return torch.clamp(q / scale, -3.0, 3.0)


class AdjointLatentRefiner(nn.Module):
    def __init__(self, in_ch, latent_ch=64, latent_grid=32, channels=96, dropout=0.05, delta_limit=0.65):
        super().__init__()
        self.delta_limit = float(delta_limit)
        self.encoder = ConditionalLatentEncoder(
            in_ch=int(in_ch),
            latent_ch=int(latent_ch),
            latent_grid=int(latent_grid),
            channels=int(channels),
            dropout=float(dropout),
        )
        groups = max(1, min(8, int(latent_ch) // 4))
        self.body = nn.Sequential(
            nn.Conv2d(int(latent_ch) * 2, int(latent_ch) * 2, 3, padding=1),
            nn.GroupNorm(groups, int(latent_ch) * 2),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(int(latent_ch) * 2, int(latent_ch), 3, padding=1),
            nn.GroupNorm(groups, int(latent_ch)),
            nn.SiLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(int(latent_ch), int(latent_ch), 1)
        self.gate_head = nn.Sequential(nn.Conv2d(int(latent_ch), int(latent_ch), 1), nn.Sigmoid())

    def forward(self, features, z0):
        h = self.encoder(features)
        body = self.body(torch.cat([h, z0], dim=1))
        raw_delta = self.delta_limit * torch.tanh(self.delta_head(body))
        gate = self.gate_head(body)
        dz = gate * raw_delta
        return z0 + dz, dz, gate


def split_dataset(ds, data_path, splits_path):
    if splits_path is None:
        splits_path = make_cache_paths(data_path)["splits_path"]
    if not os.path.exists(splits_path):
        n = len(ds)
        return IndexedSubset(ds, list(range(int(0.8 * n)))), IndexedSubset(ds, list(range(int(0.8 * n), n)))
    splits = load_splits(splits_path)
    return IndexedSubset(ds, list(map(int, splits["train_idx"]))), IndexedSubset(ds, list(map(int, splits["val_idx"])))


def unpack(batch, device):
    idx = batch[-1].to(device).long()
    sos = batch[0].to(device).float().unsqueeze(1)
    tof = batch[1].to(device).float()
    if len(batch) > 3 and torch.is_tensor(batch[2]) and batch[2].shape == batch[1].shape:
        tof_mask = batch[2].to(device).float()
    else:
        tof_mask = torch.ones_like(tof)
    return sos, tof, tof_mask, idx


def corrupt_acquisition(tof, tof_mask, args):
    """Apply hardware-like timing errors before the Stage 4 adjoint operation."""
    physical = denormalize_tof_seconds(tof, args.tof_norm)
    mask = tof_mask.clone()
    b, ne, nr = physical.shape
    dev, dtype = physical.device, physical.dtype
    ea = 2.0 * torch.pi * torch.arange(ne, device=dev, dtype=dtype) / max(1, ne)
    ra = 2.0 * torch.pi * torch.arange(nr, device=dev, dtype=dtype) / max(1, nr)
    emitter = torch.zeros((b, ne), device=dev, dtype=dtype)
    receiver = torch.zeros((b, nr), device=dev, dtype=dtype)
    for harmonic in range(1, int(args.augmentation_harmonics) + 1):
        scale = 1.0 / harmonic
        ec = torch.randn((b, 2), device=dev, dtype=dtype)
        rc = torch.randn((b, 2), device=dev, dtype=dtype)
        emitter += scale * (ec[:, :1] * torch.sin(harmonic * ea) + ec[:, 1:] * torch.cos(harmonic * ea))
        receiver += scale * (rc[:, :1] * torch.sin(harmonic * ra) + rc[:, 1:] * torch.cos(harmonic * ra))
    emitter = emitter / emitter.std(dim=1, keepdim=True).clamp_min(1e-6)
    receiver = receiver / receiver.std(dim=1, keepdim=True).clamp_min(1e-6)
    emitter *= float(args.augmentation_emitter_us) * 1e-6
    receiver *= float(args.augmentation_receiver_us) * 1e-6
    global_delay = (2.0 * torch.rand((b, 1, 1), device=dev, dtype=dtype) - 1.0) * float(args.augmentation_global_us) * 1e-6
    noise = torch.randn_like(physical) * float(args.augmentation_noise_us) * 1e-6
    physical = physical + global_delay + emitter[:, :, None] + receiver[:, None, :] + noise
    if float(args.augmentation_sector_drop_probability) > 0:
        sectors = max(1, int(args.augmentation_sectors))
        ew, rw = max(1, ne // sectors), max(1, nr // sectors)
        for sample in range(b):
            if torch.rand((), device=dev) < float(args.augmentation_sector_drop_probability):
                sector = int(torch.randint(sectors, (1,), device=dev))
                mask[sample, sector * ew:min(ne, (sector + 1) * ew), :] = 0
            if torch.rand((), device=dev) < float(args.augmentation_sector_drop_probability):
                sector = int(torch.randint(sectors, (1,), device=dev))
                mask[sample, :, sector * rw:min(nr, (sector + 1) * rw)] = 0
    if float(args.augmentation_channel_drop_probability) > 0:
        mask *= (torch.rand_like(mask) > float(args.augmentation_channel_drop_probability)).to(mask.dtype)
    return normalize_tof_seconds(physical, args.tof_norm) * mask, mask


def detail_l1(a, b, mask, kernel_size):
    return masked_l1(highpass(a, kernel_size), highpass(b, kernel_size), mask)


def tof_loss(pred_tof, target_tof, tof_mask, args):
    return args.lambda_tof_l1 * masked_l1(pred_tof, target_tof, tof_mask) + args.lambda_tof_mse * masked_mse(pred_tof, target_tof, tof_mask)


def tof_mae_us(pred_tof, target_tof, tof_mask, tof_norm):
    pred_s = denormalize_tof_seconds(pred_tof, tof_norm)
    target_s = denormalize_tof_seconds(target_tof, tof_norm)
    err_us = (pred_s - target_s).abs() / 1e-6
    m = tof_mask.to(device=err_us.device, dtype=err_us.dtype)
    return float((err_us * m).sum().detach().item() / m.sum().clamp_min(1.0).detach().item())


def tof_us_display(tof_norm_sample, tof_mask_sample, tof_norm, fill_mode="finite_median"):
    tof_us = denormalize_tof_seconds(tof_norm_sample, tof_norm) / 1e-6
    valid = torch.isfinite(tof_us) & (tof_mask_sample.to(device=tof_us.device) > 0.5)
    fill = tof_us[valid].median() if valid.any() else torch.tensor(0.0, device=tof_us.device, dtype=tof_us.dtype)
    return torch.where(valid, tof_us, fill).detach().cpu()


def tof_residual_us_display(pred_tof_norm_sample, target_tof_norm_sample, tof_mask_sample, tof_norm):
    pred_us = denormalize_tof_seconds(pred_tof_norm_sample, tof_norm) / 1e-6
    target_us = denormalize_tof_seconds(target_tof_norm_sample, tof_norm) / 1e-6
    residual = pred_us - target_us
    valid = torch.isfinite(residual) & (tof_mask_sample.to(device=residual.device) > 0.5)
    return torch.where(valid, residual, torch.zeros_like(residual)).detach().cpu()


def grad_mag(x):
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-8)


def make_refiner_features(x_model, base, adjoint, support_mask, args):
    b, _c, h, w = base.shape
    tof_img = F.interpolate(x_model, size=(h, w), mode="bilinear", align_corners=False)
    mask_chan = support_mask.to(device=base.device, dtype=base.dtype).unsqueeze(0).expand(b, 1, h, w) if support_mask is not None else torch.ones_like(base)
    return torch.cat([
        tof_img,
        base,
        blur(base, args.lowpass_kernel),
        highpass(base, args.detail_kernel),
        grad_mag(base),
        adjoint,
        adjoint.abs(),
        blur(adjoint, args.adjoint_blur_kernel),
        highpass(adjoint, args.detail_kernel),
        mask_chan,
    ], dim=1)


def forward_reconstruction(model, adjoint_op, auto, op, base_model, target, tof, tof_mask, support_mask, water_tof_norm, args):
    x_model = make_model_input(tof, tof_mask, args.use_tof_mask_channel, args.tof_feature_mode, water_tof_norm)
    with torch.no_grad():
        base = support_apply(base_model(x_model), support_mask, args.water_norm)
        z0 = auto.encode(base)
        base_tof = op(base)
        residual_s = denormalize_tof_seconds(tof, args.tof_norm) - denormalize_tof_seconds(base_tof, args.tof_norm)
        base_mps = base * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
        adj = adjoint_op(residual_s, base_mps, tof_mask, support_mask)
    feat = make_refiner_features(x_model, base, adj, support_mask, args)
    z_pred, dz, gate = model(feat, z0)
    pred = support_apply(auto.decoder(z_pred), support_mask, args.water_norm)
    pred_tof = op(pred)
    return base, adj, z0, z_pred, dz, pred, pred_tof, gate


@torch.no_grad()
def evaluate(model, adjoint_op, auto, op, base_model, loader, device, support_mask, water_tof_norm, args):
    model.eval()
    sums = {
        "base_mse": 0.0, "pred_mse": 0.0, "pred_l1": 0.0, "detail_l1": 0.0,
        "tof_loss": 0.0, "tof_mae_us": 0.0, "score": 0.0, "adj_abs": 0.0,
    }
    n = 0
    for batch in loader:
        target, tof, tof_mask, _idx = unpack(batch, device)
        target = support_apply(target, support_mask, args.water_norm)
        base, adj, _z0, _z_pred, _dz, pred, pred_tof, _gate = forward_reconstruction(
            model, adjoint_op, auto, op, base_model, target, tof, tof_mask, support_mask, water_tof_norm, args
        )
        base_mse = masked_mse(base, target, support_mask)
        pred_mse = masked_mse(pred, target, support_mask)
        det = detail_l1(pred, target, support_mask, args.detail_kernel)
        tloss = tof_loss(pred_tof, tof, tof_mask, args)
        no_improve = torch.relu(pred_mse - base_mse)
        score = pred_mse + args.val_detail_weight * det + args.val_no_improve_penalty * no_improve
        sums["base_mse"] += float(base_mse.item())
        sums["pred_mse"] += float(pred_mse.item())
        sums["pred_l1"] += float(masked_l1(pred, target, support_mask).item())
        sums["detail_l1"] += float(det.item())
        sums["tof_loss"] += float(tloss.item())
        sums["tof_mae_us"] += tof_mae_us(pred_tof, tof, tof_mask, args.tof_norm)
        sums["score"] += float(score.item())
        sums["adj_abs"] += float(adj.abs().mean().item())
        n += 1
    out = {k: v / max(1, n) for k, v in sums.items()}
    out["improvement_pct"] = 100.0 * (out["base_mse"] - out["pred_mse"]) / max(out["base_mse"], 1e-12)
    return out


@torch.no_grad()
def plot_preview(model, adjoint_op, auto, op, base_model, loader, device, support_mask, water_tof_norm, args, title):
    model.eval()
    batch = next(iter(loader))
    target, tof, tof_mask, idx = unpack(batch, device)
    target = support_apply(target, support_mask, args.water_norm)
    base, adj, _z0, _z_pred, _dz, pred, pred_tof, _gate = forward_reconstruction(
        model, adjoint_op, auto, op, base_model, target, tof, tof_mask, support_mask, water_tof_norm, args
    )
    n = min(4, target.shape[0])
    vmin, vmax = float(args.sos_min), float(args.sos_max)
    extent = [0.0, float(args.phys_x), float(args.phys_y), 0.0]
    fig, axes = plt.subplots(n, 8, figsize=(32, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        gt = target[i, 0].cpu() * (vmax - vmin) + vmin
        b = base[i, 0].cpu() * (vmax - vmin) + vmin
        q = adj[i, 0].cpu()
        p = pred[i, 0].cpu() * (vmax - vmin) + vmin
        err = (pred[i, 0] - target[i, 0]).abs().cpu() * (vmax - vmin)
        stored_us = tof_us_display(tof[i], tof_mask[i], args.tof_norm, args.tof_plot_fill)
        residual_us = tof_residual_us_display(pred_tof[i], tof[i], tof_mask[i], args.tof_norm)
        panels = [
            (gt, "Synthetic GT SoS", "gray", vmin, vmax, "image"),
            (b, "Initial learned reconstruction", "gray", vmin, vmax, "image"),
            (q, "Straight-ray adjoint evidence", "coolwarm", -2.5, 2.5, "image"),
            (p, "Adjoint-guided latent reconstruction", "gray", vmin, vmax, "image"),
            (err, "|adjoint-guided-GT| [m/s]", "magma", 0.0, None, "image"),
            (stored_us, "Stored ToF [us]\ninvalid filled for display only", "viridis", None, None, "tof"),
            (residual_us, "Matched-ToF residual [us]\ninvalid shown as 0", "coolwarm", -args.residual_plot_clip_us, args.residual_plot_clip_us, "tof"),
        ]
        for j, (img, name, cmap, lo, hi, kind) in enumerate(panels):
            ax = axes[i, j]
            if kind == "image":
                im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=extent, origin="upper", aspect="equal")
                ax.set_xlabel("Lateral [m]")
                ax.set_ylabel("Axial [m]")
            else:
                im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="lower", aspect="equal")
                ax.set_xlabel("Receiver index")
                ax.set_ylabel("Emitter index")
            ax.set_title(name)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[i, 7]
        ax.axis("off")
        ax.text(0.02, 0.85, f"sample index: {int(idx[i].item())}", transform=ax.transAxes)
        ax.text(0.02, 0.65, "physics feature: adjoint of initial ToF residual", transform=ax.transAxes)
        ax.text(0.02, 0.45, f"ray samples: {args.adjoint_ray_samples}", transform=ax.transAxes)
    fig.suptitle(title)
    fig.tight_layout()
    log_image(fig)
    plt.close(fig)


def save_ckpt(path, args, model, best_state, best_score, best_epoch, epoch, hist, opt=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "script_family": SCRIPT_FAMILY,
        "script_version": SCRIPT_VERSION,
        "kind": "adjoint_latent_physics_reconstructor",
        "config": vars(args),
        "model_state_dict": model.state_dict(),
        "best_model_state_dict": best_state,
        "best_val_score": float(best_score),
        "best_epoch": int(best_epoch),
        "last_epoch": int(epoch),
        "history": hist,
        "optimizer_state_dict": opt.state_dict() if opt is not None else None,
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--splits_path", default=None)
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--ae_path", required=True)
    p.add_argument("--operator_path", required=True)
    p.add_argument("--reconstructor_path", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init_reconstructor_path", default=None)
    p.add_argument("--model_type", default="deep")
    p.add_argument("--latent_res", type=int, default=36)
    p.add_argument("--use_tof_mask_channel", action="store_true")
    p.add_argument("--tof_feature_mode", choices=["raw", "residual_stack"], default="residual_stack")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=9e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--delta_limit", type=float, default=0.65)
    p.add_argument("--adjoint_ray_samples", type=int, default=128)
    p.add_argument("--adjoint_ray_chunk", type=int, default=4096)
    p.add_argument("--adjoint_blur_kernel", type=int, default=9)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--augmentation_probability", type=float, default=0.75)
    p.add_argument("--augmentation_global_us", type=float, default=0.35)
    p.add_argument("--augmentation_emitter_us", type=float, default=0.30)
    p.add_argument("--augmentation_receiver_us", type=float, default=0.30)
    p.add_argument("--augmentation_noise_us", type=float, default=0.035)
    p.add_argument("--augmentation_sector_drop_probability", type=float, default=0.20)
    p.add_argument("--augmentation_channel_drop_probability", type=float, default=0.01)
    p.add_argument("--augmentation_sectors", type=int, default=8)
    p.add_argument("--augmentation_harmonics", type=int, default=3)
    p.add_argument("--lambda_corrupt_reconstruction", type=float, default=0.45)
    p.add_argument("--lambda_clean_corrupt_consistency", type=float, default=0.20)
    p.add_argument("--lowpass_kernel", type=int, default=21)
    p.add_argument("--detail_kernel", type=int, default=7)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=0.65)
    p.add_argument("--lambda_grad", type=float, default=0.55)
    p.add_argument("--lambda_detail", type=float, default=0.80)
    p.add_argument("--lambda_base_low_anchor", type=float, default=0.015)
    p.add_argument("--lambda_base_full_anchor", type=float, default=0.005)
    p.add_argument("--lambda_tof_l1", type=float, default=0.10)
    p.add_argument("--lambda_tof_mse", type=float, default=0.03)
    p.add_argument("--lambda_tv", type=float, default=0.001)
    p.add_argument("--lambda_delta", type=float, default=0.002)
    p.add_argument("--lambda_gate_sparsity", type=float, default=0.0005)
    p.add_argument("--val_detail_weight", type=float, default=0.60)
    p.add_argument("--val_no_improve_penalty", type=float, default=5.0)
    p.add_argument("--early_stop_patience", type=int, default=18)
    p.add_argument("--early_stop_min_delta", type=float, default=2e-5)
    p.add_argument("--stop_if_not_better_than_base_patience", type=int, default=14)
    p.add_argument("--preview_every", type=int, default=4)
    p.add_argument("--batch_log_every", type=int, default=25)
    p.add_argument("--residual_plot_clip_us", type=float, default=8.0)
    p.add_argument("--tof_plot_fill", choices=["finite_median"], default="finite_median")
    p.add_argument("--use_circular_mask", action="store_true")
    p.add_argument("--mask_radius", type=float, default=0.1091)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--phys_x", type=float, default=0.24)
    p.add_argument("--phys_y", type=float, default=0.24)
    p.add_argument("--radius", type=float, default=0.1091)
    p.add_argument("--n_emitters", type=int, default=64)
    p.add_argument("--n_receivers", type=int, default=64)
    p.add_argument("--sos_min", type=float, default=1380.0)
    p.add_argument("--sos_max", type=float, default=1660.0)
    p.add_argument("--sos_water", type=float, default=1500.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_dir:
        args.output_dir = _normalize_output_dir(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        set_output_folder(args.output_dir)
    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message(f"[adjoint_latent_physics] script_version={SCRIPT_VERSION}")
    requested = str(args.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        msg = "[adjoint_latent_physics] CUDA requested but unavailable. Actual device will be CPU."
        if args.require_cuda:
            raise RuntimeError(msg)
        log_message(msg)
        requested = "cpu"
    device = torch.device(requested)
    log_message(f"[adjoint_latent_physics] Actual device used: {device}")

    ds = PairsCacheDataset(args.data_path, sos_min=args.sos_min, sos_max=args.sos_max, return_tof_mask=True)
    args.tof_norm = ds.tof_norm
    meta = dict(getattr(ds, "metadata", {}) or {})
    args.nx = int(meta.get("nx", args.nx))
    args.ny = int(meta.get("ny", args.ny))
    args.phys_x = float(meta.get("phys_x", args.phys_x))
    args.phys_y = float(meta.get("phys_y", args.phys_y))
    args.radius = float(meta.get("radius", args.radius))
    args.n_emitters = int(meta.get("n_emitters", args.n_emitters))
    args.n_receivers = int(meta.get("n_receivers", args.n_receivers))
    args.sos_water = float(meta.get("sos_water", args.sos_water))
    args.water_norm = float(max(0.0, min(1.0, (args.sos_water - args.sos_min) / (args.sos_max - args.sos_min))))

    train_ds, val_ds = split_dataset(ds, args.data_path, args.splits_path)
    if args.max_train_samples > 0:
        train_ds = IndexedSubset(ds, train_ds.indices[: min(args.max_train_samples, len(train_ds))])
    if args.max_val_samples > 0:
        val_ds = IndexedSubset(ds, val_ds.indices[: min(args.max_val_samples, len(val_ds))])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    support_mask = None
    if args.use_circular_mask:
        support_mask = make_circular_support_mask(args.nx, args.ny, args.phys_x, args.phys_y, args.mask_radius, device=device)
    water_tof_norm = None
    if args.tof_feature_mode == "residual_stack":
        water_tof_norm = make_homogeneous_water_tof_norm(
            args.nx, args.ny, args.phys_x, args.phys_y, args.radius,
            args.n_emitters, args.n_receivers, args.sos_water, args.tof_norm, device,
        )
    tof_input_channels = 4 if args.use_tof_mask_channel and args.tof_feature_mode == "residual_stack" else (2 if args.use_tof_mask_channel else 1)
    base_model, _ = load_inverse_model(
        args.base_model_path, args.model_type, args.nx, args.ny, args.latent_res,
        args.phys_x, args.phys_y, args.radius, args.n_emitters, args.n_receivers,
        tof_input_channels, device,
    )
    auto, ae_cfg, _ae_ckpt = load_ae_checkpoint(args.ae_path, device)
    op, _op_cfg, op_ckpt = load_matched_operator_checkpoint(args.operator_path, device)
    if "best_operator_state_dict" in op_ckpt:
        op.load_state_dict(op_ckpt["best_operator_state_dict"])
    for m in (base_model, auto, op):
        m.eval()
        for param in m.parameters():
            param.requires_grad_(False)

    adjoint_op = StraightRayAdjoint(
        args.nx, args.ny, args.phys_x, args.phys_y, args.radius,
        args.n_emitters, args.n_receivers, args.adjoint_ray_samples,
        ray_chunk=args.adjoint_ray_chunk, device=device,
    )
    log_message(f"[adjoint_latent_physics] Chunked adjoint ready: rays={args.n_emitters * args.n_receivers} samples={args.adjoint_ray_samples} chunk={args.adjoint_ray_chunk}; {cuda_memory_text(device)}")

    first = next(iter(train_loader))
    target0, tof0, tof_mask0, _idx0 = unpack(first, device)
    x0 = make_model_input(tof0, tof_mask0, args.use_tof_mask_channel, args.tof_feature_mode, water_tof_norm)
    with torch.no_grad():
        base0 = support_apply(base_model(x0), support_mask, args.water_norm)
        base_tof0 = op(base0)
        residual_s0 = denormalize_tof_seconds(tof0, args.tof_norm) - denormalize_tof_seconds(base_tof0, args.tof_norm)
        base_mps0 = base0 * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
        adj0 = adjoint_op(residual_s0, base_mps0, tof_mask0, support_mask)
    in_ch = int(make_refiner_features(x0, base0, adj0, support_mask, args).shape[1])
    latent_ch = int(ae_cfg.get("latent_ch", 64))
    latent_grid = int(ae_cfg.get("latent_grid", 32))
    model = AdjointLatentRefiner(in_ch, latent_ch, latent_grid, args.channels, args.dropout, args.delta_limit).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = {
        "train": [], "val_score": [], "val_base_mse": [], "val_pred_mse": [], "val_pred_l1": [],
        "val_detail_l1": [], "val_tof_loss": [], "val_tof_mae_us": [], "val_adj_abs": [],
        "val_improvement_pct": [],
    }
    best_score, best_epoch, stale, not_better = float("inf"), 0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    start_epoch = 1
    if args.resume and os.path.exists(args.reconstructor_path):
        ckpt = torch.load(args.reconstructor_path, map_location=device, weights_only=False)
        if ckpt.get("kind") != "adjoint_latent_physics_reconstructor":
            raise ValueError(f"--resume found incompatible checkpoint: {args.reconstructor_path}")
        state = ckpt["model_state_dict"]
        model.load_state_dict(state)
        best_state = copy.deepcopy(ckpt.get("best_model_state_dict", state))
        best_score = float(ckpt.get("best_val_score", best_score))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        hist = copy.deepcopy(ckpt.get("history", hist))
        if ckpt.get("optimizer_state_dict") is not None:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("last_epoch", len(hist.get("val_score", [])))) + 1
        log_message(f"[adjoint_latent_physics] Resuming from epoch {start_epoch}; best={best_score:.6g}@{best_epoch}.")
    elif args.resume:
        log_message(f"[adjoint_latent_physics] --resume requested but checkpoint does not exist: {args.reconstructor_path}. Starting from scratch.")
    elif args.init_reconstructor_path:
        if not os.path.exists(args.init_reconstructor_path):
            raise FileNotFoundError(f"Stage 4 initialization checkpoint is missing: {args.init_reconstructor_path}")
        init_ckpt = torch.load(args.init_reconstructor_path, map_location=device, weights_only=False)
        if init_ckpt.get("kind") != "adjoint_latent_physics_reconstructor":
            raise ValueError("Stage 4 initialization checkpoint has an incompatible kind")
        init_state = init_ckpt.get("best_model_state_dict", init_ckpt.get("model_state_dict"))
        model.load_state_dict(init_state)
        best_state = copy.deepcopy(init_state)
        log_message(f"[adjoint_latent_physics] Warm-started weights only from {args.init_reconstructor_path}; optimizer and stopping history are fresh.")

    phase_t0 = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_t0 = time.time()
            running, used_batches = 0.0, 0
            accumulation = max(1, int(args.gradient_accumulation))
            opt.zero_grad(set_to_none=True)
            for bi, batch in enumerate(train_loader, start=1):
                target, tof, tof_mask, _idx = unpack(batch, device)
                target = support_apply(target, support_mask, args.water_norm)
                base, adj, _z0, _z_pred, dz, pred, pred_tof, gate = forward_reconstruction(
                    model, adjoint_op, auto, op, base_model, target, tof, tof_mask, support_mask, water_tof_norm, args
                )
                low_anchor = masked_l1(blur(pred, args.lowpass_kernel), blur(base, args.lowpass_kernel), support_mask)
                full_anchor = masked_l1(pred, base, support_mask)
                loss_tof = tof_loss(pred_tof, tof, tof_mask, args)
                loss = (
                    args.lambda_l1 * masked_l1(pred, target, support_mask)
                    + args.lambda_mse * masked_mse(pred, target, support_mask)
                    + args.lambda_grad * grad_l1(pred, target, support_mask)
                    + args.lambda_detail * detail_l1(pred, target, support_mask, args.detail_kernel)
                    + args.lambda_base_low_anchor * low_anchor
                    + args.lambda_base_full_anchor * full_anchor
                    + loss_tof
                    + args.lambda_tv * tv_l1(pred, support_mask)
                    + args.lambda_delta * dz.square().mean()
                    + args.lambda_gate_sparsity * gate.mean()
                )
                clean_loss_value = float(loss.detach().item())
                (loss / accumulation).backward()
                total_loss_value = clean_loss_value
                if float(args.augmentation_probability) > 0 and torch.rand((), device=device) < float(args.augmentation_probability):
                    # Use half a batch for the second view. This trains the actual Stage 4
                    # input path while limiting the runtime increase to roughly 50%.
                    n_aug = max(1, int(target.shape[0]) // 2)
                    tof_aug, mask_aug = corrupt_acquisition(tof[:n_aug], tof_mask[:n_aug], args)
                    _base_aug, _adj_aug, _z0_aug, _zp_aug, _dz_aug, pred_aug, _pt_aug, _gate_aug = forward_reconstruction(
                        model, adjoint_op, auto, op, base_model, target[:n_aug], tof_aug, mask_aug,
                        support_mask, water_tof_norm, args
                    )
                    corrupt_reconstruction = (
                        masked_l1(pred_aug, target[:n_aug], support_mask)
                        + 0.65 * masked_mse(pred_aug, target[:n_aug], support_mask)
                        + 0.55 * grad_l1(pred_aug, target[:n_aug], support_mask)
                        + 0.80 * detail_l1(pred_aug, target[:n_aug], support_mask, args.detail_kernel)
                    )
                    consistency = masked_l1(pred_aug, pred[:n_aug].detach(), support_mask)
                    augmentation_loss = args.lambda_corrupt_reconstruction * corrupt_reconstruction + args.lambda_clean_corrupt_consistency * consistency
                    total_loss_value += float(augmentation_loss.detach().item())
                    # The clean graph has already been released, so this second
                    # graph does not overlap it in GPU memory.
                    (augmentation_loss / accumulation).backward()
                if bi % accumulation == 0 or bi == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                running += total_loss_value
                used_batches += 1
                if args.batch_log_every > 0 and (bi == 1 or bi % args.batch_log_every == 0 or bi == len(train_loader)):
                    log_message(
                        batch_progress("[adjoint_latent_physics]", epoch, args.epochs, bi, len(train_loader), epoch_t0, phase_t0)
                        + f" loss={total_loss_value:.6g} adj_abs={float(adj.abs().mean().detach().item()):.6g} tof={float(loss_tof.detach().item()):.6g} {cuda_memory_text(device)}"
                    )
            train_loss = running / max(1, used_batches)
            val = evaluate(model, adjoint_op, auto, op, base_model, val_loader, device, support_mask, water_tof_norm, args)
            hist["train"].append(train_loss)
            hist["val_score"].append(val["score"])
            hist["val_base_mse"].append(val["base_mse"])
            hist["val_pred_mse"].append(val["pred_mse"])
            hist["val_pred_l1"].append(val["pred_l1"])
            hist["val_detail_l1"].append(val["detail_l1"])
            hist["val_tof_loss"].append(val["tof_loss"])
            hist["val_tof_mae_us"].append(val["tof_mae_us"])
            hist["val_adj_abs"].append(val["adj_abs"])
            hist["val_improvement_pct"].append(val["improvement_pct"])
            if val["score"] < best_score - args.early_stop_min_delta:
                best_score = float(val["score"])
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                save_ckpt(args.reconstructor_path, args, model, best_state, best_score, best_epoch, epoch, hist, opt)
                log_message(f"[adjoint_latent_physics] New best: epoch={epoch}, val_score={best_score:.6g}, improvement={val['improvement_pct']:.2f}%")
            else:
                stale += 1
            not_better = not_better + 1 if val["pred_mse"] >= val["base_mse"] else 0
            elapsed = time.time() - phase_t0
            eta = elapsed / max(1, epoch) * max(0, args.epochs - epoch)
            log_message(
                f"[adjoint_latent_physics] epoch {epoch:04d}: train={train_loss:.6g} score={val['score']:.6g} "
                f"val_initial={val['base_mse']:.6g} val_adjoint={val['pred_mse']:.6g} improve={val['improvement_pct']:.2f}% "
                f"detail={val['detail_l1']:.6g} tof={val['tof_loss']:.6g} tof_mae_us={val['tof_mae_us']:.4g} "
                f"adj_abs={val['adj_abs']:.6g} best={best_score:.6g}@{best_epoch} "
                f"stale={stale}/{args.early_stop_patience} not_better_than_initial={not_better}/{args.stop_if_not_better_than_base_patience} ETA={format_duration(eta)}"
            )
            save_ckpt(args.reconstructor_path, args, model, best_state, best_score, best_epoch, epoch, hist, opt)
            if args.preview_every > 0 and (epoch == 1 or epoch % args.preview_every == 0):
                plot_preview(model, adjoint_op, auto, op, base_model, val_loader, device, support_mask, water_tof_norm, args, f"Adjoint-guided latent reconstruction epoch {epoch}")
            if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                log_message(f"[adjoint_latent_physics] Early stopping at epoch {epoch}. Best epoch={best_epoch}.")
                break
            if args.stop_if_not_better_than_base_patience > 0 and not_better >= args.stop_if_not_better_than_base_patience:
                log_message(f"[adjoint_latent_physics] Stopping: prediction failed to beat the initial reconstruction for {not_better} consecutive epochs.")
                break
    except KeyboardInterrupt:
        log_message("[adjoint_latent_physics] Ctrl-C received. Saving current and best checkpoint.")
        save_ckpt(args.reconstructor_path, args, model, best_state, best_score, best_epoch, len(hist["val_score"]), hist, opt)
        return
    model.load_state_dict(best_state)
    plot_preview(model, adjoint_op, auto, op, base_model, val_loader, device, support_mask, water_tof_norm, args, f"Adjoint-guided latent reconstruction best epoch {best_epoch}")
    save_ckpt(args.reconstructor_path, args, model, best_state, best_score, best_epoch, len(hist["val_score"]), hist, opt)
    log_message(f"[adjoint_latent_physics] Done. Best epoch={best_epoch}, best val_score={best_score:.6g}. Saved: {args.reconstructor_path}")


if __name__ == "__main__":
    main()










