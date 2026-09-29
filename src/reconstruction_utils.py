# Reconstruction evaluation utilities
"""
Independent evaluation script for testing the reconstructed model.

Uses:
- pairs cache at --data_path (SoS/ToF/(optional PDE))
- saved splits file (preferred) so test membership is stable across runs/branches
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow mixed OpenMP (unsafe but unblocks)

import config
from logger import log_message, log_image


def _canonicalize_path_str(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(str(path)).expanduser().resolve()).replace('\\', '/')


def _log_dataset_path_consistency(meta: dict, data_path: str, splits_path: str | None) -> dict:
    m = dict(meta) if isinstance(meta, dict) else {}
    actual_data_path = _canonicalize_path_str(data_path)
    actual_splits_path = _canonicalize_path_str(splits_path) if splits_path is not None else None
    recorded_pairs = _canonicalize_path_str(m.get('pairs_cache_path')) if isinstance(m.get('pairs_cache_path'), str) else None
    recorded_splits = _canonicalize_path_str(m.get('splits_path')) if isinstance(m.get('splits_path'), str) else None
    if recorded_pairs is not None and actual_data_path is not None and recorded_pairs != actual_data_path:
        log_message(
            f"[evaluation] WARNING: dataset metadata recorded pairs_cache_path={m.get('pairs_cache_path')} but the file actually loaded is {data_path}. "
            "Using the actually loaded file; the cache was likely copied or renamed after creation."
        )
    if recorded_splits is not None and actual_splits_path is not None and recorded_splits != actual_splits_path:
        log_message(
            f"[evaluation] WARNING: dataset metadata recorded splits_path={m.get('splits_path')} but the file actually used is {splits_path}. "
            "Using the actually loaded splits file."
        )
    m['loaded_from_path'] = data_path
    if splits_path is not None:
        m['loaded_splits_path'] = splits_path
    return m


def _checkpoint_config_value(train_cfg: dict, key: str, default):
    """Return a saved training/config value using current and legacy key names.

    The evaluation routine must instantiate the network with the same configuration recorded in
    the checkpoint.  For DeepGatedReconstructionNet this includes the dropout
    probabilities logged at construction time.  Dropout is disabled in eval(),
    but logging different values from the trained model is misleading and can
    hide real configuration mismatches in future variants.
    """
    if not isinstance(train_cfg, dict):
        return default
    if key in train_cfg and train_cfg[key] is not None:
        return train_cfg[key]
    aliases = {
        "fc_dropout": ["model_fc_dropout"],
        "decoder_dropout": ["model_decoder_dropout"],
        "model_fc_dropout": ["fc_dropout"],
        "model_decoder_dropout": ["decoder_dropout"],
    }.get(key, [])
    for alt in aliases:
        if alt in train_cfg and train_cfg[alt] is not None:
            return train_cfg[alt]
    return default


def _log_if_cli_differs_from_checkpoint(name: str, cli_value, ckpt_value):
    if cli_value is None or ckpt_value is None:
        return
    try:
        same = abs(float(cli_value) - float(ckpt_value)) < 1e-12
    except Exception:
        same = str(cli_value) == str(ckpt_value)
    if not same:
        log_message(
            f"[evaluation] NOTE: CLI {name}={cli_value} differs from checkpoint value {ckpt_value}; using CLI override."
        )

from settings import app_settings, set_output_folder

import torch
import time
import numpy as np
import sys
import pprint
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from utils import plot_results, plot_physics_setup, plot_learning_curves
from anatomy import generate_sensor_positions
from dataset import (
    PairsCacheDataset,
    load_splits,
    make_cache_paths,
    load_pairs_cache_metadata,
)
def _safe_float_list(seq):
    out = []
    for x in seq or []:
        try:
            out.append(float(x))
        except Exception:
            out.append(float('nan'))
    return out


def _plot_checkpoint_learning_curves(checkpoint):
    """Plot the checkpoint learning curve used for model selection.

    The evaluation routine does not recompute training/validation losses. It only visualizes
    the history saved by train_initial_reconstruction.py. To avoid confusing incomparable protocols,
    the default plot contains only deterministic eval-mode SoS losses:

        Train eval SoS      = training split evaluated with model.eval()
        Validation eval SoS = validation split evaluated with model.eval()

    The stochastic backprop loss is intentionally not plotted here.
    """
    if not isinstance(checkpoint, dict):
        return

    comp = checkpoint.get('component_history', {}) or {}
    train_eval_sos = _safe_float_list(comp.get('train_eval_sos_history', []))
    val_sos = _safe_float_list(comp.get('val_sos_history', []))

    # Preferred new-format checkpoint: directly comparable eval-mode curves.
    if train_eval_sos or val_sos:
        fig, ax = plt.subplots(figsize=(8, 6))
        if train_eval_sos:
            ax.plot(train_eval_sos, label='Train eval SoS')
        if val_sos:
            ax.plot(val_sos, label='Validation eval SoS')
            # Mark the checkpoint-selection epoch if available.
            try:
                best_epoch = int(checkpoint.get('best_epoch', 0))
                if best_epoch > 0 and best_epoch <= len(val_sos):
                    ax.axvline(best_epoch - 1, linestyle='--', linewidth=1.0, label=f'Best epoch {best_epoch}')
            except Exception:
                pass
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SoS reconstruction loss')
        ax.set_title('Checkpoint learning curve: train vs validation eval SoS')
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        log_image(fig)
        plt.close(fig)
        log_message('[evaluation] Plotted checkpoint learning curve: Train eval SoS vs Validation eval SoS. Stochastic Train backprop is not plotted.')
        return

    # Legacy checkpoint fallback: only plot if no new eval-mode history exists.
    train_hist = _safe_float_list(checkpoint.get('train_history', []))
    val_hist = _safe_float_list(checkpoint.get('val_history', []))
    if train_hist or val_hist:
        fig, ax = plt.subplots(figsize=(8, 6))
        if train_hist:
            ax.plot(train_hist, label='Legacy train history')
        if val_hist:
            ax.plot(val_hist, label='Legacy validation history')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Checkpoint learning curve: legacy history')
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        log_image(fig)
        plt.close(fig)
        log_message('[evaluation] WARNING: checkpoint has legacy history only; train/validation protocols may not be directly comparable.')


# ----------------------------
# Circular support mask utilities
# ----------------------------
def make_circular_support_mask(nx: int, ny: int, phys_x: float, phys_y: float, radius: float, device=None) -> torch.Tensor:
    """Return mask with shape (1,nx,ny), value 1 inside radius and 0 outside.

    Project convention: spatial tensors are (nx, ny). The physical domain is
    centered at (0,0), with coordinates x in [-phys_x/2, phys_x/2] and
    y in [-phys_y/2, phys_y/2].
    """
    x = torch.linspace(-float(phys_x) / 2.0, float(phys_x) / 2.0, int(nx), device=device, dtype=torch.float32)
    y = torch.linspace(-float(phys_y) / 2.0, float(phys_y) / 2.0, int(ny), device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    mask = ((xx * xx + yy * yy) <= float(radius) ** 2).to(torch.float32)
    return mask.unsqueeze(0)


def apply_circular_support_to_normalized_sos(
    sos_norm: torch.Tensor,
    mask: torch.Tensor,
    sos_min: float,
    sos_max: float,
    sos_water: float,
) -> torch.Tensor:
    """Force normalized SoS outside the circular support to normalized water."""
    water_norm = (float(sos_water) - float(sos_min)) / (float(sos_max) - float(sos_min))
    water_norm = float(max(0.0, min(1.0, water_norm)))
    m = mask.to(device=sos_norm.device, dtype=sos_norm.dtype)
    while m.ndim < sos_norm.ndim:
        m = m.unsqueeze(0)
    return sos_norm * m + water_norm * (1.0 - m)


def _prepare_sos_orientation_variants(raw_data: dict | None) -> dict[str, torch.Tensor]:
    """GT orientation diagnostics are disabled by design.

    The experimental ground truth is kept in the single, code-resolved
    MATLAB/Ali display convention stored under key ``sos``.  Evaluation must
    never choose an alternate GT orientation by minimizing MSE, because that
    would make the metric depend on the prediction rather than on the known
    coordinate convention.
    """
    return {}

def _normalize_sos_mps_to_01(sos_mps: torch.Tensor, sos_min: float, sos_max: float) -> torch.Tensor:
    return (sos_mps.float() - float(sos_min)) / (float(sos_max) - float(sos_min) + 1e-12)


def _log_orientation_mse_table(orientation_sums: dict[str, float], orientation_counts: dict[str, int]):
    # Intentionally disabled.  See _prepare_sos_orientation_variants().
    return

def unpack_reconstruction_batch(batch):
    """Return sos, tof, tof_mask, pde, colloc from all supported dataset return formats."""
    sos_b = batch[0]
    tof_b = batch[1]
    tof_mask_b = None
    pde_b = None
    colloc = None
    for item in batch[2:]:
        if isinstance(item, dict):
            colloc = item
        elif torch.is_tensor(item) and item.shape == tof_b.shape:
            tof_mask_b = item
        else:
            pde_b = item
    return sos_b, tof_b, tof_mask_b, pde_b, colloc


def normalize_physical_tof_like_dataset(tof_phys: torch.Tensor, tof_norm: dict | None) -> torch.Tensor:
    tn = tof_norm or {"type": "none"}
    kind = str(tn.get("type", "none")).lower()
    eps = float(tn.get("eps", 1e-6))
    if kind == "zscore":
        return (tof_phys - float(tn.get("mean", 0.0))) / (float(tn.get("std", 1.0)) + eps)
    if kind == "max":
        return tof_phys / (float(tn.get("max", 1.0)) + eps)
    return tof_phys


def make_homogeneous_water_tof_norm(nx, ny, phys_x, phys_y, radius, n_emitters, n_receivers, sos_water, tof_norm, device):
    dx = float(phys_x) / max(1, int(nx) - 1)
    dy = float(phys_y) / max(1, int(ny) - 1)
    emitters, receivers = generate_sensor_positions(
        int(nx), int(ny), dx, dy, float(radius), int(n_emitters), int(n_receivers)
    )
    e_xy = torch.tensor(np.asarray(emitters, dtype=np.float32), device=device)
    r_xy = torch.tensor(np.asarray(receivers, dtype=np.float32), device=device)
    e_xy = torch.stack([e_xy[:, 0] * dx, e_xy[:, 1] * dy], dim=1)
    r_xy = torch.stack([r_xy[:, 0] * dx, r_xy[:, 1] * dy], dim=1)
    dist = torch.linalg.norm(e_xy[:, None, :] - r_xy[None, :, :], dim=-1)
    return normalize_physical_tof_like_dataset(dist / float(sos_water), tof_norm).float()


def masked_double_center(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        row_mean = x.mean(dim=2, keepdim=True)
        col_mean = x.mean(dim=1, keepdim=True)
        global_mean = x.mean(dim=(1, 2), keepdim=True)
        return x - row_mean - col_mean + global_mean
    m = mask.to(device=x.device, dtype=x.dtype)
    x_m = x * m
    row_mean = x_m.sum(dim=2, keepdim=True) / m.sum(dim=2, keepdim=True).clamp_min(1.0)
    col_mean = x_m.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
    global_mean = x_m.sum(dim=(1, 2), keepdim=True) / m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return (x - row_mean - col_mean + global_mean) * m


def make_model_input(
    tof_b: torch.Tensor,
    tof_mask_b: torch.Tensor | None,
    use_tof_mask_channel: bool,
    tof_feature_mode: str = "raw",
    water_tof_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build model input [B,C,Ne,Nr] with optional residual ToF features and mask."""
    tof = tof_b.float()
    mask = tof_mask_b.to(device=tof.device, dtype=torch.float32) if tof_mask_b is not None else None
    mode = str(tof_feature_mode or "raw").lower()
    if mode == "residual_stack":
        if water_tof_norm is None:
            raise ValueError("[evaluation] residual_stack requires a homogeneous-water ToF reference.")
        water = water_tof_norm.to(device=tof.device, dtype=tof.dtype).unsqueeze(0)
        if mask is not None:
            # Invalid/masked channels are not measurements.  Present them as the
            # homogeneous-water reference in the raw acoustic channel, while the
            # residual channels and the explicit mask still indicate that they
            # should not contribute as measured data.
            raw = tof * mask + water * (1.0 - mask)
            residual = (tof - water) * mask
        else:
            raw = tof
            residual = tof - water
        residual_centered = masked_double_center(residual, mask)
        x = torch.stack([raw, residual, residual_centered], dim=1)
    else:
        x = tof.unsqueeze(1)

    if not bool(use_tof_mask_channel):
        return x
    if tof_mask_b is None:
        mask = torch.ones_like(tof_b, dtype=torch.float32, device=tof_b.device)
    else:
        mask = tof_mask_b.to(device=tof_b.device, dtype=torch.float32)
    return torch.cat([x, mask.unsqueeze(1)], dim=1)


class _OperatorResidualModel(nn.Module):
    def __init__(self, baseline_model: nn.Module | None, operator_model: nn.Module, nx: int, ny: int, residual_limit: float, device: torch.device, uses_baseline_residual: bool):
        super().__init__()
        self.baseline_model = baseline_model
        self.operator_model = operator_model
        self.nx = int(nx)
        self.ny = int(ny)
        self.residual_limit = float(residual_limit)
        self.uses_baseline_residual = bool(uses_baseline_residual)

        xs = torch.linspace(-1.0, 1.0, steps=self.ny, device=device, dtype=torch.float32)
        ys = torch.linspace(-1.0, 1.0, steps=self.nx, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([xx, yy], dim=-1).reshape(self.nx * self.ny, 2)
        self.register_buffer("xy_full", xy)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            tof = x[:, 0, :, :]
        elif x.dim() == 3:
            tof = x
        else:
            raise ValueError(f"Unexpected ToF tensor shape for operator model: {tuple(x.shape)}")

        op01 = self.operator_model(tof, self.xy_full)
        if not self.uses_baseline_residual:
            return op01.reshape(tof.shape[0], 1, self.nx, self.ny)

        if self.baseline_model is None:
            raise ValueError("Operator checkpoint requires baseline_model when uses_baseline_residual=True")

        base = self.baseline_model(tof.unsqueeze(1)).squeeze(1)
        residual = self.residual_limit * (2.0 * op01 - 1.0)
        composed = torch.clamp(base.reshape(base.shape[0], -1) + residual, 0.0, 1.0)
        return composed.reshape(base.shape[0], 1, self.nx, self.ny)


class _EdgeResidualRefinerModel(nn.Module):
    def __init__(self, base_model, refiner_model, support_mask, water_norm, lowpass_kernel):
        super().__init__()
        self.base_model = base_model
        self.refiner_model = refiner_model
        self.water_norm = float(water_norm)
        self.lowpass_kernel = int(lowpass_kernel)
        if support_mask is not None:
            self.register_buffer("support_mask", support_mask.float())
        else:
            self.support_mask = None

    def forward(self, x):
        from edge_residual_refiner import make_refiner_features

        base = self.base_model(x)
        feat = make_refiner_features(x, base, self.support_mask, self.lowpass_kernel)
        return self.refiner_model(feat, base, self.support_mask, self.water_norm)


class _PiecewiseSoSRefinerModel(nn.Module):
    def __init__(self, base_model, refiner_model, support_mask, water_norm, lowpass_kernel):
        super().__init__()
        self.base_model = base_model
        self.refiner_model = refiner_model
        self.water_norm = float(water_norm)
        self.lowpass_kernel = int(lowpass_kernel)
        if support_mask is not None:
            self.register_buffer("support_mask", support_mask.float())
        else:
            self.support_mask = None

    def forward(self, x):
        from piecewise_sos_refiner import make_piecewise_features

        base = self.base_model(x)
        feat = make_piecewise_features(x, base, self.support_mask, self.lowpass_kernel)
        return self.refiner_model(feat, self.support_mask, self.water_norm)


class _AnchoredPiecewiseRefinerModel(nn.Module):
    def __init__(self, base_model, refiner_model, support_mask, water_norm, lowpass_kernel, base_quant_temperature):
        super().__init__()
        self.base_model = base_model
        self.refiner_model = refiner_model
        self.water_norm = float(water_norm)
        self.lowpass_kernel = int(lowpass_kernel)
        self.base_quant_temperature = float(base_quant_temperature)
        if support_mask is not None:
            self.register_buffer("support_mask", support_mask.float())
        else:
            self.support_mask = None

    def forward(self, x):
        from anchored_piecewise_refiner import make_anchored_piecewise_features

        base = self.base_model(x)
        feat = make_anchored_piecewise_features(x, base, self.support_mask, self.lowpass_kernel)
        return self.refiner_model(feat, base, self.support_mask, self.water_norm, self.base_quant_temperature)


def run_evaluation(model, test_loader, device, phys_x, phys_y, epochs_label, sos_min, sos_max,
                   use_circular_mask=False, mask_radius=None, sos_water=1500.0,
                   use_tof_mask_channel=False, sos_orientation_variants=None,
                   tof_feature_mode="raw", water_tof_norm=None):
    model.eval()
    test_loss, criterion, samples_plotted = 0.0, torch.nn.MSELoss(), 0
    circular_mask = None
    sos_orientation_variants = sos_orientation_variants or {}
    orientation_sums = {name: 0.0 for name in sos_orientation_variants.keys()}
    orientation_counts = {name: 0 for name in sos_orientation_variants.keys()}
    sample_offset = 0

    if bool(sos_orientation_variants):
        log_message("[evaluation] GT orientation diagnostic enabled for variants: " + ", ".join(sorted(sos_orientation_variants.keys())))

    if bool(use_circular_mask):
        radius_eff = float(mask_radius) if mask_radius is not None else min(float(phys_x), float(phys_y)) / 2.0
        log_message(
            f"[evaluation] Circular support mask enabled. radius={radius_eff:.6g} m, "
            f"outside set to sos_water={float(sos_water):.6g} m/s"
        )

    inference_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            sos_t, tof_t, tof_mask_t, _pde_t, _colloc = unpack_reconstruction_batch(batch)

            sos_dev = sos_t.to(device)
            tof_dev = tof_t.to(device).float()
            mask_dev = tof_mask_t.to(device).float() if tof_mask_t is not None else None
            out = model(make_model_input(tof_dev, mask_dev, use_tof_mask_channel, tof_feature_mode, water_tof_norm))
            out_view = out.view_as(sos_dev)

            if bool(use_circular_mask):
                if circular_mask is None:
                    radius_eff = float(mask_radius) if mask_radius is not None else min(float(phys_x), float(phys_y)) / 2.0
                    circular_mask = make_circular_support_mask(
                        int(sos_dev.shape[-2]), int(sos_dev.shape[-1]),
                        float(phys_x), float(phys_y), radius_eff, device=device,
                    )
                out_view = apply_circular_support_to_normalized_sos(
                    out_view, circular_mask, sos_min, sos_max, sos_water
                )
                sos_dev = apply_circular_support_to_normalized_sos(
                    sos_dev, circular_mask, sos_min, sos_max, sos_water
                )

            test_loss += float(criterion(out_view, sos_dev).item())

            if bool(sos_orientation_variants):
                bsz = int(out_view.shape[0])
                for name, variant_all in sos_orientation_variants.items():
                    if sample_offset >= int(variant_all.shape[0]):
                        continue
                    variant_slice = variant_all[sample_offset:sample_offset + bsz].to(device=device, dtype=torch.float32)
                    if variant_slice.shape[0] != bsz:
                        continue
                    variant_norm = _normalize_sos_mps_to_01(variant_slice, sos_min, sos_max)
                    if variant_norm.shape != out_view.shape:
                        # Expected shape is (B,nx,ny) or (B,1,nx,ny) matching out_view.
                        if variant_norm.ndim == 3 and out_view.ndim == 4 and out_view.shape[1] == 1:
                            variant_norm = variant_norm.unsqueeze(1)
                        elif variant_norm.ndim == 4 and out_view.ndim == 3 and variant_norm.shape[1] == 1:
                            variant_norm = variant_norm[:, 0]
                    if variant_norm.shape != out_view.shape:
                        log_message(f"[evaluation] Skipping orientation variant {name}: shape {tuple(variant_norm.shape)} does not match prediction {tuple(out_view.shape)}")
                        continue
                    if bool(use_circular_mask) and circular_mask is not None:
                        variant_norm = apply_circular_support_to_normalized_sos(
                            variant_norm, circular_mask, sos_min, sos_max, sos_water
                        )
                    orientation_sums[name] += float(criterion(out_view, variant_norm).item())
                    orientation_counts[name] += 1
                sample_offset += bsz

            if samples_plotted < 3:
                for b in range(sos_t.shape[0]):
                    if samples_plotted < 3:
                        plot_results(
                            sos_dev[b].detach().cpu().numpy().squeeze(),
                            out_view[b].detach().cpu().numpy().squeeze(),
                            tof_t[b].detach().cpu().numpy(),
                            epochs_label,
                            phys_x, phys_y,
                            sos_min, sos_max,
                            title_suffix=f"(Test Sample {samples_plotted+1})",
                            tof_norm=getattr(test_loader.dataset, "dataset", test_loader.dataset).tof_norm
                            if hasattr(getattr(test_loader.dataset, "dataset", test_loader.dataset), "tof_norm")
                            else None,
                            tof_mask=tof_mask_t[b].detach().cpu().numpy() if tof_mask_t is not None else None,
                            water_tof_norm=water_tof_norm.detach().cpu().numpy() if water_tof_norm is not None else None,
                        )
                        samples_plotted += 1

    inference_duration = time.time() - inference_start
    log_message(f"[evaluation] Inference on {len(test_loader.dataset)} samples completed in {inference_duration:.4f}s")
    log_message(f"[evaluation] Final Test MSE: {test_loss/max(1,len(test_loader)):.6f}")
    _log_orientation_mse_table(orientation_sums, orientation_counts)


if __name__ == "__main__":
    # Import both reconstruction architectures.  The default ReconstructionNet
    # corresponds to the baseline model, while ImprovedReconstructionNet adds
    # a learned gating module for fusing ToF and geometry information.  See
    # model.py for details.
    from model import ReconstructionNet, ImprovedReconstructionNet, DeepGatedReconstructionNet
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Data/reconstruction_model.pth")
    p.add_argument("--data_path", type=str, default="Data/ultrasound_data.pt")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for logger artifacts (terminal/HTML logs, figures)")
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--splits_path", type=str, default=None,
                   help="Splits path. Default: <data_path_stem>_splits.pt")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--base_model_path", type=str, default=None,
                   help="Optional override for the baseline model used by operator checkpoints.")
    p.add_argument("--baseline_latent_res", type=int, default=None,
                   help="Optional override for the baseline latent resolution used by operator checkpoints.")
    p.add_argument("--model_fc_dropout", type=float, default=None,
                   help="Optional override for DeepGated fc dropout. Default is read from checkpoint train_config.")
    p.add_argument("--model_decoder_dropout", type=float, default=None,
                   help="Optional override for DeepGated decoder dropout. Default is read from checkpoint train_config.")
    p.add_argument("--op_residual_limit", type=float, default=None,
                   help="Optional override for the maximum absolute normalized residual of operator checkpoints.")
    p.add_argument("--use_circular_mask", action="store_true",
                   help="Apply a circular support mask to prediction and target before metrics/plots.")
    p.add_argument("--mask_radius", type=float, default=None,
                   help="Circular support radius in meters. Default: dataset metadata radius.")
    p.add_argument("--use_tof_mask_channel", action="store_true",
                   help="Use a second model input channel containing the ToF validity mask. If omitted, inferred from checkpoint train_config when available.")
    p.add_argument("--tof_feature_mode", choices=["raw", "residual_stack"], default=None,
                   help="ToF feature representation. If omitted, inferred from checkpoint train_config when available.")

    # Select which reconstruction model architecture to use when loading a
    # standard (non‑operator) checkpoint.  This must match the architecture
    # used during training; if unspecified the script attempts to infer it
    # from the checkpoint's saved training configuration.  Choices are
    # 'baseline' (ReconstructionNet) and 'improved' (ImprovedReconstructionNet).
    p.add_argument(
        "--model_type",
        type=str,
        choices=["baseline", "improved", "deep"],
        default=None,
        help=(
            "Model architecture used by the checkpoint.\n"
            "If not provided, the script uses the value stored in the checkpoint's train_config if available."
        ),
    )

    args = p.parse_args()

    if args.output_dir is not None:
        set_output_folder(args.output_dir)

    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message("...................")

    load_start = time.time()
    device = torch.device(args.device)

    if not os.path.exists(args.data_path):
        log_message(f"[evaluation] ERROR: Data path {args.data_path} not found.")
        sys.exit(1)
    m = load_pairs_cache_metadata(
        args.data_path
    )

    if not isinstance(m, dict) or not m:
        log_message(
            "[evaluation] ERROR: "
            "No metadata found in data file."
        )
        sys.exit(1)

    # The function currently returns an empty dictionary by design.
    sos_orientation_variants = (
        _prepare_sos_orientation_variants(
            None
        )
    )
    log_message(f"[evaluation] Dataset loaded in {time.time() - load_start:.2f}s")
    log_message("[evaluation] RETRIEVED DATASET METADATA:\n" + pprint.pformat(m))

    phys_x, phys_y = m['phys_x'], m['phys_y']
    sos_min, sos_max = m['sos_min'], m['sos_max']
    nx, ny = m['nx'], m['ny']
    radius = m.get('radius', 0.04)
    n_emitters = int(m.get('n_emitters', 32))
    n_receivers = int(m.get('n_receivers', 32))
    sos_water = float(m.get("sos_water", 1500.0))

    # Load checkpoint before constructing the dataset so that mask-aware checkpoints
    # can request return_tof_mask=True from PairsCacheDataset.
    model_load_start = time.time()
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    train_cfg_early = checkpoint.get("train_config", {}) if isinstance(checkpoint, dict) else {}
    use_tof_mask_channel = bool(args.use_tof_mask_channel or train_cfg_early.get("use_tof_mask_channel", False))
    tof_feature_mode = str(args.tof_feature_mode or train_cfg_early.get("tof_feature_mode", "raw")).lower()
    acoustic_tof_channels = 3 if tof_feature_mode == "residual_stack" else 1
    tof_input_channels = acoustic_tof_channels + (1 if use_tof_mask_channel else 0)
    water_tof_norm = None
    if use_tof_mask_channel:
        log_message("[evaluation] ToF validity mask enabled for inference (gating/prior normalization only).")

    pairs_ds = PairsCacheDataset(args.data_path, sos_min=sos_min, sos_max=sos_max, return_tof_mask=use_tof_mask_channel)
    if tof_feature_mode == "residual_stack":
        water_tof_norm = make_homogeneous_water_tof_norm(
            nx, ny, phys_x, phys_y, radius, n_emitters, n_receivers,
            sos_water, getattr(pairs_ds, "tof_norm", None), device
        )
        log_message(
            "[evaluation] ToF feature mode: residual_stack "
            f"(model input channels={tof_input_channels})."
        )
    else:
        log_message(f"[evaluation] ToF feature mode: raw (model input channels={tof_input_channels}).")

    if args.splits_path is None:
        args.splits_path = make_cache_paths(args.data_path)["splits_path"]
    if os.path.exists(args.splits_path):
        splits = load_splits(args.splits_path)
        idx = splits[f"{args.split}_idx"]
        eval_ds = Subset(pairs_ds, list(map(int, idx)))
        log_message(f"[evaluation] Using saved splits: {args.splits_path}  split={args.split}  n={len(eval_ds)}")
    else:
        eval_ds = pairs_ds
        log_message(f"[evaluation] WARNING: splits file not found: {args.splits_path}. Evaluating on full dataset.")

    epochs_trained = "Unknown"

    if isinstance(checkpoint, dict) and "anchored_piecewise_refiner_state_dict" in checkpoint:
        from inverse_model_io import load_inverse_model
        from anchored_piecewise_refiner import AnchoredPiecewisePhysicsRefiner

        train_cfg = checkpoint.get("train_config", {}) or {}
        base_model_path = args.base_model_path or checkpoint.get("base_model_path", None)
        if base_model_path is None:
            log_message("[evaluation] ERROR: Anchored-piecewise checkpoint requires base_model_path.")
            sys.exit(1)

        base_model_type = str(train_cfg.get("model_type", args.model_type or "deep"))
        base_latent_res = int(
            args.baseline_latent_res
            if args.baseline_latent_res is not None
            else train_cfg.get("latent_res", 36)
        )
        base_model, _base_cfg = load_inverse_model(
            base_model_path, base_model_type, nx, ny, base_latent_res,
            phys_x, phys_y, radius, n_emitters, n_receivers,
            tof_input_channels, device,
        )

        support_mask = None
        refiner_use_circular_mask = bool(args.use_circular_mask or train_cfg.get("use_circular_mask", False))
        refiner_mask_radius = float(
            args.mask_radius
            if args.mask_radius is not None
            else train_cfg.get("mask_radius", radius)
        )
        if refiner_use_circular_mask:
            support_mask = make_circular_support_mask(
                int(nx), int(ny), float(phys_x), float(phys_y),
                refiner_mask_radius,
                device=device,
            )
        water_norm = (float(sos_water) - float(sos_min)) / (float(sos_max) - float(sos_min) + 1e-12)
        water_norm = float(max(0.0, min(1.0, water_norm)))
        lowpass_kernel = int(train_cfg.get("lowpass_kernel", 15))
        base_quant_temperature = float(train_cfg.get("base_quant_temperature", 0.035))
        bin_centers_norm = checkpoint.get("bin_centers_norm", None)
        if bin_centers_norm is None:
            raw_centers = train_cfg.get("bin_centers_norm_resolved", None)
            if raw_centers is None:
                raise KeyError("Anchored-piecewise checkpoint is missing bin_centers_norm.")
            bin_centers_norm = torch.as_tensor(raw_centers, dtype=torch.float32)
        bin_centers_norm = torch.as_tensor(bin_centers_norm, dtype=torch.float32)
        in_channels = int(tof_input_channels) + 5
        refiner = AnchoredPiecewisePhysicsRefiner(
            in_channels=in_channels,
            bin_centers_norm=bin_centers_norm,
            width=int(train_cfg.get("width", 48)),
            residual_limit=float(train_cfg.get("residual_limit", 0.035)),
            gate_limit=float(train_cfg.get("gate_limit", 0.75)),
            dropout=float(train_cfg.get("dropout", 0.0)),
        ).to(device)
        refiner.load_state_dict(checkpoint["anchored_piecewise_refiner_state_dict"])
        refiner.eval()
        model = _AnchoredPiecewiseRefinerModel(
            base_model, refiner, support_mask, water_norm, lowpass_kernel, base_quant_temperature
        ).to(device)
        epochs_trained = checkpoint.get("epoch", "Unknown")
        log_message(f"[evaluation] Loaded anchored_piecewise_physics_refiner checkpoint in {time.time() - model_load_start:.2f}s.")
        log_message(f"[evaluation] Anchored piecewise base model path: {base_model_path}")
        log_message(f"[evaluation] Anchored piecewise residual_limit: {float(train_cfg.get('residual_limit', 0.035)):.6g}")
        log_message(f"[evaluation] Anchored piecewise gate_limit: {float(train_cfg.get('gate_limit', 0.75)):.6g}")
        log_message(f"[evaluation] Anchored piecewise bins: {[float(x) for x in bin_centers_norm.cpu().view(-1)]}")
        hist = checkpoint.get("history", {}) or {}
        if isinstance(hist, dict) and hist.get("val_pred_mse"):
            legacy_ckpt = {
                "training_losses": hist.get("train_mse", []),
                "validation_losses": hist.get("val_pred_mse", []),
            }
            _plot_checkpoint_learning_curves(legacy_ckpt)
    elif isinstance(checkpoint, dict) and "piecewise_refiner_state_dict" in checkpoint:
        from inverse_model_io import load_inverse_model
        from piecewise_sos_refiner import PiecewiseSoSRefiner

        train_cfg = checkpoint.get("train_config", {}) or {}
        base_model_path = args.base_model_path or checkpoint.get("base_model_path", None)
        if base_model_path is None:
            log_message("[evaluation] ERROR: Piecewise-refiner checkpoint requires base_model_path.")
            sys.exit(1)

        base_model_type = str(train_cfg.get("model_type", args.model_type or "deep"))
        base_latent_res = int(
            args.baseline_latent_res
            if args.baseline_latent_res is not None
            else train_cfg.get("latent_res", 36)
        )
        base_model, _base_cfg = load_inverse_model(
            base_model_path, base_model_type, nx, ny, base_latent_res,
            phys_x, phys_y, radius, n_emitters, n_receivers,
            tof_input_channels, device,
        )

        support_mask = None
        refiner_use_circular_mask = bool(args.use_circular_mask or train_cfg.get("use_circular_mask", False))
        refiner_mask_radius = float(
            args.mask_radius
            if args.mask_radius is not None
            else train_cfg.get("mask_radius", radius)
        )
        if refiner_use_circular_mask:
            support_mask = make_circular_support_mask(
                int(nx), int(ny), float(phys_x), float(phys_y),
                refiner_mask_radius,
                device=device,
            )
        water_norm = (float(sos_water) - float(sos_min)) / (float(sos_max) - float(sos_min) + 1e-12)
        water_norm = float(max(0.0, min(1.0, water_norm)))
        lowpass_kernel = int(train_cfg.get("lowpass_kernel", 13))
        bin_centers_norm = checkpoint.get("bin_centers_norm", None)
        if bin_centers_norm is None:
            bin_centers_norm = train_cfg.get("bin_centers_norm_resolved", None)
        if bin_centers_norm is None:
            n_bins = int(train_cfg.get("n_bins", 14))
            bin_centers_norm = torch.linspace(0.0, 1.0, n_bins)
        bin_centers_norm = torch.as_tensor(bin_centers_norm, dtype=torch.float32)
        in_channels = int(tof_input_channels) + 4
        refiner = PiecewiseSoSRefiner(
            in_channels=in_channels,
            bin_centers_norm=bin_centers_norm,
            width=int(train_cfg.get("width", 56)),
            residual_limit=float(train_cfg.get("residual_limit", 0.05)),
            dropout=float(train_cfg.get("dropout", 0.0)),
        ).to(device)
        refiner.load_state_dict(checkpoint["piecewise_refiner_state_dict"])
        refiner.eval()
        model = _PiecewiseSoSRefinerModel(base_model, refiner, support_mask, water_norm, lowpass_kernel).to(device)
        epochs_trained = checkpoint.get("epoch", "Unknown")
        log_message(f"[evaluation] Loaded piecewise_sos_refiner checkpoint in {time.time() - model_load_start:.2f}s.")
        log_message(f"[evaluation] Piecewise refiner base model path: {base_model_path}")
        log_message(f"[evaluation] Piecewise refiner residual_limit: {float(train_cfg.get('residual_limit', 0.05)):.6g}")
        log_message(f"[evaluation] Piecewise refiner bins: {[float(x) for x in bin_centers_norm.cpu().view(-1)]}")
        hist = checkpoint.get("history", {}) or {}
        if isinstance(hist, dict) and hist.get("val_pred_mse"):
            legacy_ckpt = {
                "train_history": list(map(float, hist.get("train_mse", []))),
                "val_history": list(map(float, hist.get("val_pred_mse", []))),
            }
            _plot_checkpoint_learning_curves(legacy_ckpt)
    elif isinstance(checkpoint, dict) and "edge_refiner_state_dict" in checkpoint:
        from edge_residual_refiner import BoundaryEdgeResidualRefiner, EdgeResidualRefiner
        from inverse_model_io import load_inverse_model

        train_cfg = checkpoint.get("train_config", {}) or {}
        ckpt_kind = str(checkpoint.get("kind", "edge_residual_refiner"))
        base_model_path = args.base_model_path or checkpoint.get("base_model_path", None)
        if base_model_path is None:
            log_message("[evaluation] ERROR: Edge-refiner checkpoint requires base_model_path.")
            sys.exit(1)

        base_model_type = str(train_cfg.get("model_type", args.model_type or "deep"))
        base_latent_res = int(
            args.baseline_latent_res
            if args.baseline_latent_res is not None
            else train_cfg.get("latent_res", 36)
        )
        base_model, _base_cfg = load_inverse_model(
            base_model_path, base_model_type, nx, ny, base_latent_res,
            phys_x, phys_y, radius, n_emitters, n_receivers,
            tof_input_channels, device,
        )

        support_mask = None
        refiner_use_circular_mask = bool(args.use_circular_mask or train_cfg.get("use_circular_mask", False))
        refiner_mask_radius = float(
            args.mask_radius
            if args.mask_radius is not None
            else train_cfg.get("mask_radius", radius)
        )
        if refiner_use_circular_mask:
            support_mask = make_circular_support_mask(
                int(nx), int(ny), float(phys_x), float(phys_y),
                refiner_mask_radius,
                device=device,
            )
        water_norm = (float(sos_water) - float(sos_min)) / (float(sos_max) - float(sos_min) + 1e-12)
        lowpass_kernel = int(train_cfg.get("lowpass_kernel", 17))
        in_channels = int(tof_input_channels) + 4
        refiner_cls = BoundaryEdgeResidualRefiner if ckpt_kind == "boundary_edge_residual_refiner" else EdgeResidualRefiner
        refiner = refiner_cls(
            in_channels=in_channels,
            width=int(train_cfg.get("width", 48)),
            residual_limit=float(train_cfg.get("residual_limit", 0.16)),
            lowpass_kernel=lowpass_kernel,
            dropout=float(train_cfg.get("dropout", 0.0)),
        ).to(device)
        refiner.load_state_dict(checkpoint["edge_refiner_state_dict"])
        refiner.eval()
        model = _EdgeResidualRefinerModel(base_model, refiner, support_mask, water_norm, lowpass_kernel).to(device)
        epochs_trained = checkpoint.get("epoch", "Unknown")
        log_message(f"[evaluation] Loaded {ckpt_kind} checkpoint in {time.time() - model_load_start:.2f}s.")
        log_message(f"[evaluation] Edge refiner base model path: {base_model_path}")
        log_message(f"[evaluation] Edge refiner residual_limit: {float(train_cfg.get('residual_limit', 0.16)):.6g}")
        hist = checkpoint.get("history", {}) or {}
        if isinstance(hist, dict) and hist.get("val_pred_mse"):
            legacy_ckpt = {
                "train_history": list(map(float, hist.get("train_mse", []))),
                "val_history": list(map(float, hist.get("val_pred_mse", []))),
            }
            _plot_checkpoint_learning_curves(legacy_ckpt)
    elif isinstance(checkpoint, dict) and "operator_state_dict" in checkpoint:
        from operator_deeponet import DeepONetConfig, DeepONetSoS

        dcfg_dict = checkpoint.get("deeponet_config", {})
        dcfg = DeepONetConfig(
            n_emitters=int(dcfg_dict.get("n_emitters", n_emitters)),
            n_receivers=int(dcfg_dict.get("n_receivers", n_receivers)),
            latent=int(dcfg_dict.get("latent", 128)),
            width=int(dcfg_dict.get("width", 256)),
            depth=int(dcfg_dict.get("depth", 3)),
            phys_x=float(dcfg_dict.get("phys_x", phys_x)),
            phys_y=float(dcfg_dict.get("phys_y", phys_y)),
        )
        operator_model = DeepONetSoS(dcfg).to(device)
        operator_model.load_state_dict(checkpoint["operator_state_dict"])
        operator_model.eval()

        uses_baseline_residual = bool(checkpoint.get("uses_baseline_residual", True))
        baseline_model = None
        base_model_path = args.base_model_path or checkpoint.get("base_model_path", None)
        if uses_baseline_residual:
            if base_model_path is None:
                log_message("[evaluation] ERROR: Operator checkpoint requires a baseline model path.")
                sys.exit(1)
            baseline_latent_res = int(
                args.baseline_latent_res
                if args.baseline_latent_res is not None
                else checkpoint.get("baseline_latent_res", 24)
            )
            baseline_model = ReconstructionNet(
                nx=nx, ny=ny, latent_res=baseline_latent_res,
                phys_x=phys_x, phys_y=phys_y, radius=radius,
                n_emitters=n_emitters, n_receivers=n_receivers,
                tof_input_channels=tof_input_channels,
            ).to(device)
            base_ckpt = torch.load(base_model_path, map_location=device, weights_only=False)
            if isinstance(base_ckpt, dict) and "model_state_dict" in base_ckpt:
                baseline_model.load_state_dict(base_ckpt["model_state_dict"])
                epochs_trained = base_ckpt.get("epochs", "Unknown")
            else:
                baseline_model.load_state_dict(base_ckpt)
            baseline_model.eval()

        residual_limit = float(
            args.op_residual_limit
            if args.op_residual_limit is not None
            else checkpoint.get("residual_limit", 0.15)
        )
        model = _OperatorResidualModel(
            baseline_model=baseline_model,
            operator_model=operator_model,
            nx=nx,
            ny=ny,
            residual_limit=residual_limit,
            device=device,
            uses_baseline_residual=uses_baseline_residual,
        ).to(device)

        history = checkpoint.get("history", [])
        if history:
            # Operator-model legacy checkpoints use a different history schema.
            # Plot it only as legacy history, not as a comparable supervised curve.
            legacy_ckpt = {
                'train_history': [float(row.get('train_loss', row.get('loss', 0.0))) for row in history],
                'val_history': [float(row.get('val_loss', np.nan)) for row in history],
            }
            _plot_checkpoint_learning_curves(legacy_ckpt)
        log_message(f"[evaluation] Loaded operator checkpoint in {time.time() - model_load_start:.2f}s.")
        log_message(f"[evaluation] Operator baseline path: {base_model_path}")
        log_message(f"[evaluation] Operator uses_baseline_residual: {uses_baseline_residual}")
        log_message(f"[evaluation] Operator residual_limit: {residual_limit}")
    else:
        # Determine latent resolution and model architecture from checkpoint contract or CLI.
        latent_res = 24
        model_type = args.model_type
        ctor_cfg = {}
        train_cfg = {}
        if isinstance(checkpoint, dict):
            train_cfg = checkpoint.get("train_config", {}) or {}
            ctor_cfg = checkpoint.get("model_ctor_config", {}) or {}
            if isinstance(ctor_cfg, dict) and ctor_cfg:
                log_message("[evaluation] Using checkpoint model_ctor_config to instantiate the network.")
            else:
                log_message("[evaluation] WARNING: checkpoint has no model_ctor_config; falling back to train_config/defaults.")
            if isinstance(ctor_cfg, dict) and ctor_cfg.get("latent_res", None) is not None:
                latent_res = int(ctor_cfg.get("latent_res", latent_res))
            elif "latent_res" in train_cfg:
                latent_res = int(train_cfg.get("latent_res", latent_res))
            if model_type is None:
                if isinstance(ctor_cfg, dict) and ctor_cfg.get("model_type", None) is not None:
                    model_type = str(ctor_cfg.get("model_type"))
                else:
                    model_type = str(train_cfg.get("model_type", "baseline"))
        # fall back to baseline if still None
        if model_type is None:
            model_type = "baseline"
        # instantiate the requested reconstruction architecture
        if model_type == "improved":
            model = ImprovedReconstructionNet(
                nx=nx, ny=ny, latent_res=latent_res,
                phys_x=phys_x, phys_y=phys_y, radius=radius,
                n_emitters=n_emitters, n_receivers=n_receivers,
                tof_input_channels=tof_input_channels,
            ).to(device)
            log_message(f"[evaluation] Instantiated ImprovedReconstructionNet (latent_res={latent_res}).")
        elif model_type == "deep":
            cfg_for_ctor = ctor_cfg if isinstance(ctor_cfg, dict) and ctor_cfg else train_cfg
            ckpt_fc_dropout = float(_checkpoint_config_value(cfg_for_ctor, "model_fc_dropout", 0.08))
            ckpt_decoder_dropout = float(_checkpoint_config_value(cfg_for_ctor, "model_decoder_dropout", 0.04))
            fc_dropout = ckpt_fc_dropout if args.model_fc_dropout is None else float(args.model_fc_dropout)
            decoder_dropout = ckpt_decoder_dropout if args.model_decoder_dropout is None else float(args.model_decoder_dropout)
            _log_if_cli_differs_from_checkpoint("model_fc_dropout", args.model_fc_dropout, ckpt_fc_dropout)
            _log_if_cli_differs_from_checkpoint("model_decoder_dropout", args.model_decoder_dropout, ckpt_decoder_dropout)
            model = DeepGatedReconstructionNet(
                nx=nx, ny=ny, latent_res=latent_res,
                phys_x=phys_x, phys_y=phys_y, radius=radius,
                n_emitters=n_emitters, n_receivers=n_receivers,
                tof_input_channels=tof_input_channels,
                fc_dropout=fc_dropout,
                decoder_dropout=decoder_dropout,
            ).to(device)
            log_message(
                f"[evaluation] Instantiated DeepGatedReconstructionNet "
                f"(latent_res={latent_res}, fc_dropout={fc_dropout}, decoder_dropout={decoder_dropout})."
            )
        else:
            model = ReconstructionNet(
                nx=nx, ny=ny, latent_res=latent_res,
                phys_x=phys_x, phys_y=phys_y, radius=radius,
                n_emitters=n_emitters, n_receivers=n_receivers,
                tof_input_channels=tof_input_channels,
            ).to(device)
            log_message(f"[evaluation] Instantiated baseline ReconstructionNet (latent_res={latent_res}).")
        # Load weights from the checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', None)
            if state_dict is None:
                # if not in dict, the checkpoint might itself be the state dict
                state_dict = checkpoint
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint/model architecture mismatch: missing={missing}, unexpected={unexpected}"
                )
            epochs_trained = checkpoint.get('epochs', "Unknown")
            log_message(f"[evaluation] Model loaded in {time.time() - model_load_start:.2f}s. Trained for {epochs_trained} epochs.")
            if isinstance(checkpoint, dict) and "model_ctor_config" in checkpoint:
                log_message(f"[evaluation] Checkpoint model_ctor_config: {pprint.pformat(checkpoint.get('model_ctor_config'))}")
            else:
                log_message("[evaluation] NOTE: this checkpoint predates model_ctor_config; exact constructor contract cannot be verified beyond train_config/state_dict.")
            _plot_checkpoint_learning_curves(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            log_message(f"[evaluation] Model loaded in {time.time() - model_load_start:.2f}s.")

    dx, dy = phys_x/(nx - 1), phys_y/(ny - 1)
    emitters, receivers = generate_sensor_positions(nx, ny, dx, dy, radius, n_emitters, n_receivers)
    # Plotting uses a CPU NumPy array even when evaluation runs on CUDA.
    plot_physics_setup(eval_ds[0][0].detach().cpu().numpy().squeeze()*(sos_max-sos_min)+sos_min, emitters, receivers, dx, dy, phys_x, phys_y, sos_min, sos_max)
    mask_radius_eff = float(args.mask_radius) if args.mask_radius is not None else float(radius)
    run_evaluation(
        model, DataLoader(eval_ds, batch_size=32), device, phys_x, phys_y,
        f"Trained {epochs_trained} Epochs", sos_min, sos_max,
        use_circular_mask=bool(args.use_circular_mask),
        mask_radius=mask_radius_eff,
        sos_water=sos_water,
        use_tof_mask_channel=use_tof_mask_channel,
        sos_orientation_variants=sos_orientation_variants,
        tof_feature_mode=tof_feature_mode,
        water_tof_norm=water_tof_norm,
    )










