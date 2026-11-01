# FILE: train_initial_reconstruction.py
"""
Pipeline supporting training, model saving, interruption handling, and resuming.

Preserves:
- SoS bank + saved splits (persistent)
- pairs cache at --data_path remains (enables --resume across epoch runs)
- logger behavior
- plot_learning_curves(...) and plot_physics_setup(...)

Adds:
- --latent_res CLI argument (no hardcoding)
- Passes latent_res into ReconstructionNet
- Enforces latent_res consistency on --resume / --mode test when checkpoint metadata exists
- --exclude_frac for limited-view masking during pairs-cache creation (if supported by dataset.py)
- Graceful Ctrl+C stop WITHOUT try/except (signal flag, stop after current epoch)

Fixes:
- ToF-consistency is no longer computed with a straight-line path matrix for
  Eikonal-generated data.  When requested, it is routed through the
  differentiable Eikonal surrogate trained from stored MSFM collocation targets.
- Validation uses the SAME objective components as training (SoS + ToF-consistency).
- Adds per-epoch logging of SoS/ToF components and forward-model terms (PDE/BC/supT).
- Adds --tof_ramp_epochs (linearly ramps lambda_tof after warmup).
"""

import config
from runtime_context import *  # project globals (kept)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # allow mixed OpenMP (unsafe but unblocks)

# Defaults for sensor geometry (used as argparse defaults).
SENSOR_RADIUS_DEFAULT = 0.04
DEFAULT_N_EMITTERS = 32
DEFAULT_N_RECEIVERS = 32

from settings import app_settings, set_output_folder
from logger import log_message, log_image

import functools
import torch
# Apply the global patch
torch.save = functools.partial(torch.save, pickle_protocol=4)

import argparse
import sys
import pprint
import time
import gc
import random
import numpy as np
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import signal
import copy

from dataset import (
    get_or_make_sos_bank,
    get_or_make_splits,
    build_pairs_cache_from_sos_bank,
    PairsCacheDataset,
    make_cache_paths,
    load_pairs_cache_metadata,
)
from model import ReconstructionNet, ImprovedReconstructionNet, DeepGatedReconstructionNet
from utils import plot_physics_setup, plot_learning_curves, plot_results
from reconstruction_utils import run_evaluation, make_circular_support_mask, apply_circular_support_to_normalized_sos
from anatomy import generate_sensor_positions
from eikonal_forward_model import EikonalForwardModel, ForwardLossWeights


def _release_transient_memory(reason: str = "") -> None:
    """Release unreachable CPU objects and unused CUDA allocator blocks.

    This never changes live model, optimizer, batch, or data tensors. CUDA's
    empty_cache only returns currently unused cached blocks to the allocator.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if reason:
        log_message(f"[train_initial_reconstruction.py] Released transient memory: {reason}")


def _format_seconds(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}min"
    return f"{minutes / 60.0:.2f}h"


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train", "test"], default='train')
    p.add_argument("--model_path", type=str, default="Data/reconstruction_model.pth")

    # pairs cache path (kept; required for resume)
    p.add_argument("--data_path", type=str, default="Data/ultrasound_data.pt")

    # logger output folder (kept)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for logger artifacts (terminal/HTML logs, figures)")

    p.add_argument("--resume", action="store_true")

    p.add_argument("--nx", type=int, default=NX_DEFAULT)
    p.add_argument("--ny", type=int, default=NY_DEFAULT)

    # NEW: user-tunable latent bottleneck resolution (no hardcoding in model.py needed)
    p.add_argument("--latent_res", type=int, default=24,
                   help="Latent bottleneck resolution for ReconstructionNet (speed/quality trade-off).")
    p.add_argument("--model_fc_dropout", type=float, default=0.08,
                   help="Dropout probability in the global ToF encoder for the deep model.")
    p.add_argument("--model_decoder_dropout", type=float, default=0.04,
                   help="Spatial dropout probability in the deep model decoder.")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay for the reconstruction model.")

    # Choose which reconstruction architecture to train.  The default
    # 'baseline' corresponds to the original geometry‑aware ReconstructionNet.
    # The alternative 'improved' selects the ImprovedReconstructionNet that
    # introduces a learned gating mechanism for fusing ToF and geometry
    # information (defined in model.py).  See the documentation of
    # ImprovedReconstructionNet for details.
    p.add_argument(
        "--model_type",
        type=str,
        choices=["baseline", "improved", "deep"],
        default="baseline",
        help=(
            "Select the reconstruction model architecture.\n"
            "'baseline': use the original ReconstructionNet.\n"
            "'improved': use the ImprovedReconstructionNet with a gating module.\n"
            "'deep': use the DeepGatedReconstructionNet (U-Net style) with gating and skip connections."
        ),
    )

    p.add_argument("--phys_x", type=float, default=PHYSICAL_SIZE_X)
    p.add_argument("--phys_y", type=float, default=PHYSICAL_SIZE_Y)
    p.add_argument("--radius", type=float, default=SENSOR_RADIUS_DEFAULT)
    p.add_argument("--n_emitters", type=int, default=DEFAULT_N_EMITTERS)
    p.add_argument("--n_receivers", type=int, default=DEFAULT_N_RECEIVERS)
    p.add_argument("--sos_min", type=float, default=SOS_MIN)
    p.add_argument("--sos_max", type=float, default=SOS_MAX)

    p.add_argument("--data_seed", type=int, default=0,
                   help="Seed for deterministic phantom generation and dataset splits")
    p.add_argument("--num_samples", type=int, default=NUM_SAMPLES_DEFAULT)
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--lr", type=float, default=LEARNING_RATE_DEFAULT)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--device", type=str, default=DEVICE_DEFAULT)

    # Limited-view exclusion (applied during pairs-cache creation if dataset.py supports it)
    p.add_argument("--exclude_frac", type=float, default=0.0,
                   help="Fraction of receivers (closest arc) to exclude around each emitter (e.g., 0.25).")

    # Optional PDE representation (saved into pairs cache)
    p.add_argument("--include_pde", action="store_true")
    p.add_argument("--pde_emitters_k", type=int, default=1)

    # ToF normalization (saved in cache metadata)
    p.add_argument("--tof_norm", type=str, default="zscore", choices=["none", "zscore", "max"],
                   help="Normalize ToF values in dataset outputs.")
    p.add_argument("--tof_norm_eps", type=float, default=1e-6)

    # Collocation travel-time samples for forward model supervision
    p.add_argument("--store_collocation_T", action="store_true",
                   help="When building pairs cache, store collocation T samples for the differentiable forward model")
    p.add_argument("--n_collocation", type=int, default=512)
    p.add_argument("--colloc_emitters_k", type=int, default=4)
    p.add_argument("--colloc_seed", type=int, default=0)

    # Synthetic domain timing model for experimental transfer.
    # This modifies the saved synthetic ToF in the pairs cache so training sees
    # the same ROI-referenced measurement representation used by extract_measured_tof.py
    # for experimental RF ToF extraction.
    p.add_argument("--synthetic_roi_tof_enable", action="store_true",
                   help="When building the pairs cache, store ROI-referenced synthetic ToF instead of raw Eikonal Tx/Rx ToF.")
    p.add_argument("--synthetic_roi_center_x", type=float, default=None,
                   help="ROI center x [m] for synthetic ToF re-referencing. Default: phys_x/2.")
    p.add_argument("--synthetic_roi_center_y", type=float, default=None,
                   help="ROI center y [m] for synthetic ToF re-referencing. Default: phys_y/2.")
    p.add_argument("--synthetic_roi_radius_min", type=float, default=0.055,
                   help="Minimum sampled synthetic ROI radius [m].")
    p.add_argument("--synthetic_roi_radius_max", type=float, default=0.070,
                   help="Maximum sampled synthetic ROI radius [m].")
    p.add_argument("--synthetic_roi_outer_c_min", type=float, default=1460.0,
                   help="Minimum sampled outer/background speed [m/s] for synthetic ROI ToF re-referencing.")
    p.add_argument("--synthetic_roi_outer_c_max", type=float, default=1500.0,
                   help="Maximum sampled outer/background speed [m/s] for synthetic ROI ToF re-referencing.")
    p.add_argument("--synthetic_roi_reference_c", type=float, default=1500.0,
                   help="Reference speed [m/s] used for the ROI portion of the two-region background model.")
    p.add_argument("--synthetic_roi_synthetic_c", type=float, default=1500.0,
                   help="Homogeneous synthetic reference speed [m/s] used after residualization.")
    p.add_argument("--synthetic_roi_min_chord_frac", type=float, default=0.05,
                   help="Minimum ROI chord fraction for a channel to retain residual ToF information.")
    p.add_argument("--synthetic_roi_residual_clip_us", type=float, default=8.0,
                   help="Clip synthetic ROI residual ToF to +/- this many microseconds. Use <=0 to disable.")
    p.add_argument("--synthetic_roi_nonintersect_mode", choices=["reference", "keep", "nan"], default="reference",
                   help="How to handle channels that barely intersect the synthetic ROI.")
    p.add_argument("--synthetic_roi_seed", type=int, default=0,
                   help="Seed for sampled synthetic ROI radius and outer speed.")

    # Differentiable forward model + ToF consistency term
    p.add_argument("--use_differentiable_eikonal", action="store_true",
                   help="Enable differentiable eikonal forward model for ToF-consistency and/or forward training")
    p.add_argument("--lr_forward", type=float, default=None)

    # weights for forward-model training (only used if collocation_T exists + forward model enabled)
    p.add_argument("--lambda_forward_total", type=float, default=1.0)
    p.add_argument("--lambda_forward_pde", type=float, default=1.0)
    p.add_argument("--lambda_forward_bc", type=float, default=1.0)
    p.add_argument("--lambda_forward_supT", type=float, default=1.0)
    p.add_argument("--lambda_forward_receiver_tof", type=float, default=0.0,
                   help=(
                       "Additional forward-surrogate training loss on actual emitter-receiver MSFM ToFs, "
                       "normalized by the surrogate time scale. This makes the surrogate useful at the sensors, "
                       "not only at interior collocation points."
                   ))
    p.add_argument("--lambda_forward_receiver_residual_tof", type=float, default=0.0,
                   help=(
                       "Additional receiver-ToF training loss after removing each emitter's receiver mean. "
                       "This emphasizes object-sensitive angular residuals."
                   ))

    # ToF-consistency loss (only used if lambda_tof > 0)
    # The synthetic cache is generated by MSFM/Eikonal travel times. Therefore
    # the reconstruction-side ToF consistency is routed through the
    # differentiable Eikonal surrogate rather than the legacy straight-line
    # geometry-prior path matrix.
    p.add_argument(
        "--lambda_tof",
        type=float,
        default=0.01,
        help=(
            "Base weight for Eikonal-surrogate ToF consistency between reconstructed SoS and measured times. "
            "Requires --use_differentiable_eikonal and a cache built with --store_collocation_T."
        ),
    )
    p.add_argument("--tof_emitters_k", type=int, default=4,
                   help="Number of emitters to sample per batch for ToF-consistency loss (only used if lambda_tof>0)")
    p.add_argument("--tof_emitter_sampling", choices=["random", "cycle"], default="cycle",
                   help="Emitter selection for ToF-consistency. 'cycle' gives deterministic ring coverage across batches/epochs.")
    p.add_argument("--tof_loss_mode", choices=["absolute", "residual", "combined"], default="residual",
                   help=(
                       "Physics ToF loss mode. 'residual' removes masked per-emitter mean ToF before comparing, "
                       "so the Eikonal term constrains angular structure rather than global clock/surrogate bias."
                   ))
    p.add_argument("--tof_absolute_weight", type=float, default=0.15,
                   help="Absolute-ToF contribution used only when --tof_loss_mode combined.")
    p.add_argument("--lambda_grad", type=float, default=0.05,
                   help="Weight for image-gradient consistency loss on the reconstructed SoS.")
    p.add_argument("--lambda_contrast", type=float, default=0.0,
                   help="Weight for multi-scale local contrast loss on reconstructed SoS.")
    p.add_argument("--lambda_detail", type=float, default=0.0,
                   help="Weight for edge-weighted high-frequency detail-band loss on reconstructed SoS.")
    p.add_argument("--lambda_laplacian", type=float, default=0.0,
                   help="Weight for Laplacian/second-derivative loss on reconstructed SoS.")
    p.add_argument("--detail_kernels", type=str, default="5,11,21",
                   help="Comma-separated blur kernels used by the detail-band loss.")
    p.add_argument("--detail_edge_gain", type=float, default=4.0,
                   help="Extra detail-loss weight on target edges and inclusions.")
    p.add_argument("--selection_detail_weight", type=float, default=0.35,
                   help="Extra validation-detail weight used when --early_stop_metric val_sos_detail.")
    p.add_argument("--lambda_teacher", type=float, default=0.0,
                   help=(
                       "2B-only proximal weight to keep the reconstruction close to the resumed "
                       "2A-best teacher while physics coupling is tested."
                   ))
    # Main reconstruction loss weighting.  This is important when the synthetic
    # anatomy contains a water background/support envelope: an unweighted full-image
    # MSE can be minimized by a nearly water-valued prediction.
    p.add_argument("--use_sos_weighted_loss", action="store_true",
                   help="Use contrast-weighted SoS MSE instead of plain full-image MSE.")
    p.add_argument("--sos_weight_water", type=float, default=1.0,
                   help="Base weight assigned to all pixels in weighted SoS MSE.")
    p.add_argument("--sos_weight_contrast_gain", type=float, default=8.0,
                   help="Additional weight on pixels whose target SoS differs from water.")
    p.add_argument("--sos_weight_power", type=float, default=1.0,
                   help="Power applied to normalized absolute deviation from water in weighted SoS MSE.")
    p.add_argument("--use_circular_mask", action="store_true",
                   help="Apply circular support to prediction and target during train/validation/test.")
    p.add_argument("--mask_radius", type=float, default=None,
                   help="Circular support radius in meters. Default: --radius.")
    # Domain-shift robustness for experimental ToF.
    p.add_argument("--use_tof_mask_channel", action="store_true",
                   help="Use a second input channel containing the ToF validity mask (1=measured, 0=excluded/missing).")
    p.add_argument("--tof_feature_mode", choices=["raw", "residual_stack"], default="raw",
                   help=(
                       "Input representation for the inverse model. 'raw' keeps the legacy normalized ToF input. "
                       "'residual_stack' adds homogeneous-water residual and emitter/receiver double-centered "
                       "residual channels so the network sees object-sensitive ToF perturbations."
                   ))
    p.add_argument("--tof_input_domain_mix", choices=["roi_only", "roi_raw"], default="roi_only",
                   help=(
                       "Training input domain for caches that store both ROI-referenced ToF and raw Eikonal ToF. "
                       "'roi_only' uses the primary cache ToF. 'roi_raw' randomly trains the inverse model on "
                       "either ROI-referenced ToF or raw Eikonal ToF for the same SoS target."
                   ))
    p.add_argument("--tof_raw_input_prob", type=float, default=0.0,
                   help="Probability of using normalized raw Eikonal ToF as inverse input when --tof_input_domain_mix roi_raw.")
    p.add_argument("--lambda_domain_consistency", type=float, default=0.0,
                   help=(
                       "If >0 and raw Eikonal ToF is available, penalize disagreement between predictions from "
                       "ROI-referenced and raw-Eikonal ToF for the same synthetic phantom."
                   ))
    p.add_argument("--tof_aug_enable", action="store_true",
                   help="Apply stochastic ToF-domain augmentation during training only.")
    p.add_argument("--tof_aug_scale_min", type=float, default=0.85,
                   help="Minimum multiplicative scale for normalized ToF augmentation.")
    p.add_argument("--tof_aug_scale_max", type=float, default=1.20,
                   help="Maximum multiplicative scale for normalized ToF augmentation.")
    p.add_argument("--tof_aug_shift_std", type=float, default=0.35,
                   help="Std. dev. of additive normalized ToF shift, sampled per image.")
    p.add_argument("--tof_aug_noise_std", type=float, default=0.03,
                   help="Std. dev. of additive normalized ToF noise, sampled per entry.")
    p.add_argument("--tof_feature_channel_dropout", type=float, default=0.0,
                   help=(
                       "Training-only dropout probability for acoustic ToF feature channels. "
                       "For residual_stack, this randomly removes raw/residual/centered-residual channels "
                       "per sample while always keeping at least one acoustic channel."
                   ))
    p.add_argument("--tof_emitter_dropout", type=float, default=0.0,
                   help="Training-only probability of dropping each emitter row from acoustic ToF feature channels.")
    p.add_argument("--tof_receiver_dropout", type=float, default=0.0,
                   help="Training-only probability of dropping each receiver column from acoustic ToF feature channels.")
    p.add_argument("--ring_quarter_aug_enable", action="store_true",
                   help=(
                       "Training-only circular-array symmetry augmentation for 2A. "
                       "For square grids with emitter/receiver counts divisible by 4, randomly applies an exact "
                       "0/90/180/270-degree rotation to the SoS target and the matching circular roll to ToF "
                       "emitter/receiver axes. Disabled automatically when differentiable-Eikonal 2B is active, "
                       "because stored collocation targets would otherwise no longer match the rotated image."
                   ))
    p.add_argument("--ring_quarter_aug_prob", type=float, default=0.75,
                   help="Probability of applying a nonzero quarter-turn ring augmentation to each training batch.")
    p.add_argument("--use_ema", action="store_true",
                   help="Maintain an exponential moving average model and use it for validation/best checkpoints.")
    p.add_argument("--ema_decay", type=float, default=0.995,
                   help="EMA decay used when --use_ema is enabled.")
    p.add_argument("--preview_every", type=int, default=10,
                   help="Plot a few validation reconstructions every this many epochs (0 disables).")

    # ToF warmup/ramp (stabilizes enabling ToF-consistency)
    # ToF warmup and ramp schedule.  During the warmup period the ToF loss is
    # disabled (effective weight zero).  After warmup, the weight ramps
    # linearly from zero to the base value over the specified number of
    # epochs.  Using a nonzero ramp allows the network to learn a good
    # reconstruction before the physics term is introduced.
    p.add_argument("--tof_warmup_epochs", type=int, default=10,
                   help=(
                       "Number of initial epochs with effective lambda_tof=0.  "
                       "Set to zero to apply the ToF loss from the start."
                   ))
    p.add_argument("--tof_ramp_epochs", type=int, default=40,
                   help=(
                       "Number of epochs over which the ToF weight ramps from 0 to the base value after warmup.  "
                       "If zero, the full weight is applied immediately after warmup."
                   ))

    p.add_argument("--tof_fwd_pretrain_epochs", type=int, default=20,
                   help=(
                       "Epochs after warmup with lamToF_eff=0 but forward-model training enabled "
                       "from stored MSFM collocation targets.  Default is 20 because the ToF "
                       "consistency term should not be coupled before the differentiable "
                       "Eikonal surrogate has started to learn."
                   ))
    p.add_argument("--tof_forward_only_epochs", type=int, default=-1,
                   help=(
                       "Maximum number of Stage-2B schedule epochs for which the reconstruction model is frozen "
                       "while only the differentiable Eikonal surrogate trains. Use a small value such as 3-5. "
                       "Negative preserves the older behavior: warmup + forward-pretrain + extra gate epochs."
                   ))
    p.add_argument("--force_tof_coupling_after_forward_only", action="store_true",
                   help=(
                       "After --tof_forward_only_epochs, unfreeze reconstruction and apply a gentle ToF coupling "
                       "even if the forward-surrogate readiness gate is still closed. This prevents 2B from "
                       "ending as pure forward-surrogate training."
                   ))
    p.add_argument("--tof_min_coupling_after_unfreeze", type=float, default=0.15,
                   help=(
                       "Minimum fraction of --lambda_tof applied after forced unfreeze when the scheduled ramp "
                       "or readiness gate would otherwise keep lamToF_eff at zero."
                   ))
    p.add_argument("--require_2b_reconstruction_update", action="store_true",
                   help=(
                       "For resumed differentiable-Eikonal stages, raise an error if no epoch updates the "
                       "reconstruction model. This catches failed 2B runs before the evaluation routine evaluates a copied 2A model."
                   ))
    p.add_argument("--stage2b_residual_adapter", action="store_true",
                   help=(
                       "For resumed differentiable-Eikonal 2B, freeze most of the inverse model and train only "
                       "late spatial correction modules. This keeps 2B as a constrained residual correction "
                       "rather than full retraining."
                   ))
    p.add_argument("--stage2b_adapter_modules", type=str, default="up3,out_head",
                   help=(
                       "Comma-separated module-name prefixes kept trainable by --stage2b_residual_adapter. "
                       "For the deep model, 'up3,out_head' adapts only the last upsampling block and output head."
                   ))
    p.add_argument("--stage2b_delta_limit_mps", type=float, default=35.0,
                   help=(
                       "Soft bound on 2B changes relative to the resumed 2A teacher, in m/s. "
                       "Only values above the bound are penalized."
                   ))
    p.add_argument("--lambda_delta_barrier", type=float, default=0.0,
                   help=(
                       "Weight for the soft 2B delta barrier. Requires a 2A teacher, which is enabled by "
                       "--lambda_teacher > 0 during resumed 2B."
                   ))

    p.add_argument("--tof_gate_forward_supT_threshold", type=float, default=0.08,
                   help=(
                       "Safety gate for Eikonal-ToF coupling.  If >0, lambda_tof remains zero "
                       "until the previous epoch's mean forward supervised collocation loss "
                       "drops below this threshold.  This prevents the learning curves from "
                       "rising mechanically when the Eikonal surrogate is still inaccurate."
                   ))
    p.add_argument("--tof_gate_forward_pde_threshold", type=float, default=0.0,
                   help=(
                       "Additional safety gate for Eikonal-ToF coupling. If >0, lambda_tof remains zero "
                       "until the previous epoch's mean forward PDE residual loss is below this value."
                   ))
    p.add_argument("--tof_gate_forward_total_threshold", type=float, default=0.0,
                   help=(
                       "Additional safety gate for Eikonal-ToF coupling. If >0, lambda_tof remains zero "
                       "until the previous epoch's mean total forward surrogate validation/training loss is below this value."
                   ))
    p.add_argument("--tof_gate_receiver_rmse_us_threshold", type=float, default=0.0,
                   help=(
                       "Hard readiness gate for Stage 2B. If >0, reconstruction-side ToF coupling is disabled "
                       "until the differentiable Eikonal surrogate predicts validation receiver ToFs from true SoS "
                       "with physical RMSE below this many microseconds."
                   ))
    p.add_argument("--tof_gate_receiver_residual_rmse_us_threshold", type=float, default=0.0,
                   help=(
                       "Hard readiness gate using per-emitter receiver-mean-removed ToF residuals. If >0, "
                       "coupling is disabled until validation residual RMSE is below this many microseconds."
                   ))

    p.add_argument("--disable_tof_forward_gate", action="store_true",
                   help="Disable the forward-surrogate readiness gate and use only the epoch schedule.")
    p.add_argument("--unsafe_force_tof_coupling_before_receiver_gate", action="store_true",
                   help=(
                       "Compatibility escape hatch. Allows --force_tof_coupling_after_forward_only to override "
                       "receiver-ToF readiness gates. Normally leave disabled."
                   ))
    p.add_argument("--forward_best_metric", choices=["receiver_residual_rmse", "receiver_rmse"], default="receiver_residual_rmse",
                   help=(
                       "Metric used to remember the best differentiable Eikonal surrogate during 2B forward-only training."
                   ))
    p.add_argument("--restore_best_forward_on_coupling", action="store_true",
                   help=(
                       "When Stage 2B first opens reconstruction-side ToF coupling, restore the forward surrogate "
                       "state that had the best validation receiver-ToF metric."
                   ))

    p.add_argument("--stop_2b_if_tof_gate_closed", action="store_true",
                   help=(
                       "For guarded Stage 2B, stop cleanly once scheduled ToF coupling should begin "
                       "but the forward-surrogate readiness gate is still closed. This preserves the "
                       "resumed 2A-best checkpoint instead of drifting into ordinary supervised overfitting."
                   ))
    p.add_argument("--tof_gate_extra_forward_only_epochs", type=int, default=0,
                   help=(
                       "Optional extra forward-only epochs after warmup+forward-pretrain before "
                       "--stop_2b_if_tof_gate_closed is allowed to stop the run. The reconstruction "
                       "network remains frozen during these extra epochs."
                   ))

    p.add_argument("--forward_zero_grad_patience", type=int, default=3,
                   help="Abort if this many consecutive forward-surrogate batches have zero parameter-gradient norm.")

    p.add_argument("--skip_forward_during_tof_warmup", action="store_true",
                   help=(
                       "If set, skip forward-model training during ToF warmup.  This is kept only "
                       "for ablation; it is normally not recommended because warmup should train "
                       "the Eikonal surrogate before ToF coupling."
                   ))

    p.add_argument("--early_stop_patience", type=int, default=0,
                   help=(
                       "Stop training after this many consecutive epochs without validation "
                       "improvement. Use 0 to disable. The best checkpoint is still saved."
                   ))
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="Minimum validation-loss decrease required to reset early-stopping patience.")
    p.add_argument("--early_stop_metric", choices=["val_total", "val_sos", "val_detail", "val_sos_detail"], default="val_sos",
                   help="Validation metric used for best-checkpoint selection and early stopping.")
    p.add_argument("--allow_noop_resume", action="store_true",
                   help="Allow --resume when the checkpoint already has >= --epochs completed epochs.")

    # Persistent SoS bank + splits
    p.add_argument("--sos_bank_path", type=str, default=None,
                   help="SoS-only bank path. Default: <data_path>_sos_bank.pt")
    p.add_argument("--splits_path", type=str, default=None,
                   help="Splits path. Default: <sos_bank_path>_splits.pt")
    p.add_argument("--make_sos_bank", action="store_true",
                   help="Create SoS bank if missing (otherwise require it exists)")
    p.add_argument("--make_splits", action="store_true",
                   help="Create splits file if missing (otherwise require it exists)")

    # split fractions
    p.add_argument("--frac_train", type=float, default=0.8)
    p.add_argument("--frac_val", type=float, default=0.1)

    # Create pairs cache if missing
    p.add_argument("--make_pairs_cache", action="store_true",
                   help="If data_path missing, compute pairs from SoS bank and save to data_path")
    p.add_argument(
        "--pairs_chunk_size",
        type=int,
        default=2000,
        help=(
            "Number of samples stored in each "
            "progressive pairs-cache chunk."
        ),
    )

    p.add_argument(
        "--no_resume_pairs_chunks",
        action="store_true",
        help=(
            "Ignore existing completed pairs-cache "
            "chunks and rebuild them from the beginning."
        ),
    )

    p.add_argument(
        "--pairs_workers",
        type=int,
        default=1,
        help=(
            "Number of CPU processes used while "
            "building the pairs cache."
        ),
    )
    p.add_argument('--sos_water', type=float, default=SOS_WATER)
    p.add_argument('--max_shapes', type=int, default=MAX_SHAPES)
    p.add_argument('--min_shapes', type=int, default=MIN_SHAPES,
                   help='Minimum number of internal shapes per synthetic anatomy. Use 0 to auto-set to max_shapes//2.')
    p.add_argument("--use_random_support_envelope", action="store_true")
    p.add_argument("--support_radius_min_frac", type=float, default=0.65)
    p.add_argument("--support_radius_max_frac", type=float, default=0.95)
    p.add_argument("--support_center_jitter_frac", type=float, default=0.05)
    p.add_argument("--support_ellipse_prob", type=float, default=0.35)
    p.add_argument("--support_ellipse_axis_jitter_frac", type=float, default=0.15)
    p.add_argument("--shape_mode", type=str, choices=["ellipses", "mixed"], default="ellipses")
    p.add_argument("--p_ellipse", type=float, default=0.65)
    p.add_argument("--p_triangle", type=float, default=0.20)
    p.add_argument("--p_polygon", type=float, default=0.15)
    p.add_argument("--p_rod", type=float, default=0.0)
    p.add_argument("--shape_size_min_px", type=int, default=4)
    p.add_argument("--shape_size_max_px", type=int, default=24)
    p.add_argument("--polygon_vertices_min", type=int, default=4)
    p.add_argument("--polygon_vertices_max", type=int, default=8)
    return p


def _set_reproducibility(seed: int) -> None:
    seed = int(seed)
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_lambda_tof(epoch_idx: int, args) -> float:
    """Epoch-dependent effective lambda_tof with warmup, optional forward-pretrain, and optional ramp.

    Schedule:
      - epochs [0, warmup): lamToF=0
      - epochs [warmup, warmup+fwd_pretrain): lamToF=0
      - then ramp over ramp_epochs (or jump to base if ramp_epochs<=0)
    """
    base = float(getattr(args, "lambda_tof", 0.0))
    if base <= 0.0:
        return 0.0
    warm = int(getattr(args, "tof_warmup_epochs", 0))
    fwd_pre = int(getattr(args, "tof_fwd_pretrain_epochs", 0))
    ramp = int(getattr(args, "tof_ramp_epochs", 0))

    if epoch_idx < warm:
        return 0.0
    if epoch_idx < warm + fwd_pre:
        return 0.0

    if ramp <= 0:
        return base
    # ramp starts after warmup+fwd_pretrain
    t = (epoch_idx - (warm + fwd_pre) + 1) / float(ramp)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return base * t


def _tof_meta_scalar(tof_norm: dict, key: str, default: float) -> float:
    """Read a scalar from ToF metadata, including DataLoader-collated tensors."""
    if not isinstance(tof_norm, dict):
        return float(default)
    value = tof_norm.get(key, default)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().flatten()[0].cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float(default)
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return float(default)
        return _tof_meta_scalar({key: value[0]}, key, default)
    return float(value)


def _tof_meta_string(tof_norm: dict, key: str, default: str) -> str:
    """Read a string from ToF metadata, including DataLoader-collated lists."""
    if not isinstance(tof_norm, dict):
        return str(default)
    value = tof_norm.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return str(value)


def configure_stage2b_residual_adapter(model: torch.nn.Module, args) -> dict:
    """Freeze most inverse-model parameters and keep only late correction modules trainable."""
    raw_prefixes = str(getattr(args, "stage2b_adapter_modules", "up3,out_head") or "up3,out_head")
    prefixes = [p.strip() for p in raw_prefixes.split(",") if p.strip()]
    if not prefixes:
        raise ValueError("[train_initial_reconstruction.py] --stage2b_residual_adapter requires at least one trainable module prefix.")

    trainable_names = []
    frozen_names = []
    n_trainable = 0
    n_frozen = 0
    for name, param in model.named_parameters():
        keep = any(name == pref or name.startswith(pref + ".") for pref in prefixes)
        param.requires_grad_(keep)
        if keep:
            trainable_names.append(name)
            n_trainable += int(param.numel())
        else:
            frozen_names.append(name)
            n_frozen += int(param.numel())

    if n_trainable <= 0:
        raise ValueError(
            "[train_initial_reconstruction.py] --stage2b_residual_adapter selected no trainable parameters. "
            f"Requested prefixes={prefixes}. Check model parameter names."
        )

    return {
        "prefixes": prefixes,
        "trainable_param_count": int(n_trainable),
        "frozen_param_count": int(n_frozen),
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }


def _normalize_tof_like_dataset(tof_pred: torch.Tensor, tof_norm: dict) -> torch.Tensor:
    """Normalize predicted ToF using dataset metadata (matches PairsCacheDataset __getitem__)."""
    if not isinstance(tof_norm, dict):
        return tof_pred
    ttype = _tof_meta_string(tof_norm, "type", "none").lower()
    eps = _tof_meta_scalar(tof_norm, "eps", 1e-6)
    if ttype == "zscore":
        mu = _tof_meta_scalar(tof_norm, "mean", 0.0)
        sig = _tof_meta_scalar(tof_norm, "std", 1.0)
        return (tof_pred - mu) / (sig + eps)
    if ttype == "max":
        mx = _tof_meta_scalar(tof_norm, "max", 1.0)
        return tof_pred / (mx + eps)
    return tof_pred


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Masked MSE over entries where mask==1 (mask broadcastable to pred)."""
    if mask is None:
        return torch.mean((pred - target) ** 2)
    diff2 = (pred - target) ** 2 * mask
    denom = torch.sum(mask) + 1e-12
    return torch.sum(diff2) / denom

def _masked_remove_receiver_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Remove the receiver-axis mean independently for each sample/emitter row."""
    if mask is None:
        return x - x.mean(dim=-1, keepdim=True)
    m = mask.to(device=x.device, dtype=x.dtype)
    denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(dim=-1, keepdim=True) / denom
    return (x - mean) * m

def select_tof_emitters(args, *, epoch_idx: int, batch_idx: int, device: torch.device) -> torch.Tensor:
    """Select emitters for the reconstruction-side ToF loss."""
    ne = int(getattr(args, "n_emitters", 1))
    k_use = min(int(getattr(args, "tof_emitters_k", 4)), ne)
    if str(getattr(args, "tof_emitter_sampling", "cycle")).lower() == "random":
        return torch.randperm(ne, device=device)[:k_use]
    start = (int(epoch_idx) * k_use + int(batch_idx) * k_use) % ne
    return (torch.arange(k_use, device=device, dtype=torch.long) + start) % ne

def sos_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """SoS image loss for the main reconstruction task.

    The target and prediction are normalized to [0,1]. When random support
    envelopes are used, a large fraction of the image is exactly water. Plain
    full-image MSE over-rewards a nearly uniform water-valued prediction and
    under-penalizes missing inclusions. The optional weighted loss keeps a base
    penalty everywhere but increases the weight where the target differs from
    water.
    """
    if not bool(getattr(args, "use_sos_weighted_loss", False)):
        return torch.mean((pred - target) ** 2)

    sos_min = float(getattr(args, "sos_min", 1400.0))
    sos_max = float(getattr(args, "sos_max", 1650.0))
    sos_water = float(getattr(args, "sos_water", 1500.0))
    water_n = (sos_water - sos_min) / (sos_max - sos_min + 1e-12)

    base = max(0.0, float(getattr(args, "sos_weight_water", 1.0)))
    gain = max(0.0, float(getattr(args, "sos_weight_contrast_gain", 8.0)))
    power = max(0.1, float(getattr(args, "sos_weight_power", 1.0)))

    contrast = torch.abs(target - water_n)
    contrast_scale = max(abs(0.0 - water_n), abs(1.0 - water_n), 1e-6)
    contrast = torch.clamp(contrast / contrast_scale, 0.0, 1.0)
    weights = base + gain * torch.pow(contrast, power)
    diff2 = (pred - target) ** 2
    return torch.sum(weights * diff2) / (torch.sum(weights) + 1e-12)

def unpack_reconstruction_batch(batch):
    """Return sos, inverse-input ToF, mask, pde, colloc, and optional raw-physics ToF."""
    sos_b = batch[0]
    tof_b = batch[1]
    tof_mask_b = None
    pde_b = None
    colloc = None
    raw_physics = None
    for item in batch[2:]:
        if isinstance(item, dict):
            if "raw_eikonal_tof_phys" in item:
                raw_physics = item
            else:
                colloc = item
        elif torch.is_tensor(item) and item.shape == tof_b.shape:
            tof_mask_b = item
        else:
            pde_b = item
    return sos_b, tof_b, tof_mask_b, pde_b, colloc, raw_physics


def normalize_physical_tof_like_dataset(tof_phys: torch.Tensor, tof_norm: dict | None) -> torch.Tensor:
    """Apply the cache ToF normalization to a physical-time matrix."""
    tn = tof_norm or {"type": "none"}
    kind = _tof_meta_string(tn, "type", "none").lower()
    eps = _tof_meta_scalar(tn, "eps", 1e-6)
    if kind == "zscore":
        return (tof_phys - _tof_meta_scalar(tn, "mean", 0.0)) / (_tof_meta_scalar(tn, "std", 1.0) + eps)
    if kind == "max":
        return tof_phys / (_tof_meta_scalar(tn, "max", 1.0) + eps)
    return tof_phys


def make_homogeneous_water_tof_norm(args, pairs_ds: PairsCacheDataset, device: torch.device) -> torch.Tensor:
    """Return normalized homogeneous-water ToF for the configured ring geometry."""
    return make_homogeneous_water_tof_norm_with_meta(args, getattr(pairs_ds, "tof_norm", None), device)


def make_homogeneous_water_tof_norm_with_meta(args, tof_norm_meta: dict | None, device: torch.device) -> torch.Tensor:
    """Return homogeneous-water ToF normalized with a specific ToF-normalization metadata dictionary."""
    dx = float(args.phys_x) / max(1, int(args.nx) - 1)
    dy = float(args.phys_y) / max(1, int(args.ny) - 1)
    emitters, receivers = generate_sensor_positions(
        int(args.nx), int(args.ny), dx, dy, float(args.radius), int(args.n_emitters), int(args.n_receivers)
    )
    e_xy = torch.tensor(np.asarray(emitters, dtype=np.float32), device=device)
    r_xy = torch.tensor(np.asarray(receivers, dtype=np.float32), device=device)
    e_xy = torch.stack([e_xy[:, 0] * dx, e_xy[:, 1] * dy], dim=1)
    r_xy = torch.stack([r_xy[:, 0] * dx, r_xy[:, 1] * dy], dim=1)
    dist = torch.linalg.norm(e_xy[:, None, :] - r_xy[None, :, :], dim=-1)
    tof_phys = dist / float(args.sos_water)
    water_norm = normalize_physical_tof_like_dataset(tof_phys, tof_norm_meta)
    return water_norm.float()


def choose_inverse_tof_domain(
    tof_roi_norm: torch.Tensor,
    tof_mask_b: torch.Tensor | None,
    raw_tof_phys: torch.Tensor | None,
    raw_tof_norm_meta: dict | None,
    args,
    *,
    training: bool,
) -> tuple[torch.Tensor, str]:
    """Choose the ToF domain used as inverse input for this batch."""
    mode = str(getattr(args, "tof_input_domain_mix", "roi_only") or "roi_only").lower()
    if (not training) or mode == "roi_only" or raw_tof_phys is None:
        return tof_roi_norm, "roi"
    if mode != "roi_raw":
        return tof_roi_norm, "roi"
    p_raw = float(max(0.0, min(1.0, getattr(args, "tof_raw_input_prob", 0.0))))
    if p_raw <= 0.0:
        return tof_roi_norm, "roi"
    if torch.rand((), device=tof_roi_norm.device).item() >= p_raw:
        return tof_roi_norm, "roi"
    raw_norm = normalize_physical_tof_like_dataset(raw_tof_phys, raw_tof_norm_meta).to(
        device=tof_roi_norm.device, dtype=tof_roi_norm.dtype
    )
    if tof_mask_b is not None:
        raw_norm = raw_norm * tof_mask_b.to(device=tof_roi_norm.device, dtype=tof_roi_norm.dtype)
    return raw_norm, "raw"


def masked_double_center(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Remove additive emitter and receiver trends from a ToF residual matrix."""
    if mask is None:
        row_mean = x.mean(dim=2, keepdim=True)
        col_mean = x.mean(dim=1, keepdim=True)
        global_mean = x.mean(dim=(1, 2), keepdim=True)
        return x - row_mean - col_mean + global_mean

    m = mask.to(device=x.device, dtype=x.dtype)
    x_m = x * m
    row_count = m.sum(dim=2, keepdim=True).clamp_min(1.0)
    col_count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    global_count = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    row_mean = x_m.sum(dim=2, keepdim=True) / row_count
    col_mean = x_m.sum(dim=1, keepdim=True) / col_count
    global_mean = x_m.sum(dim=(1, 2), keepdim=True) / global_count
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
    mask = None
    if tof_mask_b is not None:
        mask = tof_mask_b.to(device=tof.device, dtype=torch.float32)

    mode = str(tof_feature_mode or "raw").lower()
    if mode == "residual_stack":
        if water_tof_norm is None:
            raise ValueError("[train_initial_reconstruction.py] residual_stack requires a homogeneous-water ToF reference.")
        water = water_tof_norm.to(device=tof.device, dtype=tof.dtype).unsqueeze(0)
        residual = tof - water
        if mask is not None:
            residual = residual * mask
            raw = tof * mask
        else:
            raw = tof
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


def apply_tof_feature_dropout(model_input: torch.Tensor, use_tof_mask_channel: bool, args) -> torch.Tensor:
    """Training-only dropout on acoustic ToF feature channels, preserving the mask channel."""
    if model_input.ndim != 4:
        return model_input

    channel_p = float(max(0.0, min(0.95, getattr(args, "tof_feature_channel_dropout", 0.0))))
    emitter_p = float(max(0.0, min(0.95, getattr(args, "tof_emitter_dropout", 0.0))))
    receiver_p = float(max(0.0, min(0.95, getattr(args, "tof_receiver_dropout", 0.0))))
    if channel_p <= 0.0 and emitter_p <= 0.0 and receiver_p <= 0.0:
        return model_input

    x = model_input.clone()
    n_acoustic = int(x.shape[1]) - (1 if bool(use_tof_mask_channel) else 0)
    if n_acoustic <= 0:
        return x

    acoustic = x[:, :n_acoustic, :, :]
    B, C, Ne, Nr = acoustic.shape

    if channel_p > 0.0 and C > 1:
        keep = (torch.rand((B, C, 1, 1), device=x.device, dtype=x.dtype) >= channel_p).to(x.dtype)
        # Avoid all-zero acoustic input for any sample.
        empty = keep.sum(dim=1, keepdim=True) <= 0.0
        if bool(empty.any()):
            chosen = torch.randint(low=0, high=C, size=(B,), device=x.device)
            repair = torch.zeros_like(keep)
            repair[torch.arange(B, device=x.device), chosen, 0, 0] = 1.0
            keep = torch.where(empty.expand_as(keep), repair, keep)
        acoustic = acoustic * keep

    if emitter_p > 0.0:
        e_keep = (torch.rand((B, 1, Ne, 1), device=x.device, dtype=x.dtype) >= emitter_p).to(x.dtype)
        acoustic = acoustic * e_keep

    if receiver_p > 0.0:
        r_keep = (torch.rand((B, 1, 1, Nr), device=x.device, dtype=x.dtype) >= receiver_p).to(x.dtype)
        acoustic = acoustic * r_keep

    x[:, :n_acoustic, :, :] = acoustic
    return x


def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """In-place EMA update for validation/checkpoint smoothing."""
    d = float(max(0.0, min(0.9999, decay)))
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            src = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(d).add_(src.to(dtype=ema_value.dtype), alpha=1.0 - d)
            else:
                ema_value.copy_(src)


def apply_ring_quarter_augmentation(
    sos_b: torch.Tensor,
    tof_b: torch.Tensor,
    tof_mask_b: torch.Tensor | None,
    args,
    *,
    allow: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply exact 90-degree circular-array symmetry augmentation to a training batch."""
    if not allow or not bool(getattr(args, "ring_quarter_aug_enable", False)):
        return sos_b, tof_b, tof_mask_b

    prob = float(max(0.0, min(1.0, getattr(args, "ring_quarter_aug_prob", 0.75))))
    if prob <= 0.0 or torch.rand((), device=tof_b.device).item() >= prob:
        return sos_b, tof_b, tof_mask_b

    if sos_b.shape[-2] != sos_b.shape[-1]:
        return sos_b, tof_b, tof_mask_b
    ne = int(tof_b.shape[1])
    nr = int(tof_b.shape[2])
    if ne % 4 != 0 or nr % 4 != 0:
        return sos_b, tof_b, tof_mask_b

    q = int(torch.randint(1, 4, (1,), device=tof_b.device).item())
    e_shift = q * (ne // 4)
    r_shift = q * (nr // 4)
    sos_aug = torch.rot90(sos_b, k=q, dims=(-2, -1))
    tof_aug = torch.roll(tof_b, shifts=(e_shift, r_shift), dims=(1, 2))
    if tof_mask_b is not None:
        tof_mask_aug = torch.roll(tof_mask_b, shifts=(e_shift, r_shift), dims=(1, 2))
    else:
        tof_mask_aug = None
    return sos_aug, tof_aug, tof_mask_aug


def augment_normalized_tof_for_training(tof_b: torch.Tensor, tof_mask_b: torch.Tensor | None, args) -> torch.Tensor:
    """Apply global scale/shift and local noise to normalized ToF, only on measured entries."""
    if not bool(getattr(args, "tof_aug_enable", False)):
        return tof_b
    b = tof_b.shape[0]
    device = tof_b.device
    dtype = tof_b.dtype
    smin = float(getattr(args, "tof_aug_scale_min", 1.0))
    smax = float(getattr(args, "tof_aug_scale_max", 1.0))
    if smax < smin:
        smin, smax = smax, smin
    scale = torch.empty((b, 1, 1), device=device, dtype=dtype).uniform_(smin, smax)
    shift_std = float(getattr(args, "tof_aug_shift_std", 0.0))
    noise_std = float(getattr(args, "tof_aug_noise_std", 0.0))
    shift = torch.randn((b, 1, 1), device=device, dtype=dtype) * shift_std if shift_std > 0.0 else 0.0
    noise = torch.randn_like(tof_b) * noise_std if noise_std > 0.0 else 0.0
    aug = tof_b * scale + shift + noise
    if tof_mask_b is not None:
        m = tof_mask_b.to(device=device, dtype=dtype)
        aug = aug * m
    return aug


def eikonal_tof_consistency_loss(
    *,
    pred_norm_sos: torch.Tensor,
    tof_target_norm: torch.Tensor,
    tof_mask_b: torch.Tensor | None,
    args,
    pairs_ds: PairsCacheDataset,
    forward_model: EikonalForwardModel,
    emitters_t: torch.Tensor,
    receivers_t: torch.Tensor,
    emitter_idx: torch.Tensor | None = None,
    tof_target_phys: torch.Tensor | None = None,
    tof_target_norm_meta: dict | None = None,
) -> torch.Tensor:
    """ToF consistency through the learned Eikonal surrogate, not straight rays."""
    if forward_model is None or emitters_t is None or receivers_t is None:
        raise RuntimeError("[train_initial_reconstruction.py] Eikonal ToF consistency requires --use_differentiable_eikonal.")

    ne = int(getattr(args, "n_emitters", emitters_t.shape[0]))
    k_use = min(int(getattr(args, "tof_emitters_k", 4)), ne)
    device = pred_norm_sos.device
    if emitter_idx is None:
        emitter_idx = torch.randperm(ne, device=device)[:k_use]
    else:
        emitter_idx = emitter_idx.to(device=device, dtype=torch.long)

    c_pred_phys = pred_norm_sos * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
    c_pred_phys = c_pred_phys.unsqueeze(1)

    tof_pred_phys = forward_model.predict_tof_matrix(
        c_img=c_pred_phys,
        emitters_xy=emitters_t.index_select(0, emitter_idx),
        receivers_xy=receivers_t,
        emitters_k=None,
    )
    norm_meta = tof_target_norm_meta if isinstance(tof_target_norm_meta, dict) else getattr(pairs_ds, "tof_norm", {})
    tof_pred_norm = _normalize_tof_like_dataset(tof_pred_phys, norm_meta)
    if tof_target_phys is not None:
        target_phys = tof_target_phys.to(device=device, dtype=torch.float32).index_select(1, emitter_idx)
        tof_target_use = _normalize_tof_like_dataset(target_phys, norm_meta)
    else:
        tof_target_use = tof_target_norm.index_select(1, emitter_idx)

    if tof_mask_b is not None:
        mask_use = tof_mask_b.to(device=device, dtype=torch.float32).index_select(1, emitter_idx)
    elif getattr(pairs_ds, "tof_mask", None) is not None:
        m_np = np.array(pairs_ds.tof_mask, dtype=np.float32)
        m_t = torch.tensor(m_np, dtype=torch.float32, device=device)
        mask_use = m_t.index_select(0, emitter_idx).unsqueeze(0).expand_as(tof_pred_norm)
    else:
        mask_use = None

    mode = str(getattr(args, "tof_loss_mode", "residual")).lower()
    if mode == "absolute":
        return masked_mse(tof_pred_norm, tof_target_use, mask_use)

    pred_res = _masked_remove_receiver_mean(tof_pred_norm, mask_use)
    target_res = _masked_remove_receiver_mean(tof_target_use, mask_use)
    residual_loss = masked_mse(pred_res, target_res, mask_use)
    if mode == "combined":
        abs_weight = float(max(0.0, getattr(args, "tof_absolute_weight", 0.15)))
        return residual_loss + abs_weight * masked_mse(tof_pred_norm, tof_target_use, mask_use)
    return residual_loss


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple first-order image-gradient loss for sharper structure recovery."""
    dx_pred = pred[..., 1:, :] - pred[..., :-1, :]
    dx_true = target[..., 1:, :] - target[..., :-1, :]
    dy_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dy_true = target[..., :, 1:] - target[..., :, :-1]
    return torch.mean((dx_pred - dx_true) ** 2) + torch.mean((dy_pred - dy_true) ** 2)

def multiscale_local_contrast_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare local contrast at several spatial scales to discourage blurred conditional averages."""
    loss = pred.new_tensor(0.0)
    n = 0
    for k in (5, 11, 21):
        pad = k // 2
        pred_low = F.avg_pool2d(pred, kernel_size=k, stride=1, padding=pad)
        target_low = F.avg_pool2d(target, kernel_size=k, stride=1, padding=pad)
        loss = loss + torch.mean((pred - pred_low - (target - target_low)) ** 2)
        n += 1
    return loss / float(max(1, n))


def _parse_int_list(text: str, default=(5, 11, 21)) -> list[int]:
    vals = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            continue
        if v > 1:
            vals.append(v if v % 2 == 1 else v + 1)
    return vals or list(default)


def _nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Expected 2D/3D/4D image tensor, got shape={tuple(x.shape)}")


def _avg_blur_reflect(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x4 = _nchw(x)
    k = int(kernel_size)
    if k <= 1:
        return x4
    if k % 2 == 0:
        k += 1
    pad = k // 2
    return F.avg_pool2d(F.pad(x4, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)


def _edge_detail_weight(target: torch.Tensor, args) -> torch.Tensor:
    """Weight pixels where sharp SoS structure exists instead of over-rewarding water background."""
    with torch.no_grad():
        t = _nchw(target)
        dx = torch.zeros_like(t)
        dy = torch.zeros_like(t)
        dx[..., :, 1:] = torch.abs(t[..., :, 1:] - t[..., :, :-1])
        dy[..., 1:, :] = torch.abs(t[..., 1:, :] - t[..., :-1, :])
        edge = dx + dy

        sos_min = float(getattr(args, "sos_min", 1400.0))
        sos_max = float(getattr(args, "sos_max", 1650.0))
        sos_water = float(getattr(args, "sos_water", 1500.0))
        water_n = (sos_water - sos_min) / (sos_max - sos_min + 1e-12)
        contrast = torch.abs(t - water_n)
        contrast_scale = max(abs(0.0 - water_n), abs(1.0 - water_n), 1e-6)
        score = edge + 0.35 * torch.clamp(contrast / contrast_scale, 0.0, 1.0)

        flat = score.flatten(1)
        scale = flat.quantile(0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        gain = float(max(0.0, getattr(args, "detail_edge_gain", 4.0)))
        return 1.0 + gain * torch.clamp(score / scale, 0.0, 1.0)


def detail_band_loss(pred: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """High-frequency band loss, edge weighted, to train 2A to keep internal detail."""
    p = _nchw(pred)
    t = _nchw(target)
    w = _edge_detail_weight(t, args)
    loss = p.new_tensor(0.0)
    kernels = _parse_int_list(getattr(args, "detail_kernels", "5,11,21"))
    for k in kernels:
        p_hi = p - _avg_blur_reflect(p, k)
        t_hi = t - _avg_blur_reflect(t, k)
        loss = loss + torch.sum(w * (p_hi - t_hi) ** 2) / torch.sum(w).clamp_min(1.0)
    return loss / float(max(1, len(kernels)))


def laplacian_detail_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Second-derivative detail loss; useful for thin rods and sharp interfaces."""
    p = _nchw(pred)
    t = _nchw(target)
    kernel = p.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    p_lap = F.conv2d(F.pad(p, (1, 1, 1, 1), mode="reflect"), kernel)
    t_lap = F.conv2d(F.pad(t, (1, 1, 1, 1), mode="reflect"), kernel)
    return torch.mean((p_lap - t_lap) ** 2)


def validation_selection_value(metric_name: str, val_total: float, val_sos: float, val_detail: float, args) -> tuple[float, str]:
    metric_name = str(metric_name or "val_sos")
    if metric_name == "val_total":
        return float(val_total), "ValTot"
    if metric_name == "val_detail":
        return float(val_detail), "ValDetail"
    if metric_name == "val_sos_detail":
        value = float(val_sos) + float(getattr(args, "selection_detail_weight", 0.35)) * float(val_detail)
        return value, "ValSoSDetail"
    return float(val_sos), "ValSoS"


def eikonal_forward_collocation_metrics(
    forward_model: EikonalForwardModel,
    c_true_norm: torch.Tensor,
    colloc,
    emitters_t: torch.Tensor,
    args,
):
    """Evaluate the differentiable Eikonal surrogate on cached MSFM collocation targets."""
    if colloc is None:
        return None

    device = c_true_norm.device
    c_true_phys = (
        c_true_norm.view(-1, 1, int(args.nx), int(args.ny))
        * (float(args.sos_max) - float(args.sos_min))
        + float(args.sos_min)
    )
    B_fwd = int(c_true_phys.shape[0])

    colloc_xy = colloc["colloc_xy"].to(device)
    emit_idx = colloc["colloc_emitters_idx"].to(device)
    colloc_T = (colloc.get("colloc_T_phys", None) if isinstance(colloc, dict) else None)
    if colloc_T is None:
        colloc_T = colloc["colloc_T"]
    colloc_T = colloc_T.to(device)

    if colloc_xy.ndim == 2:
        colloc_xy = colloc_xy.unsqueeze(0).expand(B_fwd, -1, 2)
    elif colloc_xy.ndim == 3 and int(colloc_xy.shape[0]) == B_fwd and int(colloc_xy.shape[-1]) == 2:
        pass
    else:
        raise RuntimeError(f"[train_initial_reconstruction.py] Unexpected colloc_xy shape {tuple(colloc_xy.shape)} for batch size {B_fwd}.")

    if colloc_T.ndim == 2:
        colloc_T = colloc_T.unsqueeze(0).expand(B_fwd, -1, -1)
    elif colloc_T.ndim == 3 and int(colloc_T.shape[0]) == B_fwd:
        pass
    else:
        raise RuntimeError(f"[train_initial_reconstruction.py] Unexpected colloc_T shape {tuple(colloc_T.shape)} for batch size {B_fwd}.")

    if emit_idx.ndim == 1:
        emit_idx_batch = emit_idx.unsqueeze(0).expand(B_fwd, -1)
    elif emit_idx.ndim == 2 and int(emit_idx.shape[0]) == B_fwd:
        emit_idx_batch = emit_idx
    else:
        raise RuntimeError(
            f"[train_initial_reconstruction.py] Unexpected colloc_emitters_idx shape {tuple(emit_idx.shape)} for batch size {B_fwd}. "
            "Rebuild the pairs cache with the corrected collocation loader."
        )

    if int(colloc_T.shape[1]) != int(emit_idx_batch.shape[1]):
        raise RuntimeError(
            f"[train_initial_reconstruction.py] colloc_T/emitters mismatch: colloc_T shape {tuple(colloc_T.shape)}, "
            f"emitters shape {tuple(emit_idx_batch.shape)}."
        )

    weights = ForwardLossWeights(
        lambda_pde=float(getattr(args, "lambda_forward_pde", 1.0)),
        lambda_bc=float(getattr(args, "lambda_forward_bc", 1.0)),
        lambda_supT=float(getattr(args, "lambda_forward_supT", 1.0)),
    )

    sums = {"total": 0.0, "pde": 0.0, "bc": 0.0, "supT": 0.0, "count": 0}
    kF = int(colloc_T.shape[1])

    # forward_losses() computes an Eikonal PDE residual by differentiating T
    # with respect to collocation coordinates.  This diagnostic is called from
    # the validation block, which otherwise uses torch.no_grad(), so autograd
    # must be re-enabled locally even though no optimizer step is performed.
    with torch.enable_grad():
        c_true_phys_eval = c_true_phys.detach()
        colloc_xy_eval = colloc_xy.detach()
        colloc_T_eval = colloc_T.detach()
        for j in range(kF):
            src_xy = emitters_t[emit_idx_batch[:, j].long()].detach()
            losses = forward_model.forward_losses(
                c_img=c_true_phys_eval,
                source_xy=src_xy,
                colloc_xy=colloc_xy_eval,
                T_target=colloc_T_eval[:, j, :],
                weights=weights,
            )
            sums["total"] += float(losses["loss_forward_total"].detach().item())
            sums["pde"] += float(losses["loss_pde"].detach().item())
            sums["bc"] += float(losses["loss_bc"].detach().item())
            sums["supT"] += float(losses["loss_supT"].detach().item())
            sums["count"] += 1
    return sums


def eikonal_forward_receiver_tof_metrics(
    *,
    forward_model: EikonalForwardModel,
    c_true_norm: torch.Tensor,
    tof_target_phys: torch.Tensor | None,
    tof_mask_b: torch.Tensor | None,
    emitters_t: torch.Tensor,
    receivers_t: torch.Tensor,
    args,
    epoch_idx: int,
    batch_idx: int,
):
    """Validate the surrogate on actual receiver ToFs, not only collocation T."""
    if forward_model is None or tof_target_phys is None:
        return None

    device = c_true_norm.device
    ne = int(getattr(args, "n_emitters", emitters_t.shape[0]))
    k_use = min(int(getattr(args, "tof_emitters_k", 4)), ne)
    emitter_idx = select_tof_emitters(args, epoch_idx=epoch_idx, batch_idx=batch_idx, device=device)
    if int(emitter_idx.numel()) > k_use:
        emitter_idx = emitter_idx[:k_use]

    c_true_phys = (
        c_true_norm.view(-1, 1, int(args.nx), int(args.ny))
        * (float(args.sos_max) - float(args.sos_min))
        + float(args.sos_min)
    )
    pred_phys = forward_model.predict_tof_matrix(
        c_img=c_true_phys,
        emitters_xy=emitters_t.index_select(0, emitter_idx),
        receivers_xy=receivers_t,
        emitters_k=None,
    )
    target_phys = tof_target_phys.to(device=device, dtype=torch.float32).index_select(1, emitter_idx)

    if tof_mask_b is not None:
        mask = tof_mask_b.to(device=device, dtype=torch.float32).index_select(1, emitter_idx)
    else:
        mask = torch.ones_like(pred_phys)

    valid = mask > 0.5
    if not torch.any(valid):
        return None

    diff = pred_phys - target_phys
    diff_valid = diff[valid]
    mse = torch.mean(diff_valid * diff_valid)
    mae = torch.mean(torch.abs(diff_valid))

    pred_res = _masked_remove_receiver_mean(pred_phys, mask)
    target_res = _masked_remove_receiver_mean(target_phys, mask)
    res_diff = pred_res - target_res
    res_valid = res_diff[valid]
    res_mse = torch.mean(res_valid * res_valid)
    res_mae = torch.mean(torch.abs(res_valid))

    return {
        "rmse_us": float(torch.sqrt(mse.detach()).item() * 1.0e6),
        "mae_us": float(mae.detach().item() * 1.0e6),
        "residual_rmse_us": float(torch.sqrt(res_mse.detach()).item() * 1.0e6),
        "residual_mae_us": float(res_mae.detach().item() * 1.0e6),
        "count": int(valid.sum().detach().item()),
    }


def eikonal_forward_receiver_tof_training_loss(
    *,
    forward_model: EikonalForwardModel,
    c_true_norm: torch.Tensor,
    tof_target_phys: torch.Tensor | None,
    tof_mask_b: torch.Tensor | None,
    emitters_t: torch.Tensor,
    receivers_t: torch.Tensor,
    args,
    epoch_idx: int,
    batch_idx: int,
):
    """Endpoint ToF supervision for the differentiable Eikonal surrogate."""
    if forward_model is None or tof_target_phys is None:
        return None

    lambda_abs = float(max(0.0, getattr(args, "lambda_forward_receiver_tof", 0.0)))
    lambda_res = float(max(0.0, getattr(args, "lambda_forward_receiver_residual_tof", 0.0)))
    if lambda_abs <= 0.0 and lambda_res <= 0.0:
        return None

    device = c_true_norm.device
    ne = int(getattr(args, "n_emitters", emitters_t.shape[0]))
    k_use = min(int(getattr(args, "tof_emitters_k", 4)), ne)
    emitter_idx = select_tof_emitters(args, epoch_idx=epoch_idx, batch_idx=batch_idx, device=device)
    if int(emitter_idx.numel()) > k_use:
        emitter_idx = emitter_idx[:k_use]

    c_true_phys = (
        c_true_norm.view(-1, 1, int(args.nx), int(args.ny))
        * (float(args.sos_max) - float(args.sos_min))
        + float(args.sos_min)
    )
    pred_phys = forward_model.predict_tof_matrix(
        c_img=c_true_phys,
        emitters_xy=emitters_t.index_select(0, emitter_idx),
        receivers_xy=receivers_t,
        emitters_k=None,
    )
    target_phys = tof_target_phys.to(device=device, dtype=torch.float32).index_select(1, emitter_idx)

    if tof_mask_b is not None:
        mask = tof_mask_b.to(device=device, dtype=torch.float32).index_select(1, emitter_idx)
    else:
        mask = torch.ones_like(pred_phys)

    t0 = getattr(forward_model, "T0", None)
    scale = t0.to(device=pred_phys.device, dtype=pred_phys.dtype) if torch.is_tensor(t0) else pred_phys.new_tensor(1.0)
    pred_dimless = pred_phys / torch.clamp(scale, min=1e-9)
    target_dimless = target_phys / torch.clamp(scale, min=1e-9)

    loss_abs = masked_mse(pred_dimless, target_dimless, mask)
    pred_res = _masked_remove_receiver_mean(pred_dimless, mask)
    target_res = _masked_remove_receiver_mean(target_dimless, mask)
    loss_res = masked_mse(pred_res, target_res, mask)
    loss = lambda_abs * loss_abs + lambda_res * loss_res
    return {
        "loss": loss,
        "abs": loss_abs.detach(),
        "residual": loss_res.detach(),
    }


def _plot_training_curves_consistent(train_opt_total_hist, train_eval_total_hist, val_total_hist,
                                      train_opt_sos_hist, train_eval_sos_hist, val_sos_hist):
    """Plot the single learning curve used for model selection.

    The plotted quantities are comparable: both are deterministic eval-mode
    SoS reconstruction losses.  The stochastic Train backprop loss and the
    full-objective totals remain in the textual log/checkpoint history, but are
    intentionally not plotted because they are not the early-stopping criterion
    and mixing them with validation is confusing.
    """
    tevs = np.asarray(train_eval_sos_hist, dtype=float)
    vs = np.asarray(val_sos_hist, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    if tevs.size:
        ax.plot(tevs, label="Train eval SoS")
    if vs.size:
        ax.plot(vs, label="Validation eval SoS")
        finite = np.isfinite(vs)
        if finite.any():
            best_idx_rel = int(np.argmin(vs[finite]))
            finite_idx = np.flatnonzero(finite)
            best_idx = int(finite_idx[best_idx_rel])
            ax.axvline(best_idx, linestyle="--", alpha=0.35, label=f"Best validation epoch {best_idx + 1}")
            ax.scatter([best_idx], [vs[best_idx]], zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SoS reconstruction loss")
    ax.set_title("Supervised learning curve: deterministic evaluation metric")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    log_image(fig)
    plt.close(fig)

def _model_ctor_config_from_args(args, metadata=None):
    """Minimal architecture/inference contract saved with every checkpoint.

    This is separate from train_config.  train_config records the full command
    line; model_ctor_config records only the values needed by the evaluation routine to rebuild
    the exact network architecture.
    """
    meta = metadata if isinstance(metadata, dict) else {}

    def _get(name, default):
        value = getattr(args, name, None)
        if value is None and isinstance(meta, dict):
            value = meta.get(name, None)
        return default if value is None else value

    return {
        "model_type": str(_get("model_type", "baseline")),
        "latent_res": int(_get("latent_res", 24)),
        "nx": int(_get("nx", 0)),
        "ny": int(_get("ny", 0)),
        "phys_x": float(_get("phys_x", 0.0)),
        "phys_y": float(_get("phys_y", 0.0)),
        "radius": float(_get("radius", 0.0)),
        "n_emitters": int(_get("n_emitters", 0)),
        "n_receivers": int(_get("n_receivers", 0)),
        "tof_feature_mode": str(_get("tof_feature_mode", "raw")),
        "use_tof_mask_channel": bool(_get("use_tof_mask_channel", False)),
        "model_fc_dropout": float(_get("model_fc_dropout", 0.0)),
        "model_decoder_dropout": float(_get("model_decoder_dropout", 0.0)),
        "sos_min": float(_get("sos_min", 0.0)),
        "sos_max": float(_get("sos_max", 0.0)),
        "sos_water": float(_get("sos_water", 1500.0)),
    }

def save_checkpoint(model, optimizer, train_hist, val_hist, args, metadata, forward_model=None, optimizer_forward=None, path=None, component_history=None):
    model_ctor_config = _model_ctor_config_from_args(args, metadata)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_hist,
        'val_history': val_hist,
        'epochs': len(train_hist),
        'train_config': vars(args),
        'model_ctor_config': model_ctor_config,
        'dataset_metadata': metadata
    }
    if component_history is not None:
        payload['component_history'] = component_history
    if hasattr(args, "tof_schedule_start_epoch"):
        payload['tof_schedule_start_epoch'] = int(getattr(args, "tof_schedule_start_epoch"))
    if forward_model is not None:
        payload['forward_model_state_dict'] = forward_model.state_dict()
        if hasattr(forward_model, "metadata"):
            payload['forward_model_metadata'] = forward_model.metadata()
    if optimizer_forward is not None:
        payload['optimizer_forward_state_dict'] = optimizer_forward.state_dict()
    out_path = args.model_path if path is None else path
    torch.save(payload, out_path)


def best_checkpoint_path(model_path: str) -> str:
    root, ext = os.path.splitext(str(model_path))
    return root + "_best" + (ext if ext else ".pth")


def _supports_exclude_frac() -> bool:
    """Return True if dataset.build_pairs_cache_from_sos_bank supports exclude_frac."""
    return ("exclude_frac" in getattr(build_pairs_cache_from_sos_bank, "__code__").co_varnames)


def train_model(args):
    device = torch.device(args.device)

    if args.output_dir is not None:
        set_output_folder(args.output_dir)

    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message("...................")

    _set_reproducibility(args.data_seed)

    # Defaults for bank/splits paths derived from data_path
    paths = make_cache_paths(args.data_path)
    if args.sos_bank_path is None:
        args.sos_bank_path = paths["sos_bank_path"]
    if args.splits_path is None:
        args.splits_path = paths["splits_path"]

    # A completed Stage 0 pairs cache is authoritative for Stage 1 and resume.
    # Do not reload the multi-GiB SoS bank merely to obtain N. In particular,
    # np.array(sos_bank["sos"]) could create another complete in-memory copy.
    pairs_cache_exists = os.path.isfile(args.data_path)
    sos_bank = None

    if pairs_cache_exists:
        t0 = time.time()
        cache_index_metadata = load_pairs_cache_metadata(args.data_path)
        N = int(cache_index_metadata.get("N", 0))
        if N <= 0:
            raise ValueError(
                "[train_initial_reconstruction.py] Existing pairs-cache index does not contain a valid "
                f"metadata['N']: {args.data_path}"
            )
        log_message(
            f"[train_initial_reconstruction.py] Existing pairs cache detected (N={N}). "
            "Skipping the full SoS-bank load for Stage 1/resume. "
            f"Index read time: {time.time() - t0:.2f}s"
        )
    else:
        t0 = time.time()
        sos_bank = get_or_make_sos_bank(
            sos_bank_path=args.sos_bank_path,
            make_if_missing=bool(args.make_sos_bank),
            num_samples=args.num_samples,
            nx=args.nx, ny=args.ny,
            phys_x=args.phys_x, phys_y=args.phys_y,
            sos_water=args.sos_water,
            sos_min=args.sos_min, sos_max=args.sos_max,
            max_shapes=args.max_shapes,
            min_shapes=args.min_shapes,
            radius=args.radius,
            n_emitters=args.n_emitters,
            n_receivers=args.n_receivers,
            data_seed=args.data_seed,
            use_random_support_envelope=bool(args.use_random_support_envelope),
            support_radius_min_frac=float(args.support_radius_min_frac),
            support_radius_max_frac=float(args.support_radius_max_frac),
            support_center_jitter_frac=float(args.support_center_jitter_frac),
            support_ellipse_prob=float(args.support_ellipse_prob),
            support_ellipse_axis_jitter_frac=float(args.support_ellipse_axis_jitter_frac),
            shape_mode=str(args.shape_mode),
            p_ellipse=float(args.p_ellipse),
            p_triangle=float(args.p_triangle),
            p_polygon=float(args.p_polygon),
            p_rod=float(args.p_rod),
            shape_size_min_px=int(args.shape_size_min_px),
            shape_size_max_px=int(args.shape_size_max_px),
            polygon_vertices_min=int(args.polygon_vertices_min),
            polygon_vertices_max=int(args.polygon_vertices_max),
        )
        log_message(f"[train_initial_reconstruction.py] SoS bank ready. Lapsed time: {time.time() - t0:.2f}s")
        N = int(sos_bank["sos"].shape[0])

    # Load/create splits (persistent)
    splits = get_or_make_splits(
        splits_path=args.splits_path,
        N=N,
        seed=int(args.data_seed),
        make_if_missing=bool(args.make_splits),
        frac_train=args.frac_train,
        frac_val=args.frac_val,
        extra={"sos_bank_path": str(args.sos_bank_path)},
    )
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    log_message(f"[train_initial_reconstruction.py] Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Ensure pairs cache exists at data_path (enables resume)
    if not os.path.exists(args.data_path):
        if not args.make_pairs_cache:
            raise FileNotFoundError(
                f"[train_initial_reconstruction.py] pairs cache missing: {args.data_path}\n"
                f"Run with --make_pairs_cache to build it from the SoS bank."
            )

        exclude_frac = float(getattr(args, "exclude_frac", 0.0))
        supports_excl = _supports_exclude_frac()
        if (exclude_frac > 0.0) and (not supports_excl):
            raise ValueError(
                "[train_initial_reconstruction.py] You requested --exclude_frac > 0, but your current src/dataset.py "
                "does not support exclude_frac in build_pairs_cache_from_sos_bank().\n"
                "Update dataset.py accordingly."
            )

        kwargs = dict(
            sos_bank=sos_bank,
            cache_path=args.data_path,
            include_pde=bool(args.include_pde),
            pde_emitters_k=int(args.pde_emitters_k),
            tof_norm=str(getattr(args, "tof_norm", "zscore")),
            tof_norm_eps=float(getattr(args, "tof_norm_eps", 1e-6)),
            store_collocation_T=bool(getattr(args, "store_collocation_T", False)),
            n_collocation=int(getattr(args, "n_collocation", 512)),
            colloc_emitters_k=int(getattr(args, "colloc_emitters_k", 4)),
            colloc_seed=int(getattr(args, "colloc_seed", 0)),
            synthetic_roi_tof_enable=bool(
                getattr(args, "synthetic_roi_tof_enable", False)
            ),
            synthetic_roi_center_x=getattr(args, "synthetic_roi_center_x", None),
            synthetic_roi_center_y=getattr(args, "synthetic_roi_center_y", None),
            synthetic_roi_radius_min=float(
                getattr(args, "synthetic_roi_radius_min", 0.055)
            ),
            synthetic_roi_radius_max=float(
                getattr(args, "synthetic_roi_radius_max", 0.070)
            ),
            synthetic_roi_outer_c_min=float(
                getattr(args, "synthetic_roi_outer_c_min", 1460.0)
            ),
            synthetic_roi_outer_c_max=float(
                getattr(args, "synthetic_roi_outer_c_max", 1500.0)
            ),
            synthetic_roi_reference_c=float(
                getattr(args, "synthetic_roi_reference_c", 1500.0)
            ),
            synthetic_roi_synthetic_c=float(
                getattr(args, "synthetic_roi_synthetic_c", 1500.0)
            ),
            synthetic_roi_min_chord_frac=float(
                getattr(args, "synthetic_roi_min_chord_frac", 0.05)
            ),
            synthetic_roi_residual_clip_us=float(
                getattr(args, "synthetic_roi_residual_clip_us", 8.0)
            ),
            synthetic_roi_nonintersect_mode=str(
                getattr(args, "synthetic_roi_nonintersect_mode", "reference")
            ),
            synthetic_roi_seed=int(
                getattr(
                    args,
                    "synthetic_roi_seed",
                    0,
                )
            ),
            chunk_size=int(
                getattr(
                    args,
                    "pairs_chunk_size",
                    2000,
                )
            ),
            resume_chunks=not bool(
                getattr(
                    args,
                    "no_resume_pairs_chunks",
                    False,
                )
            ),
            pairs_workers=int(
                getattr(
                    args,
                    "pairs_workers",
                    1,
                )
            ),
        )
        if supports_excl:
            kwargs["exclude_frac"] = float(exclude_frac)

        if sos_bank is None:
            raise RuntimeError(
                "[train_initial_reconstruction.py] Internal error: cache construction requires the SoS bank."
            )
        build_pairs_cache_from_sos_bank(**kwargs)

    # Cache-building commands are often invoked through train_initial_reconstruction.py with --epochs 1
    # and a dummy model path.  In that case the requested artifacts have already
    # been produced, and continuing into the training section can incorrectly
    # trigger training-time loss guards (for example the legacy-ToF guard).
    # This return is intentionally conservative: real training should be run as
    # a separate command after the cache exists.
    if bool(getattr(args, "make_pairs_cache", False)):
        log_message(
            "[train_initial_reconstruction.py] --make_pairs_cache completed. Exiting before model training. "
            "Run a separate training command using the generated cache."
        )
        return

    # Load pairs cache and subset by saved indices (no random_split)
    pairs_ds = PairsCacheDataset(
        args.data_path,
        sos_min=args.sos_min,
        sos_max=args.sos_max,
        return_tof_mask=bool(getattr(args, "use_tof_mask_channel", False)),
    )
    if bool(getattr(args, "use_differentiable_eikonal", False)):
        missing_colloc = not bool(
            getattr(
                pairs_ds,
                "has_collocation",
                False,
            )
        )
        if missing_colloc:
            raise ValueError(
                "[train_initial_reconstruction.py] --use_differentiable_eikonal requires a pairs cache with stored "
                "MSFM collocation targets. Rebuild the pairs cache with --store_collocation_T. "
                "This guard is intentional: otherwise the Eikonal surrogate can remain frozen "
                "or untrained while the ToF term is ramped into the reconstruction loss."
            )

    if float(getattr(args, "lambda_tof", 0.0)) > 0.0 and not bool(getattr(args, "use_differentiable_eikonal", False)):
        raise ValueError(
            "[train_initial_reconstruction.py] Refusing to apply --lambda_tof with the legacy straight-line loss. "
            "Use --use_differentiable_eikonal with a cache built using --store_collocation_T, "
            "or set --lambda_tof 0."
        )
    cache_md = getattr(pairs_ds, "metadata", {}) if hasattr(pairs_ds, "metadata") else {}
    roi_cache_enabled = bool(isinstance(cache_md, dict) and cache_md.get("synthetic_roi_tof", {}).get("enabled", False))
    if roi_cache_enabled and float(getattr(args, "lambda_tof", 0.0)) > 0.0:
        if not bool(
            getattr(
                pairs_ds,
                "has_raw_eikonal",
                False,
            )
        ):
            raise ValueError(
                "[train_initial_reconstruction.py] ROI-referenced synthetic ToF cache is active, "
                "but raw Eikonal ToF targets are missing. "
                "Rebuild the cache with synthetic ROI ToF enabled."
            )

        log_message(
            "[train_initial_reconstruction.py] ROI-referenced synthetic ToF cache detected. "
            "Inverse model input uses ROI-referenced ToF; "
            "2B Eikonal consistency uses raw_eikonal_tof_phys targets."
        )
        log_message(
            "[train_initial_reconstruction.py] ROI-referenced synthetic ToF cache detected. "
            "Inverse model input uses ROI-referenced ToF; 2B Eikonal consistency uses raw_eikonal_tof_phys targets."
        )
    train_ds = Subset(pairs_ds, list(map(int, train_idx)))
    val_ds = Subset(pairs_ds, list(map(int, val_idx)))
    test_ds = Subset(pairs_ds, list(map(int, test_idx)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    # Deterministic training-set evaluation loader.  This is deliberately
    # separate from train_loader: no shuffle protocol is needed and, more
    # importantly, no training-time stochastic augmentations/dropout are used
    # when computing TrainEval curves.
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model (supports --latent_res and model selection)
    # Choose which reconstruction model to instantiate based on --model_type.
    use_tof_mask_channel = bool(getattr(args, "use_tof_mask_channel", False))
    tof_feature_mode = str(getattr(args, "tof_feature_mode", "raw")).lower()
    acoustic_tof_channels = 3 if tof_feature_mode == "residual_stack" else 1
    tof_input_channels = acoustic_tof_channels + (1 if use_tof_mask_channel else 0)
    water_tof_norm = None
    raw_water_tof_norm = None
    if tof_feature_mode == "residual_stack":
        water_tof_norm = make_homogeneous_water_tof_norm(args, pairs_ds, device)
        raw_water_tof_norm = make_homogeneous_water_tof_norm_with_meta(
            args, getattr(pairs_ds, "raw_eikonal_tof_norm", getattr(pairs_ds, "tof_norm", None)), device
        )
        log_message(
            "[train_initial_reconstruction.py] ToF feature mode: residual_stack "
            f"(channels: raw, water_residual, double_centered_residual"
            f"{', mask' if use_tof_mask_channel else ''})."
        )
    else:
        log_message(
            "[train_initial_reconstruction.py] ToF feature mode: raw "
            f"(channels: raw{', mask' if use_tof_mask_channel else ''})."
        )
    if str(getattr(args, "tof_input_domain_mix", "roi_only")).lower() == "roi_raw":
        if not bool(
            getattr(
                pairs_ds,
                "has_raw_eikonal",
                False,
            )
        ):
            raise ValueError(
                "[train_initial_reconstruction.py] --tof_input_domain_mix roi_raw requires "
                "a cache containing tof_raw_eikonal. "
                "Rebuild the pairs cache with synthetic ROI ToF enabled."
            )

        log_message(
            "[train_initial_reconstruction.py] Dual-domain inverse input enabled: "
            "training batches may use ROI-referenced ToF or raw Eikonal ToF "
            f"(raw probability={float(getattr(args, 'tof_raw_input_prob', 0.0)):.3g}, "
            f"lambda_domain_consistency="
            f"{float(getattr(args, 'lambda_domain_consistency', 0.0)):.3g})."
        )
    mtype = getattr(args, "model_type", "baseline")
    if tof_feature_mode == "residual_stack" and mtype != "deep":
        raise ValueError("[train_initial_reconstruction.py] --tof_feature_mode residual_stack currently requires --model_type deep.")
    if mtype == "improved":
        model = ImprovedReconstructionNet(
            nx=args.nx,
            ny=args.ny,
            latent_res=int(args.latent_res),
            phys_x=float(args.phys_x),
            phys_y=float(args.phys_y),
            radius=float(args.radius),
            n_emitters=int(args.n_emitters),
            n_receivers=int(args.n_receivers),
            tof_input_channels=tof_input_channels,
        ).to(device)
        log_message("[train_initial_reconstruction.py] Using ImprovedReconstructionNet (gated).")
    elif mtype == "deep":
        model = DeepGatedReconstructionNet(
            nx=args.nx,
            ny=args.ny,
            latent_res=int(args.latent_res),
            phys_x=float(args.phys_x),
            phys_y=float(args.phys_y),
            radius=float(args.radius),
            n_emitters=int(args.n_emitters),
            n_receivers=int(args.n_receivers),
            tof_input_channels=tof_input_channels,
            fc_dropout=float(getattr(args, "model_fc_dropout", 0.08)),
            decoder_dropout=float(getattr(args, "model_decoder_dropout", 0.04)),
        ).to(device)
        # Avoid using a non-breaking hyphen (\u2011) in log messages.  Use a standard hyphen instead.
        log_message("[train_initial_reconstruction.py] Using DeepGatedReconstructionNet (U-Net style).")
    else:
        model = ReconstructionNet(
            nx=args.nx,
            ny=args.ny,
            latent_res=int(args.latent_res),
            phys_x=float(args.phys_x),
            phys_y=float(args.phys_y),
            radius=float(args.radius),
            n_emitters=int(args.n_emitters),
            n_receivers=int(args.n_receivers),
            tof_input_channels=tof_input_channels,
        ).to(device)
        log_message("[train_initial_reconstruction.py] Using baseline ReconstructionNet.")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(max(0.0, getattr(args, "weight_decay", 0.0))),
    )
    criterion = torch.nn.MSELoss()

    # Circular support mask. In normalized SoS space, pixels outside the support
    # are forced to the normalized water value. This makes the support constraint
    # part of training/evaluation, not only a visualization fix.
    use_circular_mask = bool(getattr(args, "use_circular_mask", False))
    mask_radius_eff = float(args.mask_radius) if getattr(args, "mask_radius", None) is not None else float(args.radius)
    circular_mask = None
    if use_circular_mask:
        circular_mask = make_circular_support_mask(
            int(args.nx), int(args.ny), float(args.phys_x), float(args.phys_y), mask_radius_eff, device=device
        )
        log_message(
            f"[train_initial_reconstruction.py] Circular support mask enabled. radius={mask_radius_eff:.6g} m, "
            f"outside set to sos_water={float(args.sos_water):.6g} m/s"
        )

    if bool(getattr(args, "use_sos_weighted_loss", False)):
        log_message(
            f"[train_initial_reconstruction.py] Weighted SoS loss enabled: base={float(args.sos_weight_water):.3g}, "
            f"contrast_gain={float(args.sos_weight_contrast_gain):.3g}, power={float(args.sos_weight_power):.3g}"
        )
    if use_tof_mask_channel:
        log_message("[train_initial_reconstruction.py] ToF validity mask enabled: mask gates ToF/prior only; it is not learned as an acoustic feature.")
    if bool(getattr(args, "tof_aug_enable", False)):
        log_message(
            f"[train_initial_reconstruction.py] ToF augmentation enabled: scale=[{float(args.tof_aug_scale_min):.3g},{float(args.tof_aug_scale_max):.3g}], "
            f"shift_std={float(args.tof_aug_shift_std):.3g}, noise_std={float(args.tof_aug_noise_std):.3g}"
        )

    # Optional differentiable Eikonal forward model.  When --lambda_tof > 0 this
    # surrogate is also the only allowed reconstruction-side ToF consistency
    # path; the old straight-line matrix loss is deliberately disabled above.
    use_fwd = bool(getattr(args, "use_differentiable_eikonal", False))
    if bool(getattr(args, "ring_quarter_aug_enable", False)):
        if use_fwd:
            log_message(
                "[train_initial_reconstruction.py] Ring quarter-turn augmentation requested but disabled for differentiable-Eikonal 2B "
                "because stored collocation travel-time targets are tied to the unrotated images."
            )
        else:
            log_message(
                f"[train_initial_reconstruction.py] Ring quarter-turn augmentation enabled for supervised 2A: "
                f"prob={float(getattr(args, 'ring_quarter_aug_prob', 0.75)):.3g}, shifts=0/90/180/270 degrees."
            )
    forward_model = None
    optimizer_forward = None
    emitters_t = receivers_t = None

    def _set_requires_grad(m: torch.nn.Module, flag: bool) -> None:
        for p in m.parameters():
            p.requires_grad_(flag)

    if use_fwd:
        dx = float(args.phys_x) / float(args.nx - 1)
        dy = float(args.phys_y) / float(args.ny - 1)
        emitters_np, receivers_np = generate_sensor_positions(args.nx, args.ny, dx, dy, args.radius, args.n_emitters, args.n_receivers)
        emitters_t = torch.tensor(emitters_np, dtype=torch.float32, device=device)
        receivers_t = torch.tensor(receivers_np, dtype=torch.float32, device=device)

        forward_model = EikonalForwardModel(nx=args.nx, ny=args.ny, dx=dx, dy=dy).to(device)
        lr_fwd = float(args.lr) if (getattr(args, "lr_forward", None) is None) else float(args.lr_forward)
        optimizer_forward = optim.Adam(forward_model.parameters(), lr=lr_fwd)
        log_message(f"[train_initial_reconstruction.py] Differentiable forward model enabled. lr_forward={lr_fwd}")

    train_hist, val_hist = [], []
    # train_hist is the stochastic optimization objective actually backpropagated
    # during training.  train_eval_hist is the same deterministic evaluation
    # protocol used for validation, run on the training split.
    train_eval_hist = []
    train_sos_hist, val_sos_hist = [], []
    train_eval_sos_hist = []
    train_detail_hist, train_eval_detail_hist, val_detail_hist = [], [], []
    start_epoch = 0
    # For two-stage training, epoch numbering should continue after --resume,
    # but ToF warmup/pretrain/ramp must be measured from the beginning of the
    # Eikonal fine-tuning stage, not from the absolute supervised epoch count.
    tof_schedule_start_epoch = 0

    # Graceful Ctrl+C (no try/except): stop after current epoch
    interrupted_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        interrupted_flag["stop"] = True
        log_message("[train_initial_reconstruction.py] Ctrl+C detected. Will stop after current epoch and save.")

    signal.signal(signal.SIGINT, _sigint_handler)

    # Resume
    if (args.resume or args.mode == 'test') and os.path.exists(args.model_path):
        checkpoint_t0 = time.time()
        log_message(f"[train_initial_reconstruction.py] Loading resume checkpoint on CPU: {args.model_path}")
        # CPU staging avoids a transient CUDA copy of the complete checkpoint.
        # load_state_dict preserves every stored value and dtype while moving
        # model and optimizer tensors to their parameter devices.
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        log_message(
            f"[train_initial_reconstruction.py] Checkpoint read completed in {time.time() - checkpoint_t0:.2f}s"
        )
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ck_cfg = checkpoint.get('train_config', {})
            ck_latent = ck_cfg.get('latent_res', None)
            if ck_latent is not None and int(ck_latent) != int(args.latent_res):
                raise ValueError(
                    f"Checkpoint latent_res={ck_latent} but current --latent_res={args.latent_res}. "
                    f"Resume/test requires matching latent_res."
                )

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for pg in optimizer.param_groups:
                pg['lr'] = float(args.lr)
                pg['weight_decay'] = float(max(0.0, getattr(args, "weight_decay", 0.0)))
            log_message(
                f"[train_initial_reconstruction.py] Optimizer reset after resume: lr={float(args.lr):.6g}, "
                f"weight_decay={float(max(0.0, getattr(args, 'weight_decay', 0.0))):.6g}"
            )

            if use_fwd:
                loaded_forward = False
                if forward_model is not None and 'forward_model_state_dict' in checkpoint:
                    ck_fwd_md = checkpoint.get('forward_model_metadata', None)
                    if hasattr(forward_model, "is_compatible_metadata") and forward_model.is_compatible_metadata(ck_fwd_md):
                        forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
                        loaded_forward = True
                    else:
                        log_message(
                            "[train_initial_reconstruction.py] Forward-surrogate checkpoint is missing or incompatible metadata; "
                            "keeping the freshly initialized scaled Eikonal surrogate."
                        )
                if loaded_forward and optimizer_forward is not None and 'optimizer_forward_state_dict' in checkpoint:
                    optimizer_forward.load_state_dict(checkpoint['optimizer_forward_state_dict'])
                    lr_fwd = float(args.lr) if (getattr(args, "lr_forward", None) is None) else float(args.lr_forward)
                    for pg in optimizer_forward.param_groups:
                        pg['lr'] = lr_fwd
                    log_message(f"[train_initial_reconstruction.py] Forward optimizer learning rate reset after resume to lr_forward={lr_fwd:.6g}")

            train_hist = checkpoint.get('train_history', [])
            val_hist = checkpoint.get('val_history', [])
            comp_hist = checkpoint.get('component_history', {}) if isinstance(checkpoint.get('component_history', {}), dict) else {}
            train_eval_hist = list(comp_hist.get('train_eval_total_history', comp_hist.get('train_total_history', [])))
            train_sos_hist = list(comp_hist.get('train_opt_sos_history', comp_hist.get('train_sos_history', [])))
            train_eval_sos_hist = list(comp_hist.get('train_eval_sos_history', comp_hist.get('train_sos_history', [])))
            val_sos_hist = list(comp_hist.get('val_sos_history', []))
            train_detail_hist = list(comp_hist.get('train_opt_detail_history', []))
            train_eval_detail_hist = list(comp_hist.get('train_eval_detail_history', []))
            val_detail_hist = list(comp_hist.get('val_detail_history', []))
            if len(train_eval_hist) != len(train_hist):
                train_eval_hist = [float('nan')] * len(train_hist)
            if len(train_sos_hist) != len(train_hist):
                train_sos_hist = [float('nan')] * len(train_hist)
            if len(train_eval_sos_hist) != len(train_hist):
                train_eval_sos_hist = [float('nan')] * len(train_hist)
            if len(val_sos_hist) != len(val_hist):
                val_sos_hist = [float('nan')] * len(val_hist)
            if len(train_detail_hist) != len(train_hist):
                train_detail_hist = [float('nan')] * len(train_hist)
            if len(train_eval_detail_hist) != len(train_hist):
                train_eval_detail_hist = [float('nan')] * len(train_hist)
            if len(val_detail_hist) != len(val_hist):
                val_detail_hist = [float('nan')] * len(val_hist)
            start_epoch = len(train_hist)
            # If 2B starts by copying a supervised 2A checkpoint, keep absolute
            # epoch numbering but start the ToF schedule from this resume point.
            # If resuming an already-running 2B checkpoint, reuse the saved
            # schedule origin so warmup/ramp continue correctly.
            if use_fwd and float(getattr(args, "lambda_tof", 0.0)) > 0.0:
                ck_lambda_tof = float(ck_cfg.get('lambda_tof', 0.0) or 0.0)
                ck_use_fwd = bool(ck_cfg.get('use_differentiable_eikonal', False))
                saved_sched0 = checkpoint.get('tof_schedule_start_epoch', None)
                if saved_sched0 is None and isinstance(checkpoint.get('dataset_metadata', None), dict):
                    saved_sched0 = checkpoint['dataset_metadata'].get('tof_schedule_start_epoch', None)
                if ck_use_fwd and ck_lambda_tof > 0.0 and saved_sched0 is not None:
                    tof_schedule_start_epoch = int(saved_sched0)
                else:
                    tof_schedule_start_epoch = int(start_epoch)
            log_message(
                f"[train_initial_reconstruction.py] Resuming from epoch {start_epoch}; "
                f"ToF schedule origin epoch={tof_schedule_start_epoch}"
            )
        else:
            model.load_state_dict(checkpoint)

    stage2b_adapter_info = None
    if (
        args.mode == "train"
        and bool(getattr(args, "stage2b_residual_adapter", False))
        and bool(getattr(args, "resume", False))
        and use_fwd
        and float(getattr(args, "lambda_tof", 0.0)) > 0.0
    ):
        stage2b_adapter_info = configure_stage2b_residual_adapter(model, args)
        log_message(
            "[train_initial_reconstruction.py] Stage-2B residual adapter enabled: "
            f"trainable_prefixes={stage2b_adapter_info['prefixes']}, "
            f"trainable_params={stage2b_adapter_info['trainable_param_count']}, "
            f"frozen_params={stage2b_adapter_info['frozen_param_count']}."
        )
        shown = ", ".join(stage2b_adapter_info["trainable_names"][:12])
        if len(stage2b_adapter_info["trainable_names"]) > 12:
            shown += ", ..."
        log_message(f"[train_initial_reconstruction.py] Stage-2B trainable parameter names: {shown}")

    setattr(args, "tof_schedule_start_epoch", int(tof_schedule_start_epoch))

    ema_model = None
    use_ema = bool(getattr(args, "use_ema", False))
    ema_decay = float(getattr(args, "ema_decay", 0.995))
    if use_ema and args.mode == 'train':
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p_ema in ema_model.parameters():
            p_ema.requires_grad_(False)
        log_message(f"[train_initial_reconstruction.py] EMA validation/checkpoint model enabled: decay={ema_decay:.6g}.")
    elif args.mode == 'train':
        log_message("[train_initial_reconstruction.py] EMA validation/checkpoint model disabled.")

    if args.mode == 'train':
        last_fwd_supT_mean = float("inf")
        last_fwd_pde_mean = float("inf")
        last_fwd_total_mean = float("inf")
        last_fwd_receiver_rmse_us = float("inf")
        last_fwd_receiver_residual_rmse_us = float("inf")
        best_forward_metric_value = float("inf")
        best_forward_epoch = None
        best_forward_state = None
        restored_best_forward_for_coupling = False
        consecutive_zero_fwd_grad = 0
        selection_metric_name = str(getattr(args, "early_stop_metric", "val_sos"))
        best_val = min(val_hist) if len(val_hist) > 0 else float("inf")
        if isinstance(locals().get("checkpoint", None), dict):
            ck_meta = checkpoint.get("dataset_metadata", {}) or {}
            ck_best_metric = checkpoint.get("best_selection_metric", ck_meta.get("best_selection_metric", None))
            ck_best_value = checkpoint.get("best_selection_value", ck_meta.get("best_selection_value", None))
            ck_best_sos = checkpoint.get("best_val_sos", ck_meta.get("best_val_sos", None))
            ck_best_total = checkpoint.get("best_val_total", ck_meta.get("best_val_total", None))
            if selection_metric_name in ("val_sos", "val_sos_detail", "val_detail"):
                if ck_best_sos is not None:
                    best_val = float(ck_best_sos)
                elif ck_best_metric == "val_sos" and ck_best_value is not None:
                    best_val = float(ck_best_value)
            elif selection_metric_name == "val_total":
                if ck_best_total is not None:
                    best_val = float(ck_best_total)
                elif ck_best_metric == "val_total" and ck_best_value is not None:
                    best_val = float(ck_best_value)
        best_path = best_checkpoint_path(str(args.model_path))
        early_stop_patience = int(max(0, getattr(args, "early_stop_patience", 0)))
        early_stop_min_delta = float(max(0.0, getattr(args, "early_stop_min_delta", 0.0)))
        epochs_without_improvement = 0
        teacher_model = None
        if (
            bool(getattr(args, "resume", False))
            and use_fwd
            and float(getattr(args, "lambda_tof", 0.0)) > 0.0
            and float(getattr(args, "lambda_teacher", 0.0)) > 0.0
        ):
            teacher_model = copy.deepcopy(model).to(device)
            teacher_model.eval()
            for p_teacher in teacher_model.parameters():
                p_teacher.requires_grad_(False)
            log_message(
                f"[train_initial_reconstruction.py] 2B teacher/proximal anchor enabled: "
                f"lambda_teacher={float(getattr(args, 'lambda_teacher', 0.0)):.6g}. "
                "The resumed 2A-best model is kept frozen as the teacher."
            )
        log_message(
            f"[train_initial_reconstruction.py] Best-checkpoint and early-stopping metric: {selection_metric_name}. "
            f"Initial best metric value={float(best_val):.6f}. "
            "Use val_sos_detail for detail-aware 2A; use val_sos for 2B."
        )
        log_message(
            "[train_initial_reconstruction.py] Curve protocol: Train backprop is the stochastic optimization loss; "
            "Only deterministic Train eval SoS and Validation eval SoS are plotted. "
            "Train backprop and full objectives are logged but not plotted."
        )

        def _eval_inverse_split(eval_model_for_split, loader_for_split):
            """Deterministic inverse-model evaluation used for both TrainEval and ValEval."""
            eval_model_for_split.eval()
            if teacher_model is not None:
                teacher_model.eval()
            total_sum = sos_sum_e = grad_sum_e = contrast_sum_e = detail_sum_e = lap_sum_e = 0.0
            teacher_sum_e = delta_barrier_sum_e = domain_sum_e = tof_sum_e = 0.0
            teacher_count_e = delta_barrier_count_e = domain_count_e = tof_count_e = 0
            with torch.no_grad():
                for eval_batch_idx, ebatch in enumerate(loader_for_split):
                    s, t, tmask, _p, _c, raw_physics_eval = unpack_reconstruction_batch(ebatch)
                    s = s.to(device)
                    t = t.to(device)
                    if tmask is not None:
                        tmask = tmask.to(device)
                    raw_eval_tof_phys = None
                    raw_eval_tof_norm_meta = None
                    if raw_physics_eval is not None:
                        raw_eval_tof_phys = raw_physics_eval["raw_eikonal_tof_phys"].to(device)
                        raw_eval_tof_norm_meta = raw_physics_eval.get("raw_eikonal_tof_norm", None)

                    eval_input = make_model_input(t, tmask, use_tof_mask_channel, tof_feature_mode, water_tof_norm)
                    eval_pred = eval_model_for_split(eval_input).view_as(s)
                    eval_target = s
                    if use_circular_mask:
                        eval_pred = apply_circular_support_to_normalized_sos(
                            eval_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                        eval_target = apply_circular_support_to_normalized_sos(
                            s, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                    eval_sos_loss = sos_reconstruction_loss(eval_pred, eval_target, args)
                    eval_grad_loss = gradient_consistency_loss(eval_pred, eval_target)
                    eval_contrast_loss = multiscale_local_contrast_loss(eval_pred, eval_target)
                    eval_detail_loss = detail_band_loss(eval_pred, eval_target, args)
                    eval_lap_loss = laplacian_detail_loss(eval_pred, eval_target)
                    eval_total_loss = (
                        eval_sos_loss
                        + float(getattr(args, "lambda_grad", 0.0)) * eval_grad_loss
                        + float(getattr(args, "lambda_contrast", 0.0)) * eval_contrast_loss
                        + float(getattr(args, "lambda_detail", 0.0)) * eval_detail_loss
                        + float(getattr(args, "lambda_laplacian", 0.0)) * eval_lap_loss
                    )

                    eval_teacher_loss = None
                    eval_delta_barrier_loss = None
                    if teacher_model is not None:
                        teacher_eval_pred = teacher_model(eval_input).view_as(s)
                        if use_circular_mask:
                            teacher_eval_pred = apply_circular_support_to_normalized_sos(
                                teacher_eval_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                        eval_teacher_loss = torch.mean((eval_pred - teacher_eval_pred.detach()) ** 2)
                        eval_total_loss = eval_total_loss + float(getattr(args, "lambda_teacher", 0.0)) * eval_teacher_loss
                        lambda_delta_barrier = float(getattr(args, "lambda_delta_barrier", 0.0))
                        delta_limit_mps = float(getattr(args, "stage2b_delta_limit_mps", 35.0))
                        if lambda_delta_barrier > 0.0 and delta_limit_mps > 0.0:
                            sos_range = float(args.sos_max) - float(args.sos_min)
                            limit_norm = delta_limit_mps / max(sos_range, 1e-12)
                            eval_excess = torch.relu(torch.abs(eval_pred - teacher_eval_pred.detach()) - limit_norm)
                            eval_delta_barrier_loss = torch.mean(eval_excess * eval_excess)
                            eval_total_loss = eval_total_loss + lambda_delta_barrier * eval_delta_barrier_loss

                    eval_domain_consistency_loss = None
                    lambda_domain_consistency = float(getattr(args, "lambda_domain_consistency", 0.0))
                    if lambda_domain_consistency > 0.0 and raw_eval_tof_phys is not None:
                        raw_eval_norm_for_consistency = normalize_physical_tof_like_dataset(
                            raw_eval_tof_phys, raw_eval_tof_norm_meta
                        ).to(device=t.device, dtype=t.dtype)
                        if tmask is not None:
                            raw_eval_norm_for_consistency = raw_eval_norm_for_consistency * tmask.to(
                                device=t.device, dtype=t.dtype
                            )
                        roi_eval_input_for_consistency = make_model_input(
                            t, tmask, use_tof_mask_channel, tof_feature_mode, water_tof_norm
                        )
                        raw_eval_input_for_consistency = make_model_input(
                            raw_eval_norm_for_consistency,
                            tmask,
                            use_tof_mask_channel,
                            tof_feature_mode,
                            raw_water_tof_norm if raw_water_tof_norm is not None else water_tof_norm,
                        )
                        roi_eval_pred = eval_model_for_split(roi_eval_input_for_consistency).view_as(s)
                        raw_eval_pred = eval_model_for_split(raw_eval_input_for_consistency).view_as(s)
                        if use_circular_mask:
                            roi_eval_pred = apply_circular_support_to_normalized_sos(
                                roi_eval_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                            raw_eval_pred = apply_circular_support_to_normalized_sos(
                                raw_eval_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                        eval_domain_consistency_loss = torch.mean((roi_eval_pred - raw_eval_pred) ** 2)
                        eval_total_loss = eval_total_loss + lambda_domain_consistency * eval_domain_consistency_loss

                    total_sum += float(eval_total_loss.item())
                    sos_sum_e += float(eval_sos_loss.item())
                    grad_sum_e += float(eval_grad_loss.item())
                    contrast_sum_e += float(eval_contrast_loss.item())
                    detail_sum_e += float(eval_detail_loss.item())
                    lap_sum_e += float(eval_lap_loss.item())
                    if eval_teacher_loss is not None:
                        teacher_sum_e += float(eval_teacher_loss.item())
                        teacher_count_e += 1
                    if eval_delta_barrier_loss is not None:
                        delta_barrier_sum_e += float(eval_delta_barrier_loss.item())
                        delta_barrier_count_e += 1
                    if eval_domain_consistency_loss is not None:
                        domain_sum_e += float(eval_domain_consistency_loss.item())
                        domain_count_e += 1

            denom = max(1, len(loader_for_split))
            return {
                "total": total_sum / denom,
                "sos": sos_sum_e / denom,
                "grad": grad_sum_e / denom,
                "contrast": contrast_sum_e / denom,
                "detail": detail_sum_e / denom,
                "laplacian": lap_sum_e / denom,
                "teacher": (teacher_sum_e / float(max(1, teacher_count_e))) if teacher_count_e > 0 else 0.0,
                "delta_barrier": (delta_barrier_sum_e / float(max(1, delta_barrier_count_e))) if delta_barrier_count_e > 0 else 0.0,
                "domain": (domain_sum_e / float(max(1, domain_count_e))) if domain_count_e > 0 else 0.0,
                "tof": (tof_sum_e / float(max(1, tof_count_e))) if tof_count_e > 0 else 0.0,
            }

        reconstruction_update_epochs = 0
        physics_coupled_epochs = 0
        forced_coupling_epochs = 0
        if args.resume and len(val_hist) > 0 and not os.path.exists(best_path):
            meta = dict(pairs_ds.metadata) if isinstance(pairs_ds.metadata, dict) else {}
            meta["sos_bank_path"] = str(args.sos_bank_path)
            meta["splits_path"] = str(args.splits_path)
            meta["pairs_cache_path"] = str(args.data_path)
            meta["tof_schedule_start_epoch"] = int(tof_schedule_start_epoch)
            if isinstance(locals().get("checkpoint", None), dict):
                ck_meta = checkpoint.get("dataset_metadata", {}) or {}
                meta["best_epoch"] = int(checkpoint.get("best_epoch", ck_meta.get("best_epoch", np.argmin(np.asarray(val_hist, dtype=np.float64)) + 1)))
                if checkpoint.get("best_val_total", ck_meta.get("best_val_total", None)) is not None:
                    meta["best_val_total"] = float(checkpoint.get("best_val_total", ck_meta.get("best_val_total")))
                if checkpoint.get("best_val_sos", ck_meta.get("best_val_sos", None)) is not None:
                    meta["best_val_sos"] = float(checkpoint.get("best_val_sos", ck_meta.get("best_val_sos")))
            else:
                meta["best_epoch"] = int(np.argmin(np.asarray(val_hist, dtype=np.float64)) + 1)
                meta["best_val_total"] = float(best_val)
            meta["best_selection_metric"] = selection_metric_name
            meta["best_selection_value"] = float(best_val)
            meta["validation_weights"] = "ema" if ema_model is not None else "raw"
            meta["ema_decay"] = float(ema_decay) if ema_model is not None else None
            component_history = {
                "train_opt_total_history": list(train_hist),
                "train_eval_total_history": list(train_eval_hist),
                "val_total_history": list(val_hist),
                "train_opt_sos_history": list(train_sos_hist),
                "train_eval_sos_history": list(train_eval_sos_hist),
                "val_sos_history": list(val_sos_hist),
                "train_opt_detail_history": list(train_detail_hist),
                "train_eval_detail_history": list(train_eval_detail_hist),
                "val_detail_history": list(val_detail_hist),
            }
            save_checkpoint(
                ema_model if ema_model is not None else model, optimizer, train_hist, val_hist, args, meta,
                forward_model=forward_model if use_fwd else None,
                optimizer_forward=optimizer_forward if use_fwd else None,
                path=best_path,
                component_history=component_history,
            )
            log_message(
                f"[train_initial_reconstruction.py] Initialized missing best checkpoint from resumed model: "
                f"{best_path} (initial best metric={best_val:.6f})"
            )

        # The checkpoint dictionary still contains complete CPU copies of the
        # model and Adam state after load_state_dict has finished. Keeping that
        # dictionary alive throughout training is unnecessary and can consume
        # several additional GiB. Histories and scalar metadata used below have
        # already been copied out.
        if "checkpoint" in locals():
            del checkpoint
        if "comp_hist" in locals():
            del comp_hist
        if "ck_cfg" in locals():
            del ck_cfg
        if "ck_meta" in locals():
            del ck_meta
        _release_transient_memory("resume checkpoint payload")

        if int(start_epoch) >= int(args.epochs):
            msg = (
                f"[train_initial_reconstruction.py] No training epochs would run: checkpoint has start_epoch={start_epoch}, "
                f"but --epochs={args.epochs}. Increase --epochs above {start_epoch} for a resumed "
                "stage, or pass --allow_noop_resume only if this no-op is intentional."
            )
            if bool(getattr(args, "allow_noop_resume", False)):
                log_message("WARNING: " + msg)
            else:
                raise ValueError(msg)
        training_loop_start = time.time()
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            tof_schedule_epoch = max(0, int(epoch) - int(tof_schedule_start_epoch))
            lam_tof_sched = effective_lambda_tof(tof_schedule_epoch, args)
            lam_tof_eff = lam_tof_sched
            protected_forward_only_epochs = (
                int(getattr(args, "tof_warmup_epochs", 0))
                + int(getattr(args, "tof_fwd_pretrain_epochs", 0))
                + int(max(0, getattr(args, "tof_gate_extra_forward_only_epochs", 0)))
            )
            explicit_forward_only_epochs = int(getattr(args, "tof_forward_only_epochs", -1))
            forward_only_limit_epochs = (
                explicit_forward_only_epochs if explicit_forward_only_epochs >= 0 else protected_forward_only_epochs
            )
            force_after_forward_only = (
                use_fwd
                and bool(getattr(args, "force_tof_coupling_after_forward_only", False))
                and float(getattr(args, "lambda_tof", 0.0)) > 0.0
                and int(tof_schedule_epoch) >= int(forward_only_limit_epochs)
            )
            gate_supT_thr = float(getattr(args, "tof_gate_forward_supT_threshold", 0.0))
            gate_pde_thr = float(getattr(args, "tof_gate_forward_pde_threshold", 0.0))
            gate_total_thr = float(getattr(args, "tof_gate_forward_total_threshold", 0.0))
            gate_receiver_rmse_thr = float(getattr(args, "tof_gate_receiver_rmse_us_threshold", 0.0))
            gate_receiver_resid_rmse_thr = float(getattr(args, "tof_gate_receiver_residual_rmse_us_threshold", 0.0))
            gate_enabled = (
                use_fwd
                and (not bool(getattr(args, "disable_tof_forward_gate", False)))
                and (
                    gate_supT_thr > 0.0
                    or gate_pde_thr > 0.0
                    or gate_total_thr > 0.0
                    or gate_receiver_rmse_thr > 0.0
                    or gate_receiver_resid_rmse_thr > 0.0
                )
                and (lam_tof_sched > 0.0 or force_after_forward_only)
            )
            gate_reasons = []
            gate_closed_this_epoch = False
            if gate_enabled:
                if gate_supT_thr > 0.0 and not (last_fwd_supT_mean <= gate_supT_thr):
                    gate_reasons.append(f"supT={last_fwd_supT_mean:.6g}>{gate_supT_thr:.6g}")
                if gate_pde_thr > 0.0 and not (last_fwd_pde_mean <= gate_pde_thr):
                    gate_reasons.append(f"PDE={last_fwd_pde_mean:.6g}>{gate_pde_thr:.6g}")
                if gate_total_thr > 0.0 and not (last_fwd_total_mean <= gate_total_thr):
                    gate_reasons.append(f"Tot={last_fwd_total_mean:.6g}>{gate_total_thr:.6g}")
                if gate_receiver_rmse_thr > 0.0 and not (last_fwd_receiver_rmse_us <= gate_receiver_rmse_thr):
                    gate_reasons.append(
                        f"receiverRMSE_us={last_fwd_receiver_rmse_us:.6g}>{gate_receiver_rmse_thr:.6g}"
                    )
                if gate_receiver_resid_rmse_thr > 0.0 and not (
                    last_fwd_receiver_residual_rmse_us <= gate_receiver_resid_rmse_thr
                ):
                    gate_reasons.append(
                        f"receiverResidualRMSE_us={last_fwd_receiver_residual_rmse_us:.6g}>"
                        f"{gate_receiver_resid_rmse_thr:.6g}"
                    )
            if gate_enabled and gate_reasons:
                gate_closed_this_epoch = True
                lam_tof_eff = 0.0
                log_message(
                    f"[train_initial_reconstruction.py] Delaying ToF coupling at epoch {epoch+1}: "
                    f"scheduled lamToF={lam_tof_sched:.6g}; forward surrogate not ready: "
                    + ", ".join(gate_reasons)
                )

            if force_after_forward_only and lam_tof_eff <= 0.0:
                min_frac = float(max(0.0, min(1.0, getattr(args, "tof_min_coupling_after_unfreeze", 0.15))))
                forced_lambda = float(getattr(args, "lambda_tof", 0.0)) * min_frac
                if forced_lambda > 0.0:
                    if gate_closed_this_epoch:
                        if bool(getattr(args, "unsafe_force_tof_coupling_before_receiver_gate", False)):
                            lam_tof_eff = forced_lambda
                            forced_coupling_epochs += 1
                            log_message(
                                f"[train_initial_reconstruction.py] UNSAFE forced ToF coupling at epoch {epoch+1}: "
                                f"forward gate is closed ({', '.join(gate_reasons)}), but the explicit "
                                f"override was requested. Using lamToF_eff={lam_tof_eff:.6g}."
                            )
                            gate_closed_this_epoch = False
                        else:
                            log_message(
                                f"[train_initial_reconstruction.py] Refusing forced ToF coupling at epoch {epoch+1}: "
                                f"forward gate is still closed ({', '.join(gate_reasons)}). "
                                "The reconstruction model remains protected."
                            )
                    else:
                        lam_tof_eff = forced_lambda
                        log_message(
                            f"[train_initial_reconstruction.py] Starting gentle ToF coupling at epoch {epoch+1}: "
                            f"forward-only limit of {forward_only_limit_epochs} schedule epochs elapsed "
                            f"while the schedule/ramp was still zero. Using lamToF_eff={lam_tof_eff:.6g}."
                        )
            if (
                gate_closed_this_epoch
                and bool(getattr(args, "stop_2b_if_tof_gate_closed", False))
                and int(tof_schedule_epoch) >= protected_forward_only_epochs
            ):
                log_message(
                    f"[train_initial_reconstruction.py] Stopping guarded Stage 2B before epoch {epoch+1}: "
                    "scheduled ToF coupling is due, but the forward surrogate failed the readiness gate "
                    f"({', '.join(gate_reasons)}). The resumed/best 2A checkpoint is preserved at {best_path}. "
                    "No reconstruction update is applied for this failed 2B epoch."
                )
                interrupted_flag["stop"] = True
                break

            if bool(getattr(args, "skip_forward_during_tof_warmup", False)) and (tof_schedule_epoch < int(getattr(args, "tof_warmup_epochs", 0))) and (lam_tof_eff == 0.0) and use_fwd and forward_model is not None:
                log_message(f"[train_initial_reconstruction.py] Warmup epoch {epoch+1} (2B schedule epoch {tof_schedule_epoch+1}): lamToF_eff=0 -> skipping forward-model training steps to save time. This is not recommended for production training.")

            # During Eikonal-surrogate calibration, a failed receiver-ToF
            # readiness gate keeps the reconstruction network frozen and the
            # epoch in forward-only mode.
            forward_only_phase = (
                use_fwd
                and forward_model is not None
                and float(getattr(args, "lambda_tof", 0.0)) > 0.0
                and lam_tof_eff == 0.0
                and (tof_schedule_epoch < forward_only_limit_epochs or gate_closed_this_epoch)
            )
            if forward_only_phase:
                model.eval()
                if gate_closed_this_epoch:
                    log_message(
                        f"[train_initial_reconstruction.py] 2B forward-only phase at epoch {epoch+1} "
                        f"(tof_sched_epoch={tof_schedule_epoch+1}): receiver/collocation readiness gate is closed; "
                        "freezing reconstruction model and continuing to train only the differentiable Eikonal surrogate."
                    )
                else:
                    log_message(
                        f"[train_initial_reconstruction.py] 2B forward-only phase at epoch {epoch+1} "
                        f"(tof_sched_epoch={tof_schedule_epoch+1}): freezing reconstruction model; "
                        "training only the differentiable Eikonal surrogate."
                    )
            else:
                model.train()
                if use_fwd and forward_model is not None and float(getattr(args, "lambda_tof", 0.0)) > 0.0:
                    if (
                        bool(getattr(args, "restore_best_forward_on_coupling", False))
                        and not restored_best_forward_for_coupling
                        and best_forward_state is not None
                    ):
                        forward_model.load_state_dict(best_forward_state)
                        restored_best_forward_for_coupling = True
                        log_message(
                            f"[train_initial_reconstruction.py] Restored best forward surrogate before 2B coupling: "
                            f"epoch={best_forward_epoch}, "
                            f"{getattr(args, 'forward_best_metric', 'receiver_residual_rmse')}="
                            f"{best_forward_metric_value:.6g} us."
                        )
                    reconstruction_update_epochs += 1
                    if lam_tof_eff > 0.0:
                        physics_coupled_epochs += 1

            epoch_loss = 0.0

            # loss accumulators for logging
            sos_sum = 0.0
            grad_sum = 0.0
            contrast_sum = 0.0
            detail_sum = 0.0
            lap_sum = 0.0
            teacher_sum = 0.0
            teacher_count = 0
            delta_barrier_sum = 0.0
            delta_barrier_count = 0
            domain_consistency_sum = 0.0
            domain_consistency_count = 0
            raw_input_batches = 0
            roi_input_batches = 0
            tof_sum = 0.0
            tof_count = 0

            fwd_tot_sum = 0.0
            fwd_pde_sum = 0.0
            fwd_bc_sum = 0.0
            fwd_supT_sum = 0.0
            fwd_count = 0
            fwd_receiver_abs_sum = 0.0
            fwd_receiver_residual_sum = 0.0
            fwd_receiver_count = 0
            fwd_grad_norm_sum = 0.0
            fwd_grad_count = 0

            for batch_idx, batch in enumerate(train_loader):
                sos_b, tof_b, tof_mask_b, _pde_b, colloc, raw_physics = unpack_reconstruction_batch(batch)

                sos_b = sos_b.to(device)
                tof_b = tof_b.to(device)
                if tof_mask_b is not None:
                    tof_mask_b = tof_mask_b.to(device)
                raw_tof_phys = None
                raw_tof_norm_meta = None
                if raw_physics is not None:
                    raw_tof_phys = raw_physics["raw_eikonal_tof_phys"].to(device)
                    raw_tof_norm_meta = raw_physics.get("raw_eikonal_tof_norm", None)
                sos_b, tof_b, tof_mask_b = apply_ring_quarter_augmentation(
                    sos_b,
                    tof_b,
                    tof_mask_b,
                    args,
                    allow=(not use_fwd),
                )

                if not forward_only_phase:
                    optimizer.zero_grad(set_to_none=True)

                # Main reconstruction loss: augmented normalized ToF -> SoS.
                # During the forward-only phase this is computed only for logging,
                # without autograd and without updating the reconstruction model.
                tof_domain_base, tof_domain_name = choose_inverse_tof_domain(
                    tof_b,
                    tof_mask_b,
                    raw_tof_phys,
                    raw_tof_norm_meta,
                    args,
                    training=True,
                )
                if tof_domain_name == "raw":
                    raw_input_batches += 1
                    water_for_input = raw_water_tof_norm if raw_water_tof_norm is not None else water_tof_norm
                else:
                    roi_input_batches += 1
                    water_for_input = water_tof_norm

                tof_in = augment_normalized_tof_for_training(tof_domain_base, tof_mask_b, args)
                model_input = make_model_input(
                    tof_in, tof_mask_b, use_tof_mask_channel, tof_feature_mode, water_for_input
                )
                if not forward_only_phase:
                    model_input = apply_tof_feature_dropout(model_input, use_tof_mask_channel, args)
                domain_consistency_loss = None
                delta_barrier_loss = None
                with torch.set_grad_enabled(not forward_only_phase):
                    pred = model(model_input)
                    pred_view = pred.view_as(sos_b)
                    sos_target = sos_b
                    if use_circular_mask:
                        pred_view = apply_circular_support_to_normalized_sos(
                            pred_view, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                        sos_target = apply_circular_support_to_normalized_sos(
                            sos_b, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                    sos_loss = sos_reconstruction_loss(pred_view, sos_target, args)
                    grad_loss = gradient_consistency_loss(pred_view, sos_target)
                    contrast_loss = multiscale_local_contrast_loss(pred_view, sos_target)
                    detail_loss = detail_band_loss(pred_view, sos_target, args)
                    lap_loss = laplacian_detail_loss(pred_view, sos_target)
                    total_loss = (
                        sos_loss
                        + float(getattr(args, "lambda_grad", 0.0)) * grad_loss
                        + float(getattr(args, "lambda_contrast", 0.0)) * contrast_loss
                        + float(getattr(args, "lambda_detail", 0.0)) * detail_loss
                        + float(getattr(args, "lambda_laplacian", 0.0)) * lap_loss
                    )

                    teacher_loss = None
                    if teacher_model is not None and not forward_only_phase:
                        with torch.no_grad():
                            teacher_pred = teacher_model(model_input).view_as(sos_b)
                            if use_circular_mask:
                                teacher_pred = apply_circular_support_to_normalized_sos(
                                    teacher_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                                )
                        teacher_loss = torch.mean((pred_view - teacher_pred.detach()) ** 2)
                        total_loss = total_loss + float(getattr(args, "lambda_teacher", 0.0)) * teacher_loss

                        delta_barrier_loss = None
                        lambda_delta_barrier = float(getattr(args, "lambda_delta_barrier", 0.0))
                        delta_limit_mps = float(getattr(args, "stage2b_delta_limit_mps", 35.0))
                        if lambda_delta_barrier > 0.0 and delta_limit_mps > 0.0:
                            sos_range = float(args.sos_max) - float(args.sos_min)
                            limit_norm = delta_limit_mps / max(sos_range, 1e-12)
                            excess = torch.relu(torch.abs(pred_view - teacher_pred.detach()) - limit_norm)
                            delta_barrier_loss = torch.mean(excess * excess)
                            total_loss = total_loss + lambda_delta_barrier * delta_barrier_loss

                    domain_consistency_loss = None
                    lambda_domain_consistency = float(getattr(args, "lambda_domain_consistency", 0.0))
                    if (
                        lambda_domain_consistency > 0.0
                        and raw_tof_phys is not None
                        and not forward_only_phase
                    ):
                        raw_norm_for_consistency = normalize_physical_tof_like_dataset(
                            raw_tof_phys, raw_tof_norm_meta
                        ).to(device=tof_b.device, dtype=tof_b.dtype)
                        if tof_mask_b is not None:
                            raw_norm_for_consistency = raw_norm_for_consistency * tof_mask_b.to(
                                device=tof_b.device, dtype=tof_b.dtype
                            )
                        roi_input_for_consistency = make_model_input(
                            tof_b, tof_mask_b, use_tof_mask_channel, tof_feature_mode, water_tof_norm
                        )
                        raw_input_for_consistency = make_model_input(
                            raw_norm_for_consistency,
                            tof_mask_b,
                            use_tof_mask_channel,
                            tof_feature_mode,
                            raw_water_tof_norm if raw_water_tof_norm is not None else water_tof_norm,
                        )
                        roi_pred_for_consistency = model(roi_input_for_consistency).view_as(sos_b)
                        raw_pred_for_consistency = model(raw_input_for_consistency).view_as(sos_b)
                        if use_circular_mask:
                            roi_pred_for_consistency = apply_circular_support_to_normalized_sos(
                                roi_pred_for_consistency, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                            raw_pred_for_consistency = apply_circular_support_to_normalized_sos(
                                raw_pred_for_consistency, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                        domain_consistency_loss = torch.mean(
                            (roi_pred_for_consistency - raw_pred_for_consistency) ** 2
                        )
                        total_loss = total_loss + lambda_domain_consistency * domain_consistency_loss

                    # Physics: Eikonal-compatible ToF consistency (if lam_tof_eff > 0).
                    # The forward path is the differentiable Eikonal surrogate trained
                    # from MSFM collocation targets, not a straight-line path matrix.
                    tof_loss = None
                    if lam_tof_eff > 0.0:
                        train_emit_idx = select_tof_emitters(
                            args, epoch_idx=tof_schedule_epoch, batch_idx=batch_idx, device=device
                        )
                        _set_requires_grad(forward_model, False)
                        forward_model.eval()
                        tof_loss = eikonal_tof_consistency_loss(
                            pred_norm_sos=pred_view,
                            tof_target_norm=tof_b,
                            tof_mask_b=tof_mask_b,
                            args=args,
                            pairs_ds=pairs_ds,
                            forward_model=forward_model,
                            emitters_t=emitters_t,
                            receivers_t=receivers_t,
                            emitter_idx=train_emit_idx,
                            tof_target_phys=raw_tof_phys,
                            tof_target_norm_meta=raw_tof_norm_meta,
                        )
                        total_loss = total_loss + lam_tof_eff * tof_loss

                if not forward_only_phase:
                    # Backprop to reconstruction model
                    total_loss.backward()
                    optimizer.step()
                    if ema_model is not None:
                        update_ema_model(ema_model, model, ema_decay)
                if use_fwd and forward_model is not None:
                    _set_requires_grad(forward_model, True)

                epoch_loss += float(total_loss.detach().item())
                sos_sum += float(sos_loss.item())
                grad_sum += float(grad_loss.item())
                contrast_sum += float(contrast_loss.item())
                detail_sum += float(detail_loss.item())
                lap_sum += float(lap_loss.item())
                if teacher_loss is not None:
                    teacher_sum += float(teacher_loss.item())
                    teacher_count += 1
                if delta_barrier_loss is not None:
                    delta_barrier_sum += float(delta_barrier_loss.item())
                    delta_barrier_count += 1
                if domain_consistency_loss is not None:
                    domain_consistency_sum += float(domain_consistency_loss.item())
                    domain_consistency_count += 1
                if tof_loss is not None:
                    tof_sum += float(tof_loss.item())
                    tof_count += 1

                # Forward model training (optional), supervised by stored MSFM collocation T.
                # Warmup is normally the period in which this surrogate learns, so do not
                # skip it unless the explicit ablation flag is set.
                do_skip_fwd = bool(getattr(args, "skip_forward_during_tof_warmup", False)) and (tof_schedule_epoch < int(getattr(args, "tof_warmup_epochs", 0)))

                if (not do_skip_fwd) and use_fwd and forward_model is not None and optimizer_forward is not None:
                    if colloc is None:
                        raise RuntimeError(
                            "[train_initial_reconstruction.py] Missing collocation batch although --use_differentiable_eikonal is enabled. "
                            "The pairs cache must be built with --store_collocation_T."
                        )

                    forward_model.train()
                    _set_requires_grad(forward_model, True)
                    optimizer_forward.zero_grad(set_to_none=True)

                    c_true_phys = sos_b.view(-1, 1, int(args.nx), int(args.ny)) * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)

                    colloc_xy = colloc["colloc_xy"].to(device)
                    emit_idx = colloc["colloc_emitters_idx"].to(device)
                    colloc_T = (colloc.get("colloc_T_phys", None) if isinstance(colloc, dict) else None)
                    if colloc_T is None:
                        colloc_T = colloc["colloc_T"]
                    colloc_T = colloc_T.to(device)

                    B_fwd = int(c_true_phys.shape[0])

                    if colloc_xy.ndim == 2:
                        colloc_xy = colloc_xy.unsqueeze(0).expand(B_fwd, -1, 2)
                    elif colloc_xy.ndim == 3 and int(colloc_xy.shape[0]) == B_fwd and int(colloc_xy.shape[-1]) == 2:
                        pass
                    else:
                        raise RuntimeError(f"[train_initial_reconstruction.py] Unexpected colloc_xy shape {tuple(colloc_xy.shape)} for batch size {B_fwd}.")

                    if colloc_T.ndim == 2:
                        colloc_T = colloc_T.unsqueeze(0).expand(B_fwd, -1, -1)
                    elif colloc_T.ndim == 3 and int(colloc_T.shape[0]) == B_fwd:
                        pass
                    else:
                        raise RuntimeError(f"[train_initial_reconstruction.py] Unexpected colloc_T shape {tuple(colloc_T.shape)} for batch size {B_fwd}.")

                    # Emitter indices may be global (k,) or batched (B,k).
                    # Reject other shapes to avoid training against mismatched sources.
                    if emit_idx.ndim == 1:
                        emit_idx_batch = emit_idx.unsqueeze(0).expand(B_fwd, -1)
                    elif emit_idx.ndim == 2 and int(emit_idx.shape[0]) == B_fwd:
                        emit_idx_batch = emit_idx
                    else:
                        raise RuntimeError(
                            f"[train_initial_reconstruction.py] Unexpected colloc_emitters_idx shape {tuple(emit_idx.shape)} for batch size {B_fwd}. "
                            "The cache or dataset loader supplied the full (N,k) array for every sample. "
                            "Rebuild the pairs cache with this pipeline."
                        )

                    if int(colloc_T.shape[1]) != int(emit_idx_batch.shape[1]):
                        raise RuntimeError(
                            f"[train_initial_reconstruction.py] colloc_T/emitters mismatch: colloc_T shape {tuple(colloc_T.shape)}, "
                            f"emitters shape {tuple(emit_idx_batch.shape)}."
                        )

                    w = ForwardLossWeights(
                        lambda_pde=float(getattr(args, "lambda_forward_pde", 1.0)),
                        lambda_bc=float(getattr(args, "lambda_forward_bc", 1.0)),
                        lambda_supT=float(getattr(args, "lambda_forward_supT", 1.0)),
                    )

                    fwd_loss_total = None
                    kF = int(colloc_T.shape[1])
                    for j in range(kF):
                        src_xy = emitters_t[emit_idx_batch[:, j].long()]
                        fwd_losses = forward_model.forward_losses(
                            c_img=c_true_phys,
                            source_xy=src_xy,
                            colloc_xy=colloc_xy,
                            T_target=colloc_T[:, j, :],
                            weights=w,
                        )
                        loss_j = fwd_losses["loss_forward_total"]
                        fwd_loss_total = loss_j if fwd_loss_total is None else (fwd_loss_total + loss_j)

                        fwd_tot_sum += float(loss_j.detach().item())
                        fwd_pde_sum += float(fwd_losses["loss_pde"].detach().item())
                        fwd_bc_sum += float(fwd_losses["loss_bc"].detach().item())
                        fwd_supT_sum += float(fwd_losses["loss_supT"].detach().item())
                        fwd_count += 1

                    if fwd_loss_total is None:
                        raise RuntimeError("[train_initial_reconstruction.py] No forward Eikonal collocation emitters were available in this batch.")

                    fwd_loss_total = fwd_loss_total / float(max(1, kF))
                    fwd_receiver_loss = eikonal_forward_receiver_tof_training_loss(
                        forward_model=forward_model,
                        c_true_norm=sos_b,
                        tof_target_phys=raw_tof_phys,
                        tof_mask_b=tof_mask_b,
                        emitters_t=emitters_t,
                        receivers_t=receivers_t,
                        args=args,
                        epoch_idx=tof_schedule_epoch,
                        batch_idx=batch_idx,
                    )
                    if fwd_receiver_loss is not None:
                        fwd_loss_total = fwd_loss_total + fwd_receiver_loss["loss"]
                        fwd_receiver_abs_sum += float(fwd_receiver_loss["abs"].item())
                        fwd_receiver_residual_sum += float(fwd_receiver_loss["residual"].item())
                        fwd_receiver_count += 1
                    scaled_fwd_loss = float(getattr(args, "lambda_forward_total", 1.0)) * fwd_loss_total
                    scaled_fwd_loss.backward()

                    # Diagnostic: if this remains exactly zero, the surrogate cannot learn.
                    grad_sq = 0.0
                    for p_fwd in forward_model.parameters():
                        if p_fwd.grad is not None:
                            g = p_fwd.grad.detach()
                            grad_sq += float(torch.sum(g * g).item())
                    fwd_grad_norm = grad_sq ** 0.5
                    fwd_grad_norm_sum += fwd_grad_norm
                    fwd_grad_count += 1

                    if fwd_grad_norm <= 0.0:
                        consecutive_zero_fwd_grad += 1
                    else:
                        consecutive_zero_fwd_grad = 0
                    if consecutive_zero_fwd_grad >= int(getattr(args, "forward_zero_grad_patience", 3)):
                        raise RuntimeError(
                            "[train_initial_reconstruction.py] Forward Eikonal surrogate has zero parameter-gradient norm for "
                            f"{consecutive_zero_fwd_grad} consecutive batches. This indicates a real code/data problem; "
                            "training is stopped instead of silently wasting CPU time."
                        )

                    optimizer_forward.step()

            # Validation (same objective components as training)
            model.eval()
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            if use_fwd and forward_model is not None:
                forward_model.eval()
            v_tot = 0.0
            v_sos = 0.0
            v_grad = 0.0
            v_contrast = 0.0
            v_detail = 0.0
            v_lap = 0.0
            v_teacher = 0.0
            v_teacher_count = 0
            v_delta_barrier = 0.0
            v_delta_barrier_count = 0
            v_domain_consistency = 0.0
            v_domain_consistency_count = 0
            v_tof = 0.0
            v_tof_count = 0
            v_fwd_tot_sum = 0.0
            v_fwd_pde_sum = 0.0
            v_fwd_bc_sum = 0.0
            v_fwd_supT_sum = 0.0
            v_fwd_count = 0
            v_fwd_receiver_rmse_us_sum = 0.0
            v_fwd_receiver_mae_us_sum = 0.0
            v_fwd_receiver_residual_rmse_us_sum = 0.0
            v_fwd_receiver_residual_mae_us_sum = 0.0
            v_fwd_receiver_batches = 0

            with torch.no_grad():
                for val_batch_idx, vbatch in enumerate(val_loader):
                    s, t, tmask, _p, _c, _raw = unpack_reconstruction_batch(vbatch)

                    s = s.to(device)
                    t = t.to(device)
                    if tmask is not None:
                        tmask = tmask.to(device)
                    raw_val_tof_phys = None
                    raw_val_tof_norm_meta = None
                    if _raw is not None:
                        raw_val_tof_phys = _raw["raw_eikonal_tof_phys"].to(device)
                        raw_val_tof_norm_meta = _raw.get("raw_eikonal_tof_norm", None)

                    v_input = make_model_input(t, tmask, use_tof_mask_channel, tof_feature_mode, water_tof_norm)
                    v_pred_raw = eval_model(v_input)
                    v_pred = v_pred_raw.view_as(s)
                    v_target = s
                    if use_circular_mask:
                        v_pred = apply_circular_support_to_normalized_sos(
                            v_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                        v_target = apply_circular_support_to_normalized_sos(
                            s, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                    v_sos_loss = sos_reconstruction_loss(v_pred, v_target, args)
                    v_grad_loss = gradient_consistency_loss(v_pred, v_target)
                    v_contrast_loss = multiscale_local_contrast_loss(v_pred, v_target)
                    v_detail_loss = detail_band_loss(v_pred, v_target, args)
                    v_lap_loss = laplacian_detail_loss(v_pred, v_target)

                    v_total_loss = (
                        v_sos_loss
                        + float(getattr(args, "lambda_grad", 0.0)) * v_grad_loss
                        + float(getattr(args, "lambda_contrast", 0.0)) * v_contrast_loss
                        + float(getattr(args, "lambda_detail", 0.0)) * v_detail_loss
                        + float(getattr(args, "lambda_laplacian", 0.0)) * v_lap_loss
                    )
                    v_teacher_loss = None
                    v_delta_barrier_loss = None
                    if teacher_model is not None:
                        teacher_v_pred = teacher_model(v_input).view_as(s)
                        if use_circular_mask:
                            teacher_v_pred = apply_circular_support_to_normalized_sos(
                                teacher_v_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                        v_teacher_loss = torch.mean((v_pred - teacher_v_pred.detach()) ** 2)
                        v_total_loss = v_total_loss + float(getattr(args, "lambda_teacher", 0.0)) * v_teacher_loss
                        lambda_delta_barrier = float(getattr(args, "lambda_delta_barrier", 0.0))
                        delta_limit_mps = float(getattr(args, "stage2b_delta_limit_mps", 35.0))
                        if lambda_delta_barrier > 0.0 and delta_limit_mps > 0.0:
                            sos_range = float(args.sos_max) - float(args.sos_min)
                            limit_norm = delta_limit_mps / max(sos_range, 1e-12)
                            v_excess = torch.relu(torch.abs(v_pred - teacher_v_pred.detach()) - limit_norm)
                            v_delta_barrier_loss = torch.mean(v_excess * v_excess)
                            v_total_loss = v_total_loss + lambda_delta_barrier * v_delta_barrier_loss

                    v_domain_consistency_loss = None
                    lambda_domain_consistency = float(getattr(args, "lambda_domain_consistency", 0.0))
                    if lambda_domain_consistency > 0.0 and raw_val_tof_phys is not None:
                        raw_val_norm_for_consistency = normalize_physical_tof_like_dataset(
                            raw_val_tof_phys, raw_val_tof_norm_meta
                        ).to(device=t.device, dtype=t.dtype)
                        if tmask is not None:
                            raw_val_norm_for_consistency = raw_val_norm_for_consistency * tmask.to(
                                device=t.device, dtype=t.dtype
                            )
                        roi_val_input_for_consistency = make_model_input(
                            t, tmask, use_tof_mask_channel, tof_feature_mode, water_tof_norm
                        )
                        raw_val_input_for_consistency = make_model_input(
                            raw_val_norm_for_consistency,
                            tmask,
                            use_tof_mask_channel,
                            tof_feature_mode,
                            raw_water_tof_norm if raw_water_tof_norm is not None else water_tof_norm,
                        )
                        roi_val_pred = eval_model(roi_val_input_for_consistency).view_as(s)
                        raw_val_pred = eval_model(raw_val_input_for_consistency).view_as(s)
                        if use_circular_mask:
                            roi_val_pred = apply_circular_support_to_normalized_sos(
                                roi_val_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                            raw_val_pred = apply_circular_support_to_normalized_sos(
                                raw_val_pred, circular_mask, args.sos_min, args.sos_max, args.sos_water
                            )
                        v_domain_consistency_loss = torch.mean((roi_val_pred - raw_val_pred) ** 2)
                        v_total_loss = v_total_loss + lambda_domain_consistency * v_domain_consistency_loss
                    v_tof_loss = None

                    # Validation ToF loss through the Eikonal surrogate.
                    if lam_tof_eff > 0.0:
                        val_emit_idx = select_tof_emitters(
                            args, epoch_idx=tof_schedule_epoch, batch_idx=val_batch_idx, device=device
                        )
                        v_tof_loss = eikonal_tof_consistency_loss(
                            pred_norm_sos=v_pred,
                            tof_target_norm=t,
                            tof_mask_b=tmask,
                            args=args,
                            pairs_ds=pairs_ds,
                            forward_model=forward_model,
                            emitters_t=emitters_t,
                            receivers_t=receivers_t,
                            emitter_idx=val_emit_idx,
                            tof_target_phys=raw_val_tof_phys,
                            tof_target_norm_meta=raw_val_tof_norm_meta,
                        )
                        v_total_loss = v_total_loss + lam_tof_eff * v_tof_loss

                    if use_fwd and forward_model is not None and _c is not None:
                        fwd_val_metrics = eikonal_forward_collocation_metrics(
                            forward_model=forward_model,
                            c_true_norm=s,
                            colloc=_c,
                            emitters_t=emitters_t,
                            args=args,
                        )
                        if fwd_val_metrics is not None:
                            v_fwd_tot_sum += fwd_val_metrics["total"]
                            v_fwd_pde_sum += fwd_val_metrics["pde"]
                            v_fwd_bc_sum += fwd_val_metrics["bc"]
                            v_fwd_supT_sum += fwd_val_metrics["supT"]
                            v_fwd_count += int(fwd_val_metrics["count"])
                    if use_fwd and forward_model is not None and raw_val_tof_phys is not None:
                        fwd_receiver_metrics = eikonal_forward_receiver_tof_metrics(
                            forward_model=forward_model,
                            c_true_norm=s,
                            tof_target_phys=raw_val_tof_phys,
                            tof_mask_b=tmask,
                            emitters_t=emitters_t,
                            receivers_t=receivers_t,
                            args=args,
                            epoch_idx=tof_schedule_epoch,
                            batch_idx=val_batch_idx,
                        )
                        if fwd_receiver_metrics is not None:
                            v_fwd_receiver_rmse_us_sum += fwd_receiver_metrics["rmse_us"]
                            v_fwd_receiver_mae_us_sum += fwd_receiver_metrics["mae_us"]
                            v_fwd_receiver_residual_rmse_us_sum += fwd_receiver_metrics["residual_rmse_us"]
                            v_fwd_receiver_residual_mae_us_sum += fwd_receiver_metrics["residual_mae_us"]
                            v_fwd_receiver_batches += 1

                    v_tot += float(v_total_loss.item())
                    v_sos += float(v_sos_loss.item())
                    v_grad += float(v_grad_loss.item())
                    v_contrast += float(v_contrast_loss.item())
                    v_detail += float(v_detail_loss.item())
                    v_lap += float(v_lap_loss.item())
                    if v_teacher_loss is not None:
                        v_teacher += float(v_teacher_loss.item())
                        v_teacher_count += 1
                    if v_delta_barrier_loss is not None:
                        v_delta_barrier += float(v_delta_barrier_loss.item())
                        v_delta_barrier_count += 1
                    if v_domain_consistency_loss is not None:
                        v_domain_consistency += float(v_domain_consistency_loss.item())
                        v_domain_consistency_count += 1
                    if v_tof_loss is not None:
                        v_tof += float(v_tof_loss.item())
                        v_tof_count += 1

            # Deterministic training-set evaluation with exactly the same protocol as validation.
            # This is the curve to compare against ValEval.  TrainOpt above remains
            # the true stochastic objective used for backpropagation.
            train_eval_metrics = _eval_inverse_split(eval_model, train_eval_loader)

            train_hist.append(epoch_loss / max(1, len(train_loader)))
            train_eval_hist.append(float(train_eval_metrics["total"]))
            val_hist.append(v_tot / max(1, len(val_loader)))

            train_sos_mean = sos_sum / max(1, len(train_loader))
            train_grad_mean = grad_sum / max(1, len(train_loader))
            train_contrast_mean = contrast_sum / max(1, len(train_loader))
            train_detail_mean = detail_sum / max(1, len(train_loader))
            train_lap_mean = lap_sum / max(1, len(train_loader))
            train_teacher_mean = (teacher_sum / float(max(1, teacher_count))) if (teacher_count > 0) else 0.0
            train_delta_barrier_mean = (
                delta_barrier_sum / float(max(1, delta_barrier_count))
            ) if (delta_barrier_count > 0) else 0.0
            train_domain_consistency_mean = (
                domain_consistency_sum / float(max(1, domain_consistency_count))
            ) if (domain_consistency_count > 0) else 0.0
            train_tof_mean = (tof_sum / float(max(1, tof_count))) if (tof_count > 0) else 0.0
            train_eval_sos_mean = float(train_eval_metrics["sos"])
            train_eval_grad_mean = float(train_eval_metrics["grad"])
            train_eval_contrast_mean = float(train_eval_metrics["contrast"])
            train_eval_detail_mean = float(train_eval_metrics["detail"])
            train_eval_lap_mean = float(train_eval_metrics["laplacian"])
            train_eval_teacher_mean = float(train_eval_metrics["teacher"])
            train_eval_delta_barrier_mean = float(train_eval_metrics["delta_barrier"])
            train_eval_domain_consistency_mean = float(train_eval_metrics["domain"])
            train_eval_tof_mean = float(train_eval_metrics["tof"])
            val_sos_mean = v_sos / max(1, len(val_loader))
            val_grad_mean = v_grad / max(1, len(val_loader))
            val_contrast_mean = v_contrast / max(1, len(val_loader))
            val_detail_mean = v_detail / max(1, len(val_loader))
            val_lap_mean = v_lap / max(1, len(val_loader))
            val_teacher_mean = (v_teacher / float(max(1, v_teacher_count))) if (v_teacher_count > 0) else 0.0
            val_delta_barrier_mean = (
                v_delta_barrier / float(max(1, v_delta_barrier_count))
            ) if (v_delta_barrier_count > 0) else 0.0
            val_domain_consistency_mean = (
                v_domain_consistency / float(max(1, v_domain_consistency_count))
            ) if (v_domain_consistency_count > 0) else 0.0
            val_tof_mean = (v_tof / float(max(1, v_tof_count))) if (v_tof_count > 0) else 0.0
            train_sos_hist.append(float(train_sos_mean))
            train_eval_sos_hist.append(float(train_eval_sos_mean))
            val_sos_hist.append(float(val_sos_mean))
            train_detail_hist.append(float(train_detail_mean))
            train_eval_detail_hist.append(float(train_eval_detail_mean))
            val_detail_hist.append(float(val_detail_mean))

            if fwd_count > 0:
                fwd_tot_mean = fwd_tot_sum / float(fwd_count)
                fwd_pde_mean = fwd_pde_sum / float(fwd_count)
                fwd_bc_mean = fwd_bc_sum / float(fwd_count)
                fwd_supT_mean = fwd_supT_sum / float(fwd_count)
                fwd_grad_mean = fwd_grad_norm_sum / float(max(1, fwd_grad_count))
                last_fwd_supT_mean = fwd_supT_mean
                last_fwd_pde_mean = fwd_pde_mean
                last_fwd_total_mean = fwd_tot_mean
                fwd_msg = (
                    f" | Fwd(mean): Tot={fwd_tot_mean:.6g}, PDE={fwd_pde_mean:.6g}, "
                    f"BC={fwd_bc_mean:.6g}, supT={fwd_supT_mean:.6g}, gradNorm={fwd_grad_mean:.6g}"
                )
                if fwd_receiver_count > 0:
                    fwd_msg += (
                        f", receiverAbs={fwd_receiver_abs_sum / float(fwd_receiver_count):.6g}, "
                        f"receiverResidual={fwd_receiver_residual_sum / float(fwd_receiver_count):.6g}"
                    )
            else:
                fwd_msg = ""

            if v_fwd_count > 0:
                v_fwd_msg = (
                    f" | ValFwd(mean): Tot={v_fwd_tot_sum / float(v_fwd_count):.6g}, "
                    f"PDE={v_fwd_pde_sum / float(v_fwd_count):.6g}, "
                    f"BC={v_fwd_bc_sum / float(v_fwd_count):.6g}, "
                    f"supT={v_fwd_supT_sum / float(v_fwd_count):.6g}"
                )
            else:
                v_fwd_msg = ""

            if v_fwd_receiver_batches > 0:
                last_fwd_receiver_rmse_us = v_fwd_receiver_rmse_us_sum / float(v_fwd_receiver_batches)
                last_fwd_receiver_residual_rmse_us = (
                    v_fwd_receiver_residual_rmse_us_sum / float(v_fwd_receiver_batches)
                )
                forward_metric_name = str(getattr(args, "forward_best_metric", "receiver_residual_rmse"))
                if forward_metric_name == "receiver_rmse":
                    current_forward_metric_value = float(last_fwd_receiver_rmse_us)
                    current_forward_metric_label = "receiver_rmse_us"
                else:
                    current_forward_metric_value = float(last_fwd_receiver_residual_rmse_us)
                    current_forward_metric_label = "receiver_residual_rmse_us"
                if (
                    use_fwd
                    and forward_model is not None
                    and current_forward_metric_value < float(best_forward_metric_value)
                ):
                    best_forward_metric_value = current_forward_metric_value
                    best_forward_epoch = int(epoch + 1)
                    best_forward_state = {
                        k: v.detach().cpu().clone()
                        for k, v in forward_model.state_dict().items()
                    }
                    log_message(
                        f"[train_initial_reconstruction.py] New best forward surrogate at epoch {epoch+1}: "
                        f"{current_forward_metric_label}={best_forward_metric_value:.6g} us."
                    )
                v_fwd_receiver_msg = (
                    f" | ValFwdReceiver: RMSE_us={last_fwd_receiver_rmse_us:.6g}, "
                    f"MAE_us={v_fwd_receiver_mae_us_sum / float(v_fwd_receiver_batches):.6g}, "
                    f"ResidualRMSE_us={last_fwd_receiver_residual_rmse_us:.6g}, "
                    f"ResidualMAE_us={v_fwd_receiver_residual_mae_us_sum / float(v_fwd_receiver_batches):.6g}"
                )
            else:
                v_fwd_receiver_msg = ""

            epoch_elapsed = time.time() - epoch_start
            total_elapsed = time.time() - training_loop_start
            epochs_done_this_run = int(epoch) - int(start_epoch) + 1
            epochs_remaining_this_run = max(0, int(args.epochs) - int(epoch) - 1)
            eta_seconds = (total_elapsed / float(max(1, epochs_done_this_run))) * float(epochs_remaining_this_run)

            log_message(
                f"Epoch {epoch+1}/{args.epochs} (tof_sched_epoch={tof_schedule_epoch+1}, lamToF_eff={lam_tof_eff:.6g}) | "
                f"TrainBackpropTot={train_hist[-1]:.6f} (SoS={train_sos_mean:.6f}, Grad={train_grad_mean:.6f}, "
                f"Contrast={train_contrast_mean:.6f}, Detail={train_detail_mean:.6f}, Lap={train_lap_mean:.6f}, Teacher={train_teacher_mean:.6f}, "
                f"DeltaBarrier={train_delta_barrier_mean:.6f}, Domain={train_domain_consistency_mean:.6f}, ToF={train_tof_mean:.6f}, "
                f"InputBatches=roi:{roi_input_batches}/raw:{raw_input_batches}) | "
                f"TrainEvalTot={train_eval_hist[-1]:.6f} (SoS={train_eval_sos_mean:.6f}, Grad={train_eval_grad_mean:.6f}, "
                f"Contrast={train_eval_contrast_mean:.6f}, Detail={train_eval_detail_mean:.6f}, Lap={train_eval_lap_mean:.6f}, Teacher={train_eval_teacher_mean:.6f}, "
                f"DeltaBarrier={train_eval_delta_barrier_mean:.6f}, Domain={train_eval_domain_consistency_mean:.6f}, ToF={train_eval_tof_mean:.6f}) | "
                f"ValEvalTot={val_hist[-1]:.6f} (SoS={val_sos_mean:.6f}, Grad={val_grad_mean:.6f}, "
                f"Contrast={val_contrast_mean:.6f}, Detail={val_detail_mean:.6f}, Lap={val_lap_mean:.6f}, Teacher={val_teacher_mean:.6f}, "
                f"DeltaBarrier={val_delta_barrier_mean:.6f}, Domain={val_domain_consistency_mean:.6f}, ToF={val_tof_mean:.6f})"
                f"{fwd_msg}{v_fwd_msg}{v_fwd_receiver_msg} | "
                f"EpochTime={_format_seconds(epoch_elapsed)}, "
                f"TotalElapsed={_format_seconds(total_elapsed)}, ETA={_format_seconds(eta_seconds)}"
            )

            if forward_only_phase:
                if early_stop_patience > 0:
                    log_message(
                        "[train_initial_reconstruction.py] Early-stopping patience unchanged during forward-only "
                        "Eikonal pretraining; reconstruction model is frozen."
                    )
            else:
                current_selection_value, selection_label = validation_selection_value(
                    selection_metric_name, float(val_hist[-1]), float(val_sos_mean), float(val_detail_mean), args
                )
                improved = current_selection_value < (float(best_val) - early_stop_min_delta)
                if improved:
                    best_val = current_selection_value
                    epochs_without_improvement = 0
                    meta = dict(pairs_ds.metadata) if isinstance(pairs_ds.metadata, dict) else {}
                    meta["sos_bank_path"] = str(args.sos_bank_path)
                    meta["splits_path"] = str(args.splits_path)
                    meta["pairs_cache_path"] = str(args.data_path)
                    meta["tof_schedule_start_epoch"] = int(tof_schedule_start_epoch)
                    meta["best_epoch"] = int(epoch + 1)
                    meta["best_val_total"] = float(val_hist[-1])
                    meta["best_val_sos"] = float(val_sos_mean)
                    meta["best_val_detail"] = float(val_detail_mean)
                    meta["best_val_laplacian"] = float(val_lap_mean)
                    meta["best_selection_metric"] = selection_metric_name
                    meta["best_selection_value"] = float(best_val)
                    meta["validation_weights"] = "ema" if ema_model is not None else "raw"
                    meta["ema_decay"] = float(ema_decay) if ema_model is not None else None
                    if stage2b_adapter_info is not None:
                        meta["stage2b_residual_adapter"] = {
                            "enabled": True,
                            "trainable_prefixes": list(stage2b_adapter_info["prefixes"]),
                            "trainable_param_count": int(stage2b_adapter_info["trainable_param_count"]),
                            "frozen_param_count": int(stage2b_adapter_info["frozen_param_count"]),
                            "delta_limit_mps": float(getattr(args, "stage2b_delta_limit_mps", 35.0)),
                            "lambda_delta_barrier": float(getattr(args, "lambda_delta_barrier", 0.0)),
                        }
                    meta["early_stop_metric"] = selection_metric_name
                    meta["learning_curve_primary"] = "train_eval_sos_vs_val_eval_sos"
                    meta["checkpoint_selection_metric"] = selection_metric_name
                    component_history = {
                        "train_opt_total_history": list(train_hist),
                        "train_eval_total_history": list(train_eval_hist),
                        "val_total_history": list(val_hist),
                        "train_opt_sos_history": list(train_sos_hist),
                        "train_eval_sos_history": list(train_eval_sos_hist),
                        "val_sos_history": list(val_sos_hist),
                        "train_opt_detail_history": list(train_detail_hist),
                        "train_eval_detail_history": list(train_eval_detail_hist),
                        "val_detail_history": list(val_detail_hist),
                    }
                    save_checkpoint(
                        ema_model if ema_model is not None else model, optimizer, train_hist, val_hist, args, meta,
                        forward_model=forward_model if use_fwd else None,
                        optimizer_forward=optimizer_forward if use_fwd else None,
                        path=best_path,
                        component_history=component_history,
                    )
                    log_message(
                        f"[train_initial_reconstruction.py] New best validation checkpoint at epoch {epoch+1}: {best_path} "
                        f"({selection_label}={best_val:.6f}, ValEvalTot={float(val_hist[-1]):.6f}, ValSoS={val_sos_mean:.6f})"
                    )
                else:
                    epochs_without_improvement += 1
                    if early_stop_patience > 0:
                        log_message(
                            f"[train_initial_reconstruction.py] No validation improvement for {epochs_without_improvement}/"
                            f"{early_stop_patience} epochs. Best {selection_label}={best_val:.6f}; "
                            f"current {selection_label}={current_selection_value:.6f}; min_delta={early_stop_min_delta:.6g}."
                        )
                        if epochs_without_improvement >= early_stop_patience:
                            log_message(
                                f"[train_initial_reconstruction.py] Early stopping at epoch {epoch+1}: validation {selection_label} did not improve "
                                f"for {early_stop_patience} consecutive epochs. Best checkpoint remains: {best_path}"
                            )
                            interrupted_flag["stop"] = True

            if int(getattr(args, "preview_every", 0)) > 0 and ((epoch + 1) % int(args.preview_every) == 0):
                with torch.no_grad():
                    preview_batch = next(iter(val_loader))
                    ps, pt, pmask, _p, _c, _raw = unpack_reconstruction_batch(preview_batch)
                    ps = ps.to(device)
                    pt = pt.to(device)
                    if pmask is not None:
                        pmask = pmask.to(device)
                    p_input = make_model_input(pt, pmask, use_tof_mask_channel, tof_feature_mode, water_tof_norm)
                    preview_model = ema_model if ema_model is not None else model
                    preview_model.eval()
                    pout = preview_model(p_input).view_as(ps)
                    if use_circular_mask:
                        pout = apply_circular_support_to_normalized_sos(
                            pout, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                        ps = apply_circular_support_to_normalized_sos(
                            ps, circular_mask, args.sos_min, args.sos_max, args.sos_water
                        )
                    nshow = min(2, int(ps.shape[0]))
                    tof_norm_preview = getattr(pairs_ds, "tof_norm", None)
                    for ib in range(nshow):
                        plot_results(
                            ps[ib].detach().cpu().numpy().squeeze(),
                            pout[ib].detach().cpu().numpy().squeeze(),
                            pt[ib].detach().cpu().numpy(),
                            f"Epoch {epoch+1}",
                            args.phys_x, args.phys_y,
                            args.sos_min, args.sos_max,
                            title_suffix=f"(Val Preview {ib+1})",
                            tof_norm=tof_norm_preview,
                        )

            # Periodic save
            if (epoch + 1) % 10 == 0:
                meta = dict(pairs_ds.metadata) if isinstance(pairs_ds.metadata, dict) else {}
                meta["sos_bank_path"] = str(args.sos_bank_path)
                meta["splits_path"] = str(args.splits_path)
                meta["pairs_cache_path"] = str(args.data_path)
                meta["tof_schedule_start_epoch"] = int(tof_schedule_start_epoch)
                if stage2b_adapter_info is not None:
                    meta["stage2b_residual_adapter"] = {
                        "enabled": True,
                        "trainable_prefixes": list(stage2b_adapter_info["prefixes"]),
                        "trainable_param_count": int(stage2b_adapter_info["trainable_param_count"]),
                        "frozen_param_count": int(stage2b_adapter_info["frozen_param_count"]),
                        "delta_limit_mps": float(getattr(args, "stage2b_delta_limit_mps", 35.0)),
                        "lambda_delta_barrier": float(getattr(args, "lambda_delta_barrier", 0.0)),
                    }
                meta["early_stop_metric"] = selection_metric_name
                meta["learning_curve_primary"] = "train_eval_sos_vs_val_eval_sos"
                meta["checkpoint_selection_metric"] = selection_metric_name
                component_history = {
                    "train_opt_total_history": list(train_hist),
                    "train_eval_total_history": list(train_eval_hist),
                    "val_total_history": list(val_hist),
                    "train_opt_sos_history": list(train_sos_hist),
                    "train_eval_sos_history": list(train_eval_sos_hist),
                    "val_sos_history": list(val_sos_hist),
                    "train_opt_detail_history": list(train_detail_hist),
                    "train_eval_detail_history": list(train_eval_detail_hist),
                    "val_detail_history": list(val_detail_hist),
                }
                save_checkpoint(
                    model, optimizer, train_hist, val_hist, args, meta,
                    forward_model=forward_model if use_fwd else None,
                    optimizer_forward=optimizer_forward if use_fwd else None,
                    component_history=component_history,
                )
                log_message(f"[train_initial_reconstruction.py] Periodic save at epoch {epoch+1}")

            if interrupted_flag["stop"]:
                break

        if use_fwd and float(getattr(args, "lambda_tof", 0.0)) > 0.0:
            log_message(
                "[train_initial_reconstruction.py] Stage-2B reconstruction update summary: "
                f"reconstruction_update_epochs={reconstruction_update_epochs}, "
                f"physics_coupled_epochs={physics_coupled_epochs}, "
                f"forced_coupling_epochs={forced_coupling_epochs}."
            )
            if bool(getattr(args, "require_2b_reconstruction_update", False)) and reconstruction_update_epochs <= 0:
                raise RuntimeError(
                    "[train_initial_reconstruction.py] Stage 2B ended without any reconstruction-network update. "
                    "The best 2B checkpoint would be only a copied 2A model, so the evaluation routine should not be run yet. "
                    "Lower --tof_forward_only_epochs or use --force_tof_coupling_after_forward_only."
                )

        # Final save + learning curves
        meta = dict(pairs_ds.metadata) if isinstance(pairs_ds.metadata, dict) else {}
        meta["sos_bank_path"] = str(args.sos_bank_path)
        meta["splits_path"] = str(args.splits_path)
        meta["pairs_cache_path"] = str(args.data_path)
        meta["tof_schedule_start_epoch"] = int(tof_schedule_start_epoch)
        if use_fwd and float(getattr(args, "lambda_tof", 0.0)) > 0.0:
            meta["stage2b_reconstruction_update_epochs"] = int(reconstruction_update_epochs)
            meta["stage2b_physics_coupled_epochs"] = int(physics_coupled_epochs)
            meta["stage2b_forced_coupling_epochs"] = int(forced_coupling_epochs)
        if stage2b_adapter_info is not None:
            meta["stage2b_residual_adapter"] = {
                "enabled": True,
                "trainable_prefixes": list(stage2b_adapter_info["prefixes"]),
                "trainable_param_count": int(stage2b_adapter_info["trainable_param_count"]),
                "frozen_param_count": int(stage2b_adapter_info["frozen_param_count"]),
                "delta_limit_mps": float(getattr(args, "stage2b_delta_limit_mps", 35.0)),
                "lambda_delta_barrier": float(getattr(args, "lambda_delta_barrier", 0.0)),
            }
        meta["early_stop_metric"] = selection_metric_name
        meta["learning_curve_primary"] = "train_eval_sos_vs_val_eval_sos"
        meta["checkpoint_selection_metric"] = selection_metric_name
        component_history = {
            "train_opt_total_history": list(train_hist),
            "train_eval_total_history": list(train_eval_hist),
            "val_total_history": list(val_hist),
            "train_opt_sos_history": list(train_sos_hist),
            "train_eval_sos_history": list(train_eval_sos_hist),
            "val_sos_history": list(val_sos_hist),
            "train_opt_detail_history": list(train_detail_hist),
            "train_eval_detail_history": list(train_eval_detail_hist),
            "val_detail_history": list(val_detail_hist),
        }
        save_checkpoint(
            model, optimizer, train_hist, val_hist, args, meta,
            forward_model=forward_model if use_fwd else None,
            optimizer_forward=optimizer_forward if use_fwd else None,
            component_history=component_history,
        )
        _plot_training_curves_consistent(train_hist, train_eval_hist, val_hist, train_sos_hist, train_eval_sos_hist, val_sos_hist)
        if os.path.exists(best_path):
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            if isinstance(best_checkpoint, dict) and "model_state_dict" in best_checkpoint:
                model.load_state_dict(best_checkpoint["model_state_dict"])
                if use_fwd and forward_model is not None and "forward_model_state_dict" in best_checkpoint:
                    ck_fwd_md = best_checkpoint.get("forward_model_metadata", None)
                    if not hasattr(forward_model, "is_compatible_metadata") or forward_model.is_compatible_metadata(ck_fwd_md):
                        forward_model.load_state_dict(best_checkpoint["forward_model_state_dict"])
                best_epoch_for_eval = None
                if isinstance(best_checkpoint.get("dataset_metadata", None), dict):
                    best_epoch_for_eval = best_checkpoint["dataset_metadata"].get("best_epoch", None)
                log_message(
                    f"[train_initial_reconstruction.py] Loaded best checkpoint for final plots/evaluation: {best_path}"
                    + (f" (best_epoch={best_epoch_for_eval})" if best_epoch_for_eval is not None else "")
                )
            del best_checkpoint
            _release_transient_memory("final best-checkpoint payload")

    # Physics setup preview and evaluation on saved test split
    dx, dy = args.phys_x / (args.nx - 1), args.phys_y / (args.ny - 1)
    emitters, receivers = generate_sensor_positions(args.nx, args.ny, dx, dy, args.radius, args.n_emitters, args.n_receivers)
    item0 = test_ds[0]
    s_norm = item0[0]
    plot_physics_setup(
        s_norm.numpy().squeeze() * (args.sos_max - args.sos_min) + args.sos_min,
        emitters, receivers, dx, dy, args.phys_x, args.phys_y, args.sos_min, args.sos_max
    )

    run_evaluation(
        model, test_loader, device, args.phys_x, args.phys_y,
        f"Epoch {len(train_hist)}", args.sos_min, args.sos_max,
        use_circular_mask=use_circular_mask,
        mask_radius=mask_radius_eff,
        sos_water=float(args.sos_water),
        use_tof_mask_channel=use_tof_mask_channel,
        tof_feature_mode=tof_feature_mode,
        water_tof_norm=water_tof_norm,
    )


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    if args.output_dir is not None:
        set_output_folder(args.output_dir)

    train_model(args)










