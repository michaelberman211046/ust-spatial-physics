import argparse
import pprint
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairsCacheDataset
from logger import log_message
from settings import set_output_folder
from reconstruction_utils import make_circular_support_mask, make_homogeneous_water_tof_norm
from frozen_stack import load_frozen_reconstruction_stack
from train_background_refiner import IndexedSubset, split_dataset
from train_spatial_physics import (
    LOG_PREFIX,
    SpatialPhysicsResidualNet,
    evaluate,
    evaluate_corrupted,
    make_spatial_physics_features,
    plot_preview,
    support_apply,
)


SCRIPT_VERSION = "synthetic-evaluation-multisample-report-v1"


def _orientation_variants(x):
    """Eight square-grid dihedral transforms, applied to model output only."""
    return {
        "identity": x,
        "rot90": torch.rot90(x, 1, dims=(-2, -1)),
        "rot180": torch.rot90(x, 2, dims=(-2, -1)),
        "rot270": torch.rot90(x, 3, dims=(-2, -1)),
        "flip_x": torch.flip(x, dims=(-2,)),
        "flip_y": torch.flip(x, dims=(-1,)),
        "transpose": x.transpose(-2, -1),
        "transpose_flip_xy": torch.flip(x.transpose(-2, -1), dims=(-2, -1)),
    }


@torch.no_grad()
def log_component_orientation_diagnostics(model, stack, loader, device, support_mask, water_tof_norm, args):
    batch = next(iter(loader))
    target, _tof, _tof_mask, _idx, initial, background, feat, _bg = make_spatial_physics_features(
        batch, stack, device, support_mask, water_tof_norm, args
    )
    pred = support_apply(background + model(feat), support_mask, args.water_norm)
    mask = support_mask
    if mask is None:
        mask = torch.ones_like(target[:, :1])
    if mask.shape[0] == 1 and target.shape[0] > 1:
        mask = mask.expand(target.shape[0], -1, -1, -1)
    denom = torch.clamp(mask.sum(), min=1.0)
    report = {}
    for label, image in (("initial", initial), ("background_refiner", background), ("final", pred)):
        scores = {}
        for orientation, transformed in _orientation_variants(image).items():
            scores[orientation] = float((((transformed - target) ** 2) * mask).sum().item() / denom.item())
        report[label] = {
            "canonical_mse": scores["identity"],
            "best_orientation": min(scores, key=scores.get),
            "best_orientation_mse": min(scores.values()),
            "all_orientation_mse": scores,
        }
    log_message(f"[experimental-orientation-diagnostic]\n{pprint.pformat(report)}")


@torch.no_grad()
def collect_report_samples(model, stack, loader, device, support_mask,
                           water_tof_norm, args, sample_count):
    """Collect distinct clean reconstructions for a publication/report NPZ.

    Evaluation metrics are still calculated by ``evaluate`` over the complete
    selected loader.  This routine only changes the number of examples retained
    for visualization.  Pipeline tensors remain in their native
    (lateral, axial) order; plotting code performs the established display
    transpose.
    """
    requested = max(1, int(sample_count))
    targets, initials, backgrounds, predictions, sample_indices = [], [], [], [], []

    for batch in loader:
        target, _tof, _tof_mask, index, initial, background, feat, _bg = (
            make_spatial_physics_features(
                batch, stack, device, support_mask, water_tof_norm, args
            )
        )
        prediction = support_apply(
            background + model(feat), support_mask, args.water_norm
        )
        take = min(requested - len(sample_indices), int(target.shape[0]))
        if take <= 0:
            break

        targets.append(target[:take, 0].detach().cpu())
        initials.append(initial[:take, 0].detach().cpu())
        backgrounds.append(background[:take, 0].detach().cpu())
        predictions.append(prediction[:take, 0].detach().cpu())

        if torch.is_tensor(index):
            indices = index.detach().cpu().reshape(-1).tolist()
        elif isinstance(index, (list, tuple, np.ndarray)):
            indices = np.asarray(index).reshape(-1).tolist()
        else:
            indices = [index] * int(target.shape[0])
        sample_indices.extend(int(value) for value in indices[:take])

        if len(sample_indices) >= requested:
            break

    if len(sample_indices) < requested:
        raise RuntimeError(
            f"Requested {requested} report samples, but only "
            f"{len(sample_indices)} distinct samples were available."
        )

    return {
        "target": torch.cat(targets, dim=0),
        "initial": torch.cat(initials, dim=0),
        "background": torch.cat(backgrounds, dim=0),
        "prediction": torch.cat(predictions, dim=0),
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--splits_path", default=None)
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--ae_path", required=True)
    p.add_argument("--operator_path", required=True)
    p.add_argument("--background_reconstructor_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--baseline_model_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--report_npz", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_type", default="deep")
    p.add_argument("--latent_res", type=int, default=36)
    p.add_argument("--use_tof_mask_channel", action="store_true")
    p.add_argument("--tof_feature_mode", choices=["raw", "residual_stack"], default="residual_stack")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--preview_count", type=int, default=4)
    p.add_argument(
        "--report_sample_count", type=int, default=4,
        help=("Number of distinct clean synthetic reconstructions to retain in "
              "report_npz. The publication generator requires at least two."),
    )
    p.add_argument("--use_val_split", action="store_true")
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--delta_limit", type=float, default=0.55)
    p.add_argument("--physics_residual_norm_scale", type=float, default=0.35)
    p.add_argument("--background_delta_limit", type=float, default=0.65)
    p.add_argument("--adjoint_ray_samples", type=int, default=128)
    p.add_argument("--num_emitter_sectors", type=int, default=8)
    p.add_argument("--num_receiver_sectors", type=int, default=0)
    p.add_argument("--sector_highpass", action="store_true")
    p.add_argument("--lowpass_kernel", type=int, default=21)
    p.add_argument("--detail_kernel", type=int, default=7)
    p.add_argument("--adjoint_blur_kernel", type=int, default=7)
    p.add_argument("--nuisance_global_delay_us", type=float, default=0.80)
    p.add_argument("--nuisance_emitter_delay_us", type=float, default=0.70)
    p.add_argument("--nuisance_receiver_delay_us", type=float, default=0.70)
    p.add_argument("--nuisance_noise_us", type=float, default=0.080)
    p.add_argument("--nuisance_sector_drop_prob", type=float, default=0.40)
    p.add_argument("--nuisance_channel_drop_prob", type=float, default=0.030)
    p.add_argument("--nuisance_harmonics", type=int, default=4)
    p.add_argument("--directional_artifact_amplitude", type=float, default=0.65)
    p.add_argument("--directional_artifact_width_min", type=float, default=0.025)
    p.add_argument("--directional_artifact_width_max", type=float, default=0.10)
    p.add_argument("--directional_artifact_count", type=int, default=3)
    p.add_argument("--directional_artifact_probability", type=float, default=0.50)
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
        set_output_folder(args.output_dir)
    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message(f"[test_spatial_physics_reconstructor] script_version={SCRIPT_VERSION}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        log_message("[test_spatial_physics_reconstructor] CUDA requested but unavailable. Actual device will be CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    log_message(f"[test_spatial_physics_reconstructor] Actual device used: {device}")

    ds = PairsCacheDataset(args.data_path, sos_min=args.sos_min, sos_max=args.sos_max, return_tof_mask=True)
    args.tof_norm = ds.tof_norm
    meta = dict(getattr(ds, "metadata", {}) or {})
    for key in ["nx", "ny", "n_emitters", "n_receivers"]:
        setattr(args, key, int(meta.get(key, getattr(args, key))))
    for key in ["phys_x", "phys_y", "radius", "sos_water"]:
        setattr(args, key, float(meta.get(key, getattr(args, key))))
    args.water_norm = float(max(0.0, min(1.0, (args.sos_water - args.sos_min) / (args.sos_max - args.sos_min))))

    if args.use_val_split:
        _train_ds, use_ds = split_dataset(ds, args.data_path, args.splits_path)
        if args.max_test_samples > 0:
            use_ds = IndexedSubset(ds, use_ds.indices[:min(int(args.max_test_samples),len(use_ds))])
    else:
        use_ds = IndexedSubset(
            ds,
            list(range(min(
                len(ds),
                max(int(args.batch_size), int(args.preview_count),
                    int(args.report_sample_count)),
            ))),
        )
    loader = DataLoader(use_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    support_mask = make_circular_support_mask(args.nx, args.ny, args.phys_x, args.phys_y, args.mask_radius, device=device) if args.use_circular_mask else None
    water_tof_norm = make_homogeneous_water_tof_norm(
        args.nx, args.ny, args.phys_x, args.phys_y, args.radius,
        args.n_emitters, args.n_receivers, args.sos_water, args.tof_norm, device
    ) if args.tof_feature_mode == "residual_stack" else None
    stack = load_frozen_reconstruction_stack(args, loader, device, support_mask, water_tof_norm)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if ckpt.get("kind") != "spatial_physics_reconstructor":
        raise ValueError(f"Incompatible checkpoint: {args.model_path}")
    cfg = dict(ckpt.get("config", {}) or {})
    for key in [
        "channels", "dropout", "delta_limit", "num_emitter_sectors", "num_receiver_sectors",
        "sector_highpass", "physics_residual_norm_scale", "background_delta_limit",
        "adjoint_ray_samples", "lowpass_kernel", "detail_kernel", "adjoint_blur_kernel",
        "nuisance_global_delay_us", "nuisance_emitter_delay_us", "nuisance_receiver_delay_us",
        "nuisance_noise_us", "nuisance_sector_drop_prob", "nuisance_channel_drop_prob",
        "nuisance_harmonics", "directional_artifact_amplitude",
        "directional_artifact_width_min", "directional_artifact_width_max", "directional_artifact_count",
        "directional_artifact_probability",
    ]:
        if key in cfg:
            setattr(args, key, cfg[key])
    first = next(iter(loader))
    *_unused, feat0, _res0 = make_spatial_physics_features(first, stack, device, support_mask, water_tof_norm, args)
    in_ch = int(feat0.shape[1])
    if int(ckpt.get("in_ch", -1)) != in_ch:
        raise RuntimeError(f"Checkpoint expects in_ch={ckpt.get('in_ch')}, but runtime features produced {in_ch}.")
    model = SpatialPhysicsResidualNet(in_ch, args.channels, args.dropout, args.delta_limit).to(device)
    state = ckpt["best_model_state_dict"] if "best_model_state_dict" in ckpt else ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    metrics = evaluate(model, stack, loader, device, support_mask, water_tof_norm, args)
    torch.manual_seed(230923)
    if device.type == "cuda": torch.cuda.manual_seed_all(230923)
    robust_metrics = evaluate_corrupted(model, stack, loader, device, support_mask, water_tof_norm, args)
    metrics.update(robust_metrics)
    baseline_model = None
    if args.baseline_model_path:
        baseline_ckpt=torch.load(args.baseline_model_path,map_location=device,weights_only=False)
        baseline_model=SpatialPhysicsResidualNet(in_ch,args.channels,args.dropout,args.delta_limit).to(device)
        baseline_model.load_state_dict(baseline_ckpt.get("best_model_state_dict",baseline_ckpt["model_state_dict"]));baseline_model.eval()
        baseline_metrics=evaluate(baseline_model,stack,loader,device,support_mask,water_tof_norm,args)
    else:
        baseline_metrics=dict(metrics)
    torch.manual_seed(230923)
    if device.type == "cuda": torch.cuda.manual_seed_all(230923)
    if baseline_model is not None:
        baseline_metrics.update(evaluate_corrupted(baseline_model,stack,loader,device,support_mask,water_tof_norm,args))
        log_message(f"[synthetic_evaluation] optional baseline metrics:\n{pprint.pformat(baseline_metrics)}")
    log_message(f"[test_spatial_physics_reconstructor] metrics:\n{pprint.pformat(metrics)}")
    log_component_orientation_diagnostics(model, stack, loader, device, support_mask, water_tof_norm, args)
    report = collect_report_samples(
        model, stack, loader, device, support_mask, water_tof_norm, args,
        args.report_sample_count,
    )
    lo, hi = float(args.sos_min), float(args.sos_max)
    target_mps = report["target"].numpy() * (hi - lo) + lo
    initial_mps = report["initial"].numpy() * (hi - lo) + lo
    background_mps = report["background"].numpy() * (hi - lo) + lo
    prediction_mps = report["prediction"].numpy() * (hi - lo) + lo
    absolute_error_mps = np.abs(prediction_mps - target_mps)
    np.savez_compressed(
        args.report_npz,
        target_mps=target_mps,
        initial_mps=initial_mps,
        background_mps=background_mps,
        prediction_mps=prediction_mps,
        absolute_error_mps=absolute_error_mps,
        sample_indices=report["sample_indices"],
        report_sample_count=np.asarray(len(report["sample_indices"])),
        phys_x=np.asarray(float(args.phys_x)),
        phys_y=np.asarray(float(args.phys_y)),
        sos_min=np.asarray(lo),
        sos_max=np.asarray(hi),
        background_mse=np.asarray(metrics["background_mse"]),
        prediction_mse=np.asarray(metrics["pred_mse"]),
        prediction_l1=np.asarray(metrics["pred_l1"]),
        tof_mse=np.asarray(metrics["tof_mse"]),
        corrupt_prediction_mse=np.asarray(metrics["corrupt_pred_mse"]),
        clean_corrupt_l1=np.asarray(metrics["clean_corrupt_l1"]),
        stripe_sensitivity=np.asarray(metrics["stripe_sensitivity"]),
        baseline_prediction_mse=np.asarray(baseline_metrics["pred_mse"]),
        baseline_corrupt_prediction_mse=np.asarray(baseline_metrics["corrupt_pred_mse"]),
        baseline_clean_corrupt_l1=np.asarray(baseline_metrics["clean_corrupt_l1"]),
        baseline_stripe_sensitivity=np.asarray(baseline_metrics["stripe_sensitivity"]),
    )
    log_message(
        f"[synthetic_evaluation] numerical validation and {len(report['sample_indices'])} "
        f"distinct report samples saved={args.report_npz}; "
        f"sample_indices={report['sample_indices'].tolist()}"
    )


if __name__ == "__main__":
    main()












