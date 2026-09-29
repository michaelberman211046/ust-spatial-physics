import torch

from inverse_model_io import load_inverse_model
from latent_operator_models import (
    denormalize_tof_seconds,
    load_ae_checkpoint,
    load_matched_operator_checkpoint,
    support_apply,
)
from reconstruction_utils import make_model_input
from train_background_refiner import AdjointLatentRefiner, StraightRayAdjoint, make_refiner_features, unpack


def load_frozen_reconstruction_stack(args, train_loader, device, support_mask, water_tof_norm):
    """Load Stages 1-4 with exactly the feature ordering used by clean Stage 4."""
    tof_channels = 4 if args.use_tof_mask_channel and args.tof_feature_mode == "residual_stack" else (2 if args.use_tof_mask_channel else 1)
    base_model, _ = load_inverse_model(
        args.base_model_path, args.model_type, args.nx, args.ny, args.latent_res,
        args.phys_x, args.phys_y, args.radius, args.n_emitters, args.n_receivers,
        tof_channels, device,
    )
    auto, ae_cfg, _ = load_ae_checkpoint(args.ae_path, device)
    op, _, op_ckpt = load_matched_operator_checkpoint(args.operator_path, device)
    if "best_operator_state_dict" in op_ckpt:
        op.load_state_dict(op_ckpt["best_operator_state_dict"])
    for module in (base_model, auto, op):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    adjoint = StraightRayAdjoint(
        args.nx, args.ny, args.phys_x, args.phys_y, args.radius,
        args.n_emitters, args.n_receivers, args.adjoint_ray_samples, device=device,
    )
    first = next(iter(train_loader))
    _target, tof, tof_mask, _index = unpack(first, device)
    x_model = make_model_input(tof, tof_mask, args.use_tof_mask_channel, args.tof_feature_mode, water_tof_norm)
    with torch.no_grad():
        initial = support_apply(base_model(x_model), support_mask, args.water_norm)
        residual_s = denormalize_tof_seconds(tof, args.tof_norm) - denormalize_tof_seconds(op(initial), args.tof_norm)
        initial_mps = initial * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
        adjoint_image = adjoint(residual_s, initial_mps, tof_mask, support_mask)

    checkpoint = torch.load(args.background_reconstructor_path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "adjoint_latent_physics_reconstructor":
        raise ValueError("The Stage 4 checkpoint has an incompatible kind")
    config = dict(checkpoint.get("config", {}) or {})
    state = checkpoint.get("best_model_state_dict", checkpoint["model_state_dict"])
    computed_channels = int(make_refiner_features(x_model, initial, adjoint_image, support_mask, args).shape[1])
    stored_channels = int(state["encoder.net.0.weight"].shape[1])
    if computed_channels != stored_channels:
        raise RuntimeError(f"Stage 4 feature mismatch: checkpoint={stored_channels}, computed={computed_channels}")
    background = AdjointLatentRefiner(
        stored_channels,
        int(ae_cfg.get("latent_ch", 64)),
        int(ae_cfg.get("latent_grid", 32)),
        int(config.get("channels", 96)),
        float(config.get("dropout", 0.05)),
        float(config.get("delta_limit", args.background_delta_limit)),
    ).to(device)
    background.load_state_dict(state)
    background.eval()
    for parameter in background.parameters():
        parameter.requires_grad_(False)
    return base_model, auto, op, background, adjoint










