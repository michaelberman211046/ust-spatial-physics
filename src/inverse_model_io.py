import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow mixed OpenMP (unsafe but unblocks)

import torch

from model import ReconstructionNet, ImprovedReconstructionNet, DeepGatedReconstructionNet


def unpack_batch(batch):
    sos, tof = batch[0], batch[1]
    tof_mask = None
    raw_item = None
    for item in batch[2:]:
        if torch.is_tensor(item) and item.shape == tof.shape:
            tof_mask = item
        elif isinstance(item, dict) and "raw_eikonal_tof_phys" in item:
            raw_item = item
    return sos, tof, tof_mask, raw_item


def load_inverse_model(
    checkpoint_path,
    model_type,
    nx,
    ny,
    latent_res,
    phys_x,
    phys_y,
    radius,
    n_emitters,
    n_receivers,
    tof_input_channels,
    device,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("train_config", {}) if isinstance(ckpt, dict) else {}
    latent_res = int(cfg.get("latent_res", latent_res))
    model_type = str(model_type or cfg.get("model_type", "deep"))
    fc_dropout = float(cfg.get("model_fc_dropout", 0.0))
    dec_dropout = float(cfg.get("model_decoder_dropout", 0.0))

    if model_type == "deep":
        model = DeepGatedReconstructionNet(
            nx=nx,
            ny=ny,
            latent_res=latent_res,
            phys_x=phys_x,
            phys_y=phys_y,
            radius=radius,
            n_emitters=n_emitters,
            n_receivers=n_receivers,
            tof_input_channels=tof_input_channels,
            fc_dropout=fc_dropout,
            decoder_dropout=dec_dropout,
        ).to(device)
    elif model_type == "improved":
        model = ImprovedReconstructionNet(
            nx=nx,
            ny=ny,
            latent_res=latent_res,
            phys_x=phys_x,
            phys_y=phys_y,
            radius=radius,
            n_emitters=n_emitters,
            n_receivers=n_receivers,
            tof_input_channels=tof_input_channels,
        ).to(device)
    else:
        model = ReconstructionNet(
            nx=nx,
            ny=ny,
            latent_res=latent_res,
            phys_x=phys_x,
            phys_y=phys_y,
            radius=radius,
            n_emitters=n_emitters,
            n_receivers=n_receivers,
            tof_input_channels=tof_input_channels,
        ).to(device)

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cfg










