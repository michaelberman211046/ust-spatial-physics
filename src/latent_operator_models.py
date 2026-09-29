import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_FAMILY = "latent-operator-matched-sos-v1"


def format_duration(seconds):
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}min"
    return f"{minutes / 60.0:.2f}h"


def ensure_image_batch(x):
    x = x.float()
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(1)
    if x.ndim == 3:
        if x.shape[0] == 1:
            return x.unsqueeze(0)
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected image with 2, 3, or 4 dims, got {tuple(x.shape)}")


def ensure_tof_batch(x):
    x = x.float()
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected ToF matrix with 2 or 3 dims, got {tuple(x.shape)}")


def denormalize_tof_seconds(tof_normed, tof_norm):
    x = tof_normed.float()
    meta = tof_norm or {"type": "none"}
    kind = str(meta.get("type", "none")).lower()
    if kind == "zscore":
        return x * (float(meta.get("std", 1.0)) + float(meta.get("eps", 1e-6))) + float(meta.get("mean", 0.0))
    if kind == "max":
        return x * (float(meta.get("max", 1.0)) + float(meta.get("eps", 1e-6)))
    return x


def normalize_tof_seconds(tof_seconds, tof_norm):
    x = tof_seconds.float()
    meta = tof_norm or {"type": "none"}
    kind = str(meta.get("type", "none")).lower()
    if kind == "zscore":
        return (x - float(meta.get("mean", 0.0))) / (float(meta.get("std", 1.0)) + float(meta.get("eps", 1e-6)))
    if kind == "max":
        return x / (float(meta.get("max", 1.0)) + float(meta.get("eps", 1e-6)))
    return x


def masked_mse(a, b, mask=None):
    if mask is None:
        return F.mse_loss(a, b)
    m = mask.to(device=a.device, dtype=a.dtype)
    while m.ndim < a.ndim:
        m = m.unsqueeze(1)
    m = m.expand_as(a)
    return (((a - b) ** 2) * m).sum() / m.sum().clamp_min(1.0)


def masked_l1(a, b, mask=None):
    if mask is None:
        return F.l1_loss(a, b)
    m = mask.to(device=a.device, dtype=a.dtype)
    while m.ndim < a.ndim:
        m = m.unsqueeze(1)
    m = m.expand_as(a)
    return (torch.abs(a - b) * m).sum() / m.sum().clamp_min(1.0)


def support_apply(x, support_mask, water_norm):
    if support_mask is None:
        return x
    m = support_mask.to(device=x.device, dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(0)
    return x * m + float(water_norm) * (1.0 - m)


def blur(x, kernel_size):
    k = int(kernel_size)
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k, stride=1)


def highpass(x, kernel_size):
    return x - blur(x, kernel_size)


def grad_l1(a, b, mask=None):
    ax = a[..., :, 1:] - a[..., :, :-1]
    bx = b[..., :, 1:] - b[..., :, :-1]
    ay = a[..., 1:, :] - a[..., :-1, :]
    by = b[..., 1:, :] - b[..., :-1, :]
    if mask is None:
        return F.l1_loss(ax, bx) + F.l1_loss(ay, by)
    mx = mask[..., :, 1:].to(device=a.device, dtype=a.dtype)
    my = mask[..., 1:, :].to(device=a.device, dtype=a.dtype)
    return masked_l1(ax, bx, mx) + masked_l1(ay, by, my)


def tv_l1(x, mask=None):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    if mask is None:
        return torch.sqrt(dx * dx + 1e-8).mean() + torch.sqrt(dy * dy + 1e-8).mean()
    mx = mask[..., :, 1:].to(device=x.device, dtype=x.dtype)
    my = mask[..., 1:, :].to(device=x.device, dtype=x.dtype)
    return (torch.sqrt(dx * dx + 1e-8) * mx).sum() / mx.sum().clamp_min(1.0) + (
        torch.sqrt(dy * dy + 1e-8) * my
    ).sum() / my.sum().clamp_min(1.0)


def batch_progress(prefix, epoch, epochs, batch_idx, total_batches, epoch_t0, phase_t0):
    elapsed_epoch = time.time() - epoch_t0
    elapsed_phase = time.time() - phase_t0
    eta_epoch = elapsed_epoch / max(1, batch_idx) * max(0, total_batches - batch_idx)
    done = ((epoch - 1) * total_batches + batch_idx) / max(1, epochs * total_batches)
    eta_phase = elapsed_phase * (1.0 / done - 1.0) if done > 0 else 0.0
    return (
        f"{prefix} epoch {epoch:04d}/{epochs:04d} batch {batch_idx:04d}/{total_batches:04d} "
        f"elapsed_epoch={format_duration(elapsed_epoch)} ETA_epoch={format_duration(eta_epoch)} "
        f"elapsed_phase={format_duration(elapsed_phase)} ETA_phase={format_duration(eta_phase)}"
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, dilation=1):
        super().__init__()
        groups = max(1, min(8, int(out_ch) // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=int(dilation), dilation=int(dilation)),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=int(dilation), dilation=int(dilation)),
            nn.GroupNorm(groups, out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class ShapeDecoder(nn.Module):
    def __init__(self, latent_ch=64, channels=64, out_hw=(200, 200), dropout=0.02):
        super().__init__()
        c = int(channels)
        self.out_hw = tuple(map(int, out_hw))
        self.net = nn.Sequential(
            ResidualBlock(latent_ch, c * 4, dropout),
            ResidualBlock(c * 4, c * 4, dropout, dilation=2),
            ResidualBlock(c * 4, c * 4, dropout, dilation=4),
            ResidualBlock(c * 4, c * 2, dropout),
            ResidualBlock(c * 2, c, dropout),
            nn.Conv2d(c, 1, 1),
        )

    def forward(self, z):
        x = F.interpolate(z, size=self.out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.net(x))


class ShapeAutoencoder(nn.Module):
    def __init__(self, latent_ch=64, latent_grid=32, channels=64, out_hw=(200, 200), dropout=0.02):
        super().__init__()
        c = int(channels)
        self.latent_grid = int(latent_grid)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.GroupNorm(max(1, min(8, c // 4)), c),
            nn.SiLU(inplace=True),
            ResidualBlock(c, c, dropout),
            ResidualBlock(c, c * 2, dropout),
            nn.AvgPool2d(2),
            ResidualBlock(c * 2, c * 2, dropout),
            ResidualBlock(c * 2, c * 4, dropout),
            nn.AvgPool2d(2),
            ResidualBlock(c * 4, c * 4, dropout),
            nn.Conv2d(c * 4, latent_ch, 1),
        )
        self.decoder = ShapeDecoder(latent_ch, channels, out_hw, dropout)

    def encode(self, x):
        z = self.encoder(x)
        return F.interpolate(z, size=(self.latent_grid, self.latent_grid), mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.decoder(self.encode(x))


class ConditionalLatentEncoder(nn.Module):
    def __init__(self, in_ch, latent_ch=64, latent_grid=32, channels=64, dropout=0.04):
        super().__init__()
        c = int(channels)
        self.latent_grid = int(latent_grid)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(max(1, min(8, c // 4)), c),
            nn.SiLU(inplace=True),
            ResidualBlock(c, c, dropout),
            ResidualBlock(c, c * 2, dropout),
            nn.AvgPool2d(2),
            ResidualBlock(c * 2, c * 2, dropout),
            ResidualBlock(c * 2, c * 4, dropout),
            nn.AvgPool2d(2),
            ResidualBlock(c * 4, c * 4, dropout),
            ResidualBlock(c * 4, c * 4, dropout, dilation=2),
            nn.Conv2d(c * 4, latent_ch, 1),
        )

    def forward(self, x):
        z = self.net(x)
        return F.interpolate(z, size=(self.latent_grid, self.latent_grid), mode="bilinear", align_corners=False)


class MatchedToFOperator(nn.Module):
    """Differentiable surrogate for the actual processed ToF stored in the dataset."""
    def __init__(self, n_emitters=64, n_receivers=64, channels=48, latent_dim=256, dropout=0.04):
        super().__init__()
        c = int(channels)
        self.n_emitters = int(n_emitters)
        self.n_receivers = int(n_receivers)
        out_dim = self.n_emitters * self.n_receivers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 5, stride=2, padding=2),
            nn.GroupNorm(max(1, min(8, c // 4)), c),
            nn.SiLU(inplace=True),
            ResidualBlock(c, c, dropout),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(8, c // 2)), c * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(c * 2, c * 2, dropout),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(8, c)), c * 4),
            nn.SiLU(inplace=True),
            ResidualBlock(c * 4, c * 4, dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, int(latent_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(latent_dim), out_dim),
        )

    def forward(self, sos_norm):
        y = self.head(self.encoder(sos_norm.float()))
        return y.view(sos_norm.shape[0], self.n_emitters, self.n_receivers)


def make_condition_features(x_model, base, support_mask, lowpass_kernel):
    x_img = F.interpolate(x_model.float(), size=base.shape[-2:], mode="bilinear", align_corners=False)
    base_hp = highpass(base, int(lowpass_kernel))
    if support_mask is None:
        mask = torch.ones_like(base)
    else:
        mask = support_mask.to(device=base.device, dtype=base.dtype)
        while mask.ndim < base.ndim:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(base)
    return torch.cat([base, base_hp, mask, x_img], dim=1)


def load_ae_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("args", {}))
    auto = ShapeAutoencoder(
        latent_ch=int(cfg.get("latent_ch", 64)),
        latent_grid=int(cfg.get("latent_grid", 32)),
        channels=int(cfg.get("channels", cfg.get("decoder_channels", 64))),
        out_hw=(int(cfg.get("nx", 200)), int(cfg.get("ny", 200))),
        dropout=float(cfg.get("dropout", 0.02)),
    ).to(device)
    state = ckpt.get("best_autoencoder_state_dict", ckpt.get("autoencoder_state_dict", ckpt.get("model_state_dict")))
    if state is None:
        raise ValueError(f"AE checkpoint missing state dict: {path}")
    auto.load_state_dict(state)
    auto.eval()
    return auto, cfg, ckpt


def load_matched_operator_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("args", {}))
    model = MatchedToFOperator(
        n_emitters=int(cfg.get("n_emitters", 64)),
        n_receivers=int(cfg.get("n_receivers", 64)),
        channels=int(cfg.get("operator_channels", 48)),
        latent_dim=int(cfg.get("operator_latent_dim", 256)),
        dropout=float(cfg.get("dropout", 0.04)),
    ).to(device)
    state = ckpt.get("operator_state_dict", ckpt.get("model_state_dict"))
    if state is None:
        raise ValueError(f"Operator checkpoint missing state dict: {path}")
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt










