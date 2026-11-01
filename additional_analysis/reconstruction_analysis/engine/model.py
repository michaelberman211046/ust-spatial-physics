import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import log_message
from anatomy import generate_sensor_positions


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResidualRefiner(nn.Module):
    """
    Small full-resolution residual corrector.

    Input channels:
      - coarse normalized SoS
      - resized raw ToF
      - resized geometry prior
    Output:
      bounded residual in [-scale, scale]
    """
    def __init__(self, in_ch: int = 3, hidden: int = 16, scale: float = 0.10):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden),
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(x))


class GeometryPriorMixin:
    def _build_geometry_prior_matrix_core(self, prior_nx: int, prior_ny: int, log_tag: str):
        dx = self.phys_x / max(1, self.nx - 1)
        dy = self.phys_y / max(1, self.ny - 1)
        emitters, receivers = generate_sensor_positions(
            self.nx, self.ny, dx, dy, self.radius, self.n_emitters, self.n_receivers
        )

        xs = np.linspace(0.0, self.phys_x, prior_nx, dtype=np.float32)
        ys = np.linspace(0.0, self.phys_y, prior_ny, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

        sigma = 0.18 * max(self.phys_x / max(1, prior_nx), self.phys_y / max(1, prior_ny))
        sigma = max(float(sigma), 1e-4)

        rows = []
        lengths = []
        for e in emitters:
            ex = float(e[0]) * dx
            ey = float(e[1]) * dy
            p0 = np.array([ex, ey], dtype=np.float32)
            for r in receivers:
                rx = float(r[0]) * dx
                ry = float(r[1]) * dy
                p1 = np.array([rx, ry], dtype=np.float32)
                chord = p1 - p0
                denom = float(np.dot(chord, chord)) + 1e-12
                rel = pts - p0[None, :]
                t = np.sum(rel * chord[None, :], axis=1) / denom
                t_clip = np.clip(t, 0.0, 1.0)
                closest = p0[None, :] + t_clip[:, None] * chord[None, :]
                dist2 = np.sum((pts - closest) ** 2, axis=1)
                line_w = np.exp(-0.5 * dist2 / (sigma ** 2))
                seg_gate = (t >= -0.08) & (t <= 1.08)
                center_bias = 0.50 + 0.50 * np.sin(np.pi * t_clip) ** 2
                w = line_w * center_bias * seg_gate.astype(np.float32)
                s = float(w.sum())
                if s > 0.0:
                    w = w / s
                rows.append(w.astype(np.float32))
                lengths.append(np.linalg.norm(p1 - p0) + 1e-12)

        mat = np.stack(rows, axis=0)
        lengths = np.array(lengths, dtype=np.float32)
        log_message(f"[model.py] Geometry prior matrix shape{log_tag}: {mat.shape}")
        log_message(f"[model.py] Path lengths count{log_tag}: {lengths.shape}")
        return torch.from_numpy(mat), torch.from_numpy(lengths)

    def _compute_prior_core(self, tof_2d: torch.Tensor, prior_matrix: torch.Tensor, prior_nx: int, prior_ny: int, mask_2d: torch.Tensor | None = None):
        b = tof_2d.shape[0]
        flat = tof_2d.reshape(b, -1)
        p = self.n_emitters * self.n_receivers
        if flat.shape[1] != p:
            raise ValueError(
                f"[model.py] Expected ToF shape ({self.n_emitters},{self.n_receivers}) => {p} values, got {tuple(tof_2d.shape)}"
            )

        if mask_2d is not None:
            m = torch.clamp(mask_2d.reshape(b, -1).to(device=flat.device, dtype=flat.dtype), 0.0, 1.0)
            count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (flat * m).sum(dim=1, keepdim=True) / count
            var = (((flat - mean) * m) ** 2).sum(dim=1, keepdim=True) / count
            flat_scaled = (flat - mean) / torch.sqrt(var + 1e-6)
            flat_scaled = flat_scaled * m
        else:
            flat_centered = flat - flat.mean(dim=1, keepdim=True)
            flat_scaled = flat_centered / (flat.std(dim=1, keepdim=True, unbiased=False) + 1e-6)

        prior = flat_scaled @ prior_matrix
        prior = prior.view(b, 1, prior_nx, prior_ny)
        prior = prior - prior.amin(dim=(-2, -1), keepdim=True)
        prior = prior / (prior.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return prior


class ReconstructionNet(nn.Module, GeometryPriorMixin):
    """
    Coarse-to-fine geometry-aware baseline.

    Public interface preserved.
    Output remains normalized SoS in [0,1].
    """
    def __init__(
        self,
        nx,
        ny,
        latent_res=24,
        phys_x=0.10,
        phys_y=0.10,
        radius=0.04,
        n_emitters=32,
        n_receivers=32,
        tof_input_channels: int = 1,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.phys_x = float(phys_x)
        self.phys_y = float(phys_y)
        self.radius = float(radius)
        self.n_emitters = int(n_emitters)
        self.n_receivers = int(n_receivers)
        self.tof_input_channels = int(tof_input_channels)
        if self.tof_input_channels < 1:
            raise ValueError(f"[model.py] tof_input_channels must be >= 1, got {self.tof_input_channels}")

        ratio = self.nx / max(1, self.ny)
        self.latent_nx = int(latent_res)
        self.latent_ny = max(1, int(round(float(latent_res) / ratio)))
        # Keep the geometry prior deliberately coarser than the latent grid.
        self.prior_nx = min(max(12, self.latent_nx // 2), 16)
        self.prior_ny = min(max(12, self.latent_ny // 2), 16)
        self.mid_nx = max(self.latent_nx * 2, self.nx // 2)
        self.mid_ny = max(self.latent_ny * 2, self.ny // 2)

        log_message('.')
        log_message(f"[model.py] Initializing Geometry-aware ReconstructionNet for {self.nx}x{self.ny} output.")
        log_message(f"[model.py] Latent bottleneck resolution: {self.latent_nx}x{self.latent_ny}")
        log_message(f"[model.py] Prior grid: {self.prior_nx}x{self.prior_ny}")
        log_message(
            f"[model.py] Geometry: phys=({self.phys_x:.4f},{self.phys_y:.4f}) m, "
            f"radius={self.radius:.4f} m, emitters={self.n_emitters}, receivers={self.n_receivers}, "
            f"tof_input_channels={self.tof_input_channels}"
        )

        prior_matrix, path_lengths = self._build_geometry_prior_matrix_core(self.prior_nx, self.prior_ny, log_tag='')
        self.register_buffer('prior_matrix', prior_matrix, persistent=True)
        self.register_buffer('path_lengths', path_lengths, persistent=True)

        # The ToF validity mask is a gating signal rather than an acoustic feature.
        # The learnable path therefore receives only the gated normalized ToF channel.
        self.tof_stem = ConvBlock(1, 24)
        self.prior_stem = ConvBlock(1, 24)
        self.tof_proj = nn.Conv2d(24, 24, kernel_size=1)
        self.prior_proj = nn.Conv2d(24, 24, kernel_size=1)
        self.alpha_logits = nn.Parameter(torch.full((1, 24, 1, 1), 1.5))

        self.fuse = ConvBlock(24, 32)
        self.pool = nn.AvgPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 64)
        self.up = UpsampleConvBlock(64, 32)
        self.dec = ConvBlock(32 + 32, 32)
        self.up_mid = UpsampleConvBlock(32, 24)
        self.up_full = UpsampleConvBlock(24, 16)
        self.coarse_logits = nn.Conv2d(16, 1, kernel_size=1)

        self.refiner = ResidualRefiner(in_ch=3, hidden=16, scale=0.08)

    def _compute_prior(self, tof_2d: torch.Tensor, mask_2d: torch.Tensor | None = None) -> torch.Tensor:
        return self._compute_prior_core(tof_2d, self.prior_matrix, self.prior_nx, self.prior_ny, mask_2d=mask_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"[model.py] Unexpected input shape: {tuple(x.shape)}")

        if x.shape[1] not in (1, 2):
            raise ValueError(f"[model.py] Unexpected channel count in input: {tuple(x.shape)}")
        if x.shape[1] != self.tof_input_channels:
            raise ValueError(
                f"[model.py] Model expects tof_input_channels={self.tof_input_channels}, "
                f"but got input shape {tuple(x.shape)}"
            )

        tof_value = x[:, 0:1, :, :]
        tof_valid = None
        if x.shape[1] == 2:
            # The second channel is a validity mask.  Use it only to gate the ToF
            # values and the geometry prior.  Do not feed the raw mask as a
            # learnable feature channel.
            tof_valid = torch.clamp(x[:, 1:2, :, :], 0.0, 1.0)
            tof_value = tof_value * tof_valid

        tof_2d = tof_value[:, 0, :, :]
        mask_2d = tof_valid[:, 0, :, :] if tof_valid is not None else None
        x_lat = F.interpolate(tof_value, size=(self.latent_nx, self.latent_ny), mode='bilinear', align_corners=False)
        prior = self._compute_prior(tof_2d, mask_2d=mask_2d)
        prior_lat = F.interpolate(prior, size=(self.latent_nx, self.latent_ny), mode='bilinear', align_corners=False)

        f_tof = self.tof_proj(self.tof_stem(x_lat))
        f_prior = self.prior_proj(self.prior_stem(prior_lat))
        alpha = torch.sigmoid(self.alpha_logits)
        f0 = alpha * f_tof + (1.0 - alpha) * f_prior
        f0 = self.fuse(f0)

        f1 = self.enc2(self.pool(f0))
        fb = self.bottleneck(f1)
        fu = self.up(fb, size=f0.shape[-2:])
        fd = self.dec(torch.cat([fu, f0], dim=1))
        fd = self.up_mid(fd, size=(self.mid_nx, self.mid_ny))
        fd = self.up_full(fd, size=(self.nx, self.ny))

        coarse = torch.sigmoid(self.coarse_logits(fd))

        x_full = F.interpolate(tof_value, size=(self.nx, self.ny), mode='bilinear', align_corners=False)
        prior_full = F.interpolate(prior, size=(self.nx, self.ny), mode='bilinear', align_corners=False)
        refiner_inputs = [coarse, x_full, prior_full]
        delta = self.refiner(torch.cat(refiner_inputs, dim=1))
        y = torch.clamp(coarse + delta, 0.0, 1.0)
        return y


class ImprovedReconstructionNet(ReconstructionNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log_message("[model.py] ImprovedReconstructionNet uses the coarse-to-fine geometry-aware baseline core.")


class DeepGatedReconstructionNet(nn.Module):
    """
    Global ToF-to-SoS inverse model for the difficult synthetic reconstruction task.

    The complete ToF matrix is treated as a global measurement
    vector, encodes it with fully connected layers, then decodes a spatial SoS map
    with a CNN.  The optional ToF mask is used as a second measurement vector, not
    as an acoustic image feature.
    """
    def __init__(
        self,
        nx,
        ny,
        latent_res=36,
        phys_x=0.10,
        phys_y=0.10,
        radius=0.04,
        n_emitters=32,
        n_receivers=32,
        tof_input_channels: int = 1,
        fc_dropout: float = 0.08,
        decoder_dropout: float = 0.04,
        output_channels: int = 1,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.phys_x = float(phys_x)
        self.phys_y = float(phys_y)
        self.radius = float(radius)
        self.n_emitters = int(n_emitters)
        self.n_receivers = int(n_receivers)
        self.tof_input_channels = int(tof_input_channels)
        self.fc_dropout = float(max(0.0, fc_dropout))
        self.decoder_dropout = float(max(0.0, decoder_dropout))
        self.output_channels = int(output_channels)
        if self.tof_input_channels < 1:
            raise ValueError(f"[model.py] tof_input_channels must be >= 1, got {self.tof_input_channels}")
        if self.output_channels < 1:
            raise ValueError(f"[model.py] output_channels must be >= 1, got {self.output_channels}")

        # Honor --latent_res exactly (with only a small lower safety bound).
        # Earlier revisions silently capped latent_res at 25, so a command using
        # --latent_res 36 actually trained a 25x25 seed grid.  That made the
        # command/log misleading and reduced spatial capacity.
        ratio = self.nx / max(1, self.ny)
        self.latent_nx = int(max(12, latent_res))
        self.latent_ny = max(12, int(round(float(self.latent_nx) / ratio)))
        self.measurement_dim = self.n_emitters * self.n_receivers
        self.encoder_in_dim = self.measurement_dim * self.tof_input_channels
        self.seed_channels = 32
        hidden = 768

        log_message('.')
        log_message(f"[model.py] Initializing GlobalToF DeepGatedReconstructionNet for {self.nx}x{self.ny} output.")
        log_message(
            f"[model.py] Global encoder: input_dim={self.encoder_in_dim}, hidden={hidden}, "
            f"seed=({self.seed_channels},{self.latent_nx},{self.latent_ny})"
        )
        log_message(
            f"[model.py] Geometry: phys=({self.phys_x:.4f},{self.phys_y:.4f}) m, "
            f"radius={self.radius:.4f} m, emitters={self.n_emitters}, receivers={self.n_receivers}, "
            f"tof_input_channels={self.tof_input_channels}, output_channels={self.output_channels}"
        )
        log_message(
            f"[model.py] Regularization: fc_dropout={self.fc_dropout:.3g}, "
            f"decoder_dropout={self.decoder_dropout:.3g}"
        )

        self.fc = nn.Sequential(
            nn.Linear(self.encoder_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p=self.fc_dropout),
            nn.Linear(hidden, self.seed_channels * self.latent_nx * self.latent_ny),
            nn.GELU(),
        )

        self.dec0 = ConvBlock(self.seed_channels, 64)
        self.up1 = UpsampleConvBlock(64, 64)
        self.up2 = UpsampleConvBlock(64, 48)
        self.up3 = UpsampleConvBlock(48, 32)
        self.out_head = nn.Sequential(
            ConvBlock(32, 24),
            nn.Conv2d(24, self.output_channels, kernel_size=1),
        )
        self.spatial_dropout = nn.Dropout2d(p=self.decoder_dropout) if self.decoder_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze(1)
        elif x.dim() != 4:
            raise ValueError(f"[model.py] Unexpected input shape: {tuple(x.shape)}")

        if x.shape[1] != self.tof_input_channels:
            raise ValueError(
                f"[model.py] Model expects tof_input_channels={self.tof_input_channels}, "
                f"but got input shape {tuple(x.shape)}"
            )

        if self.tof_input_channels == 1:
            flat = x[:, 0:1, :, :].flatten(1)
        else:
            # Backward compatible convention:
            # channel 0..C-2 are acoustic ToF-derived features, last channel is
            # the validity mask when --use_tof_mask_channel is used.  All
            # acoustic channels are gated by the mask before flattening.
            mask = torch.clamp(x[:, -1:, :, :], 0.0, 1.0)
            acoustic = x[:, :-1, :, :] * mask
            flat = torch.cat([acoustic.flatten(1), mask.flatten(1)], dim=1)

        # Do NOT standardize each ToF sample inside the model.  The pairs cache
        # already applies the requested dataset-level ToF normalization.  A second
        # per-sample normalization removes the absolute travel-time level, which
        # is physically meaningful for estimating the mean SoS/slowness.
        if self.tof_input_channels > 1:
            m = flat[:, -self.measurement_dim:].clamp(0.0, 1.0)
            y = flat[:, :-self.measurement_dim]
            flat = torch.cat([y, m], dim=1)

        z = self.fc(flat)
        z = z.view(x.shape[0], self.seed_channels, self.latent_nx, self.latent_ny)
        z = self.dec0(z)
        z = self.spatial_dropout(z)
        z = self.up1(z, size=(max(self.latent_nx * 2, self.nx // 4), max(self.latent_ny * 2, self.ny // 4)))
        z = self.spatial_dropout(z)
        z = self.up2(z, size=(max(self.latent_nx * 4, self.nx // 2), max(self.latent_ny * 4, self.ny // 2)))
        z = self.spatial_dropout(z)
        z = self.up3(z, size=(self.nx, self.ny))
        out = self.out_head(z)
        if self.output_channels == 1:
            return torch.sigmoid(out)
        return out


class ReconstructionNetHeavy(ReconstructionNet):
    pass










