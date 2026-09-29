"""eikonal_forward_model.py

Differentiable Eikonal forward model WITHOUT torch.grid_sample.

This module is used by training and evaluation when --use_differentiable_eikonal.

Public API
----------
- ForwardLossWeights dataclass
- EikonalForwardModel(nn.Module)
    * forward(c_img, source_xy, query_xy) -> (B,P)
    * forward_losses(...)-> dict with key 'loss_forward_total'
    * predict_tof_matrix(...)-> (B,E,R)

Notes
-----
- Image tensors are (B,1,nx,ny) where the first spatial dim corresponds to x.
- Pixel coords are (x,y) with x in [0,nx-1], y in [0,ny-1].
- No torch.nn.functional.grid_sample is used (avoids grid_sampler_2d_backward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pix_to_norm_xy(xy_pix: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """Pixel (x,y) -> normalized [-1,1] (x,y)."""
    x = xy_pix[..., 0]
    y = xy_pix[..., 1]
    xd = float(max(nx - 1, 1))
    yd = float(max(ny - 1, 1))
    x_n = 2.0 * (x / xd) - 1.0
    y_n = 2.0 * (y / yd) - 1.0
    return torch.stack([x_n, y_n], dim=-1)


def bilinear_sample_2d(img_bchw: torch.Tensor, xy_pix_bph2: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear sampling without grid_sample.

    IMPORTANT: This pipeline uses an x-first layout for images:
        img_bchw is (B,C,nx,ny) where H==nx corresponds to x-index and W==ny corresponds to y-index.

    Therefore:
        xy_pix_bph2[...,0] is x-index in [0,nx-1]  -> indexes H
        xy_pix_bph2[...,1] is y-index in [0,ny-1]  -> indexes W

    Returns:
        (B,C,P)
    """
    if img_bchw.ndim != 4:
        raise ValueError(f"img must be (B,C,H,W), got {img_bchw.shape}")
    if xy_pix_bph2.ndim != 3 or xy_pix_bph2.size(-1) != 2:
        raise ValueError(f"xy must be (B,P,2), got {xy_pix_bph2.shape}")

    img = img_bchw
    if hasattr(img, "is_mkldnn") and img.is_mkldnn:
        img = img.to_dense()
    img = img.contiguous()
    if img.dtype != torch.float32:
        img = img.float()

    B, C, H, W = img.shape  # H==nx (x), W==ny (y)
    P = int(xy_pix_bph2.shape[1])

    x = xy_pix_bph2[..., 0].to(dtype=img.dtype)  # x-index -> H
    y = xy_pix_bph2[..., 1].to(dtype=img.dtype)  # y-index -> W

    x = x.clamp(0.0, float(H - 1))
    y = y.clamp(0.0, float(W - 1))

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, W - 1)

    x0_f = x0.to(dtype=img.dtype)
    y0_f = y0.to(dtype=img.dtype)
    wx = x - x0_f
    wy = y - y0_f

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    img_flat = img.view(B, C, H * W)

    def _g(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        # Linear index in x-first layout: idx = x*W + y
        idx = (xx * W + yy).view(B, 1, P).expand(B, C, P)
        return torch.gather(img_flat, 2, idx)

    v00 = _g(x0, y0)
    v10 = _g(x1, y0)
    v01 = _g(x0, y1)
    v11 = _g(x1, y1)

    w00 = w00.view(B, 1, P)
    w10 = w10.view(B, 1, P)
    w01 = w01.view(B, 1, P)
    w11 = w11.view(B, 1, P)

    return w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11


class _SoSEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, feat_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PointMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ForwardLossWeights:
    lambda_supT: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 1.0



class EikonalForwardModel(nn.Module):
    """
    Differentiable travel-time surrogate trained against MSFM collocation targets.

    The network is deliberately non-dimensionalized:
      - the SoS image is supplied to the CNN/MLP as c/c0, not raw speeds around 1500;
      - query/source coordinates are normalized to [-1,1];
      - the predicted travel time is dimensionless and converted to seconds with T0.

    The travel-time ansatz is
        T_hat(q;S,c) = ||q_n-S_n|| * softplus(NN(q_n,S_n,c/c0,features)),
    so T_hat(S)=0 exactly and the output is non-negative, while retaining
    position-sensitive travel times and useful spatial gradients.
    """

    architecture_version = "eikonal_forward_model_scaled_distance_ansatz"

    def __init__(self, *, nx: int, ny: int, dx: float, dy: float, feat_ch: int = 16, c0: float = 1500.0):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.c0 = float(c0)
        self.feat_ch = int(feat_ch)

        Lx = float(max(self.nx - 1, 1)) * self.dx
        Ly = float(max(self.ny - 1, 1)) * self.dy
        L = (Lx * Lx + Ly * Ly) ** 0.5
        T0 = max(L / max(self.c0, 1e-6), 1e-9)
        self.register_buffer("T0", torch.tensor(float(T0), dtype=torch.float32))

        self.encoder = _SoSEncoder(1, feat_ch)
        self.mlp = _PointMLP(in_dim=feat_ch + 1 + 2 + 2, hidden=128)

    def metadata(self) -> dict:
        return {
            "architecture_version": self.architecture_version,
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "c0": self.c0,
            "feat_ch": self.feat_ch,
        }

    def is_compatible_metadata(self, md: Optional[dict]) -> bool:
        if not isinstance(md, dict):
            return False
        ref = self.metadata()
        for key in ("architecture_version", "nx", "ny", "feat_ch"):
            if md.get(key) != ref[key]:
                return False
        for key in ("dx", "dy", "c0"):
            if abs(float(md.get(key, float("nan"))) - float(ref[key])) > 1e-12:
                return False
        return True

    def _normalize_c(self, c_img: torch.Tensor) -> torch.Tensor:
        """Return c/c0 clipped to a conservative range for stable CNN/MLP inputs."""
        c = c_img.to(dtype=torch.float32)
        c_scaled = c / max(self.c0, 1e-6)
        return torch.clamp(c_scaled, 0.5, 1.5)

    def _forward_dimless(self, c_img: torch.Tensor, source_xy: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        """Return dimensionless non-negative travel times at query points: (B,P)."""
        B = c_img.shape[0]
        P = query_xy.shape[1]

        c_scaled = self._normalize_c(c_img)
        feats = self.encoder(c_scaled)  # (B,F,nx,ny)
        f_q = bilinear_sample_2d(feats, query_xy)  # (B,F,P)
        c_q = bilinear_sample_2d(c_scaled, query_xy)  # (B,1,P)

        qn = _pix_to_norm_xy(query_xy, self.nx, self.ny)  # (B,P,2)
        sn_single = _pix_to_norm_xy(source_xy, self.nx, self.ny).unsqueeze(1)  # (B,1,2)
        sn = sn_single.expand(B, P, 2)

        x_in = torch.cat([
            f_q.permute(0, 2, 1),
            c_q.permute(0, 2, 1),
            qn,
            sn,
        ], dim=-1)

        rate = F.softplus(self.mlp(x_in.reshape(B * P, -1)).reshape(B, P)) + 1e-6
        dist = torch.sqrt(torch.sum((qn - sn) * (qn - sn), dim=-1) + 1e-12)
        return dist * rate

    def forward(self, c_img: torch.Tensor, source_xy: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        """Return travel times in seconds at query points: (B,P)."""
        return self._forward_dimless(c_img, source_xy, query_xy) * self.T0

    def _pde_residual(self, c_img: torch.Tensor, source_xy: torch.Tensor, colloc_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute resid = (c/c0)*||grad_{x_n} T_dimless|| - 1.

        Returns resid, gradient norm, and the already-computed dimensionless
        travel time so forward_losses() can reuse it for MSFM supervision.
        """
        colloc_req = colloc_xy.clone().detach().requires_grad_(True)
        Tt = self._forward_dimless(c_img, source_xy, colloc_req)
        grad_pix = torch.autograd.grad(
            outputs=Tt,
            inputs=colloc_req,
            grad_outputs=torch.ones_like(Tt),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        sx = float(max(self.nx - 1, 1)) / 2.0
        sy = float(max(self.ny - 1, 1)) / 2.0
        dTdx_n = grad_pix[..., 0] * sx
        dTdy_n = grad_pix[..., 1] * sy
        gnorm = torch.sqrt(dTdx_n * dTdx_n + dTdy_n * dTdy_n + 1e-12)

        c_q_scaled = bilinear_sample_2d(self._normalize_c(c_img), colloc_xy)[:, 0, :]
        resid = c_q_scaled * gnorm - 1.0
        return resid, gnorm, Tt

    def forward_losses(
        self,
        *,
        c_img: torch.Tensor,
        source_xy: torch.Tensor,
        colloc_xy: torch.Tensor,
        T_target: Optional[torch.Tensor] = None,
        weights: Optional[ForwardLossWeights] = None,
    ) -> Dict[str, torch.Tensor]:
        w = weights or ForwardLossWeights()
        losses: Dict[str, torch.Tensor] = {}

        resid, gnorm, Tt_pred = self._pde_residual(c_img, source_xy, colloc_xy)
        losses["loss_pde"] = torch.mean(resid * resid) * float(w.lambda_pde)
        losses["grad_norm_mean"] = torch.mean(gnorm.detach())

        # The distance ansatz enforces T(source)=0 exactly.
        losses["loss_bc"] = losses["loss_pde"].new_zeros(())

        if T_target is not None:
            Tt_target = T_target / self.T0
            losses["loss_supT"] = F.mse_loss(Tt_pred, Tt_target) * float(w.lambda_supT)
        else:
            losses["loss_supT"] = losses["loss_pde"].new_zeros(())

        losses["loss_forward_total"] = losses["loss_pde"] + losses["loss_bc"] + losses["loss_supT"]
        return losses

    def predict_tof_matrix(
        self,
        *,
        c_img: torch.Tensor,
        emitters_xy: torch.Tensor,
        receivers_xy: torch.Tensor,
        emitters_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict ToF in seconds at receivers for selected emitters. Returns (B,E,R)."""
        B = c_img.shape[0]
        E_all = int(emitters_xy.shape[0])
        R = int(receivers_xy.shape[0])

        if emitters_k is None:
            E = E_all
            emit_use = emitters_xy
        else:
            E = int(max(1, min(int(emitters_k), E_all)))
            emit_use = emitters_xy[:E]

        recv = receivers_xy.unsqueeze(0).expand(B, R, 2)
        out = []
        for e in range(E):
            src = emit_use[e].unsqueeze(0).expand(B, 2)
            T = self.forward(c_img, src, recv)
            out.append(T.unsqueeze(1))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def predict_tof_matrix_nograd(
        self,
        *,
        c_img: torch.Tensor,
        emitters_xy: torch.Tensor,
        receivers_xy: torch.Tensor,
        emitters_k: Optional[int] = None,
    ) -> torch.Tensor:
        return self.predict_tof_matrix(
            c_img=c_img,
            emitters_xy=emitters_xy,
            receivers_xy=receivers_xy,
            emitters_k=emitters_k,
        )











