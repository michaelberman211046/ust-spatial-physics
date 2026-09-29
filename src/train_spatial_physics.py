import argparse
import copy
import os
import pprint
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PairsCacheDataset
from latent_operator_models import (
    denormalize_tof_seconds,
    normalize_tof_seconds,
    blur,
    format_duration,
    highpass,
    masked_l1,
    masked_mse,
    support_apply,
    tv_l1,
)
from logger import log_message as pipeline_log_message
from settings import set_output_folder
from reconstruction_utils import make_circular_support_mask, make_homogeneous_water_tof_norm, make_model_input
from frozen_stack import load_frozen_reconstruction_stack
from train_background_refiner import IndexedSubset, batch_progress, make_refiner_features, split_dataset, unpack


SCRIPT_VERSION = "spatial-physics-reconstructor-v1"
LOG_PREFIX = "[spatial_physics]"


_HTML_LOG_DISABLED=False


def _resize_tof_feature(x, h, w):
    return F.interpolate(x.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)


def _double_center(x, mask):
    m = mask.to(device=x.device, dtype=x.dtype)
    xm = x * m
    row = xm.sum(dim=2, keepdim=True) / m.sum(dim=2, keepdim=True).clamp_min(1.0)
    col = xm.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
    glob = xm.sum(dim=(1, 2), keepdim=True) / m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return (x - row - col + glob) * m


@torch.no_grad()
def make_physics_condition(batch, stack, device, support_mask, water_tof_norm, args):
    """Construct the Stage 7 condition directly from the clean Stage 1-4 stack."""
    base_model, auto, op, background_model, background_adjoint = stack
    target, tof, tof_mask, index = unpack(batch, device)
    target = support_apply(target, support_mask, args.water_norm)
    x_model = make_model_input(tof, tof_mask, args.use_tof_mask_channel, args.tof_feature_mode, water_tof_norm)
    initial = support_apply(base_model(x_model), support_mask, args.water_norm)
    z_initial = auto.encode(initial)
    initial_tof = op(initial)
    initial_residual_s = denormalize_tof_seconds(tof, args.tof_norm) - denormalize_tof_seconds(initial_tof, args.tof_norm)
    initial_mps = initial * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
    initial_adjoint = background_adjoint(initial_residual_s, initial_mps, tof_mask, support_mask)
    background_features = make_refiner_features(x_model, initial, initial_adjoint, support_mask, args)
    z_background, _delta, _gate = background_model(background_features, z_initial)
    background = support_apply(auto.decoder(z_background), support_mask, args.water_norm)

    b, _c, h, w = background.shape
    tof_image = F.interpolate(x_model, size=(h, w), mode="bilinear", align_corners=False)
    support = support_mask.to(device=device, dtype=background.dtype).unsqueeze(0).expand(b, 1, h, w) if support_mask is not None else torch.ones_like(background)
    dx = F.pad(background[..., :, 1:] - background[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(background[..., 1:, :] - background[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1e-8)
    base_condition = torch.cat([
        tof_image, initial, background, background - initial,
        blur(background, args.lowpass_kernel), highpass(background, args.detail_kernel), gradient, support,
    ], dim=1)
    background_tof = op(background)
    mask = tof_mask.to(device=device, dtype=tof.dtype)
    background_residual = (tof - background_tof) * mask
    initial_residual = (tof - initial_tof) * mask
    scale = max(float(args.physics_residual_norm_scale), 1e-6)
    residual_s = denormalize_tof_seconds(tof, args.tof_norm) - denormalize_tof_seconds(background_tof, args.tof_norm)
    background_mps = background * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
    adjoint = background_adjoint(residual_s, background_mps, tof_mask, support_mask)
    extra = torch.cat([
        _resize_tof_feature(background_residual, h, w),
        _resize_tof_feature(torch.tanh(background_residual / scale), h, w),
        _resize_tof_feature(_double_center(background_residual, mask), h, w),
        _resize_tof_feature(initial_residual, h, w),
        _resize_tof_feature(_double_center(initial_residual, mask), h, w),
        _resize_tof_feature(mask, h, w), adjoint, highpass(adjoint, args.detail_kernel),
    ], dim=1)
    return target, tof, tof_mask, index, initial, background, torch.cat([base_condition, extra], dim=1).detach(), background_residual


def log_message(message):
    """Never terminate long training merely because the optional HTML log vanished."""
    global _HTML_LOG_DISABLED
    if _HTML_LOG_DISABLED:
        sys.__stdout__.write(str(message)+"\n");sys.__stdout__.flush();return
    try:
        pipeline_log_message(message)
    except Exception as exc:
        _HTML_LOG_DISABLED=True
        sys.__stdout__.write(f"{LOG_PREFIX} disabling optional HTML log after {type(exc).__name__}: {exc}\n{message}\n");sys.__stdout__.flush()


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        groups = max(1, min(8, int(channels) // 8))
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Dropout2d(float(dropout)),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SpatialPhysicsResidualNet(nn.Module):
    """Direct residual-image reconstructor from spatial physics evidence."""

    def __init__(self, in_ch, channels=96, dropout=0.04, delta_limit=0.55):
        super().__init__()
        ch = int(channels)
        self.delta_limit = float(delta_limit)
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_ch), ch, 5, padding=2),
            nn.SiLU(inplace=True),
            ResidualBlock(ch, dropout),
        )
        self.down1 = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.SiLU(inplace=True), ResidualBlock(ch, dropout))
        self.down2 = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.SiLU(inplace=True), ResidualBlock(ch * 2, dropout))
        self.mid = nn.Sequential(ResidualBlock(ch * 2, dropout), ResidualBlock(ch * 2, dropout))
        self.up1 = nn.Sequential(nn.Conv2d(ch * 2 + ch, ch, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(ch, dropout))
        self.up0 = nn.Sequential(nn.Conv2d(ch + ch, ch, 3, padding=1), nn.SiLU(inplace=True), ResidualBlock(ch, dropout))
        self.head = nn.Sequential(
            nn.GroupNorm(max(1, min(8, ch // 8)), ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        b = self.mid(self.down2(s1))
        u1 = F.interpolate(b, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, s1], dim=1))
        u0 = F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, s0], dim=1))
        return self.delta_limit * torch.tanh(self.head(u0) / max(self.delta_limit, 1e-6))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--splits_path", default=None)
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--ae_path", required=True)
    p.add_argument("--operator_path", required=True)
    p.add_argument("--background_reconstructor_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--init_model_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model_type", default="deep")
    p.add_argument("--latent_res", type=int, default=36)
    p.add_argument("--use_tof_mask_channel", action="store_true")
    p.add_argument("--tof_feature_mode", choices=["raw", "residual_stack"], default="residual_stack")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1.2e-4)
    p.add_argument("--weight_decay", type=float, default=8e-5)
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
    p.add_argument("--lambda_res_l1", type=float, default=1.4)
    p.add_argument("--lambda_res_detail", type=float, default=1.2)
    p.add_argument("--lambda_final_mse", type=float, default=0.55)
    p.add_argument("--lambda_final_l1", type=float, default=0.25)
    p.add_argument("--lambda_tof", type=float, default=0.35)
    p.add_argument("--lambda_tv", type=float, default=0.00035)
    p.add_argument("--lambda_view_consistency", type=float, default=0.35)
    p.add_argument("--lambda_stripe_sensitivity", type=float, default=0.45)
    p.add_argument("--directional_artifact_amplitude", type=float, default=0.65)
    p.add_argument("--directional_artifact_width_min", type=float, default=0.025)
    p.add_argument("--directional_artifact_width_max", type=float, default=0.10)
    p.add_argument("--directional_artifact_count", type=int, default=3)
    p.add_argument("--directional_artifact_probability", type=float, default=0.50)
    p.add_argument("--nuisance_global_delay_us", type=float, default=0.35)
    p.add_argument("--nuisance_emitter_delay_us", type=float, default=0.30)
    p.add_argument("--nuisance_receiver_delay_us", type=float, default=0.30)
    p.add_argument("--nuisance_noise_us", type=float, default=0.035)
    p.add_argument("--nuisance_sector_drop_prob", type=float, default=0.30)
    p.add_argument("--nuisance_channel_drop_prob", type=float, default=0.015)
    p.add_argument("--nuisance_harmonics", type=int, default=3)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=14)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--preview_every", type=int, default=2)
    p.add_argument("--preview_count", type=int, default=4)
    p.add_argument("--batch_log_every", type=int, default=10)
    p.add_argument("--checkpoint_every_batches", type=int, default=250)
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


def sector_mask_like(tof_mask, axis, sector_idx, num_sectors):
    b, ne, nr = tof_mask.shape
    out = torch.zeros_like(tof_mask)
    if int(num_sectors) <= 0:
        return out
    if axis == "emitter":
        start = int(round(sector_idx * ne / float(num_sectors)))
        end = int(round((sector_idx + 1) * ne / float(num_sectors)))
        out[:, start:end, :] = tof_mask[:, start:end, :]
    else:
        start = int(round(sector_idx * nr / float(num_sectors)))
        end = int(round((sector_idx + 1) * nr / float(num_sectors)))
        out[:, :, start:end] = tof_mask[:, :, start:end]
    return out


def make_spatial_physics_features(batch, stack, device, support_mask, water_tof_norm, args):
    base_model, auto, op, background_model, background_adjoint_op = stack
    target, tof, tof_mask, idx, initial, background, cond, bg_res_norm = make_physics_condition(
        batch, stack, device, support_mask, water_tof_norm, args
    )
    bg_tof = op(background)
    residual_s = denormalize_tof_seconds(tof, args.tof_norm) - denormalize_tof_seconds(bg_tof, args.tof_norm)
    bg_mps = background * (float(args.sos_max) - float(args.sos_min)) + float(args.sos_min)
    sector_maps = []
    for s in range(int(args.num_emitter_sectors)):
        m = sector_mask_like(tof_mask, "emitter", s, int(args.num_emitter_sectors))
        a = background_adjoint_op(residual_s, bg_mps, m, support_mask)
        sector_maps.append(a)
        if bool(args.sector_highpass):
            sector_maps.append(highpass(a, args.detail_kernel))
    for s in range(int(args.num_receiver_sectors)):
        m = sector_mask_like(tof_mask, "receiver", s, int(args.num_receiver_sectors))
        a = background_adjoint_op(residual_s, bg_mps, m, support_mask)
        sector_maps.append(a)
        if bool(args.sector_highpass):
            sector_maps.append(highpass(a, args.detail_kernel))
    spatial = torch.cat([background, highpass(background, args.detail_kernel), cond] + sector_maps, dim=1).detach()
    return target, tof, tof_mask, idx, initial, background, spatial, bg_res_norm


def detail_l1(a, b, mask, kernel):
    return masked_l1(highpass(a, kernel), highpass(b, kernel), mask)


def nuisance_view(batch, args):
    """Synthetic-only acquisition nuisance; anatomy/target is never changed."""
    items=list(batch);tof=items[1].clone();mask=items[2].clone();physical=denormalize_tof_seconds(tof,args.tof_norm)
    b,ne,nr=physical.shape;dev=physical.device;dtype=physical.dtype
    ea=2*torch.pi*torch.arange(ne,device=dev,dtype=dtype)/max(1,ne);ra=2*torch.pi*torch.arange(nr,device=dev,dtype=dtype)/max(1,nr)
    emitter=torch.zeros((b,ne),device=dev,dtype=dtype);receiver=torch.zeros((b,nr),device=dev,dtype=dtype)
    for h in range(1,int(args.nuisance_harmonics)+1):
        scale=1.0/h;ec=torch.randn((b,2),device=dev,dtype=dtype);rc=torch.randn((b,2),device=dev,dtype=dtype)
        emitter+=scale*(ec[:,0:1]*torch.sin(h*ea)+ec[:,1:2]*torch.cos(h*ea));receiver+=scale*(rc[:,0:1]*torch.sin(h*ra)+rc[:,1:2]*torch.cos(h*ra))
    emitter=emitter/emitter.std(dim=1,keepdim=True).clamp_min(1e-6)*float(args.nuisance_emitter_delay_us)*1e-6
    receiver=receiver/receiver.std(dim=1,keepdim=True).clamp_min(1e-6)*float(args.nuisance_receiver_delay_us)*1e-6
    global_delay=(2*torch.rand((b,1,1),device=dev,dtype=dtype)-1)*float(args.nuisance_global_delay_us)*1e-6
    noise=torch.randn_like(physical)*float(args.nuisance_noise_us)*1e-6
    physical=physical+global_delay+emitter[:,:,None]+receiver[:,None,:]+noise
    if float(args.nuisance_sector_drop_prob)>0:
        sectors=max(1,int(args.num_emitter_sectors));width=max(1,ne//sectors)
        for i in range(b):
            if torch.rand((),device=dev)<float(args.nuisance_sector_drop_prob):
                s=int(torch.randint(sectors,(1,),device=dev));mask[i,s*width:min(ne,(s+1)*width),:]=0
            if torch.rand((),device=dev)<float(args.nuisance_sector_drop_prob):
                s=int(torch.randint(sectors,(1,),device=dev));mask[i,:,s*max(1,nr//sectors):min(nr,(s+1)*max(1,nr//sectors))]=0
    mask=mask*(torch.rand_like(mask)>float(args.nuisance_channel_drop_prob)).to(mask.dtype)
    items[1]=normalize_tof_seconds(physical,args.tof_norm)*mask;items[2]=mask
    return tuple(items)


def inject_directional_feature_artifacts(features, args):
    """Corrupt only the final sector-adjoint channels, never anatomy/background channels."""
    out=features.clone();b,_c,h,w=out.shape;dev=out.device;dtype=out.dtype
    yy=torch.linspace(-1,1,h,device=dev,dtype=dtype)[None,:,None]
    xx=torch.linspace(-1,1,w,device=dev,dtype=dtype)[None,None,:]
    bands=torch.zeros((b,h,w),device=dev,dtype=dtype)
    for _ in range(max(1,int(args.directional_artifact_count))):
        theta=torch.rand((b,1,1),device=dev,dtype=dtype)*torch.pi
        offset=torch.rand((b,1,1),device=dev,dtype=dtype)*1.6-.8
        width=float(args.directional_artifact_width_min)+(float(args.directional_artifact_width_max)-float(args.directional_artifact_width_min))*torch.rand((b,1,1),device=dev,dtype=dtype)
        signed=(torch.randint(0,2,(b,1,1),device=dev)*2-1).to(dtype)
        distance=xx*torch.cos(theta)+yy*torch.sin(theta)-offset
        bands+=signed*torch.exp(-.5*torch.square(distance/width.clamp_min(1e-3)))
    bands=bands-bands.mean(dim=(1,2),keepdim=True)
    bands=bands/bands.std(dim=(1,2),keepdim=True).clamp_min(1e-6)
    per_sector=2 if bool(args.sector_highpass) else 1
    sector_channels=per_sector*(int(args.num_emitter_sectors)+int(args.num_receiver_sectors))
    if sector_channels<=0:return out
    start=max(0,out.shape[1]-sector_channels)
    active=(torch.rand((b,1,1,1),device=dev)<float(args.directional_artifact_probability)).to(dtype)
    gains=torch.empty((b,out.shape[1]-start,1,1),device=dev,dtype=dtype).uniform_(.35,1.0)
    out[:,start:]+=active*float(args.directional_artifact_amplitude)*gains*bands[:,None]
    return out


def directional_sensitivity(prediction_difference, mask):
    """Measure row/column/diagonal coherent energy caused only by corruption."""
    d=prediction_difference*mask
    row=d.mean(dim=3);col=d.mean(dim=2)
    diag1=torch.diagonal(d,dim1=2,dim2=3).mean(dim=2)
    diag2=torch.diagonal(torch.flip(d,dims=(3,)),dim1=2,dim2=3).mean(dim=2)
    return row.abs().mean()+col.abs().mean()+diag1.abs().mean()+diag2.abs().mean()


@torch.no_grad()
def evaluate_corrupted(model, stack, loader, device, support_mask, water_tof_norm, args):
    model.eval();mse=consistency=stripe=0.;n=0
    for batch in loader:
        clean=make_spatial_physics_features(batch,stack,device,support_mask,water_tof_norm,args);target=clean[0]
        corrupt=make_spatial_physics_features(nuisance_view(batch,args),stack,device,support_mask,water_tof_norm,args)
        clean_pred=support_apply(clean[5]+model(clean[6]),support_mask,args.water_norm)
        corrupt_feat=inject_directional_feature_artifacts(corrupt[6],args)
        corrupt_pred=support_apply(corrupt[5]+model(corrupt_feat),support_mask,args.water_norm)
        bs=int(target.shape[0]);mse+=float(masked_mse(corrupt_pred,target,support_mask))*bs
        consistency+=float(masked_l1(corrupt_pred,clean_pred,support_mask))*bs
        stripe+=float(directional_sensitivity(corrupt_pred-clean_pred,support_mask))*bs;n+=bs
    return {"corrupt_pred_mse":mse/max(1,n),"clean_corrupt_l1":consistency/max(1,n),"stripe_sensitivity":stripe/max(1,n)}


@torch.no_grad()
def evaluate(model, stack, loader, device, support_mask, water_tof_norm, args):
    model.eval()
    op = stack[2]
    sums = {
        "background_mse": 0.0,
        "pred_mse": 0.0,
        "pred_l1": 0.0,
        "res_detail_l1": 0.0,
        "tof_mse": 0.0,
    }
    n = 0
    for batch in loader:
        target, tof, tof_mask, _idx, _initial, background, feat, _bg_res_norm = make_spatial_physics_features(
            batch, stack, device, support_mask, water_tof_norm, args
        )
        target_delta = target - background
        pred_delta = model(feat)
        pred = support_apply(background + pred_delta, support_mask, args.water_norm)
        pred_tof = op(pred)
        bs = int(target.shape[0])
        sums["background_mse"] += float(masked_mse(background, target, support_mask).item()) * bs
        sums["pred_mse"] += float(masked_mse(pred, target, support_mask).item()) * bs
        sums["pred_l1"] += float(masked_l1(pred, target, support_mask).item()) * bs
        sums["res_detail_l1"] += float(detail_l1(pred_delta, target_delta, support_mask, args.detail_kernel).item()) * bs
        sums["tof_mse"] += float(masked_mse(pred_tof, tof, tof_mask).item()) * bs
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def plot_preview(model, stack, loader, device, support_mask, water_tof_norm, args, title):
    log_message(f"{LOG_PREFIX} preview deferred; V24 training process intentionally imports no plotting library")


def save_ckpt(path, args, model, best_state, best_score, best_epoch, epoch, hist, opt, in_ch):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "kind": "spatial_physics_reconstructor",
            "script_version": SCRIPT_VERSION,
            "config": vars(args),
            "in_ch": int(in_ch),
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_state,
            "best_val_score": float(best_score),
            "best_epoch": int(best_epoch),
            "last_epoch": int(epoch),
            "history": hist,
            "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        },
        path,
    )


def main():
    args = parse_args()
    if args.output_dir:
        set_output_folder(args.output_dir)
    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message(f"{LOG_PREFIX} script_version={SCRIPT_VERSION}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        log_message(f"{LOG_PREFIX} CUDA requested but unavailable. Actual device will be CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    log_message(f"{LOG_PREFIX} Actual device used: {device}")

    ds = PairsCacheDataset(args.data_path, sos_min=args.sos_min, sos_max=args.sos_max, return_tof_mask=True)
    args.tof_norm = ds.tof_norm
    meta = dict(getattr(ds, "metadata", {}) or {})
    for key in ["nx", "ny", "n_emitters", "n_receivers"]:
        setattr(args, key, int(meta.get(key, getattr(args, key))))
    for key in ["phys_x", "phys_y", "radius", "sos_water"]:
        setattr(args, key, float(meta.get(key, getattr(args, key))))
    args.water_norm = float(max(0.0, min(1.0, (args.sos_water - args.sos_min) / (args.sos_max - args.sos_min))))
    train_ds, val_ds = split_dataset(ds, args.data_path, args.splits_path)
    if args.max_train_samples > 0:
        train_ds = IndexedSubset(ds, train_ds.indices[: min(int(args.max_train_samples), len(train_ds))])
    if args.max_val_samples > 0:
        val_ds = IndexedSubset(ds, val_ds.indices[: min(int(args.max_val_samples), len(val_ds))])
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    support_mask = make_circular_support_mask(args.nx, args.ny, args.phys_x, args.phys_y, args.mask_radius, device=device) if args.use_circular_mask else None
    water_tof_norm = make_homogeneous_water_tof_norm(
        args.nx, args.ny, args.phys_x, args.phys_y, args.radius,
        args.n_emitters, args.n_receivers, args.sos_water, args.tof_norm, device
    ) if args.tof_feature_mode == "residual_stack" else None
    stack = load_frozen_reconstruction_stack(args, train_loader, device, support_mask, water_tof_norm)
    first = next(iter(train_loader))
    *_unused, feat0, _res0 = make_spatial_physics_features(first, stack, device, support_mask, water_tof_norm, args)
    in_ch = int(feat0.shape[1])
    log_message(f"{LOG_PREFIX} spatial feature channels={in_ch}")
    model = SpatialPhysicsResidualNet(in_ch, args.channels, args.dropout, args.delta_limit).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch = 1
    hist = []
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    if args.resume and os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        if ckpt.get("kind") == "spatial_physics_reconstructor" and int(ckpt.get("in_ch", -1)) == in_ch:
            model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("optimizer_state_dict") is not None:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            hist = list(ckpt.get("history", []))
            best_score = float(ckpt.get("best_val_score", best_score))
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_state = copy.deepcopy(ckpt.get("best_model_state_dict", model.state_dict()))
            start_epoch = int(ckpt.get("last_epoch", 0)) + 1
            log_message(f"{LOG_PREFIX} resumed from epoch {start_epoch}")
    elif args.init_model_path:
        init_ckpt=torch.load(args.init_model_path,map_location=device,weights_only=False)
        if init_ckpt.get("kind")!="spatial_physics_reconstructor" or int(init_ckpt.get("in_ch",-1))!=in_ch:
            raise RuntimeError(f"Incompatible initialization checkpoint: {args.init_model_path}")
        model.load_state_dict(init_ckpt.get("best_model_state_dict",init_ckpt["model_state_dict"]))
        best_state=copy.deepcopy(model.state_dict())
        log_message(f"{LOG_PREFIX} initialized weights from {args.init_model_path}; optimizer and epoch start are fresh")

    stale = 0
    phase_t0 = time.time()
    last_completed_epoch = start_epoch - 1
    interrupted = False
    op = stack[2]
    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            model.train()
            epoch_t0 = time.time()
            total = 0.0
            seen = 0
            for bi, batch in enumerate(train_loader, 1):
                clean_tof=batch[1].to(device);clean_mask=batch[2].to(device)
                clean=make_spatial_physics_features(batch,stack,device,support_mask,water_tof_norm,args)
                corrupt=make_spatial_physics_features(nuisance_view(batch,args),stack,device,support_mask,water_tof_norm,args)
                views=[];view_losses=[]
                for vi,items in enumerate((clean,corrupt)):
                    target, _tof_aug, _mask_aug, _idx, _initial, background, feat, _bg_res_norm=items
                    if vi==1:feat=inject_directional_feature_artifacts(feat,args)
                    target_delta=target-background;pred_delta=model(feat);pred=support_apply(background+pred_delta,support_mask,args.water_norm);pred_tof=op(pred)
                    view_losses.append(float(args.lambda_res_l1)*masked_l1(pred_delta,target_delta,support_mask)+float(args.lambda_res_detail)*detail_l1(pred_delta,target_delta,support_mask,args.detail_kernel)+float(args.lambda_final_mse)*masked_mse(pred,target,support_mask)+float(args.lambda_final_l1)*masked_l1(pred,target,support_mask)+float(args.lambda_tof)*masked_mse(pred_tof,clean_tof,clean_mask)+float(args.lambda_tv)*tv_l1(pred_delta,support_mask));views.append(pred)
                view_difference=views[1]-views[0]
                loss=.5*(view_losses[0]+view_losses[1])+float(args.lambda_view_consistency)*masked_l1(views[0],views[1],support_mask)+float(args.lambda_stripe_sensitivity)*directional_sensitivity(view_difference,support_mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                bs = int(target.shape[0])
                total += float(loss.item()) * bs
                seen += bs
                if int(args.checkpoint_every_batches)>0 and bi%int(args.checkpoint_every_batches)==0:
                    save_ckpt(args.model_path,args,model,best_state,best_score,best_epoch,epoch-1,hist,opt,in_ch)
                if bi == 1 or bi == len(train_loader) or bi % int(args.batch_log_every) == 0:
                    log_message(batch_progress(LOG_PREFIX, epoch, args.epochs, bi, len(train_loader), epoch_t0, phase_t0) + f" loss={float(loss.item()):.6g}")
            train_loss = total / max(1, seen)
            val = evaluate(model, stack, val_loader, device, support_mask, water_tof_norm, args)
            robust_val=evaluate_corrupted(model,stack,val_loader,device,support_mask,water_tof_norm,args)
            score = val["pred_mse"] + 0.15 * val["res_detail_l1"] + 0.05 * val["tof_mse"] + 0.50*robust_val["corrupt_pred_mse"] + 0.20*robust_val["clean_corrupt_l1"] + 0.10*robust_val["stripe_sensitivity"]
            improved = score < best_score - float(args.early_stop_min_delta)
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            hist.append({"epoch": epoch, "train": train_loss, "score": score, **val, **robust_val})
            last_completed_epoch = epoch
            elapsed = time.time() - phase_t0
            eta = elapsed / max(1, epoch - start_epoch + 1) * max(0, int(args.epochs) - epoch)
            log_message(
                f"{LOG_PREFIX} epoch {epoch:04d}/{int(args.epochs):04d}: train={train_loss:.6g} score={score:.6g} "
                f"background_mse={val['background_mse']:.6g} pred_mse={val['pred_mse']:.6g} "
                f"detail={val['res_detail_l1']:.6g} tof_mse={val['tof_mse']:.6g} "
                f"corrupt_mse={robust_val['corrupt_pred_mse']:.6g} consistency={robust_val['clean_corrupt_l1']:.6g} stripe={robust_val['stripe_sensitivity']:.6g} "
                f"best={best_score:.6g}@{best_epoch} stale={stale}/{int(args.early_stop_patience)} "
                f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}"
            )
            save_ckpt(args.model_path, args, model, best_state, best_score, best_epoch, epoch, hist, opt, in_ch)
            if int(args.preview_count)>0 and (epoch == start_epoch or epoch % int(args.preview_every) == 0 or improved):
                model.load_state_dict(best_state)
                plot_preview(model, stack, val_loader, device, support_mask, water_tof_norm, args, "Spatial physics residual reconstruction")
                model.train()
            if stale >= int(args.early_stop_patience):
                log_message(f"{LOG_PREFIX} early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break
    except KeyboardInterrupt:
        interrupted = True
        log_message(f"{LOG_PREFIX} Ctrl-C received. Saving last/best checkpoint before exit.")
    finally:
        save_ckpt(args.model_path, args, model, best_state, best_score, best_epoch, last_completed_epoch, hist, opt, in_ch)
        model.load_state_dict(best_state)
        if int(args.preview_count)>0 and len(val_loader) > 0:
            plot_preview(model, stack, val_loader, device, support_mask, water_tof_norm, args, "Spatial physics residual reconstruction final/best preview")
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()












