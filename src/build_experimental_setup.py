import argparse, pprint, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PairsCacheDataset
from logger import log_message
from settings import set_output_folder
from reconstruction_utils import make_circular_support_mask, make_homogeneous_water_tof_norm
from frozen_stack import load_frozen_reconstruction_stack
from train_background_refiner import IndexedSubset
from train_spatial_physics import SpatialPhysicsResidualNet, make_spatial_physics_features, support_apply

VERSION = "clean-experimental-prior"


def derive_support(baseline, water, ring, taper, radius_quantile):
    """Acquisition-centered support with radius estimated from experimental evidence.

    Stage 7 is allowed to estimate radial extent, but not center, angle or eccentricity.
    Thus a diagonal Stage-7 artifact cannot rotate or translate the anatomy mask.
    """
    ring4 = ring
    while ring4.ndim < 4:
        ring4 = ring4.unsqueeze(0)
    evidence = F.avg_pool2d(F.relu(water - baseline), 13, 1, 6) * ring4
    w = evidence[0, 0].detach().cpu().numpy()
    valid = ring4[0, 0].detach().cpu().numpy() > .5
    positive = w[valid & (w > 0)]
    if positive.size < 64:
        raise RuntimeError("Insufficient image evidence to estimate anatomical support")
    yy, xx = np.indices(w.shape)
    center = np.array([(w.shape[0]-1)/2, (w.shape[1]-1)/2], dtype=np.float64)
    radius = np.sqrt((yy-center[0])**2 + (xx-center[1])**2)
    # Suppress the weak broad halo, then take a weighted radial quantile.
    threshold = float(np.quantile(positive, .58))
    ew = np.where(valid & (w >= threshold), w-threshold, 0.0)
    if ew.sum() <= 0: raise RuntimeError("No radial support evidence survived robust threshold")
    order = np.argsort(radius.ravel()); cumulative = np.cumsum(ew.ravel()[order]); target = radius_quantile*cumulative[-1]
    radius_px = float(radius.ravel()[order[np.searchsorted(cumulative, target)]])
    ring_area = float(valid.sum()); ring_radius_px = np.sqrt(ring_area/np.pi)
    radius_px = float(np.clip(radius_px, .48*ring_radius_px, .72*ring_radius_px))
    hard = torch.as_tensor((radius <= radius_px) & valid, device=baseline.device, dtype=baseline.dtype)[None, None]
    soft = F.avg_pool2d(hard, 2 * taper + 1, 1, taper).clamp(0, 1) * ring4
    info = {
        "method": "acquisition_centered_data_radius",
        "center_pixels_yx": [float(x) for x in center],
        "radius_pixels": radius_px,
        "radius_fraction_of_ring": radius_px/ring_radius_px,
        "evidence_threshold": threshold,
        "radius_quantile": radius_quantile,
        "hard_fraction_of_grid": float(hard.mean()),
    }
    return soft, hard, evidence, info


def main():
    p = argparse.ArgumentParser()
    for key in ["data_path", "base_model_path", "ae_path", "operator_path", "background_reconstructor_path", "model_path", "output_dir", "out_pt"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--angular_sectors", type=int, default=8)
    p.add_argument("--support_taper_pixels", type=int, default=7)
    p.add_argument("--support_radius_quantile", type=float, default=.70)
    p.add_argument("--model_type", default="deep"); p.add_argument("--latent_res", type=int, default=36)
    p.add_argument("--use_tof_mask_channel", action="store_true"); p.add_argument("--tof_feature_mode", default="residual_stack")
    p.add_argument("--use_circular_mask", action="store_true"); p.add_argument("--mask_radius", type=float, default=.1091)
    p.add_argument("--nx", type=int, default=200); p.add_argument("--ny", type=int, default=200)
    p.add_argument("--phys_x", type=float, default=.24); p.add_argument("--phys_y", type=float, default=.24)
    p.add_argument("--radius", type=float, default=.1091); p.add_argument("--n_emitters", type=int, default=512); p.add_argument("--n_receivers", type=int, default=512)
    p.add_argument("--sos_min", type=float, default=1400); p.add_argument("--sos_max", type=float, default=1650); p.add_argument("--sos_water", type=float, default=1500)
    p.add_argument("--channels", type=int, default=96); p.add_argument("--dropout", type=float, default=.04); p.add_argument("--delta_limit", type=float, default=.55)
    p.add_argument("--physics_residual_norm_scale", type=float, default=.35); p.add_argument("--background_delta_limit", type=float, default=.65)
    p.add_argument("--adjoint_ray_samples", type=int, default=128); p.add_argument("--num_emitter_sectors", type=int, default=8); p.add_argument("--num_receiver_sectors", type=int, default=0)
    p.add_argument("--sector_highpass", action="store_true"); p.add_argument("--lowpass_kernel", type=int, default=21); p.add_argument("--detail_kernel", type=int, default=7); p.add_argument("--adjoint_blur_kernel", type=int, default=7)
    a = p.parse_args();ck = torch.load(a.model_path, map_location="cpu", weights_only=False);cfg = dict(ck.get("config",{}) or {})
    for key in ["sos_min","sos_max","sos_water","channels","dropout","delta_limit","num_emitter_sectors","num_receiver_sectors","sector_highpass","physics_residual_norm_scale","background_delta_limit","adjoint_ray_samples","lowpass_kernel","detail_kernel","adjoint_blur_kernel"]:
        if key in cfg:setattr(a,key,cfg[key])
    set_output_folder(a.output_dir)
    log_message("..................."); log_message(f"[CMD] {' '.join(sys.argv)}"); log_message(f"[ARGS]\n{pprint.pformat(vars(a))}"); log_message(f"[experimental_setup] version={VERSION}")
    dev = torch.device(a.device if not a.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    ds = PairsCacheDataset(a.data_path, sos_min=a.sos_min, sos_max=a.sos_max, return_tof_mask=True)
    a.tof_norm = ds.tof_norm; a.water_norm = (a.sos_water - a.sos_min) / (a.sos_max - a.sos_min)
    loader = DataLoader(IndexedSubset(ds, [0]), batch_size=1, shuffle=False)
    ring = make_circular_support_mask(a.nx, a.ny, a.phys_x, a.phys_y, a.mask_radius, device=dev)
    water_tof = make_homogeneous_water_tof_norm(a.nx, a.ny, a.phys_x, a.phys_y, a.radius, a.n_emitters, a.n_receivers, a.sos_water, a.tof_norm, dev)
    stack = load_frozen_reconstruction_stack(a, loader, dev, ring, water_tof)
    ck = torch.load(a.model_path, map_location=dev, weights_only=False); cfg = ck.get("config", {})
    for key in ["channels", "dropout", "delta_limit", "num_emitter_sectors", "num_receiver_sectors", "sector_highpass", "physics_residual_norm_scale", "background_delta_limit", "adjoint_ray_samples", "lowpass_kernel", "detail_kernel", "adjoint_blur_kernel"]:
        if key in cfg: setattr(a, key, cfg[key])
    batch = next(iter(loader))
    gt, tof, tof_mask, _idx, _initial, background, features, _res = make_spatial_physics_features(batch, stack, dev, ring, water_tof, a)
    model = SpatialPhysicsResidualNet(features.shape[1], a.channels, a.dropout, a.delta_limit).to(dev)
    model.load_state_dict(ck.get("best_model_state_dict", ck["model_state_dict"])); model.eval()
    with torch.no_grad():
        baseline_full = support_apply(background + model(features), ring, a.water_norm)
    soft, hard, evidence, support_info = derive_support(baseline_full, a.water_norm, ring, a.support_taper_pixels, a.support_radius_quantile)
    baseline = a.water_norm + soft * (baseline_full - a.water_norm)
    package = torch.load(a.data_path, map_location="cpu", weights_only=False)
    reliability = torch.as_tensor(package.get("tof_weight", np.ones((1, a.n_emitters, a.n_receivers))), dtype=tof.dtype, device=dev)
    while reliability.ndim > tof_mask.ndim: reliability = reliability.squeeze(1)
    while reliability.ndim < tof_mask.ndim: reliability = reliability.unsqueeze(0)
    reliability = reliability.expand_as(tof_mask).clamp(0, 1)
    usable = tof_mask * (reliability > 0).to(tof.dtype)
    ee = torch.arange(a.n_emitters, device=dev)[:, None]
    emitter_sector = torch.div(ee*a.angular_sectors, a.n_emitters, rounding_mode="floor").clamp_max(a.angular_sectors-1)
    sector_masks=[]
    for sector in range(a.angular_sectors):
        validation=(emitter_sector==sector)[None].to(tof.dtype).expand_as(usable)*usable
        sector_masks.append(validation.cpu())
    metadata = dict(ds.metadata or {})
    alignment_report = (metadata.get("experimental_alignment", metadata.get("experimental_alignment_v13", {})) or {})
    out = {
        "kind": "experimental_setup", "version": VERSION,
        "baseline_full_normalized": baseline_full.cpu(), "baseline_normalized": baseline.cpu(),
        "soft_support": soft.cpu(), "hard_support": hard.cpu(), "support_evidence": evidence.cpu(), "ring_support": ring.cpu(),
        "tof_normalized": tof.cpu(), "tof_mask": tof_mask.cpu(), "tof_weight": reliability.cpu(),
        "angular_sector_validation_masks": sector_masks, "angular_sectors": a.angular_sectors,
        "reference_diagnostic_normalized": gt.cpu(), "alignment_distribution_report": alignment_report,
        "support_info": support_info, "metadata": dict(ds.metadata or {}),
        "geometry": {"nx": a.nx, "ny": a.ny, "phys_x": a.phys_x, "phys_y": a.phys_y, "sos_min": a.sos_min, "sos_max": a.sos_max, "sos_water": a.sos_water},
    }
    Path(a.out_pt).parent.mkdir(parents=True, exist_ok=True); torch.save(out, a.out_pt)
    log_message(f"[experimental_setup] sector_ray_counts={[int(q.sum()) for q in sector_masks]} support={pprint.pformat(support_info)}")
    if alignment_report:
        log_message(f"[experimental_setup] input_distribution median_abs_z={alignment_report.get('post_channel_z_median_abs')} fraction_gt3={alignment_report.get('post_channel_z_fraction_gt3')}")
    log_message(f"[experimental_setup] saved={a.out_pt}; plotting is deferred to the report process")


if __name__ == "__main__": main()












