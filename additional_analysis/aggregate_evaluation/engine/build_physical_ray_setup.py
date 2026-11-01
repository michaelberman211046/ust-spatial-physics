import argparse, copy, pprint, sys
from pathlib import Path

import numpy as np
import torch

from logger import log_message
from settings import set_output_folder

VERSION="measured-coordinate-single-calibration-ray-setup-v1"


def median(x):
    x=np.asarray(x);x=x[np.isfinite(x)];return float(np.median(x)) if x.size else 0.


def fit_additive(residual,use,limit_us,iters=8):
    ne,nr=residual.shape;g=median(residual[use]);e=np.zeros(ne);r=np.zeros(nr);limit=limit_us*1e-6
    for _ in range(iters):
        for i in range(ne):
            if use[i].any():e[i]=median((residual[i]-g-r)[use[i]])
        e=np.clip(e-np.median(e),-limit,limit)
        for j in range(nr):
            if use[:,j].any():r[j]=median((residual[:,j]-g-e)[use[:,j]])
        r=np.clip(r-np.median(r),-limit,limit);g=median((residual-e[:,None]-r[None,:])[use])
    return g,e,r,g+e[:,None]+r[None,:]


def fit_outer_reference(residual, length, use, limit_us, slope_limit_us_per_m, iters=10):
    """Fit only anatomy-free outer rays; keep a chord-length term separate from electronics."""
    ne,nr=residual.shape
    if int(use.sum()) < 1000:
        raise RuntimeError("Too few outer rays for hardware/reference calibration")
    length_ref=median(length[use]); dl=length-length_ref
    g=median(residual[use]); e=np.zeros(ne); r=np.zeros(nr); slope=0.0
    lim=limit_us*1e-6; slope_lim=slope_limit_us_per_m*1e-6
    for _ in range(iters):
        base=residual-slope*dl
        for i in range(ne):
            if use[i].any(): e[i]=median((base[i]-g-r)[use[i]])
        e=np.clip(e-median(e),-lim,lim)
        for j in range(nr):
            if use[:,j].any(): r[j]=median((base[:,j]-g-e)[use[:,j]])
        r=np.clip(r-median(r),-lim,lim)
        g=median((base-e[:,None]-r[None,:])[use])
        x=dl[use]; y=(residual-g-e[:,None]-r[None,:])[use]
        err=y-slope*x; scale=max(1.4826*median(abs(err-median(err))),0.05e-6)
        w=np.minimum(1.0,1.5*scale/np.maximum(abs(err),1e-12))
        slope=float(np.clip(np.sum(w*x*y)/max(np.sum(w*x*x),1e-20),-slope_lim,slope_lim))
    correction=g+e[:,None]+r[None,:]+slope*dl
    return g,e,r,slope,length_ref,correction


def geometry(n,radius):
    from measured_geometry import require_geometry
    points, _, _, _ = require_geometry(n, n)
    emit=points;recv=points
    p=emit[:,None,:];q=recv[None,:,:];d=q-p;length=np.sqrt((d*d).sum(-1));t=np.clip(-(p*d).sum(-1)/np.maximum(length**2,1e-12),0,1);closest=p+t[...,None]*d;distance=np.sqrt((closest*closest).sum(-1))
    return emit,recv,length,distance


def main():
    p=argparse.ArgumentParser()
    for k in ["aligned_gt_pt","support_setup_pt","out_pt","output_dir"]:p.add_argument("--"+k,required=True)
    p.add_argument("--grid_size",type=int,default=100);p.add_argument("--ray_samples",type=int,default=96)
    p.add_argument("--emitter_stride",type=int,default=2);p.add_argument("--receiver_stride",type=int,default=2)
    p.add_argument("--ring_radius_m",type=float,default=.1091);p.add_argument("--phys_x",type=float,default=.24);p.add_argument("--phys_y",type=float,default=.24)
    p.add_argument("--water_sos",type=float,default=1500);p.add_argument("--outer_margin_m",type=float,default=.003)
    a=p.parse_args();set_output_folder(a.output_dir);log_message("...................");log_message(f"[CMD] {' '.join(sys.argv)}");log_message(f"[ARGS]\n{pprint.pformat(vars(a))}");log_message(f"[physical_setup] version={VERSION}")
    data=torch.load(a.aligned_gt_pt,map_location="cpu",weights_only=False);support=torch.load(a.support_setup_pt,map_location="cpu",weights_only=False)
    tof=torch.as_tensor(data["tof"])[0].numpy().astype(np.float64);mask=torch.as_tensor(data.get("tof_mask",np.ones_like(tof)));mask=(mask[0] if mask.ndim==3 else mask).numpy()>.5
    weight=torch.as_tensor(data.get("tof_weight",mask.astype(np.float32)));weight=(weight[0] if weight.ndim==3 else weight).numpy().astype(np.float64);weight=np.clip(weight,0,1)
    n=tof.shape[0];emit,recv,length,distance=geometry(n,a.ring_radius_m);support_radius=float(support["support_info"]["radius_pixels"])*float(support["geometry"]["phys_x"])/float(support["geometry"]["nx"])
    water=length/a.water_sos;outer=mask&(distance>=support_radius+a.outer_margin_m)
    ee_all,rr_all=np.indices(mask.shape);cal_train=outer&(((ee_all+3*rr_all)%5)!=0);cal_val=outer&~cal_train
    # Stage 9b already made the one GT-free electronic timing correction.
    # Do not fit another outer-ray delay against a homogeneous water model.
    g=0.0;e=np.zeros(n);r=np.zeros(n);slope=0.0;length_ref=0.0
    before=(tof-water)[cal_val]*1e6
    before_med=median(abs(before));after_med=before_med;calibration_accepted=False
    corrected=tof
    es=np.arange(0,n,a.emitter_stride);rs=np.arange(0,n,a.receiver_stride);ee,rr=np.meshgrid(es,rs,indexing="ij");selected=mask[ee,rr]&(distance[ee,rr]<support_radius);ei=ee[selected].astype(np.int64);ri=rr[selected].astype(np.int64)
    ep=emit[ei];rp=recv[ri];ray_length=np.sqrt(((rp-ep)**2).sum(1));tau=np.linspace(0,1,a.ray_samples,dtype=np.float64)[None,:,None];pts=ep[:,None,:]*(1-tau)+rp[:,None,:]*tau
    # publication uses the same measured x/y coordinate convention as synthetic
    # anatomy.generate_sensor_positions for every ray and tensor.
    ix=np.rint((pts[...,0]+a.phys_x/2)/a.phys_x*(a.grid_size-1)).astype(np.int64);iy=np.rint((pts[...,1]+a.phys_y/2)/a.phys_y*(a.grid_size-1)).astype(np.int64);ix=np.clip(ix,0,a.grid_size-1);iy=np.clip(iy,0,a.grid_size-1)
    # Image tensors are (lateral, axial), unlike imshow; this order is intentional.
    pix=ix*a.grid_size+iy
    target=(corrected[ei,ri]-water[ei,ri]).astype(np.float32);ray_weight=weight[ei,ri].astype(np.float32)
    prior_norm=torch.as_tensor(support["baseline_full_normalized"]).float();geom=support["geometry"];lo,hi=float(geom["sos_min"]),float(geom["sos_max"]);prior_mps=prior_norm*(hi-lo)+lo
    out={"kind":"physical_ray_setup","version":VERSION,"ray_pixel_indices":torch.as_tensor(pix,dtype=torch.int32),"ray_length_m":torch.as_tensor(ray_length,dtype=torch.float32),"target_residual_seconds":torch.as_tensor(target),"ray_weight":torch.as_tensor(ray_weight),"ray_emitter_index":torch.as_tensor(ei,dtype=torch.int16),"ray_receiver_index":torch.as_tensor(ri,dtype=torch.int16),"prior_mps":prior_mps,"soft_support":torch.as_tensor(support["soft_support"]).float(),"hard_support":torch.as_tensor(support["hard_support"]).float(),"reference_diagnostic_normalized":support.get("reference_diagnostic_normalized"),"geometry":geom,"physical_geometry":{"grid_size":a.grid_size,"ray_samples":a.ray_samples,"ring_radius_m":a.ring_radius_m,"support_radius_m":support_radius,"water_sos":a.water_sos,"emitter_stride":a.emitter_stride,"receiver_stride":a.receiver_stride},"hardware_calibration":{"calibration_accepted":calibration_accepted,"selected_tof":"physical_outer_length_calibrated" if calibration_accepted else "aligned_original","global_delay_us":g*1e6,"emitter_delay_us":(e*1e6).tolist(),"receiver_delay_us":(r*1e6).tolist(),"length_slope_us_per_m":slope*1e6,"length_reference_m":length_ref,"outer_ray_count":int(outer.sum()),"calibration_train_rays":int(cal_train.sum()),"calibration_holdout_rays":int(cal_val.sum()),"outer_residual_before_median_abs_us":before_med,"outer_residual_candidate_median_abs_us":after_med},"metadata":copy.deepcopy(dict(data.get("metadata",{})or{}))}
    Path(a.out_pt).parent.mkdir(parents=True,exist_ok=True);torch.save(out,a.out_pt);log_message(f"[physical_setup] support_radius_m={support_radius:.7g} selected_rays={len(ei)} outer_rays={int(outer.sum())} candidate_outer_median_abs_us={before_med:.6g}->{after_med:.6g} accepted={calibration_accepted}")
    log_message("[physical_setup] diagnostic plotting intentionally omitted from numerical setup")
    log_message(f"[physical_setup] selected_tof={out['hardware_calibration']['selected_tof']} saved={a.out_pt}")


if __name__=="__main__":main()












