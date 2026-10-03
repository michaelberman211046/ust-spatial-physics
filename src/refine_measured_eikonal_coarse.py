import argparse
import json
import pprint
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from logger import log_message
from settings import set_output_folder
import refine_synthetic_eikonal_coarse as engine
from measured_eikonal_refinement import refine as corrected_refine
try:
    from measured_geometry import require_geometry
except ImportError:
    from measured_geometry import require_geometry


VERSION = "measured-eikonal-coarse-v1"


def arguments():
    p=argparse.ArgumentParser(description="Ground-truth-free Eikonal coarse refinement of Initial reconstruction")
    for name in ("setup_pt","initial_pt","out_pt","report_npz","output_dir"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--emitter_stride",type=int,default=4);p.add_argument("--receiver_stride",type=int,default=2)
    p.add_argument("--angular_sectors",type=int,default=8);p.add_argument("--holdout_sectors",default="0,2,4,6")
    p.add_argument("--outer_iterations",type=int,default=3);p.add_argument("--inner_steps",type=int,default=50)
    p.add_argument("--correction_grid",type=int,default=64);p.add_argument("--lr",type=float,default=.05)
    p.add_argument("--update_limit_mps",type=float,default=15);p.add_argument("--huber_beta_us",type=float,default=.20)
    p.add_argument("--prior_weight",type=float,default=.15);p.add_argument("--tv_weight",type=float,default=.02)
    p.add_argument("--curvature_weight",type=float,default=.01);p.add_argument("--min_holdout_improvement_us",type=float,default=.005)
    p.add_argument("--trace_step_pixels",type=float,default=.75);p.add_argument("--trace_max_steps",type=int,default=420)
    p.add_argument("--min_accepted_folds",type=int,default=3)
    p.add_argument("--max_mean_update_mps",type=float,default=1.5)
    p.add_argument("--max_rms_update_mps",type=float,default=8.0)
    return p.parse_args()


def dense_measurements(setup):
    ei=setup["ray_emitter_index"].cpu().numpy().astype(np.int64)
    ri=setup["ray_receiver_index"].cpu().numpy().astype(np.int64)
    n=int(max(ei.max(),ri.max())+1);n=max(n,512)
    observed=np.zeros((n,n),np.float64);valid=np.zeros((n,n),bool)
    water=float(setup["physical_geometry"]["water_sos"])
    total=setup["target_residual_seconds"].cpu().numpy().astype(np.float64)+setup["ray_length_m"].cpu().numpy()/water
    observed[ei,ri]=total;valid[ei,ri]=setup["ray_weight"].cpu().numpy()>0
    return observed,valid


def image4(x):
    x=torch.as_tensor(x).float()
    while x.ndim<4:x=x.unsqueeze(0)
    return x


def figure(setup,initial,coarse,outdir,phys_x,phys_y,lo,hi):
    panels=[];titles=[]
    gt=setup.get("reference_diagnostic_normalized")
    if gt is not None:
        panels.append((image4(gt)[0,0].cpu().numpy()*(hi-lo)+lo));titles.append("FWI comparison (evaluation only)")
    panels.extend((initial,coarse));titles.extend(("Initial reconstruction","Coarse Eikonal refinement"))
    fig,axes=plt.subplots(1,len(panels),figsize=(3.15*len(panels),3.0),constrained_layout=True,squeeze=False)
    extent=[0,phys_x,phys_y,0];artist=None
    for i,(im,title) in enumerate(zip(panels,titles)):
        ax=axes[0,i];artist=ax.imshow(im.T,origin="upper",extent=extent,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest")
        ax.set_title(title);ax.set_aspect("equal");ax.set_xlabel("Lateral position (m)");ax.set_ylabel("Axial position (m)" if i==0 else "")
        ax.text(.025,.975,f"({chr(97+i)})",transform=ax.transAxes,fontweight="bold",va="top",bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":1.5})
    fig.colorbar(artist,ax=axes.ravel().tolist(),shrink=.82,label="Speed of sound (m s$^{-1}$)")
    out=Path(outdir);fig.savefig(out/"measured_eikonal_coarse_refinement.pdf",bbox_inches="tight");fig.savefig(out/"measured_eikonal_coarse_refinement.png",bbox_inches="tight",dpi=600);plt.close(fig)


def main():
    a=arguments();set_output_folder(a.output_dir);log_message("...................");log_message(f"[CMD] {' '.join(sys.argv)}");log_message(f"[ARGS]\n{pprint.pformat(vars(a))}");log_message(f"[measured-eikonal-coarse] version={VERSION}")
    setup=torch.load(a.setup_pt,map_location="cpu",weights_only=False);initial=torch.load(a.initial_pt,map_location="cpu",weights_only=False)
    required_setup={"ray_emitter_index","ray_receiver_index","ray_length_m","target_residual_seconds","ray_weight","soft_support","geometry","physical_geometry"}
    missing_setup=sorted(required_setup.difference(setup))
    if missing_setup:raise RuntimeError(f"Physical setup is missing required fields: {missing_setup}")
    required_geometry={"phys_x","phys_y","sos_min","sos_max"}
    missing_geometry=sorted(required_geometry.difference(setup["geometry"]))
    if missing_geometry:raise RuntimeError(f"Physical setup geometry is missing required fields: {missing_geometry}")
    if "water_sos" not in setup["physical_geometry"]:raise RuntimeError("Physical setup is missing physical_geometry.water_sos")
    if "agreement_weighted_reconstruction_mps" not in initial:raise RuntimeError("Initial reconstruction file is missing agreement_weighted_reconstruction_mps")
    log_message(f"[measured-eikonal-coarse] accepted setup version={setup.get('version','unspecified')} by structural validation")
    start=image4(initial["agreement_weighted_reconstruction_mps"])[0,0].numpy();support=image4(setup["soft_support"])
    support=F.interpolate(support,size=start.shape,mode="bilinear",align_corners=False)[0,0].numpy().astype(np.float32)
    observed,valid=dense_measurements(setup);geom=setup["geometry"];pg=setup["physical_geometry"]
    phys_x=float(geom["phys_x"]);phys_y=float(geom["phys_y"]);lo=float(geom["sos_min"]);hi=float(geom["sos_max"]);water=float(pg["water_sos"])
    points,_order,_center,digest=require_geometry(observed.shape[0],observed.shape[1]);nx,ny=start.shape;dx=phys_x/(nx-1);dy=phys_y/(ny-1)
    sensors=np.empty_like(points);sensors[:,0]=(nx-1)/2+points[:,0]/dx;sensors[:,1]=(ny-1)/2+points[:,1]/dy
    a.mask_radius=float(pg.get("support_radius_m",.1091));a.sos_min=lo;a.sos_max=hi;a.sos_water=water
    device=torch.device(a.device if not a.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    refined,history=corrected_refine(start,start,observed,valid,sensors,support,a,device,phys_x,phys_y,False,"measured-eikonal-coarse")
    accepted_iterations=sum(bool(x["accepted"]) for x in history)
    output={"kind":"measured_eikonal_coarse_refinement","version":VERSION,"initial_mps":torch.from_numpy(start)[None,None],"reconstruction_mps":torch.from_numpy(refined)[None,None],"history":history,"accepted_iterations":accepted_iterations,"soft_support":torch.from_numpy(support)[None,None],"setup_pt":a.setup_pt,"initial_pt":a.initial_pt,"geometry_digest":digest}
    Path(a.out_pt).parent.mkdir(parents=True,exist_ok=True);torch.save(output,a.out_pt)
    np.savez_compressed(a.report_npz,initial_mps=start,refined_mps=refined,soft_support=support,accepted_iterations=accepted_iterations,phys_x=phys_x,phys_y=phys_y,sos_min=lo,sos_max=hi)
    summary={"version":VERSION,"accepted_iterations":accepted_iterations,"mean_abs_change_mps":float(np.mean(np.abs(refined-start))),"rms_change_mps":float(np.sqrt(np.mean((refined-start)**2))),"support_weighted_mean_change_mps":float(((refined-start)*support).sum()/max(support.sum(),1)),"selection":"withheld measured ToF median and RMS; bounded nuisance timing","fwi_used_in_selection":False}
    (Path(a.output_dir)/"measured_eikonal_coarse_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");figure(setup,start,refined,a.output_dir,phys_x,phys_y,lo,hi);log_message(f"[measured-eikonal-coarse] summary={summary}")


if __name__=="__main__":main()
