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
import refine_synthetic_eikonal_fine as engine
from refine_measured_eikonal_coarse import dense_measurements, image4
from measured_eikonal_refinement import refine as corrected_refine
try:
    from measured_geometry import require_geometry
except ImportError:
    from measured_geometry import require_geometry


VERSION="measured-eikonal-fine-v1"


def arguments():
    p=argparse.ArgumentParser(description="Ground-truth-free fine Eikonal refinement of the accepted coarse measured reconstruction")
    for name in ("setup_pt","coarse_pt","out_pt","report_npz","output_dir"):p.add_argument("--"+name,required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--emitter_stride",type=int,default=4);p.add_argument("--receiver_stride",type=int,default=2)
    p.add_argument("--angular_sectors",type=int,default=8);p.add_argument("--holdout_sectors",default="0,2,4,6")
    p.add_argument("--fine_iterations",type=int,default=2);p.add_argument("--inner_steps",type=int,default=45);p.add_argument("--fine_grid",type=int,default=96)
    p.add_argument("--lr",type=float,default=.035);p.add_argument("--fine_limit_mps",type=float,default=7);p.add_argument("--huber_beta_us",type=float,default=.15)
    p.add_argument("--prior_weight",type=float,default=.22);p.add_argument("--edge_tv_weight",type=float,default=.035);p.add_argument("--curvature_weight",type=float,default=.012)
    p.add_argument("--min_holdout_improvement_us",type=float,default=.002);p.add_argument("--min_accepted_folds",type=int,default=2)
    p.add_argument("--trace_step_pixels",type=float,default=.65);p.add_argument("--trace_max_steps",type=int,default=480)
    p.add_argument("--max_mean_update_mps",type=float,default=.75);p.add_argument("--max_rms_update_mps",type=float,default=3.0)
    return p.parse_args()


def figure(setup,initial,coarse,fine,outdir,phys_x,phys_y,lo,hi):
    panels=[];titles=[];gt=setup.get("reference_diagnostic_normalized")
    if gt is not None:panels.append(image4(gt)[0,0].numpy()*(hi-lo)+lo);titles.append("FWI comparison (evaluation only)")
    panels.extend((initial,coarse,fine));titles.extend(("Initial reconstruction","Coarse Eikonal refinement","Fine Eikonal refinement"))
    fig,axes=plt.subplots(1,len(panels),figsize=(3.15*len(panels),3.0),constrained_layout=True,squeeze=False);extent=[0,phys_x,phys_y,0]
    for i,(im,title) in enumerate(zip(panels,titles)):
        ax=axes[0,i];artist=ax.imshow(im.T,origin="upper",extent=extent,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest");ax.set_title(title);ax.set_aspect("equal");ax.set_xlabel("Lateral position (m)");ax.set_ylabel("Axial position (m)" if i==0 else "")
        ax.text(.025,.975,f"({chr(97+i)})",transform=ax.transAxes,fontweight="bold",va="top",bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":1.5})
    fig.colorbar(artist,ax=axes.ravel().tolist(),shrink=.82,label="Speed of sound (m s$^{-1}$)");out=Path(outdir);fig.savefig(out/"measured_eikonal_fine_refinement.pdf",bbox_inches="tight");fig.savefig(out/"measured_eikonal_fine_refinement.png",bbox_inches="tight",dpi=600);plt.close(fig)


def main():
    a=arguments();set_output_folder(a.output_dir);log_message("...................");log_message(f"[CMD] {' '.join(sys.argv)}");log_message(f"[ARGS]\n{pprint.pformat(vars(a))}");log_message(f"[measured-eikonal-fine] version={VERSION}")
    setup=torch.load(a.setup_pt,map_location="cpu",weights_only=False);coarse_result=torch.load(a.coarse_pt,map_location="cpu",weights_only=False)
    start=image4(coarse_result["reconstruction_mps"])[0,0].numpy();initial=image4(coarse_result["initial_mps"])[0,0].numpy();support=image4(coarse_result["soft_support"])
    support=F.interpolate(support,size=start.shape,mode="bilinear",align_corners=False)[0,0].numpy().astype(np.float32);observed,valid=dense_measurements(setup)
    geom=setup["geometry"];pg=setup["physical_geometry"];phys_x=float(geom["phys_x"]);phys_y=float(geom["phys_y"]);lo=float(geom["sos_min"]);hi=float(geom["sos_max"]);water=float(pg["water_sos"])
    points,_order,_center,digest=require_geometry(observed.shape[0],observed.shape[1]);nx,ny=start.shape;dx=phys_x/(nx-1);dy=phys_y/(ny-1);sensors=np.empty_like(points);sensors[:,0]=(nx-1)/2+points[:,0]/dx;sensors[:,1]=(ny-1)/2+points[:,1]/dy
    a.mask_radius=float(pg.get("support_radius_m",.1091));a.sos_min=lo;a.sos_max=hi;a.sos_water=water
    device=torch.device(a.device if not a.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    refined,history=corrected_refine(start,initial,observed,valid,sensors,support,a,device,phys_x,phys_y,True,"fine_eikonal-corrected");accepted_iterations=sum(bool(x["accepted"]) for x in history)
    output={"kind":"measured_eikonal_fine_refinement","version":VERSION,"initial_mps":torch.from_numpy(initial)[None,None],"coarse_mps":torch.from_numpy(start)[None,None],"reconstruction_mps":torch.from_numpy(refined)[None,None],"history":history,"accepted_iterations":accepted_iterations,"soft_support":torch.from_numpy(support)[None,None],"setup_pt":a.setup_pt,"coarse_pt":a.coarse_pt,"geometry_digest":digest}
    Path(a.out_pt).parent.mkdir(parents=True,exist_ok=True);torch.save(output,a.out_pt);np.savez_compressed(a.report_npz,initial_mps=initial,coarse_mps=start,refined_mps=refined,soft_support=support,accepted_iterations=accepted_iterations,phys_x=phys_x,phys_y=phys_y,sos_min=lo,sos_max=hi)
    summary={"version":VERSION,"accepted_iterations":accepted_iterations,"mean_abs_change_from_coarse_mps":float(np.mean(np.abs(refined-start))),"rms_change_from_coarse_mps":float(np.sqrt(np.mean((refined-start)**2))),"support_weighted_mean_change_mps":float(((refined-start)*support).sum()/max(support.sum(),1)),"selection":"withheld measured ToF median and RMS; bounded nuisance timing","fwi_used_in_selection":False}
    (Path(a.output_dir)/"measured_eikonal_fine_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");figure(setup,initial,start,refined,a.output_dir,phys_x,phys_y,lo,hi);log_message(f"[measured-eikonal-fine] summary={summary}")


if __name__=="__main__":main()
