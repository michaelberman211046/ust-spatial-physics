import argparse, base64, html, io, pprint, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

VERSION="measured-coordinate-report-v1"


def display(a):
    # Pipeline tensors are (lateral, axial); imshow is (axial, lateral).
    return np.asarray(a).T


def add_image(doc, fig, title, output_png):
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    b=io.BytesIO();fig.savefig(b,format="png",dpi=150,bbox_inches="tight");plt.close(fig)
    doc.append(f"<h2>{html.escape(title)}</h2><img style='max-width:100%;height:auto' src='data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}'/>")


def main():
    p=argparse.ArgumentParser();p.add_argument("--report_npz",required=True);p.add_argument("--output_dir",required=True);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);z=np.load(a.report_npz,allow_pickle=True);px=float(z["phys_x"]);py=float(z["phys_y"]);lo=float(z["sos_min"]);hi=float(z["sos_max"]);extent=[0.0,px,py,0.0]
    summary=[{"sector":int(s),"initial_us":float(i),"selected_us":float(b),"holdout_change_us":float(d),"step":int(k),"accepted":bool(q)} for s,i,b,d,k,q in zip(z["sectors"],z["initial_validation_us"],z["best_validation_us"],z["holdout_improvement_us"],z["best_steps"],z["accepted"])];passed=bool(z["physical_refinement_accepted"]);count=int(z["accepted_fold_count"]);status="accepted-fold median" if passed else "frozen prior (physical gate failed)";doc=["<!doctype html><html><head><meta charset='utf-8'><title>publication experimental reconstruction</title></head><body>",f"<h1>publication measured-coordinate physical-ray reconstruction</h1><pre>version={VERSION}\ncommand={html.escape(' '.join(sys.argv))}\narguments={html.escape(pprint.pformat(vars(a)))}\nPhysical refinement accepted={passed}; accepted folds={count}/{len(summary)}; primary={status}.\nfold diagnostics={html.escape(pprint.pformat(summary))}\nagreement scale={float(z['agreement_scale_mps']):.4g} m/s</pre>"]
    panels=[(z["gt_mps"],"GT (visual comparison only)","gray",lo,hi,"Speed of sound [m/s]"),(z["prior_mps"],"Frozen synthetic Stage 7 prior","gray",lo,hi,"Speed of sound [m/s]"),(z["reconstruction_mps"],"Primary: "+status,"gray",lo,hi,"Speed of sound [m/s]"),(z["agreement_weighted_mps"],"Secondary: agreement-weighted","gray",lo,hi,"Speed of sound [m/s]"),(z["spread_mps"],"Accepted-fold disagreement","magma",0,max(5,float(np.quantile(z["spread_mps"],.99))),"Interquartile spread [m/s]")]
    fig,axes=plt.subplots(1,5,figsize=(22.5,4.7),constrained_layout=True)
    for ax,(im,title,cmap,vmin,vmax,label) in zip(axes,panels):q=ax.imshow(display(im),origin="upper",extent=extent,cmap=cmap,vmin=vmin,vmax=vmax,aspect="equal");ax.set_title(title);ax.set_xlabel("Lateral [m]");ax.set_ylabel("Axial [m]");fig.colorbar(q,ax=ax,fraction=.046,pad=.04).set_label(label)
    add_image(doc,fig,"Reconstruction summary",out/"physical_reconstruction.png")
    histories=z["histories"];sectors=z["sectors"];best=z["best_steps"];initial=z["initial_validation_us"];accepted=z["accepted"];fig,axes=plt.subplots(1,len(histories),figsize=(4.3*len(histories),4),constrained_layout=True);axes=np.atleast_1d(axes)
    for ax,h,sector,step,start,ok in zip(axes,histories,sectors,best,initial,accepted):h=np.asarray(h,dtype=float);ax.plot(h[:,1],label="fit rays");ax.plot(h[:,2],label="held-out sector");ax.axhline(float(start),color="gray",ls=":",label="iteration-zero holdout");ax.axvline(int(step),color="k",ls="--",lw=1);ax.set(title=f"Sector {int(sector)}: {'accepted' if bool(ok) else 'rejected'}",xlabel="Iteration",ylabel="Robust residual [us]");ax.grid(alpha=.3);ax.legend()
    add_image(doc,fig,"Held-out physical-sector validation",out/"physical_sector_validation.png");doc.append("</body></html>");html_path=out/"output___physical_ray.html";html_path.write_text("\n".join(doc),encoding="utf-8");print(f"[physical-report] HTML saved: {html_path}");print(f"[physical-report] version={VERSION}")


if __name__=="__main__":main()












