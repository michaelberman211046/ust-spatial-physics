import argparse,base64,html,io,pprint,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
import numpy as np

VERSION="synthetic-evaluation-report-v1"

def shown(x):return np.asarray(x).T

def main():
    p=argparse.ArgumentParser();p.add_argument("--report_npz",required=True);p.add_argument("--output_dir",required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);z=np.load(a.report_npz);px=float(z["phys_x"]);py=float(z["phys_y"]);lo=float(z["sos_min"]);hi=float(z["sos_max"]);extent=[0.,px,py,0.]
    panels=[(z["target_mps"],"Synthetic target","gray",lo,hi,"SoS [m/s]"),(z["initial_mps"],"Stage 1 initial","gray",lo,hi,"SoS [m/s]"),(z["background_mps"],"Stage 4 background","gray",lo,hi,"SoS [m/s]"),(z["prediction_mps"],"Stage 7 prediction","gray",lo,hi,"SoS [m/s]"),(z["absolute_error_mps"],"Absolute error","magma",0,max(5.,float(np.quantile(z["absolute_error_mps"],.99))),"Absolute error [m/s]")]
    fig,axes=plt.subplots(1,5,figsize=(22,4.7),constrained_layout=True)
    for ax,(im,title,cmap,vmin,vmax,label) in zip(axes,panels):q=ax.imshow(shown(im),origin="upper",extent=extent,cmap=cmap,vmin=vmin,vmax=vmax,aspect="equal");ax.set_title(title);ax.set_xlabel("Lateral [m]");ax.set_ylabel("Axial [m]");fig.colorbar(q,ax=ax,fraction=.046,pad=.04).set_label(label)
    png=out/"synthetic_validation.png";fig.savefig(png,dpi=180,bbox_inches="tight");buf=io.BytesIO();fig.savefig(buf,format="png",dpi=150,bbox_inches="tight");plt.close(fig);metrics={k:float(z[k]) for k in ("background_mse","prediction_mse","prediction_l1","tof_mse","corrupt_prediction_mse","clean_corrupt_l1","stripe_sensitivity")};doc=f"<!doctype html><html><body><h1>Stage 8 clean and corrupted synthetic validation</h1><pre>version={VERSION}\ncommand={html.escape(' '.join(sys.argv))}\narguments={html.escape(pprint.pformat(vars(a)))}\nmetrics={html.escape(pprint.pformat(metrics))}</pre><img style='max-width:100%;height:auto' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'/></body></html>";(out/"synthetic_evaluation.html").write_text(doc,encoding="utf-8");print(f"[synthetic-evaluation-report] metrics={metrics}");print(f"[synthetic-evaluation-report] saved={out/'synthetic_evaluation.html'}")

if __name__=="__main__":main()












