import argparse
import json
import pprint
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import PairsCacheDataset
from logger import log_message
from msfm import msfm
try:
    from measured_geometry import require_geometry
except ImportError:
    from measured_geometry import require_geometry
from settings import set_output_folder

from refine_synthetic_eikonal_coarse import bilinear, metrics, ring_mask, robust, scalar, solve_times


VERSION = "synthetic-eikonal-fine-v1"


def arguments():
    p = argparse.ArgumentParser(description="Fine Eikonal refinement of accepted coarse Eikonal synthetic results")
    p.add_argument("--data_path", required=True)
    p.add_argument("--coarse_report", required=True)
    p.add_argument("--out_pt", required=True)
    p.add_argument("--report_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--emitter_stride", type=int, default=4)
    p.add_argument("--receiver_stride", type=int, default=2)
    p.add_argument("--angular_sectors", type=int, default=8)
    p.add_argument("--holdout_sectors", default="0,2,4,6")
    p.add_argument("--fine_iterations", type=int, default=2)
    p.add_argument("--inner_steps", type=int, default=45)
    p.add_argument("--fine_grid", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.035)
    p.add_argument("--fine_limit_mps", type=float, default=7.0)
    p.add_argument("--huber_beta_us", type=float, default=0.15)
    p.add_argument("--prior_weight", type=float, default=0.22)
    p.add_argument("--edge_tv_weight", type=float, default=0.035)
    p.add_argument("--curvature_weight", type=float, default=0.012)
    p.add_argument("--min_holdout_improvement_us", type=float, default=0.002)
    p.add_argument("--min_accepted_folds", type=int, default=2)
    p.add_argument("--trace_step_pixels", type=float, default=0.65)
    p.add_argument("--trace_max_steps", type=int, default=480)
    p.add_argument("--mask_radius", type=float, default=0.1091)
    p.add_argument("--sos_min", type=float, default=1380.0)
    p.add_argument("--sos_max", type=float, default=1660.0)
    p.add_argument("--sos_water", type=float, default=1500.0)
    return p.parse_args()


def trace_bilinear(travel, source, receivers, dx, dy, step_pixels, max_steps):
    gx, gy = np.gradient(travel, dx, dy, edge_order=1)
    pos = receivers.astype(np.float64).copy()
    n = len(pos); nx, ny = travel.shape
    idx = np.zeros((n, max_steps, 4), np.int32)
    weight = np.zeros((n, max_steps, 4), np.float32)
    length = np.zeros((n, max_steps), np.float32)
    active = np.ones(n, bool); reached = np.zeros(n, bool)
    step_m = float(step_pixels) * min(dx, dy)
    for k in range(max_steps):
        rows = np.flatnonzero(active)
        if not len(rows): break
        p = pos[rows]
        arrived = np.linalg.norm(p-source[None], axis=1) <= max(1.25, 1.6*step_pixels)
        reached[rows[arrived]] = True; active[rows[arrived]] = False
        rows = rows[~arrived]
        if not len(rows): continue
        p = pos[rows]
        grad = np.stack((bilinear(gx,p), bilinear(gy,p)), axis=1)
        norm = np.linalg.norm(grad, axis=1)
        good = np.isfinite(norm) & (norm > 1e-12)
        active[rows[~good]] = False; rows = rows[good]; p = p[good]
        grad = grad[good]; norm = norm[good]
        if not len(rows): continue
        x=np.clip(p[:,0],0,nx-1); y=np.clip(p[:,1],0,ny-1)
        x0=np.floor(x).astype(np.int64); y0=np.floor(y).astype(np.int64)
        x1=np.minimum(x0+1,nx-1); y1=np.minimum(y0+1,ny-1)
        wx=x-x0; wy=y-y0
        idx[rows,k]=np.stack((x0*ny+y0,x1*ny+y0,x0*ny+y1,x1*ny+y1),axis=1)
        weight[rows,k]=np.stack(((1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy),axis=1)
        length[rows,k]=step_m
        direction=-grad/norm[:,None]
        pos[rows]=p+np.stack((step_m*direction[:,0]/dx,step_m*direction[:,1]/dy),axis=1)
        outside=(pos[rows,0]<0)|(pos[rows,0]>nx-1)|(pos[rows,1]<0)|(pos[rows,1]>ny-1)
        active[rows[outside]]=False
    return idx, weight, length, reached


def build_rays(c, observed, valid, sensors, emitters, receivers, dx, dy, args):
    all_i=[]; all_w=[]; all_l=[]; all_e=[]; all_o=[]; all_p=[]
    rp=sensors[receivers].astype(np.float64); rx=rp.astype(np.intp)
    for emitter in emitters:
        travel=msfm(c,sensors[emitter],dx=dx,dy=dy)
        pred=travel[rx[:,0],rx[:,1]]
        i,w,l,reached=trace_bilinear(travel,sensors[emitter],rp,dx,dy,args.trace_step_pixels,args.trace_max_steps)
        use=valid[emitter,receivers]&reached
        if use.any():
            all_i.append(i[use]);all_w.append(w[use]);all_l.append(l[use])
            all_e.append(np.full(use.sum(),emitter,np.int32));all_o.append(observed[emitter,receivers[use]]);all_p.append(pred[use])
    if not all_i: raise RuntimeError("No valid Eikonal characteristics reached their emitters")
    return {"indices":np.concatenate(all_i),"weights":np.concatenate(all_w),"lengths":np.concatenate(all_l),
            "emitters":np.concatenate(all_e),"observed":np.concatenate(all_o),"predicted":np.concatenate(all_p)}


def median_error(c, observed, valid, sensors, emitters, receivers, dx, dy):
    pred=solve_times(c,sensors,emitters,receivers,dx,dy); values=[]
    for row,e in enumerate(emitters):
        use=valid[e,receivers]
        values.extend(np.abs((pred[row,use]-observed[e,receivers[use]])*1e6).tolist())
    return float(np.median(values)) if values else float("inf")


def fine_refine(start, initial, target, observed, valid, sensors, args, device, phys_x, phys_y, sample_index):
    nx,ny=start.shape;dx=phys_x/(nx-1);dy=phys_y/(ny-1)
    mask=ring_mask(nx,ny,phys_x,phys_y,args.mask_radius); current=args.sos_water+mask*(start-args.sos_water)
    emitters=np.arange(0,observed.shape[0],args.emitter_stride,dtype=np.int64)
    receivers=np.arange(0,observed.shape[1],args.receiver_stride,dtype=np.int64)
    heldouts=[int(x) for x in args.holdout_sectors.split(",") if x.strip()]; history=[]
    for iteration in range(args.fine_iterations):
        rays=build_rays(current,observed,valid,sensors,emitters,receivers,dx,dy,args)
        indices=torch.as_tensor(rays["indices"],dtype=torch.long,device=device)
        weights=torch.as_tensor(rays["weights"],dtype=torch.float32,device=device)
        lengths=torch.as_tensor(rays["lengths"],dtype=torch.float32,device=device)
        base=torch.as_tensor(rays["predicted"]-rays["observed"],dtype=torch.float32,device=device)
        ray_emitters=torch.as_tensor(rays["emitters"],dtype=torch.long,device=device)
        sectors=torch.div(ray_emitters*args.angular_sectors,observed.shape[0],rounding_mode="floor")
        current_t=torch.as_tensor(current,dtype=torch.float32,device=device)[None,None]
        mask_t=torch.as_tensor(mask,dtype=torch.float32,device=device)[None,None]
        # Preserve boundaries already present in the learned image; discourage new unsupported edges.
        ref=torch.as_tensor(initial,dtype=torch.float32,device=device)[None,None]
        gx=ref[...,1:,:]-ref[...,:-1,:];gy=ref[...,:,1:]-ref[...,:,:-1]
        edge_scale=torch.quantile(torch.cat((gx.abs().flatten(),gy.abs().flatten())),.80).clamp_min(2.0)
        wx=torch.exp(-gx.abs()/edge_scale);wy=torch.exp(-gy.abs()/edge_scale)
        folds=[]; records=[]
        for heldout in heldouts:
            validation=(sectors==heldout).float();fit=1-validation
            if validation.sum()<1 or fit.sum()<1: continue
            raw=torch.nn.Parameter(torch.zeros((1,1,args.fine_grid,args.fine_grid),device=device))
            opt=torch.optim.Adam([raw],lr=args.lr)
            for _ in range(args.inner_steps):
                coarse=args.fine_limit_mps*torch.tanh(raw)
                dc=F.interpolate(coarse,size=(nx,ny),mode="bilinear",align_corners=False)*mask_t
                candidate=(current_t+dc).clamp(args.sos_min,args.sos_max)
                ds=(1/candidate-1/current_t).reshape(-1)
                sampled=(ds[indices]*weights).sum(-1)
                residual=base+(sampled*lengths).sum(1)
                data=robust(residual*1e6,fit,args.huber_beta_us)
                prior=(dc.square()*mask_t).sum()/mask_t.sum().clamp_min(1)/(args.fine_limit_mps**2)
                dxdc=dc[...,1:,:]-dc[...,:-1,:];dydc=dc[...,:,1:]-dc[...,:,:-1]
                edge_tv=(torch.sqrt(dxdc.square()+.04)*wx).mean()+(torch.sqrt(dydc.square()+.04)*wy).mean()
                curvature=(F.avg_pool2d(coarse,3,1,1)-coarse).square().mean()/(args.fine_limit_mps**2)
                loss=data+args.prior_weight*prior+args.edge_tv_weight*edge_tv+args.curvature_weight*curvature
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([raw],1.0);opt.step()
            proposed=(F.interpolate(args.fine_limit_mps*torch.tanh(raw),size=(nx,ny),mode="bilinear",align_corners=False)*mask_t)[0,0].detach().cpu().numpy()
            held=np.asarray([e for e in emitters if min(args.angular_sectors-1,int(e*args.angular_sectors/observed.shape[0]))==heldout],dtype=np.int64)
            before=median_error(current,observed,valid,sensors,held,receivers,dx,dy); accepted=None
            for alpha in (1.0,.5,.25,.125):
                candidate=np.clip(current+alpha*proposed,args.sos_min,args.sos_max);candidate=args.sos_water+mask*(candidate-args.sos_water)
                after=median_error(candidate,observed,valid,sensors,held,receivers,dx,dy)
                if before-after>=args.min_holdout_improvement_us:
                    accepted=alpha*proposed;records.append({"sector":heldout,"accepted":True,"alpha":alpha,"before_us":before,"after_us":after});break
            if accepted is not None: folds.append(accepted)
            else: records.append({"sector":heldout,"accepted":False,"before_us":before})
        row={"iteration":iteration,"accepted":False,"folds":records}
        if len(folds)>=args.min_accepted_folds:
            stack=np.stack(folds);median=np.median(stack,axis=0);spread=np.quantile(stack,.75,axis=0)-np.quantile(stack,.25,axis=0)
            scale=max(1.0,float(np.quantile(spread[mask>.5],.75)));consensus=np.exp(-(spread/scale)**2)*mask
            update=consensus*median
            all_before=median_error(current,observed,valid,sensors,emitters,receivers,dx,dy)
            for alpha in (1.0,.5,.25,.125):
                candidate=np.clip(current+alpha*update,args.sos_min,args.sos_max);candidate=args.sos_water+mask*(candidate-args.sos_water)
                all_after=median_error(candidate,observed,valid,sensors,emitters,receivers,dx,dy)
                if all_before-all_after>=args.min_holdout_improvement_us:
                    current=candidate;row.update({"accepted":True,"alpha":alpha,"before_us":all_before,"after_us":all_after,"accepted_folds":len(folds)});break
        history.append(row);log_message(f"[synthetic-eikonal-fine] sample={sample_index} iteration={iteration} {row}")
        if not row["accepted"]: break
    return {"sample_index":int(sample_index),"target_mps":target,"initial_mps":initial,"coarse_mps":start,"refined_mps":current,
            "history":history,"accepted_iterations":sum(x["accepted"] for x in history),"initial_metrics":metrics(initial,target,mask),
            "coarse_metrics":metrics(start,target,mask),"refined_metrics":metrics(current,target,mask)}


def make_figure(results,outdir,phys_x,phys_y):
    plt.rcParams.update({"font.family":"serif","font.size":8.5,"axes.titlesize":9,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(len(results),4,figsize=(9.6,2.15*len(results)),constrained_layout=True,squeeze=False)
    lo=min(x["target_mps"].min() for x in results);hi=max(x["target_mps"].max() for x in results);extent=[0,phys_x,phys_y,0]
    titles=["Synthetic reference","initial learned reconstruction","coarse Eikonal coarse refinement","fine Eikonal fine refinement"];panel=0
    for r,result in enumerate(results):
        images=(result["target_mps"],result["initial_mps"],result["coarse_mps"],result["refined_mps"])
        for c,image in enumerate(images):
            ax=axes[r,c];artist=ax.imshow(image.T,origin="upper",extent=extent,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest");ax.set_aspect("equal")
            if r==0:ax.set_title(titles[c])
            ax.text(.025,.975,f"({chr(97+panel)})",transform=ax.transAxes,fontweight="bold",va="top",bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":1.5});panel+=1
            if c==0:ax.set_ylabel("Axial position (m)")
            if r==len(results)-1:ax.set_xlabel("Lateral position (m)")
        axes[r,0].text(-.34,.5,f"Test sample {r+1}",transform=axes[r,0].transAxes,rotation=90,va="center",ha="center",fontweight="bold")
    divider=make_axes_locatable(axes[-1,-1]);cax=divider.append_axes("right",size="4%",pad=.06);fig.colorbar(artist,cax=cax).set_label("Speed of sound (m s$^{-1}$)")
    out=Path(outdir);fig.savefig(out/"synthetic_eikonal_fine_refinement.pdf",bbox_inches="tight");fig.savefig(out/"synthetic_eikonal_fine_refinement.png",bbox_inches="tight",dpi=600);plt.close(fig)


def main():
    args=arguments();set_output_folder(args.output_dir);log_message("...................");log_message(f"[CMD] {' '.join(sys.argv)}");log_message(f"[ARGS]\n{pprint.pformat(vars(args))}");log_message(f"[synthetic-eikonal-fine] version={VERSION}")
    device=torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    report=np.load(args.coarse_report,allow_pickle=True);required={"sample_indices","target_mps","initial_mps","refined_mps"};missing=required.difference(report.files)
    if missing:raise RuntimeError(f"coarse Eikonal report is missing {sorted(missing)}")
    args.sos_min=scalar(report,"sos_min",args.sos_min);args.sos_max=scalar(report,"sos_max",args.sos_max);phys_x=scalar(report,"phys_x",.24);phys_y=scalar(report,"phys_y",.24)
    dataset=PairsCacheDataset(args.data_path,sos_min=args.sos_min,sos_max=args.sos_max,return_tof_mask=True);meta=dict(dataset.metadata or {})
    ne=int(meta.get("n_emitters",512));nr=int(meta.get("n_receivers",512));points,_order,_center,digest=require_geometry(ne,nr)
    coarse=np.asarray(report["refined_mps"],np.float32);initial=np.asarray(report["initial_mps"],np.float32);targets=np.asarray(report["target_mps"],np.float32);indices=np.asarray(report["sample_indices"],np.int64)
    nx,ny=coarse.shape[-2:];dx=phys_x/(nx-1);dy=phys_y/(ny-1);sensors=np.empty_like(points);sensors[:,0]=(nx-1)/2+points[:,0]/dx;sensors[:,1]=(ny-1)/2+points[:,1]/dy
    valid=np.asarray(dataset.tof_mask if dataset.tof_mask is not None else np.ones((ne,nr)),bool);results=[]
    for pos,index in enumerate(indices):
        sample=dataset._sample_arrays(int(index))
        if "tof_raw_eikonal" not in sample:raise RuntimeError(f"Sample {index} lacks tof_raw_eikonal")
        results.append(fine_refine(coarse[pos],initial[pos],targets[pos],np.asarray(sample["tof_raw_eikonal"],np.float64),valid,sensors,args,device,phys_x,phys_y,int(index)))
    Path(args.out_pt).parent.mkdir(parents=True,exist_ok=True);torch.save({"kind":"synthetic_eikonal_fine_refinement","version":VERSION,"geometry_digest":digest,"results":results},args.out_pt)
    np.savez_compressed(args.report_npz,sample_indices=indices,target_mps=targets,initial_mps=initial,coarse_mps=coarse,refined_mps=np.stack([x["refined_mps"] for x in results]),initial_rmse_mps=[x["initial_metrics"]["rmse_mps"] for x in results],coarse_rmse_mps=[x["coarse_metrics"]["rmse_mps"] for x in results],refined_rmse_mps=[x["refined_metrics"]["rmse_mps"] for x in results],initial_correlation=[x["initial_metrics"]["correlation"] for x in results],coarse_correlation=[x["coarse_metrics"]["correlation"] for x in results],refined_correlation=[x["refined_metrics"]["correlation"] for x in results],accepted_iterations=[x["accepted_iterations"] for x in results],phys_x=phys_x,phys_y=phys_y,sos_min=args.sos_min,sos_max=args.sos_max)
    summary={"version":VERSION,"samples":len(results),"mean_initial_rmse_mps":float(np.mean([x["initial_metrics"]["rmse_mps"] for x in results])),"mean_coarse_rmse_mps":float(np.mean([x["coarse_metrics"]["rmse_mps"] for x in results])),"mean_fine_rmse_mps":float(np.mean([x["refined_metrics"]["rmse_mps"] for x in results])),"accepted_iterations":[x["accepted_iterations"] for x in results]}
    (Path(args.output_dir)/"synthetic_eikonal_fine_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");make_figure(results,args.output_dir,phys_x,phys_y);log_message(f"[synthetic-eikonal-fine] summary={summary}")


if __name__=="__main__":main()
