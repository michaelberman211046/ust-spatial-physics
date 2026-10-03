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


VERSION = "synthetic-eikonal-coarse-v1"


def arguments():
    p = argparse.ArgumentParser(description="Eikonal-consistent refinement of retained initial learned synthetic reconstructions")
    p.add_argument("--data_path", required=True)
    p.add_argument("--initial_report", required=True)
    p.add_argument("--out_pt", required=True)
    p.add_argument("--report_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--emitter_stride", type=int, default=4)
    p.add_argument("--receiver_stride", type=int, default=2)
    p.add_argument("--angular_sectors", type=int, default=8)
    p.add_argument("--holdout_sectors", default="0,2,4,6")
    p.add_argument("--outer_iterations", type=int, default=3)
    p.add_argument("--inner_steps", type=int, default=50)
    p.add_argument("--correction_grid", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--update_limit_mps", type=float, default=15.0)
    p.add_argument("--huber_beta_us", type=float, default=0.20)
    p.add_argument("--prior_weight", type=float, default=0.15)
    p.add_argument("--tv_weight", type=float, default=0.02)
    p.add_argument("--curvature_weight", type=float, default=0.01)
    p.add_argument("--min_holdout_improvement_us", type=float, default=0.005)
    p.add_argument("--trace_step_pixels", type=float, default=0.75)
    p.add_argument("--trace_max_steps", type=int, default=420)
    p.add_argument("--mask_radius", type=float, default=0.1091)
    p.add_argument("--sos_min", type=float, default=1380.0)
    p.add_argument("--sos_max", type=float, default=1660.0)
    p.add_argument("--sos_water", type=float, default=1500.0)
    return p.parse_args()


def scalar(report, key, default):
    return float(np.asarray(report[key]).reshape(-1)[0]) if key in report else float(default)


def ring_mask(nx, ny, phys_x, phys_y, radius):
    x = np.linspace(-phys_x / 2, phys_x / 2, nx)
    y = np.linspace(-phys_y / 2, phys_y / 2, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return (xx * xx + yy * yy <= radius * radius).astype(np.float32)


def bilinear(field, positions):
    nx, ny = field.shape
    x = np.clip(positions[:, 0], 0, nx - 1)
    y = np.clip(positions[:, 1], 0, ny - 1)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, nx - 1); y1 = np.minimum(y0 + 1, ny - 1)
    wx = x - x0; wy = y - y0
    return ((1-wx)*(1-wy)*field[x0,y0] + wx*(1-wy)*field[x1,y0] +
            (1-wx)*wy*field[x0,y1] + wx*wy*field[x1,y1])


def solve_times(c, sensors_px, emitter_ids, receiver_ids, dx, dy):
    predicted = np.empty((len(emitter_ids), len(receiver_ids)), dtype=np.float64)
    receiver_x = sensors_px[receiver_ids, 0].astype(np.intp)
    receiver_y = sensors_px[receiver_ids, 1].astype(np.intp)
    for row, emitter in enumerate(emitter_ids):
        travel = msfm(c, sensors_px[emitter], dx=dx, dy=dy)
        predicted[row] = travel[receiver_x, receiver_y]
    return predicted


def trace_characteristics(travel, source_px, receiver_px, dx, dy, step_pixels, max_steps):
    gx, gy = np.gradient(travel, dx, dy, edge_order=1)
    positions = receiver_px.astype(np.float64).copy()
    count = len(positions)
    indices = np.zeros((count, max_steps), dtype=np.int32)
    lengths = np.zeros((count, max_steps), dtype=np.float32)
    active = np.ones(count, dtype=bool)
    reached = np.zeros(count, dtype=bool)
    step_m = float(step_pixels) * min(dx, dy)
    nx, ny = travel.shape
    for step in range(max_steps):
        if not active.any():
            break
        rows = np.flatnonzero(active)
        pos = positions[rows]
        distance_px = np.linalg.norm(pos - source_px[None], axis=1)
        arrived = distance_px <= max(1.25, 1.6 * step_pixels)
        if arrived.any():
            reached[rows[arrived]] = True
            active[rows[arrived]] = False
            rows = rows[~arrived]
            pos = positions[rows]
        if not len(rows):
            continue
        grad = np.stack([bilinear(gx, pos), bilinear(gy, pos)], axis=1)
        norm = np.linalg.norm(grad, axis=1)
        usable = np.isfinite(norm) & (norm > 1e-12)
        bad = rows[~usable]
        active[bad] = False
        rows = rows[usable]; pos = positions[rows]; grad = grad[usable]; norm = norm[usable]
        if not len(rows):
            continue
        direction_m = -grad / norm[:, None]
        delta_px = np.stack([step_m * direction_m[:,0] / dx,
                             step_m * direction_m[:,1] / dy], axis=1)
        nearest = np.rint(pos).astype(np.int64)
        nearest[:,0] = np.clip(nearest[:,0], 0, nx-1)
        nearest[:,1] = np.clip(nearest[:,1], 0, ny-1)
        indices[rows, step] = nearest[:,0] * ny + nearest[:,1]
        lengths[rows, step] = step_m
        positions[rows] = pos + delta_px
        outside = ((positions[rows,0] < 0) | (positions[rows,0] > nx-1) |
                   (positions[rows,1] < 0) | (positions[rows,1] > ny-1))
        active[rows[outside]] = False
    return indices, lengths, reached


def forward_and_rays(c, observed, valid_mask, sensors_px, emitter_ids, receiver_ids,
                     dx, dy, args, need_rays):
    predicted = np.empty((len(emitter_ids), len(receiver_ids)), dtype=np.float64)
    ray_indices, ray_lengths, ray_emitters, ray_observed, ray_predicted = [], [], [], [], []
    receiver_px = np.stack([sensors_px[receiver_ids,0].astype(np.intp),
                            sensors_px[receiver_ids,1].astype(np.intp)], axis=1).astype(np.float64)
    for row, emitter in enumerate(emitter_ids):
        travel = msfm(c, sensors_px[emitter], dx=dx, dy=dy)
        rx = receiver_px.astype(np.intp)
        predicted[row] = travel[rx[:,0], rx[:,1]]
        if not need_rays:
            continue
        ids, seg, reached = trace_characteristics(
            travel, sensors_px[emitter], receiver_px, dx, dy,
            args.trace_step_pixels, args.trace_max_steps,
        )
        usable = valid_mask[emitter, receiver_ids] & reached
        if usable.any():
            ray_indices.append(ids[usable]); ray_lengths.append(seg[usable])
            ray_emitters.append(np.full(int(usable.sum()), emitter, dtype=np.int32))
            ray_observed.append(observed[emitter, receiver_ids[usable]])
            ray_predicted.append(predicted[row, usable])
    if not need_rays:
        return predicted, None
    if not ray_indices:
        raise RuntimeError("No valid Eikonal characteristics reached their emitters")
    rays = {
        "indices": np.concatenate(ray_indices),
        "lengths": np.concatenate(ray_lengths),
        "emitters": np.concatenate(ray_emitters),
        "observed": np.concatenate(ray_observed),
        "predicted": np.concatenate(ray_predicted),
    }
    return predicted, rays


def robust(residual_us, mask, beta):
    absolute = residual_us.abs()
    huber = torch.where(absolute < beta, 0.5*absolute.square()/beta,
                         absolute-0.5*beta)
    return (huber*mask).sum()/mask.sum().clamp_min(1)


def residual_statistics(predicted, observed, valid, emitter_ids, holdout_set, sectors, ne):
    fit_values, holdout_values = [], []
    for row, emitter in enumerate(emitter_ids):
        values = np.abs((predicted[row] - observed[emitter]) * 1e6)
        values = values[valid[emitter]]
        sector = min(sectors-1, int(emitter*sectors/ne))
        (holdout_values if sector in holdout_set else fit_values).extend(values.tolist())
    return {
        "fit_median_abs_us": float(np.median(fit_values)) if fit_values else float("inf"),
        "holdout_median_abs_us": float(np.median(holdout_values)) if holdout_values else float("inf"),
    }


def metrics(image, target, mask):
    use = mask > 0.5; a = image[use].astype(np.float64); b = target[use].astype(np.float64)
    error = a-b
    return {"rmse_mps": float(np.sqrt(np.mean(error*error))),
            "mae_mps": float(np.mean(np.abs(error))),
            "correlation": float(np.corrcoef(a,b)[0,1])}


def refine_sample(initial, target, observed, valid, sensors_px, args, device,
                  phys_x, phys_y, sample_index):
    nx, ny = initial.shape; dx = phys_x/(nx-1); dy = phys_y/(ny-1)
    mask = ring_mask(nx, ny, phys_x, phys_y, args.mask_radius)
    current = args.sos_water + mask*(initial-args.sos_water)
    emitter_ids = np.arange(0, observed.shape[0], args.emitter_stride, dtype=np.int64)
    receiver_ids = np.arange(0, observed.shape[1], args.receiver_stride, dtype=np.int64)
    holdout_sectors = [int(x) for x in args.holdout_sectors.split(",") if x.strip()]
    history = []
    for outer in range(args.outer_iterations):
        predicted, rays = forward_and_rays(
            current, observed, valid, sensors_px, emitter_ids, receiver_ids,
            dx, dy, args, True,
        )
        before = residual_statistics(predicted, observed[:,receiver_ids],
                                     valid[:,receiver_ids], emitter_ids,
                                     set(holdout_sectors), args.angular_sectors, observed.shape[0])
        indices = torch.as_tensor(rays["indices"], dtype=torch.long, device=device)
        lengths = torch.as_tensor(rays["lengths"], dtype=torch.float32, device=device)
        base = torch.as_tensor(rays["predicted"]-rays["observed"], dtype=torch.float32, device=device)
        emitters = torch.as_tensor(rays["emitters"], dtype=torch.long, device=device)
        sector = torch.div(emitters*args.angular_sectors, observed.shape[0], rounding_mode="floor")
        current_t = torch.as_tensor(current, dtype=torch.float32, device=device)[None,None]
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)[None,None]
        evidence = F.avg_pool2d((current_t-args.sos_water).abs(), 11, 1, 5)
        scale = torch.quantile(evidence[mask_t>0.5], 0.75).clamp_min(3.0)
        anchor_weight = 1 + 4*torch.exp(-evidence/scale)
        fold_corrections, fold_records = [], []
        for heldout in holdout_sectors:
            validation = (sector == heldout).float()
            fit_mask = 1-validation
            if float(validation.sum()) < 1 or float(fit_mask.sum()) < 1:
                continue
            raw = torch.nn.Parameter(torch.zeros(
                (1,1,args.correction_grid,args.correction_grid),
                dtype=torch.float32, device=device,
            ))
            optimizer = torch.optim.Adam([raw], lr=args.lr)
            for _step in range(args.inner_steps):
                coarse = args.update_limit_mps*torch.tanh(raw)
                dc = F.interpolate(coarse,size=(nx,ny),mode="bilinear",align_corners=False)*mask_t
                candidate = (current_t+dc).clamp(args.sos_min,args.sos_max)
                delta_s = (1/candidate-1/current_t).reshape(-1)
                linear = base+(delta_s[indices]*lengths).sum(1)
                data_loss = robust(linear*1e6,fit_mask,args.huber_beta_us)
                anchor = ((dc/20).square()*anchor_weight*mask_t).sum()/mask_t.sum().clamp_min(1)
                tv = ((coarse[...,1:,:]-coarse[...,:-1,:]).abs().mean()+
                      (coarse[...,:,1:]-coarse[...,:,:-1]).abs().mean())/20
                curvature = (F.avg_pool2d(coarse,3,1,1)-coarse).abs().mean()/20
                loss=data_loss+args.prior_weight*anchor+args.tv_weight*tv+args.curvature_weight*curvature
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([raw],1);optimizer.step()
            proposed=(F.interpolate(args.update_limit_mps*torch.tanh(raw),size=(nx,ny),mode="bilinear",align_corners=False)*mask_t)[0,0].detach().cpu().numpy()
            heldout_emitters=np.asarray([e for e in emitter_ids if min(args.angular_sectors-1,int(e*args.angular_sectors/observed.shape[0]))==heldout],dtype=np.int64)
            if not len(heldout_emitters):
                continue
            before_hold=solve_times(current,sensors_px,heldout_emitters,receiver_ids,dx,dy)
            before_values=[]
            for rr,e in enumerate(heldout_emitters):
                use=valid[e,receiver_ids];before_values.extend(np.abs((before_hold[rr,use]-observed[e,receiver_ids[use]])*1e6).tolist())
            before_median=float(np.median(before_values))
            accepted_fold=None
            for alpha in (1.0,.5,.25,.125):
                candidate=np.clip(current+alpha*proposed,args.sos_min,args.sos_max);candidate=args.sos_water+mask*(candidate-args.sos_water)
                after_hold=solve_times(candidate,sensors_px,heldout_emitters,receiver_ids,dx,dy)
                values=[]
                for rr,e in enumerate(heldout_emitters):
                    use=valid[e,receiver_ids];values.extend(np.abs((after_hold[rr,use]-observed[e,receiver_ids[use]])*1e6).tolist())
                after_median=float(np.median(values));improvement=before_median-after_median
                if improvement>=args.min_holdout_improvement_us:
                    accepted_fold=alpha*proposed;fold_records.append({"sector":heldout,"accepted":True,"alpha":alpha,"before_us":before_median,"after_us":after_median,"improvement_us":improvement});break
            if accepted_fold is None:
                fold_records.append({"sector":heldout,"accepted":False,"before_us":before_median})
            else:
                fold_corrections.append(accepted_fold)
        row={"outer_iteration":outer,"before":before,"folds":fold_records,"accepted":False}
        if len(fold_corrections)>=2:
            stack=np.stack(fold_corrections);median=np.median(stack,axis=0);spread=np.quantile(stack,.75,axis=0)-np.quantile(stack,.25,axis=0)
            inside=spread[mask>.5];agreement_scale=max(2.0,float(np.quantile(inside,.75))) if inside.size else 2.0
            agreement=np.exp(-np.square(spread/agreement_scale))*mask
            combined=agreement*median
            selected=None
            for alpha in (1.0,.5,.25,.125):
                candidate=np.clip(current+alpha*combined,args.sos_min,args.sos_max);candidate=args.sos_water+mask*(candidate-args.sos_water)
                candidate_pred=solve_times(candidate,sensors_px,emitter_ids,receiver_ids,dx,dy)
                after=residual_statistics(candidate_pred,observed[:,receiver_ids],valid[:,receiver_ids],emitter_ids,set(holdout_sectors),args.angular_sectors,observed.shape[0])
                improvement=before["holdout_median_abs_us"]-after["holdout_median_abs_us"]
                if improvement>=args.min_holdout_improvement_us and after["fit_median_abs_us"]<=before["fit_median_abs_us"]:
                    selected=(candidate,alpha,after,improvement,agreement_scale);break
            if selected is not None:
                current,alpha,after,improvement,agreement_scale=selected;row.update({"accepted":True,"alpha":alpha,"after":after,"holdout_improvement_us":improvement,"accepted_folds":len(fold_corrections),"agreement_scale_mps":agreement_scale})
        history.append(row);log_message(f"[coarse_eikonal] sample={sample_index} outer={outer} {row}")
        if not row["accepted"]:break
    return {"sample_index":int(sample_index),"initial_mps":initial,"refined_mps":current,
            "target_mps":target,"history":history,"accepted_iterations":sum(x["accepted"] for x in history),
            "initial_metrics":metrics(initial,target,mask),"refined_metrics":metrics(current,target,mask)}


def figure(results, output_dir, phys_x, phys_y):
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
                         "font.size":8.5,"axes.titlesize":9,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(len(results),3,figsize=(7.5,2.15*len(results)),constrained_layout=True,squeeze=False)
    lo=min(np.min(x["target_mps"]) for x in results); hi=max(np.max(x["target_mps"]) for x in results)
    titles=["Synthetic reference","initial learned reconstruction","coarse Eikonal Eikonal refinement"]
    extent=[0,phys_x,phys_y,0]; panel=0; artist=None
    for row,result in enumerate(results):
        for col,image in enumerate((result["target_mps"],result["initial_mps"],result["refined_mps"])):
            ax=axes[row,col]; artist=ax.imshow(image.T,origin="upper",extent=extent,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest")
            ax.set_title(titles[col] if row==0 else ""); ax.set_aspect("equal")
            ax.text(.025,.975,f"({chr(97+panel)})",transform=ax.transAxes,fontweight="bold",fontsize=9,va="top",bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":1.5});panel+=1
            ax.set_ylabel("Axial position (m)" if col==0 else ""); ax.set_xlabel("Lateral position (m)" if row==len(results)-1 else "")
        axes[row,0].text(-.34,.5,f"Test sample {row+1}",transform=axes[row,0].transAxes,rotation=90,va="center",ha="center",fontweight="bold")
    divider=make_axes_locatable(axes[-1,-1]);cax=divider.append_axes("right",size="4%",pad=.06);bar=fig.colorbar(artist,cax=cax);bar.set_label("Speed of sound (m s$^{-1}$)")
    out=Path(output_dir); pdf=out/"coarse_eikonal_eikonal_refinement.pdf";png=out/"coarse_eikonal_eikonal_refinement.png"
    fig.savefig(pdf,bbox_inches="tight");fig.savefig(png,bbox_inches="tight",dpi=600);plt.close(fig)


def main():
    args=arguments();set_output_folder(args.output_dir);log_message("...................");log_message(f"[CMD] {' '.join(sys.argv)}");log_message(f"[ARGS]\n{pprint.pformat(vars(args))}");log_message(f"[coarse_eikonal] version={VERSION}")
    device=torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    report=np.load(args.initial_report,allow_pickle=True);required={"sample_indices","target_mps","prediction_mps"};missing=required.difference(report.files)
    if missing: raise RuntimeError(f"initial learned report is missing {sorted(missing)}")
    args.sos_min=scalar(report,"sos_min",args.sos_min);args.sos_max=scalar(report,"sos_max",args.sos_max)
    phys_x=scalar(report,"phys_x",.24);phys_y=scalar(report,"phys_y",.24)
    dataset=PairsCacheDataset(args.data_path,sos_min=args.sos_min,sos_max=args.sos_max,return_tof_mask=True);metadata=dict(dataset.metadata or {})
    ne=int(metadata.get("n_emitters",512));nr=int(metadata.get("n_receivers",512));points,_order,_center,digest=require_geometry(ne,nr)
    nx=np.asarray(report["prediction_mps"]).shape[-2];ny=np.asarray(report["prediction_mps"]).shape[-1];dx=phys_x/(nx-1);dy=phys_y/(ny-1)
    sensors_px=np.empty_like(points);sensors_px[:,0]=(nx-1)/2+points[:,0]/dx;sensors_px[:,1]=(ny-1)/2+points[:,1]/dy
    valid=np.asarray(dataset.tof_mask if dataset.tof_mask is not None else np.ones((ne,nr)),dtype=bool)
    indices=np.asarray(report["sample_indices"],dtype=np.int64);targets=np.asarray(report["target_mps"],dtype=np.float32);initial=np.asarray(report["prediction_mps"],dtype=np.float32)
    results=[]
    for position,index in enumerate(indices):
        sample=dataset._sample_arrays(int(index))
        if "tof_raw_eikonal" not in sample: raise RuntimeError(f"Sample {index} lacks tof_raw_eikonal")
        results.append(refine_sample(initial[position],targets[position],np.asarray(sample["tof_raw_eikonal"],dtype=np.float64),valid,sensors_px,args,device,phys_x,phys_y,int(index)))
    Path(args.out_pt).parent.mkdir(parents=True,exist_ok=True);torch.save({"kind":"synthetic_eikonal_coarse_refinement","version":VERSION,"geometry_digest":digest,"results":results},args.out_pt)
    np.savez_compressed(args.report_npz,sample_indices=indices,target_mps=np.stack([x["target_mps"] for x in results]),initial_mps=np.stack([x["initial_mps"] for x in results]),refined_mps=np.stack([x["refined_mps"] for x in results]),initial_rmse_mps=np.asarray([x["initial_metrics"]["rmse_mps"] for x in results]),refined_rmse_mps=np.asarray([x["refined_metrics"]["rmse_mps"] for x in results]),initial_correlation=np.asarray([x["initial_metrics"]["correlation"] for x in results]),refined_correlation=np.asarray([x["refined_metrics"]["correlation"] for x in results]),accepted_iterations=np.asarray([x["accepted_iterations"] for x in results]),phys_x=np.asarray(phys_x),phys_y=np.asarray(phys_y),sos_min=np.asarray(args.sos_min),sos_max=np.asarray(args.sos_max))
    summary={"version":VERSION,"samples":len(results),"mean_initial_rmse_mps":float(np.mean([x["initial_metrics"]["rmse_mps"] for x in results])),"mean_refined_rmse_mps":float(np.mean([x["refined_metrics"]["rmse_mps"] for x in results])),"accepted_iterations":[x["accepted_iterations"] for x in results]}
    (Path(args.output_dir)/"synthetic_eikonal_coarse_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");figure(results,args.output_dir,phys_x,phys_y);log_message(f"[synthetic-eikonal-coarse] summary={summary}");log_message(f"[synthetic-eikonal-coarse] saved={args.out_pt}; report={args.report_npz}")


if __name__=="__main__": main()
