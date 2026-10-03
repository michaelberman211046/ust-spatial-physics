import numpy as np
import torch
import torch.nn.functional as F

import refine_synthetic_eikonal_coarse as coarse_engine
import refine_synthetic_eikonal_fine as fine_engine


def _project_update(raw_full, support, limit):
    dc=support*raw_full
    denom=(support*support).sum().clamp_min(1.0)
    mean=(dc*support).sum()/denom
    dc=support*(raw_full-mean)
    dc=dc.clamp(-limit,limit)
    mean=(dc*support).sum()/denom
    return dc-support*mean


def _values(image, observed, valid, sensors, emitters, receivers, dx, dy, delay=0.0):
    predicted=coarse_engine.solve_times(image,sensors,emitters,receivers,dx,dy);values=[]
    for row,e in enumerate(emitters):
        use=valid[e,receivers]
        values.extend(((predicted[row,use]-observed[e,receivers[use]]+delay)*1e6).tolist())
    return np.asarray(values,np.float64)


def _score(values):
    if not len(values):return {"median_abs_us":float("inf"),"rms_us":float("inf")}
    return {"median_abs_us":float(np.median(np.abs(values))),"rms_us":float(np.sqrt(np.mean(values*values)))}


def _sector_emitters(emitters,sector,sectors,ne):
    return np.asarray([e for e in emitters if min(sectors-1,int(e*sectors/ne))==sector],np.int64)


def refine(base, edge_reference, observed, valid, sensors, support_np, args, device,
           phys_x, phys_y, fine=False, label="measured-coarse"):
    nx,ny=base.shape;dx=phys_x/(nx-1);dy=phys_y/(ny-1);current=base.copy()
    support_np=np.clip(support_np,0,1).astype(np.float32)
    support=torch.as_tensor(support_np,dtype=torch.float32,device=device)[None,None]
    emitters=np.arange(0,observed.shape[0],args.emitter_stride,dtype=np.int64)
    receivers=np.arange(0,observed.shape[1],args.receiver_stride,dtype=np.int64)
    sectors=[int(x) for x in args.holdout_sectors.split(",") if x.strip()]
    iterations=args.fine_iterations if fine else args.outer_iterations
    grid=args.fine_grid if fine else args.correction_grid
    limit=args.fine_limit_mps if fine else args.update_limit_mps
    history=[]
    for outer in range(iterations):
        if fine:
            rays=fine_engine.build_rays(current,observed,valid,sensors,emitters,receivers,dx,dy,args)
            indices=torch.as_tensor(rays["indices"],dtype=torch.long,device=device)
            weights=torch.as_tensor(rays["weights"],dtype=torch.float32,device=device)
        else:
            _pred,rays=coarse_engine.forward_and_rays(current,observed,valid,sensors,emitters,receivers,dx,dy,args,True)
            indices=torch.as_tensor(rays["indices"],dtype=torch.long,device=device)
            weights=None
        lengths=torch.as_tensor(rays["lengths"],dtype=torch.float32,device=device)
        base_residual=torch.as_tensor(rays["predicted"]-rays["observed"],dtype=torch.float32,device=device)
        ray_emitters=torch.as_tensor(rays["emitters"],dtype=torch.long,device=device)
        ray_sectors=torch.div(ray_emitters*args.angular_sectors,observed.shape[0],rounding_mode="floor")
        current_t=torch.as_tensor(current,dtype=torch.float32,device=device)[None,None]
        reference=torch.as_tensor(edge_reference,dtype=torch.float32,device=device)[None,None]
        gx=reference[...,1:,:]-reference[...,:-1,:];gy=reference[...,:,1:]-reference[...,:,:-1]
        edge_scale=torch.quantile(torch.cat((gx.abs().flatten(),gy.abs().flatten())),.80).clamp_min(2.0)
        edge_x=torch.exp(-gx.abs()/edge_scale);edge_y=torch.exp(-gy.abs()/edge_scale)
        fold_updates=[];fold_records=[]
        for heldout in sectors:
            validation=(ray_sectors==heldout);fit=~validation
            if int(validation.sum())<1 or int(fit.sum())<1:continue
            # The median fit residual is a bounded nuisance timing term. It is
            # removed from the image objective and never converted into SoS.
            nuisance=torch.median(base_residual[fit]).detach().clamp(-.75e-6,.75e-6)
            raw=torch.nn.Parameter(torch.zeros((1,1,grid,grid),device=device));optimizer=torch.optim.Adam([raw],lr=args.lr)
            for _step in range(args.inner_steps):
                coarse=limit*torch.tanh(raw)
                full=F.interpolate(coarse,size=(nx,ny),mode="bilinear",align_corners=False)
                dc=_project_update(full,support,limit)
                candidate=(current_t+dc).clamp(args.sos_min,args.sos_max)
                ds=(1/candidate-1/current_t).reshape(-1)
                sampled=(ds[indices]*weights).sum(-1) if fine else ds[indices]
                residual=base_residual-nuisance+(sampled*lengths).sum(1)
                data=coarse_engine.robust(residual*1e6,fit.float(),args.huber_beta_us)
                prior=(dc.square()*support).sum()/support.sum().clamp_min(1)/(limit*limit)
                dxdc=dc[...,1:,:]-dc[...,:-1,:];dydc=dc[...,:,1:]-dc[...,:,:-1]
                if fine:
                    smooth=(torch.sqrt(dxdc.square()+.04)*edge_x).mean()+(torch.sqrt(dydc.square()+.04)*edge_y).mean()
                    smooth_weight=args.edge_tv_weight
                else:
                    smooth=dxdc.abs().mean()+dydc.abs().mean();smooth_weight=args.tv_weight
                curvature=(F.avg_pool2d(coarse,3,1,1)-coarse).square().mean()/(limit*limit)
                loss=data+args.prior_weight*prior+smooth_weight*smooth+args.curvature_weight*curvature
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([raw],1.0);optimizer.step()
            with torch.no_grad():
                full=F.interpolate(limit*torch.tanh(raw),size=(nx,ny),mode="bilinear",align_corners=False)
                proposed=_project_update(full,support,limit)[0,0].cpu().numpy()
            held=_sector_emitters(emitters,heldout,args.angular_sectors,observed.shape[0])
            fit_emit=np.asarray([e for e in emitters if e not in set(held.tolist())],np.int64)
            baseline_fit=_values(current,observed,valid,sensors,fit_emit,receivers,dx,dy)
            baseline_delay=float(np.clip(-np.median(baseline_fit)*1e-6,-.75e-6,.75e-6))
            before=_score(_values(current,observed,valid,sensors,held,receivers,dx,dy,baseline_delay));accepted=None
            for alpha in (1.0,.5,.25,.125):
                update=alpha*proposed;candidate=np.clip(current+update,args.sos_min,args.sos_max)
                candidate_fit=_values(candidate,observed,valid,sensors,fit_emit,receivers,dx,dy)
                delay=float(np.clip(-np.median(candidate_fit)*1e-6,-.75e-6,.75e-6))
                after=_score(_values(candidate,observed,valid,sensors,held,receivers,dx,dy,delay))
                weighted_mean=float((update*support_np).sum()/max(support_np.sum(),1));weighted_rms=float(np.sqrt((update*update*support_np).sum()/max(support_np.sum(),1)))
                if (before["median_abs_us"]-after["median_abs_us"]>=args.min_holdout_improvement_us and
                    after["rms_us"]<=before["rms_us"] and abs(weighted_mean)<=args.max_mean_update_mps and weighted_rms<=args.max_rms_update_mps):
                    accepted=update;fold_records.append({"sector":heldout,"accepted":True,"alpha":alpha,"before":before,"after":after,"delay_us":delay*1e6,"mean_update_mps":weighted_mean,"rms_update_mps":weighted_rms});break
            if accepted is None:fold_records.append({"sector":heldout,"accepted":False,"before":before})
            else:fold_updates.append(accepted)
        row={"iteration":outer,"accepted":False,"folds":fold_records}
        if len(fold_updates)>=args.min_accepted_folds:
            stack=np.stack(fold_updates);median=np.median(stack,axis=0);spread=np.quantile(stack,.75,axis=0)-np.quantile(stack,.25,axis=0)
            scale=max(1.0,float(np.quantile(spread[support_np>.25],.75)));update=np.exp(-(spread/scale)**2)*median
            update=update*support_np
            weighted_mean=(update*support_np).sum()/max((support_np*support_np).sum(),1);update=update-support_np*weighted_mean
            before_values=_values(current,observed,valid,sensors,emitters,receivers,dx,dy)
            before_delay=float(np.clip(-np.median(before_values)*1e-6,-.75e-6,.75e-6))
            before_all=_score(before_values+before_delay*1e6)
            for alpha in (1.0,.5,.25,.125):
                applied=alpha*update;candidate=np.clip(current+applied,args.sos_min,args.sos_max)
                after_values=_values(candidate,observed,valid,sensors,emitters,receivers,dx,dy)
                after_delay=float(np.clip(-np.median(after_values)*1e-6,-.75e-6,.75e-6));after_all=_score(after_values+after_delay*1e6)
                mean=float((applied*support_np).sum()/max(support_np.sum(),1));rms=float(np.sqrt((applied*applied*support_np).sum()/max(support_np.sum(),1)))
                if (before_all["median_abs_us"]-after_all["median_abs_us"]>=args.min_holdout_improvement_us and after_all["rms_us"]<=before_all["rms_us"] and abs(mean)<=args.max_mean_update_mps and rms<=args.max_rms_update_mps):
                    current=candidate;row.update({"accepted":True,"alpha":alpha,"before":before_all,"after":after_all,"accepted_folds":len(fold_updates),"before_delay_us":before_delay*1e6,"after_delay_us":after_delay*1e6,"mean_update_mps":mean,"rms_update_mps":rms});break
        history.append(row);coarse_engine.log_message(f"[{label}] iteration={outer} {row}")
        if not row["accepted"]:break
    return current,history
