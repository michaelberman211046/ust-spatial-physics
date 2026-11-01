"""Homogeneous-initialized regularized bent-ray travel-time inversion."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def difference_matrix(shape):
    nx,ny=shape
    ids=np.arange(nx*ny).reshape(shape)
    left=np.concatenate((ids[:-1,:].ravel(),ids[:,:-1].ravel()))
    right=np.concatenate((ids[1:,:].ravel(),ids[:,1:].ravel()))
    rows=np.repeat(np.arange(len(left)),2)
    cols=np.stack((left,right),1).ravel()
    vals=np.tile([-1.0,1.0],len(left))
    return sparse.coo_matrix((vals,(rows,cols)),shape=(len(left),nx*ny)).tocsr()


def ray_system(image, observed, valid, sensors, emitter_ids, receiver_ids, px, py, ray_args):
    import refine_synthetic_eikonal_fine as engine
    nx,ny=image.shape;dx=px/(nx-1);dy=py/(ny-1)
    blocks=[];residuals=[];emitters=[]
    for emitter in emitter_ids:
        try:
            rays=engine.build_rays(image,observed,valid,sensors,np.array([emitter]),receiver_ids,dx,dy,ray_args)
        except RuntimeError as exc:
            if 'No valid' not in str(exc):
                raise
            continue
        indices=rays['indices'];weights=rays['weights'];lengths=rays['lengths']
        nr,nstep=lengths.shape
        values=(lengths[:,:,None]*weights).reshape(-1)
        cols=indices.reshape(-1)
        rows=np.repeat(np.arange(nr),nstep*4)
        keep=values>0
        h=sparse.coo_matrix((values[keep],(rows[keep],cols[keep])),shape=(nr,nx*ny)).tocsr()
        blocks.append(h);residuals.append(rays['observed']-rays['predicted'])
        emitters.extend([int(emitter)]*nr)
    if not blocks:
        raise RuntimeError('Classical inversion has no traced valid paths')
    return sparse.vstack(blocks,format='csr'),np.concatenate(residuals),np.asarray(emitters)


def score(image, observed, valid, sensors, emitters, receivers, px, py):
    from refine_synthetic_eikonal_coarse import solve_times
    prediction=solve_times(image,sensors,emitters,receivers,px/(image.shape[0]-1),py/(image.shape[1]-1))
    select=valid[np.ix_(emitters,receivers)]
    values=(prediction-observed[np.ix_(emitters,receivers)])[select]*1e6
    if not len(values):
        raise ValueError('Empty classical validation channels')
    return dict(rms_us=float(np.sqrt(np.mean(values**2))),median_abs_us=float(np.median(np.abs(values))))


def invert(observed, valid, sensors, shape, support, px, py, water, lo, hi, cfg, ray_args):
    """Fit six angular sectors; select the iterate on two disjoint sectors.

    The gradient penalty acts on the absolute deviation from homogeneous water.
    No learned image, target image, or FWI comparator enters this function.
    """
    ne,nr=observed.shape
    emit=np.arange(0,ne,ray_args.emitter_stride)
    recv=np.arange(0,nr,ray_args.receiver_stride)
    sector=np.minimum(7,emit*8//ne)
    held=emit[np.isin(sector,cfg['validation_sectors'])]
    fitted=emit[~np.isin(sector,cfg['validation_sectors'])]
    current=np.full(shape,water,np.float32);best=current.copy()
    before=score(current,observed,valid,sensors,held,recv,px,py)
    best_score=before['rms_us'];history=[]
    inside=np.flatnonzero(np.asarray(support).ravel()>.5)
    d=difference_matrix(shape)[:,inside]
    identity=sparse.eye(len(inside),format='csr')
    scale=100.0
    for iteration in range(int(cfg['max_iterations'])):
        print(f'[CLASSICAL] iteration {iteration+1}/{cfg["max_iterations"]}: tracing rays',flush=True)
        h,residual,_=ray_system(current,observed,valid,sensors,fitted,recv,px,py,ray_args)
        jac=h.multiply((-1e6*scale/current.ravel()**2)[None,:]).tocsr()[:,inside]
        n=len(residual)
        prior=(current.ravel()[inside]-water)/scale
        grad_factor=np.sqrt(cfg['gradient_weight']/max(d.shape[0],1))
        amp_factor=np.sqrt(cfg['amplitude_weight']/max(len(inside),1))
        matrix=sparse.vstack((jac/np.sqrt(n),grad_factor*d,amp_factor*identity),format='csr')
        rhs=np.concatenate((residual*1e6/np.sqrt(n),-grad_factor*d@prior,-amp_factor*prior))
        solved=lsqr(matrix,rhs,atol=1e-5,btol=1e-5,iter_lim=cfg['lsqr_iterations'])
        update=np.zeros(current.size);update[inside]=np.clip(scale*solved[0],-cfg['step_limit_mps'],cfg['step_limit_mps'])
        update=update.reshape(shape)
        fit_before=score(current,observed,valid,sensors,fitted,recv,px,py)['rms_us']
        moved=False
        for alpha in (1.,.5,.25,.125):
            candidate=np.clip(current+alpha*update,lo,hi).astype(np.float32)
            fit_after=score(candidate,observed,valid,sensors,fitted,recv,px,py)['rms_us']
            old_objective=fit_before**2+cfg['gradient_weight']*float(np.mean((d@prior)**2))+cfg['amplitude_weight']*float(np.mean(prior**2))
            new_prior=(candidate.ravel()[inside]-water)/scale
            objective=fit_after**2+cfg['gradient_weight']*float(np.mean((d@new_prior)**2))+cfg['amplitude_weight']*float(np.mean(new_prior**2))
            if objective < old_objective:
                current=candidate;moved=True;break
        val=score(current,observed,valid,sensors,held,recv,px,py)
        if val['rms_us'] < best_score-cfg['minimum_improvement_us']:
            best=current.copy();best_score=val['rms_us']
        history.append(dict(iteration=iteration,fit_rms_us=fit_before,validation=val,
                            lsqr_stop=int(solved[1]),lsqr_iterations=int(solved[2]),moved=moved))
        print(f'[CLASSICAL] validation RMS={val["rms_us"]:.6f} us; best={best_score:.6f} us; moved={moved}',flush=True)
        if not moved:
            break
    return best,dict(history=history,initial_validation=before,best_validation_rms_us=best_score,
                     selection='fixed sector holdout; minimum RMS',initialization='homogeneous water')
