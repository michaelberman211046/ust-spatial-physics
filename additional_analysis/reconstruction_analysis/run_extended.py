"""Separate saved-checkpoint reliability evaluation; all writes stay under --output."""
import argparse,gc,json,time,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_ablation as ab
import report_utils as utils


def resize(a,shape):
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.as_tensor(np.asarray(a).copy(),dtype=torch.float32)[None,None],size=shape,mode='bilinear',align_corners=True)[0,0].numpy()


def sensors(ctx,shape):
    from measured_geometry import require_geometry
    p,_,_,_=require_geometry(ctx.args.n_emitters,ctx.args.n_receivers)
    return np.column_stack(((shape[0]-1)/2+p[:,0]*(shape[0]-1)/ctx.args.phys_x,
                            (shape[1]-1)/2+p[:,1]*(shape[1]-1)/ctx.args.phys_y))


def refinement_args(ctx,fine=False,shape=None,measured=False):
    d=dict(ctx.study['fine' if fine else 'coarse'])
    d.update(emitter_stride=ctx.study['emitter_stride'],receiver_stride=ctx.study['receiver_stride'],
             sos_min=ctx.args.sos_min,sos_max=ctx.args.sos_max,sos_water=ctx.args.sos_water,mask_radius=ctx.args.mask_radius)
    if measured:
        d.update(min_accepted_folds=3,max_mean_update_mps=1.5,max_rms_update_mps=4. if fine else 8.)
    if shape is not None:
        # Keep characteristic integration increments comparable in physical units.
        d['trace_step_pixels']*= (shape[0]-1)/(ctx.args.nx-1)
    return SimpleNamespace(**d)


def synthetic_refine(ctx,initial,observed,valid,index):
    import refine_synthetic_eikonal_coarse as coarse
    import refine_synthetic_eikonal_fine as fine
    shape=initial.shape;s=sensors(ctx,shape);a=refinement_args(ctx,shape=shape)
    # True generating fields never enter either optimizer, even for diagnostics.
    first=coarse.refine_sample(initial,initial,observed,valid,s,a,ctx.device,ctx.args.phys_x,ctx.args.phys_y,index)
    second=fine.fine_refine(first['refined_mps'],initial,initial,observed,valid,s,
        refinement_args(ctx,True,shape),ctx.device,ctx.args.phys_x,ctx.args.phys_y,index)
    return second['refined_mps'],{'coarse':first['history'],'fine':second['history'],
        'accepted_coarse':first['accepted_iterations'],'accepted_fine':second['accepted_iterations']}


def measured_data(ctx):
    import torch
    from refine_measured_eikonal_coarse import dense_measurements
    setup=torch.load(ctx.extra_paths['setup'],map_location='cpu',weights_only=False)
    saved=torch.load(ctx.extra_paths['measured_initial'],map_location='cpu',weights_only=False)
    base=np.asarray(saved['agreement_weighted_reconstruction_mps']).squeeze().astype(np.float32)
    support=np.asarray(setup['soft_support']).squeeze().astype(np.float32)
    geom=setup['geometry'];pg=setup['physical_geometry']
    if not np.allclose([geom['phys_x'],geom['phys_y']],[ctx.args.phys_x,ctx.args.phys_y]):raise ValueError('Physical domain differs')
    reference=np.asarray(setup['reference_diagnostic_normalized']).squeeze()*(geom['sos_max']-geom['sos_min'])+geom['sos_min']
    reference=resize(reference,base.shape);support=resize(support,base.shape)
    obs,valid=dense_measurements(setup)
    if not np.isclose(pg['water_sos'],ctx.args.sos_water):raise ValueError('Water SoS differs')
    return base,support,reference,obs,valid


def measured_refine(ctx,base,support,obs,valid,label):
    from measured_eikonal_refinement import refine
    s=sensors(ctx,base.shape);px,py=ctx.args.phys_x,ctx.args.phys_y
    first,h1=refine(base,base,obs,valid,s,support,refinement_args(ctx,False,base.shape,True),ctx.device,px,py,False,label+'-coarse')
    final,h2=refine(first,base,obs,valid,s,support,refinement_args(ctx,True,base.shape,True),ctx.device,px,py,True,label+'-fine')
    return final,{'coarse':h1,'fine':h2}


def prepare(cli):
    ctx=ab.prepare(cli);ctx.study=ctx.cfg['studies'];ctx.utils=utils
    exp=ctx.cli.root/'experiment_run'/'experimental_geometry_v28'
    ctx.extra_paths={'measured_raw':exp/'experimental_raw.pt','setup':exp/'physical_setup.pt',
                     'measured_initial':exp/'physical_reconstruction.pt'}
    identities={k:utils.file_identity(p,True) for k,p in ctx.extra_paths.items()}
    rng=np.random.default_rng(int(ctx.cfg['selection_seed'])+901)
    ids=rng.permutation(ctx.indices['test']).astype(int).tolist()
    required=max(int(ctx.study[k]) for k in ['refinement_samples','sensitivity_samples','synthetic_grid_samples'])
    if required>len(ids):raise ValueError('Requested more test samples than available')
    ctx.plan={'refinement':ids[:ctx.study['refinement_samples']],
              'sensitivity':ids[:ctx.study['sensitivity_samples']],
              'synthetic_grid':ids[:ctx.study['synthetic_grid_samples']]}
    protocol={'extra_inputs':identities,'selected_indices':ctx.plan,'studies':ctx.study,
              'reference_use':'Known synthetic fields and FWI are used only for post-optimization evaluation.',
              'budgets':'Wall-time limits apply between complete cases; repeated commands resume the same predetermined cases.'}
    path=ctx.cli.output/'extended_protocol.json'
    if path.exists() and json.loads(path.read_text())!=ab.jsonable(protocol):raise RuntimeError('Extended protocol changed; use a new output')
    ab.save_json(path,protocol)
    ctx.verify_extra=lambda:verify_extra(ctx,identities)
    return ctx


def verify_extra(ctx,identities):
    result={k:utils.file_identity(ctx.extra_paths[k],True)==v for k,v in identities.items()}
    ab.save_json(ctx.cli.output/'extended_input_integrity.json',result)
    if not all(result.values()):raise RuntimeError('Original experimental inputs changed')


def original_model(ctx):
    import torch
    model=ab.spatial.SpatialPhysicsResidualNet(ctx.layout['channels'],ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit).to(ctx.device)
    ck=ctx.original_checkpoint
    model.load_state_dict(ck.get('best_model_state_dict',ck['model_state_dict']));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def infer(ctx,batch,stack,mask,water,model):
    import torch
    from latent_operator_models import support_apply
    feat=ab.features(ctx,batch,stack,mask,water)
    with torch.no_grad():pred=support_apply(feat['prior']+model(feat['features']),mask,ctx.args.water_norm)
    return feat,pred


def cache(ctx):
    import torch
    folder=ctx.cli.output/'synthetic_cache';folder.mkdir(exist_ok=True)
    ids=sorted(set(sum(ctx.plan.values(),[])));missing=[i for i in ids if not (folder/f'{i}.npz').exists()]
    if not missing:return
    ds,loader,mask,water,stack=ab.load_upstream(ctx,missing);model=original_model(ctx)
    scale=ctx.args.sos_max-ctx.args.sos_min
    for batch in loader:
        i=int(batch[3].item());feat,pred=infer(ctx,batch,stack,mask,water,model)
        raw=ds._sample_arrays(i)
        if 'tof_raw_eikonal' not in raw:raise ValueError(f'Missing raw Eikonal ToF for {i}')
        np.savez_compressed(folder/f'{i}.npz',initial=pred[0,0].cpu().numpy()*scale+ctx.args.sos_min,
            reference=feat['target'][0].cpu().numpy().squeeze()*scale+ctx.args.sos_min,
            observed=np.asarray(raw['tof_raw_eikonal'],float),valid=np.asarray(ds.tof_mask,bool),region=mask.cpu().numpy().squeeze())
        print(f'[CACHE] original frozen test sample {i}',flush=True)
    del ds,loader,mask,water,stack,model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def case(ctx,index):
    with np.load(ctx.cli.output/'synthetic_cache'/f'{index}.npz') as d:return {k:d[k] for k in d.files}


def summarize_cases(ctx,folder,planned):
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob('case_*.json'))]
    utils.write_csv(folder/'case_metrics.csv',rows)
    stats={k:utils.describe([r[k] for r in rows if r.get(k) is not None]) for k in (rows[0].keys() if rows else [])
           if k not in ['sample_index','condition'] and isinstance(rows[0][k],(int,float))}
    result={'planned':planned,'completed':len(rows),'complete':len(rows)==planned,'summary':stats}
    if rows and 'initial_rmse_mps' in rows[0]:
        rng=np.random.default_rng(ctx.cfg['selection_seed']);paired={}
        for metric in ['rmse_mps','nrmse_percent','mad_mps','correlation']:
            key='final_'+metric;before='initial_'+metric
            if key not in rows[0]:continue
            delta=np.array([r[key]-r[before] for r in rows if r[key] is not None and r[before] is not None],float)
            if not len(delta):continue
            boots=np.mean(rng.choice(delta,(ctx.cfg['bootstrap_replicates'],len(delta)),replace=True),axis=1)
            paired[metric]={'mean_final_minus_initial':float(delta.mean()),'bootstrap_mean_CI95':np.quantile(boots,[.025,.975]).tolist()}
        result['paired_changes']=paired
    ab.save_json(folder/'summary.json',result)
    return result


def tof_diagnostics(ctx,image,observed,valid,measured=False):
    from measured_eikonal_refinement import _values
    emitters=np.arange(0,observed.shape[0],ctx.study['emitter_stride'])
    receivers=np.arange(0,observed.shape[1],ctx.study['receiver_stride'])
    values=_values(image,observed,valid,sensors(ctx,image.shape),emitters,receivers,
                   ctx.args.phys_x/(image.shape[0]-1),ctx.args.phys_y/(image.shape[1]-1))
    if not len(values):raise ValueError('No valid ToF diagnostic channels')
    delay=float(np.clip(-np.median(values),-.75,.75)) if measured else 0.
    aligned=values+delay
    return {'tof_rms_us':float(np.sqrt(np.mean(aligned**2))),
            'tof_median_abs_us':float(np.median(np.abs(aligned))),'diagnostic_common_delay_us':delay}


def refinement(ctx):
    cache(ctx);folder=ctx.cli.output/'larger_refinement';folder.mkdir(exist_ok=True);start=time.perf_counter()
    for i in ctx.plan['refinement']:
        if (folder/f'case_{i}.json').exists():continue
        if time.perf_counter()-start>ctx.study['refinement_budget_hours']*3600:break
        d=case(ctx,i);ab.sync(ctx.device);t=time.perf_counter()
        final,history=synthetic_refine(ctx,d['initial'],d['observed'],d['valid'],i)
        ab.sync(ctx.device);elapsed=time.perf_counter()-t
        np.savez_compressed(folder/f'case_{i}.npz',initial=d['initial'],final=final,reference=d['reference'],region=d['region'])
        row={'sample_index':i,'seconds':elapsed,'accepted_coarse':history['accepted_coarse'],'accepted_fine':history['accepted_fine']}
        for label,image in [('initial',d['initial']),('final',final)]:
            row.update({label+'_'+k:v for k,v in utils.image_metrics(image,d['reference'],d['region']).items()})
        for label,image in [('initial',d['initial']),('final',final)]:
            row.update({label+'_'+k:v for k,v in tof_diagnostics(ctx,image,d['observed'],d['valid']).items()})
        ab.save_json(folder/f'history_{i}.json',history);ab.save_json(folder/f'case_{i}.json',row)
    print('[REFINEMENT]',summarize_cases(ctx,folder,len(ctx.plan['refinement'])),flush=True)


def normalized_tof(seconds,valid,norm):
    x=np.asarray(seconds,np.float32).copy();kind=norm.get('type','none');eps=float(norm.get('eps',1e-6))
    if kind=='zscore':x=(x-float(norm.get('mean',0)))/(float(norm.get('std',1))+eps)
    elif kind=='max':x/=float(norm.get('max',1))+eps
    elif kind!='none':raise ValueError(f'Unknown ToF normalization {kind}')
    return x*np.asarray(valid,np.float32)


def tof_sensitivity(ctx):
    import torch
    folder=ctx.cli.output/'tof_robustness';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    ds,loader,mask,water,stack=ab.load_upstream(ctx,ctx.plan['sensitivity']);model=original_model(ctx)
    scale=ctx.args.sos_max-ctx.args.sos_min;rows=[]
    for batch in loader:
        i=int(batch[3].item());raw=ds._sample_arrays(i)
        region=mask.cpu().numpy().squeeze();target=batch[0].numpy().squeeze()*scale+ctx.args.sos_min
        for j,(kind,amplitude) in enumerate(ctx.study['tof_conditions']):
            name=f'{kind}_{amplitude:g}';path=folder/f'case_{i}_{j}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            seconds,valid=utils.perturb_tof(raw['tof'],ds.tof_mask,kind,float(amplitude),int(ctx.cfg['selection_seed'])+i*101+j)
            modified=list(batch);modified[1]=torch.from_numpy(normalized_tof(seconds,valid,ds.tof_norm))[None]
            modified[2]=torch.from_numpy(valid.astype(np.float32))[None]
            ab.sync(ctx.device);t=time.perf_counter();_,pred=infer(ctx,modified,stack,mask,water,model);ab.sync(ctx.device)
            image=pred[0,0].cpu().numpy()*scale+ctx.args.sos_min
            row={'sample_index':i,'condition':name,'seconds':time.perf_counter()-t,**utils.image_metrics(image,target,region)}
            ab.save_json(path,row);rows.append(row)
    utils.write_csv(folder/'metrics.csv',rows)
    summary=[]
    for kind,amp in ctx.study['tof_conditions']:
        name=f'{kind}_{amp:g}';group=[r for r in rows if r['condition']==name]
        item={'condition':name,'samples':len(group)}
        for metric in ab.METRICS:
            item[metric]=utils.describe([r[metric] for r in group])
            clean={r['sample_index']:r for r in rows if r['condition']=='clean_0'}
            item[metric+'_paired_change']=utils.describe([r[metric]-clean[r['sample_index']][metric] for r in group if r[metric] is not None and clean[r['sample_index']][metric] is not None])
        summary.append(item)
    ab.save_json(folder/'complete.json',{'summary':summary,'scope':'Controlled perturbations of processed synthetic ToFs; frozen learned reconstruction; known generating fields.'})
    del ds,loader,stack,model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def transform_support(support,condition,phys_x,phys_y):
    from scipy.ndimage import map_coordinates
    s=np.asarray(support,float);gx,gy=np.indices(s.shape,dtype=float);mass=s.sum()
    if mass<=0:raise ValueError('Empty nominal support')
    cx=float((gx*s).sum()/mass);cy=float((gy*s).sum()/mass)
    radius=float(condition['radius_scale'])
    qx=cx+(gx-cx-float(condition['shift_x_m'])*(s.shape[0]-1)/phys_x)/radius
    qy=cy+(gy-cy-float(condition['shift_y_m'])*(s.shape[1]-1)/phys_y)/radius
    out=map_coordinates(s,[qx,qy],order=1,mode='constant',cval=0.)
    taper=float(condition['taper_scale']);interior=(out>0)&(out<1)
    out[interior]=1/(1+np.exp(-np.log(out[interior]/(1-out[interior]))/taper))
    return out.astype(np.float32)


def support_sensitivity(ctx):
    folder=ctx.cli.output/'support_sensitivity';folder.mkdir(exist_ok=True)
    base,support,reference,obs,valid=measured_data(ctx);region=support>.5;start=time.perf_counter()
    for condition in ctx.study['support_conditions']:
        name=condition['name'];path=folder/f'case_{name}.json'
        if path.exists():continue
        if time.perf_counter()-start>ctx.study['support_budget_hours']*3600:break
        changed=transform_support(support,condition,ctx.args.phys_x,ctx.args.phys_y)
        ab.sync(ctx.device);t=time.perf_counter();final,history=measured_refine(ctx,base,changed,obs,valid,name);ab.sync(ctx.device)
        row={'condition':name,'seconds':time.perf_counter()-t,'evaluation_pixels':int(region.sum()),**utils.image_metrics(final,reference,region)}
        row.update(tof_diagnostics(ctx,final,obs,valid,True))
        np.savez_compressed(folder/f'case_{name}.npz',initial=base,final=final,reference=reference,support=changed,evaluation_region=region)
        ab.save_json(folder/f'history_{name}.json',history);ab.save_json(path,row)
    nominal=folder/'case_nominal.npz';rows=[]
    if nominal.exists():
        with np.load(nominal) as f:nom=f['final']
        for p in sorted(folder.glob('case_*.json')):
            row=json.loads(p.read_text())
            with np.load(p.with_suffix('.npz')) as f:difference=f['final']-nom
            row['rms_change_from_nominal_mps']=float(np.sqrt(np.mean(difference[region]**2)))
            row['mean_change_from_nominal_mps']=float(np.mean(difference[region]));rows.append(row)
    utils.write_csv(folder/'metrics.csv',rows)
    ab.save_json(folder/'summary.json',{'planned':len(ctx.study['support_conditions']),'completed':len(rows),'complete':len(rows)==len(ctx.study['support_conditions']),
        'initial_FWI_agreement':utils.image_metrics(base,reference,region),'results':rows,
        'scope':'Support perturbations of measured post-inference Eikonal refinement, holding the saved entering image, measured ToFs and nominal evaluation region fixed; FWI used after optimization.'})


def forward_grid(ctx):
    from measured_geometry import require_geometry
    from refine_synthetic_eikonal_coarse import solve_times
    folder=ctx.cli.output/'forward_grid';folder.mkdir(exist_ok=True)
    points,_,_,_=require_geometry(ctx.args.n_emitters,ctx.args.n_receivers)
    emitters=np.linspace(0,len(points)-1,ctx.study['forward_emitters'],dtype=int)
    receivers=np.linspace(0,len(points)-1,ctx.study['forward_receivers'],dtype=int)
    px,py=ctx.args.phys_x,ctx.args.phys_y
    # Same analytic field sampled independently on each mesh.
    def field(n,heterogeneous):
        x=np.linspace(-px/2,px/2,n)[:,None];y=np.linspace(-py/2,py/2,n)[None,:]
        result=np.full((n,n),1500.,np.float32)
        if heterogeneous:
            result[((x+.025)/.045)**2+((y-.01)/.06)**2<=1]=1450.
            result[((x-.025)/.018)**2+((y+.012)/.023)**2<=1]=1580.
        return result
    from dataset import PairsCacheDataset
    ds=PairsCacheDataset(str(ctx.paths['data']),sos_min=ctx.args.sos_min,sos_max=ctx.args.sos_max,return_tof_mask=True)
    valid=np.asarray(ds.tof_mask,bool)[np.ix_(emitters,receivers)];del ds
    reference_n=ctx.study['forward_reference_grid'];rows=[]
    for heterogeneous in [False,True]:
        label='heterogeneous' if heterogeneous else 'homogeneous';solves={}
        for n in sorted(set(ctx.study['forward_grids']+[reference_n])):
            path=folder/f'{label}_{n}.npz'
            if not path.exists():
                t=time.perf_counter();times=solve_times(field(n,heterogeneous),sensors(ctx,(n,n)),emitters,receivers,px/(n-1),py/(n-1))
                np.savez_compressed(path,times=times,seconds=time.perf_counter()-t)
            with np.load(path) as f:solves[n]=(f['times'],float(f['seconds']))
        analytic=np.linalg.norm(points[emitters,None,:]-points[None,receivers,:],axis=-1)/1500.
        truth=solves[reference_n][0] if heterogeneous else analytic
        for n in ctx.study['forward_grids']:
            delta=(solves[n][0]-truth)[valid]*1e6
            rows.append({'field':label,'grid':n,'reference':f'numerical_{reference_n}' if heterogeneous else 'analytic_chord',
                'seconds':solves[n][1],'channels':int(valid.sum()),'tof_rmse_us':float(np.sqrt(np.mean(delta**2))),
                'tof_bias_us':float(delta.mean()),'tof_p95_abs_us':float(np.quantile(abs(delta),.95))})
    utils.write_csv(folder/'metrics.csv',rows)
    ab.save_json(folder/'complete.json',{'results':rows,'scope':'Forward numerical convergence for analytic homogeneous and heterogeneous test fields. The heterogeneous reference is a finer numerical solution.'})


def reconstruction_grid(ctx):
    cache(ctx);folder=ctx.cli.output/'reconstruction_grid';folder.mkdir(exist_ok=True);start=time.perf_counter()
    base,support,reference,obs,valid=measured_data(ctx)
    cases=[('synthetic',i) for i in ctx.plan['synthetic_grid']]+[('measured',-1)];planned=len(cases)*len(ctx.study['reconstruction_grids'])
    for kind,index in cases:
        if kind=='synthetic':
            d=case(ctx,index);initial=d['initial'];ref=d['reference'];region=d['region'];observed=d['observed'];v=d['valid']
        else:initial=base;ref=reference;region=support>.5;observed=obs;v=valid
        for n in ctx.study['reconstruction_grids']:
            name=f'{kind}_{index}_{n}';path=folder/f'case_{name}.json'
            if path.exists():continue
            if time.perf_counter()-start>ctx.study['grid_budget_hours']*3600:break
            entering=resize(initial,(n,n));ab.sync(ctx.device);t=time.perf_counter()
            if kind=='synthetic':final,history=synthetic_refine(ctx,entering,observed,v,index)
            else:final,history=measured_refine(ctx,entering,resize(support,(n,n)),observed,v,name)
            ab.sync(ctx.device);elapsed=time.perf_counter()-t;common=resize(final,initial.shape)
            row={'condition':name,'kind':kind,'sample_index':index,'grid':n,'seconds':elapsed}
            for label,image in [('resampling_control',resize(entering,initial.shape)),('final',common)]:
                row.update({label+'_'+k:value for k,value in utils.image_metrics(image,ref,region).items()})
            np.savez_compressed(folder/f'case_{name}.npz',entering=entering,final=final,final_common_grid=common,reference_common_grid=ref,evaluation_region=region)
            ab.save_json(folder/f'history_{name}.json',history);ab.save_json(path,row)
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob('case_*.json'))];utils.write_csv(folder/'metrics.csv',rows)
    ab.save_json(folder/'summary.json',{'planned':planned,'completed':len(rows),'complete':len(rows)==planned,'results':rows,
        'scope':'Fresh physical optimization at each grid; same fixed correction basis and physical characteristic step; all metrics on original common grid with resampling-only controls. This tests numerical/post-inference grid effects, not a retrained high-resolution neural model.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['preflight','rf','tof','refinement','support','forward-grid','reconstruction-grid','all'],required=True)
    p.add_argument('--root',default='Data/new_pipeline_512_20000a');p.add_argument('--output',required=True)
    p.add_argument('--geometry-mat',default='Data/dot_mat/Malignancy.mat');p.add_argument('--config',default=str(Path(__file__).resolve().parent/'extended_config.json'))
    p.add_argument('--device',default='cuda');ctx=prepare(p.parse_args())
    actions={'rf':lambda c:__import__('rf_uncertainty').run(c),'tof':tof_sensitivity,'refinement':refinement,
             'support':support_sensitivity,'forward-grid':forward_grid,'reconstruction-grid':reconstruction_grid}
    try:
        phases=list(actions) if ctx.cli.phase=='all' else [ctx.cli.phase]
        for phase in phases:
            if phase=='preflight':continue
            print(f'[STUDY] {phase}',flush=True);actions[phase](ctx)
    finally:ctx.verify_extra();ctx.verify()
    print('[OK] Completed requested pass. Check planned/completed counts; rerun the same command to finish budget-limited studies.',flush=True)


if __name__=='__main__':main()

