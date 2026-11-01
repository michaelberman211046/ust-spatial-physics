"""Fixed-checkpoint evaluation for ultrasound travel-time reconstruction."""
from pathlib import Path
import argparse
import contextlib
import copy
import gc
import importlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'engine'))
os.environ.setdefault('MPLBACKEND','Agg')

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from common import (save_json,read_json,file_identity,image_metrics,describe,summarize_rows,
                    write_csv,timed,hardware,perturb_tof,digest)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['preflight','inference','pilot','studies','measured','summarize','all'],required=True)
    p.add_argument('--root',default='Data/new_pipeline_512_20000a')
    p.add_argument('--output',default='Data/initial_refinement_results')
    p.add_argument('--geometry-mat',default='Data/dot_mat/Malignancy.mat')
    p.add_argument('--config',default=str(HERE/'validation_config.json'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-test',type=int,default=0,help='0 evaluates every saved test index')
    p.add_argument('--batch-size',type=int,default=0)
    p.add_argument('--hash-checkpoints',action='store_true')
    p.add_argument('--measured-setup',default=None)
    p.add_argument('--measured-initial',default=None)
    return p.parse_args()


def paths(a):
    root=Path(a.root);train=root/'training_v28';exp=root/'experiment_run/experimental_geometry_v28'
    return dict(data=train/'ultrasound_data_20000_pairs.pt',splits=train/'ultrasound_data_20000_pairs_splits.pt',
                inverse=train/'reconstruction_model_supervised_roi_tof_best.pth',
                autoencoder=train/'latent_sos_autoencoder_v1.pth',operator=train/'matched_tof_operator_v1.pth',
                latent=train/'adjoint_latent_background_clean.pt',spatial=train/'spatial_physics_reconstructor_clean.pt',
                measured_setup=Path(a.measured_setup) if a.measured_setup else exp/'physical_setup.pt',
                measured_initial=Path(a.measured_initial) if a.measured_initial else exp/'physical_reconstruction.pt')


def load(path):
    from checkpoint_reader import load_checkpoint
    return load_checkpoint(path)


def prepare(a,cfg):
    pp=paths(a)
    for key,p in pp.items():
        if not p.is_file():
            raise FileNotFoundError(f'{key}: {p}')
    if not Path(a.geometry_mat).is_file():
        raise FileNotFoundError(a.geometry_mat)
    from measured_geometry import require_geometry
    meta=load(pp['data'])['metadata']
    ds_size=int(meta.get('N',meta.get('num_samples',load(pp['data']).get('N',0))))
    partition=load(pp['splits'])
    arrays={k:np.asarray(partition[k],dtype=np.int64).reshape(-1) for k in ('train_idx','val_idx','test_idx')}
    for key,indices in arrays.items():
        if len(indices)!=len(np.unique(indices)) or np.any(indices<0) or np.any(indices>=ds_size):
            raise ValueError(f'Invalid partition {key}')
    for x,y in (('train_idx','val_idx'),('train_idx','test_idx'),('val_idx','test_idx')):
        if np.intersect1d(arrays[x],arrays[y]).size:
            raise ValueError(f'{x} overlaps {y}')
    if int(partition['N'])!=ds_size or sum(map(len,arrays.values()))!=ds_size:
        raise ValueError('Saved split and dataset population do not match')
    points,_,_,geometry_hash=require_geometry(int(meta['n_emitters']),int(meta['n_receivers']))
    indices=arrays['test_idx'].copy()
    np.random.default_rng(cfg['seed']).shuffle(indices)
    if a.max_test:
        indices=indices[:a.max_test]
    if len(indices)<cfg['pilot_samples']:
        raise ValueError('Test sample count is smaller than pilot size')
    if not meta.get('chunked'):
        raise ValueError('This evaluator requires the chunked cache to avoid bulk data loading')
    chunkroot=pp['data'].parent/meta['chunk_root_name']
    for chunk in np.unique(indices//int(meta['chunk_size'])):
        folder=chunkroot/f'chunk_{chunk:04d}'
        for field in ('sos','tof','tof_raw_eikonal'):
            f=folder/(field+'.npy')
            if not f.is_file():
                raise FileNotFoundError(f'Incomplete cache: {f}. Full test evaluation requires all selected chunks.')
    protocol=dict(schema=1,configuration=cfg,test_indices=indices.tolist(),pilot_indices=indices[:cfg['pilot_samples']].tolist(),
                  sensitivity_indices=indices[:min(cfg['sensitivity_samples'],len(indices))].tolist(),
                  inputs={key:file_identity(p,a.hash_checkpoints or key in ('data','splits')) for key,p in pp.items()},
                  geometry_sha256=geometry_hash,dataset_metadata=meta,
                  source_sha256={str(p.relative_to(HERE)):digest(p) for p in HERE.rglob('*.py')},
                  selection='seeded permutation of saved test_idx; no outcome-based selection',
                  test_size=len(indices),split_sizes={k:len(v) for k,v in arrays.items()},
                  output_root=str(Path(a.output).resolve()))
    spatial_config=load(pp['spatial']).get('config',{})
    effective_meta=dict(meta)
    for key in ('sos_min','sos_max','sos_water'):
        if key not in spatial_config:
            raise ValueError(f'Spatial checkpoint lacks normalization field {key}')
        effective_meta[key]=spatial_config[key]
    protocol['model_metadata']=effective_meta
    protocol['normalization_note']='Generation bounds and learned-stack normalization bounds are recorded separately; retained checkpoint normalization is preserved.'
    manifest=Path(a.output)/'protocol.json'
    if manifest.exists():
        old=read_json(manifest)
        if old!=json.loads(json.dumps(__import__('common').jsonable(protocol))):
            raise RuntimeError('Protocol or inputs changed. Use a new --output folder; existing results are not mixed.')
    else:
        save_json(manifest,protocol)
    save_json(Path(a.output)/'hardware.json',hardware())
    print(f'[PREFLIGHT] {len(indices)} verified test indices; geometry {geometry_hash}',flush=True)
    return protocol


def get_protocol(a):
    p=Path(a.output)/'protocol.json'
    if not p.is_file():
        raise RuntimeError('Run preflight first')
    result=read_json(p)
    if read_json(a.config)!=result['configuration']:
        raise RuntimeError('Configuration differs from frozen protocol')
    for key,p in paths(a).items():
        stored=result['inputs'][key]
        now=file_identity(p,'sha256' in stored)
        if now!=stored:
            raise RuntimeError(f'Input changed since preflight: {p}')
    for filename,expected in result['source_sha256'].items():
        if digest(HERE/filename)!=expected:
            raise RuntimeError(f'Source changed since preflight: {filename}; use a new result folder')
    return result


class CompactDataset(Dataset):
    def __init__(self,ds,indices):self.ds=ds;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,pos):
        index=int(self.indices[pos]);item=self.ds[index]
        return item[0],item[1],item[2],torch.tensor(index,dtype=torch.long)


def dataset(pp,meta):
    from dataset import PairsCacheDataset
    return PairsCacheDataset(str(pp['data']),sos_min=float(meta.get('sos_min',1380)),
                             sos_max=float(meta.get('sos_max',1660)),return_tof_mask=True)


def runtime_args(a,meta):
    import evaluate_synthetic_reconstruction as ev
    pp=paths(a)
    argv=sys.argv
    try:
        sys.argv=['evaluation','--data_path',str(pp['data']),'--base_model_path',str(pp['inverse']),
                  '--ae_path',str(pp['autoencoder']),'--operator_path',str(pp['operator']),
                  '--background_reconstructor_path',str(pp['latent']),'--model_path',str(pp['spatial']),
                  '--report_npz','unused.npz','--use_tof_mask_channel','--use_circular_mask']
        args=ev.parse_args()
    finally:
        sys.argv=argv
    state=load(pp['spatial'])
    for k,v in state.get('config',{}).items():
        if hasattr(args,k) and not k.endswith('_path') and k not in ('data_path','device','output_dir','use_val_split'):
            setattr(args,k,v)
    for k in ('nx','ny','phys_x','phys_y','radius','n_emitters','n_receivers','sos_water','sos_min','sos_max'):
        if k in meta:setattr(args,k,meta[k])
    args.mask_radius=float(getattr(args,'mask_radius',.1091))
    args.device=a.device
    args.water_norm=(args.sos_water-args.sos_min)/(args.sos_max-args.sos_min)
    return args,state


def extract_architecture(a,stack,spatial,args):
    out=Path(a.output)/'implementation';out.mkdir(exist_ok=True)
    names=('supervised_inverse','sos_autoencoder','forward_operator','latent_correction','fixed_adjoint')
    details={}
    for name,model in list(zip(names,stack))+[('spatial_correction',spatial)]:
        details[name]=dict(parameters=sum(p.numel() for p in model.parameters()),
                           class_name=type(model).__name__,architecture=str(model))
        (out/(name+'.txt')).write_text(str(model),encoding='utf-8')
    for name,key in (('supervised_inverse','inverse'),('sos_autoencoder','autoencoder'),
                     ('forward_operator','operator'),('latent_correction','latent'),('spatial_correction','spatial')):
        ck=load(paths(a)[key])
        details[name]['checkpoint_configuration']={k:v for k,v in ck.items() if k in
                ('config','train_config','args','epoch','best_epoch','best_val_loss','best_score','kind','version','in_ch')}
        del ck
    save_json(out/'architecture_and_configuration.json',details)
    save_json(out/'effective_inference_arguments.json',vars(args))


def load_models(a,protocol,indices):
    from reconstruction_utils import make_circular_support_mask,make_homogeneous_water_tof_norm
    from frozen_stack import load_frozen_reconstruction_stack
    from train_spatial_physics import make_spatial_physics_features,SpatialPhysicsResidualNet
    meta=protocol['model_metadata'];args,checkpoint=runtime_args(a,meta)
    ds=dataset(paths(a),meta);args.tof_norm=ds.tof_norm
    dev=torch.device(a.device)
    if dev.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    loader=DataLoader(CompactDataset(ds,indices),batch_size=1,shuffle=False,num_workers=0)
    mask=make_circular_support_mask(args.nx,args.ny,args.phys_x,args.phys_y,args.mask_radius,device=dev)
    water=make_homogeneous_water_tof_norm(args.nx,args.ny,args.phys_x,args.phys_y,args.radius,
                                         args.n_emitters,args.n_receivers,args.sos_water,args.tof_norm,dev)
    with torch.no_grad():
        stack=load_frozen_reconstruction_stack(args,loader,dev,mask,water)
        first=next(iter(loader))
        features=make_spatial_physics_features(first,stack,dev,mask,water,args)[-2]
    if int(checkpoint['in_ch'])!=features.shape[1]:
        raise RuntimeError(f'Feature count mismatch {features.shape[1]} versus {checkpoint["in_ch"]}')
    model=SpatialPhysicsResidualNet(features.shape[1],args.channels,args.dropout,args.delta_limit).to(dev)
    state=checkpoint.get('best_model_state_dict',checkpoint.get('model_state_dict'))
    model.load_state_dict(state);model.eval()
    for m in (*stack,model):
        for p in m.parameters():p.requires_grad_(False)
    extract_architecture(a,stack,model,args)
    del checkpoint,features
    return ds,args,dev,mask,water,stack,model


def infer(batch,stack,model,dev,mask,water,args):
    from train_spatial_physics import make_spatial_physics_features,support_apply
    with torch.no_grad():
        target,tof,valid,idx,initial,prior,feature,_=make_spatial_physics_features(batch,stack,dev,mask,water,args)
        final=support_apply(prior+model(feature),mask,args.water_norm)
    scale=args.sos_max-args.sos_min
    return {key:value[:,0].detach().cpu().numpy()*scale+args.sos_min for key,value in
            (('target_mps',target),('supervised_mps',initial),('latent_mps',prior),('frozen_mps',final))}


def inference(a,protocol):
    out=Path(a.output);folder=out/'frozen';folder.mkdir(exist_ok=True)
    indices=protocol['test_indices']
    missing=[i for i in indices if not (folder/f'{i:06d}.npz').exists()]
    if not missing:
        print('[INFERENCE] All sample files already present');return
    start=time.perf_counter()
    ds,args,dev,mask,water,stack,model=load_models(a,protocol,missing[:1])
    startup=time.perf_counter()-start
    batch_size=a.batch_size or protocol['configuration']['batch_size']
    loader=DataLoader(CompactDataset(ds,missing),batch_size=batch_size,shuffle=False,num_workers=0)
    for batch in loader:
        outputs,seconds=timed(lambda:infer(batch,stack,model,dev,mask,water,args),dev)
        for j,index in enumerate(batch[-1].tolist()):
            np.savez_compressed(folder/f'{index:06d}.npz',sample_index=index,
                                split='test',inference_seconds=seconds/len(batch[-1]),
                                timing_batch_size=len(batch[-1]),
                                **{k:v[j].astype(np.float32) for k,v in outputs.items()})
        print(f'[INFERENCE] saved test indices {batch[-1].tolist()} ({seconds:.3f}s/batch)',flush=True)
    save_json(out/'inference_run.json',dict(startup_seconds=startup,processed_samples=len(missing),
              note='Warm feature construction and inference include CPU transfer of output, exclude dataset I/O and NPZ writing',
              batch_size=batch_size))


def engine_args(cfg,meta,fine=False):
    d=copy.deepcopy(cfg['fine' if fine else 'coarse'])
    d.update(emitter_stride=cfg['emitter_stride'],receiver_stride=cfg['receiver_stride'],
             sos_min=float(meta.get('sos_min',1380)),sos_max=float(meta.get('sos_max',1660)),
             sos_water=float(meta.get('sos_water',1500)),mask_radius=.1091)
    return SimpleNamespace(**d)


def geometry(meta,shape):
    from measured_geometry import require_geometry
    points,_,_,_=require_geometry(meta['n_emitters'],meta['n_receivers'])
    px=float(meta['phys_x']);py=float(meta['phys_y']);nx,ny=shape
    sensors=points.copy();sensors[:,0]=(nx-1)/2+points[:,0]/(px/(nx-1))
    sensors[:,1]=(ny-1)/2+points[:,1]/(py/(ny-1))
    return sensors,px,py


def synthetic_case(a,protocol,index):
    meta=protocol['model_metadata'];ds=dataset(paths(a),meta)
    raw=ds._sample_arrays(index)
    if 'tof_raw_eikonal' not in raw:
        raise RuntimeError(f'Raw Eikonal ToFs missing for test sample {index}')
    data=dict(np.load(Path(a.output)/'frozen'/f'{index:06d}.npz',allow_pickle=False))
    if int(data['sample_index'])!=index or str(data['split'])!='test':raise ValueError('Provenance mismatch')
    observed=np.asarray(raw['tof_raw_eikonal'],float).copy()
    valid=np.asarray(ds.tof_mask,bool).copy()
    return data,observed,valid


def refine_synthetic(start,target,observed,valid,sensors,cfg,meta,dev,index,support=None):
    import refine_synthetic_eikonal_coarse as coarse
    import refine_synthetic_eikonal_fine as fine
    ca=engine_args(cfg,meta);fa=engine_args(cfg,meta,True)
    if support is not None:
        ca.support_array=support;fa.support_array=support
    px=float(meta['phys_x']);py=float(meta['phys_y'])
    r1,t1=timed(lambda:coarse.refine_sample(start,target,observed,valid,sensors,ca,dev,px,py,index),dev)
    r2,t2=timed(lambda:fine.fine_refine(r1['refined_mps'],start,target,observed,valid,sensors,fa,dev,px,py,index),dev)
    return r2['refined_mps'],dict(coarse_seconds=t1,fine_seconds=t2,seconds=t1+t2,
            coarse_history=r1['history'],fine_history=r2['history'],
            accepted_iterations=r1['accepted_iterations']+r2['accepted_iterations'])


def run_case(a,protocol,index,classical=True):
    folder=Path(a.output)/'refined';folder.mkdir(exist_ok=True)
    f=folder/f'{index:06d}.npz';record=f.with_suffix('.json')
    if record.exists():return read_json(record)
    cfg=protocol['configuration'];meta=protocol['model_metadata'];dev=torch.device(a.device)
    data,obs,valid=synthetic_case(a,protocol,index)
    sensors,px,py=geometry(meta,data['frozen_mps'].shape)
    refined,details=refine_synthetic(data['frozen_mps'],data['target_mps'],obs,valid,sensors,cfg,meta,dev,index)
    payload=dict(refined_mps=refined)
    if classical:
        from classical import invert
        from refine_synthetic_eikonal_coarse import ring_mask
        support=ring_mask(*refined.shape,px,py,.1091)
        result,secs=timed(lambda:invert(obs,valid,sensors,refined.shape,support,px,py,
                    float(meta.get('sos_water',1500)),float(meta.get('sos_min',1380)),float(meta.get('sos_max',1660)),
                    cfg['classical'],engine_args(cfg,meta,True)),dev)
        payload['classical_mps']=result[0];details['classical']=result[1];details['classical_seconds']=secs
    details.update(sample_index=index,split='test')
    np.savez_compressed(f,sample_index=index,**payload)
    save_json(record,details)
    return details


def pilot(a,protocol):
    cfg=protocol['configuration'];rows=[]
    for index in protocol['pilot_indices']:
        rows.append(run_case(a,protocol,index))
    per=[r['seconds']+r['classical_seconds'] for r in rows]
    conservative=float(max(np.quantile(per,.9),1)*1.5)
    count=min(len(protocol['test_indices']),cfg['refinement_max_samples'],
              max(cfg['pilot_samples'],int(cfg['refinement_budget_hours']*3600/conservative)))
    plan=dict(pilot_indices=protocol['pilot_indices'],pilot_pair_seconds=per,
              measured_pair_p90_seconds=float(np.quantile(per,.9)),planning_seconds_per_case=conservative,
              budget_hours=cfg['refinement_budget_hours'],cohort_indices=protocol['test_indices'][:count],
              estimated_clean_cohort_hours=count*conservative/3600,
              scope='Clean synthetic coarse+fine refinement and classical comparator only. Sensitivity and measured studies are additional.',
              policy='Fixed test permutation prefix; 1.5x pilot 90th-percentile cost; maximum count from protocol',
              caveat='Runtime is an estimate; pilot-dependent early stopping may underestimate later costs.')
    save_json(Path(a.output)/'cohort_plan.json',plan)
    print(f'[PILOT] paired processing cost {per}; planned cohort N={count}; estimated clean hours={plan["estimated_clean_cohort_hours"]:.2f}',flush=True)


def support_variant(nominal,condition,px,py):
    from scipy.ndimage import map_coordinates
    nx,ny=nominal.shape
    x,y=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
    scale=condition['radius_scale']
    xx=(x-(nx-1)/2-condition['shift_x_m']/(px/(nx-1)))/scale+(nx-1)/2
    yy=(y-(ny-1)/2)/scale+(ny-1)/2
    output=map_coordinates(nominal,[xx,yy],order=1,mode='constant',cval=0)
    if condition['taper_scale']!=1:
        from scipy.ndimage import distance_transform_edt
        signed=distance_transform_edt(output>.5)-distance_transform_edt(output<=.5)
        width=max(1.,float(np.count_nonzero((nominal>.05)&(nominal<.95)))/max(2*np.pi*np.sqrt((nominal>.5).sum()/np.pi),1))
        output=np.clip(.5+signed/(2*width*condition['taper_scale']),0,1)
    return np.asarray(output,np.float32)


def studies(a,protocol):
    plan=read_json(Path(a.output)/'cohort_plan.json')
    for index in plan['cohort_indices']:
        print(f'[COHORT] sample {index}',flush=True);run_case(a,protocol,index)
    synthetic_sensitivity(a,protocol)


def synthetic_sensitivity(a,protocol):
    from refine_synthetic_eikonal_coarse import ring_mask
    from latent_operator_models import denormalize_tof_seconds,normalize_tof_seconds
    cfg=protocol['configuration'];meta=protocol['model_metadata'];indices=protocol['sensitivity_indices']
    ds,args,dev,mask,water,stack,model=load_models(a,protocol,indices[:1])
    folder=Path(a.output)/'synthetic_sensitivity';folder.mkdir(exist_ok=True)
    for index in indices:
        base,raw_observed,valid=synthetic_case(a,protocol,index)
        sensors,px,py=geometry(meta,base['frozen_mps'].shape)
        target=base['target_mps'];region=ring_mask(*target.shape,px,py,.1091)
        for condition,amplitude in cfg['tof_conditions']:
            label=f'{condition}_{amplitude:g}';dest=folder/f'{index:06d}_tof_{label}.json'
            if dest.exists():continue
            batch=next(iter(DataLoader(CompactDataset(ds,[index]),batch_size=1)))
            physical=denormalize_tof_seconds(batch[1],args.tof_norm).numpy()[0]
            noisy,newmask=perturb_tof(physical,valid,condition,amplitude,cfg['seed']+index)
            batch[1]=normalize_tof_seconds(torch.from_numpy(noisy[None]).float(),args.tof_norm)*torch.from_numpy(newmask[None]).float()
            batch[2]=torch.from_numpy(newmask[None]).float()
            images,secs=timed(lambda:infer(batch,stack,model,dev,mask,water,args),dev)
            np.savez_compressed(dest.with_suffix('.npz'),frozen_mps=images['frozen_mps'][0])
            rows=[dict(study='synthetic_tof',condition=label,method='frozen',sample_index=index,seconds=secs,
                       **image_metrics(images['frozen_mps'][0],target,region))]
            save_json(dest,dict(rows=rows))
        del base
    del stack,model;gc.collect()
    if dev.type=='cuda':torch.cuda.empty_cache()
    for index in indices:
        base,obs,valid=synthetic_case(a,protocol,index);target=base['target_mps']
        sensors,px,py=geometry(meta,target.shape);region=ring_mask(*target.shape,px,py,.1091)
        for kind,amplitude in cfg['tof_conditions']:
            label=f'{kind}_{amplitude:g}';dest=folder/f'{index:06d}_tof_{label}.json'
            saved=read_json(dest)
            if len(saved['rows'])>1:continue
            start=np.load(dest.with_suffix('.npz'))['frozen_mps']
            noisy,newmask=perturb_tof(obs,valid,kind,amplitude,cfg['seed']+index)
            final,record=refine_synthetic(start,target,noisy,newmask,sensors,cfg,meta,dev,index)
            saved['rows'].append(dict(study='synthetic_tof',condition=label,method='final',sample_index=index,
                           seconds=record['seconds'],accepted_iterations=record['accepted_iterations'],**image_metrics(final,target,region)))
            saved['refinement']=record;save_json(dest,saved)
            np.savez_compressed(dest.with_suffix('.npz'),frozen_mps=start,refined_mps=final)
        nominal=region
        for condition in cfg['support_conditions']:
            if condition['taper_scale']!=1:continue
            dest=folder/f'{index:06d}_support_{condition["name"]}.json'
            if dest.exists():continue
            support=support_variant(nominal,condition,px,py)
            final,record=refine_synthetic(base['frozen_mps'],target,obs,valid,sensors,cfg,meta,dev,index,support)
            save_json(dest,dict(rows=[dict(study='synthetic_support',condition=condition['name'],method='final',sample_index=index,
                       seconds=record['seconds'],accepted_iterations=record['accepted_iterations'],**image_metrics(final,target,region))],refinement=record))
            np.savez_compressed(dest.with_suffix('.npz'),refined_mps=final,support=support)


def measured(a,protocol):
    from refine_measured_eikonal_coarse import dense_measurements,image4
    from measured_eikonal_refinement import refine
    from classical import score,invert
    import torch.nn.functional as F
    pp=paths(a);setup=load(pp['measured_setup']);initial=load(pp['measured_initial'])
    start=image4(initial['agreement_weighted_reconstruction_mps'])[0,0].numpy()
    support=F.interpolate(image4(setup['soft_support']),size=start.shape,mode='bilinear',align_corners=False)[0,0].numpy()
    ref=setup.get('reference_diagnostic_normalized');geom=setup['geometry'];pg=setup['physical_geometry']
    meta=dict(protocol['model_metadata']);meta.update(geom);meta['sos_water']=pg['water_sos']
    lo=float(geom['sos_min']);hi=float(geom['sos_max'])
    reference=None if ref is None else F.interpolate(image4(ref),size=start.shape,mode='bilinear',align_corners=False)[0,0].numpy()*(hi-lo)+lo
    obs,valid=dense_measurements(setup);meta['n_emitters'],meta['n_receivers']=obs.shape
    sensors,px,py=geometry(meta,start.shape);cfg=protocol['configuration'];dev=torch.device(a.device)
    folder=Path(a.output)/'measured';folder.mkdir(exist_ok=True)
    region=support>.5
    ca=engine_args(cfg,meta);fa=engine_args(cfg,meta,True)
    for ar in (ca,fa):
        ar.min_accepted_folds=3;ar.max_mean_update_mps=1.5;ar.max_rms_update_mps=8.0
    fa.max_rms_update_mps=4.0
    conditions=[('tof_'+kind+'_'+str(amp),kind,amp,None) for kind,amp in cfg['tof_conditions']]
    conditions += [('support_'+s['name'],'clean',0,s) for s in cfg['support_conditions'] if s['name']!='nominal']
    for label,kind,amp,variant in conditions:
        dest=folder/(label+'.json')
        if dest.exists():continue
        perturbed,vmask=perturb_tof(obs,valid,kind,amp,cfg['seed'])
        s=support if variant is None else support_variant(support,variant,px,py)
        r1,t1=timed(lambda:refine(start,start,perturbed,vmask,sensors,s,ca,dev,px,py,False,'measured_coarse'),dev)
        r2,t2=timed(lambda:refine(r1[0],start,perturbed,vmask,sensors,s,fa,dev,px,py,True,'measured_fine'),dev)
        emit=np.arange(0,obs.shape[0],cfg['emitter_stride']);recv=np.arange(0,obs.shape[1],cfg['receiver_stride'])
        record=dict(study='measured',condition=label,method='final',sample_index=-1,seconds=t1+t2,
                    accepted_iterations=sum(bool(x['accepted']) for x in r1[1]+r2[1]),
                    change_rms_mps=float(np.sqrt(np.mean((r2[0][region]-start[region])**2))),
                    tof_residual=score(r2[0],perturbed,vmask,sensors,emit,recv,px,py),
                    evaluation='FWI agreement; fixed nominal support' if reference is not None else 'ToF consistency and image change')
        if reference is not None:record.update(image_metrics(r2[0],reference,region))
        save_json(dest,dict(rows=[record],coarse_history=r1[1],fine_history=r2[1],coarse_seconds=t1,fine_seconds=t2))
        np.savez_compressed(dest.with_suffix('.npz'),initial_mps=start,refined_mps=r2[0],support=s)
    dest=folder/'classical.json'
    if not dest.exists():
        from refine_synthetic_eikonal_coarse import ring_mask
        classical_support=ring_mask(*start.shape,px,py,.1091)
        result,secs=timed(lambda:invert(obs,valid,sensors,start.shape,classical_support,px,py,meta['sos_water'],lo,hi,
                                cfg['classical'],fa),dev)
        row=dict(study='measured',condition='clean',method='classical',sample_index=-1,seconds=secs)
        if reference is not None:row.update(image_metrics(result[0],reference,region))
        save_json(dest,dict(rows=[row],history=result[1]));np.savez_compressed(dest.with_suffix('.npz'),classical_mps=result[0])
    save_json(folder/'scope.json',dict(perturbation_location='after timing alignment and initial physical reconstruction; entering image is fixed',
               scope='isolates post-inference Eikonal correction sensitivity; does not measure end-to-end RF picker or frozen measured-inference sensitivity',
               evaluation_support='fixed nominal S>0.5 across all conditions',FWI_used_in_selection=False))


def summarize(a,protocol):
    from refine_synthetic_eikonal_coarse import ring_mask
    out=Path(a.output);meta=protocol['model_metadata'];rows=[];paired=[]
    region=ring_mask(int(meta['nx']),int(meta['ny']),float(meta['phys_x']),float(meta['phys_y']),.1091)
    for index in protocol['test_indices']:
        path=out/'frozen'/f'{index:06d}.npz'
        if not path.exists():continue
        data=dict(np.load(path));target=data['target_mps'];frozen_metrics=None
        for name,key in (('supervised','supervised_mps'),('latent','latent_mps'),('frozen','frozen_mps')):
            metrics=image_metrics(data[key],target,region)
            rows.append(dict(study='synthetic_full_test',condition='clean',method=name,sample_index=index,**metrics))
            if name=='frozen':
                frozen_metrics=metrics;rows[-1]['seconds']=float(data['inference_seconds'])
        refined_path=out/'refined'/f'{index:06d}.npz'
        if refined_path.exists():
            images=dict(np.load(refined_path));record=read_json(refined_path.with_suffix('.json'))
            for name,key,seconds in (('frozen','frozen_mps',float(data['inference_seconds'])),
                                     ('final','refined_mps',record['seconds']),('classical','classical_mps',record['classical_seconds'])):
                im=data[key] if name=='frozen' else images[key]
                metrics=image_metrics(im,target,region)
                rows.append(dict(study='synthetic_refinement_cohort',condition='clean',method=name,sample_index=index,seconds=seconds,**metrics))
                if name=='final':
                    rows[-1]['accepted_iterations']=record['accepted_iterations']
                    delta=dict(sample_index=index,rmse_reduction_mps=frozen_metrics['rmse_mps']-metrics['rmse_mps'],
                               rmse_reduction_percent=100*(frozen_metrics['rmse_mps']-metrics['rmse_mps'])/frozen_metrics['rmse_mps'],
                               mad_reduction_mps=frozen_metrics['mad_mps']-metrics['mad_mps'],
                               correlation_increase=None if metrics['correlation'] is None or frozen_metrics['correlation'] is None
                               else metrics['correlation']-frozen_metrics['correlation'],accepted_iterations=record['accepted_iterations'])
                    paired.append(delta)
    for folder in ('synthetic_sensitivity','measured'):
        for p in (out/folder).glob('*.json'):
            rows.extend(read_json(p).get('rows',[]))
    write_csv(out/'per_sample_metrics.csv',rows);write_csv(out/'paired_refinement_changes.csv',paired)
    save_json(out/'aggregate_metrics.json',summarize_rows(rows))
    summary={key:describe([r.get(key) for r in paired]) for key in
             ('rmse_reduction_mps','rmse_reduction_percent','mad_reduction_mps','correlation_increase')}
    summary.update(n=len(paired),rmse_improved=sum(r['rmse_reduction_mps']>0 for r in paired),
                   rmse_worsened=sum(r['rmse_reduction_mps']<0 for r in paired),
                   unchanged=sum(r['rmse_reduction_mps']==0 for r in paired),
                   accepted=sum(r['accepted_iterations']>0 for r in paired),
                   completeness=dict(frozen_files=len(list((out/'frozen').glob('*.npz'))),planned_test=len(protocol['test_indices'])))
    save_json(out/'paired_summary.json',summary)
    print(f'[SUMMARY] {len(rows)} metric rows; {len(paired)} refined cases; {out}',flush=True)


def main():
    a=arguments();Path(a.output).mkdir(parents=True,exist_ok=True)
    from settings import set_output_folder
    set_output_folder(str(Path(a.output)/'logs'))
    os.environ['UST_GEOMETRY_MAT']=str(Path(a.geometry_mat).resolve())
    cfg=read_json(a.config)
    if a.phase=='all':
        for phase in ('preflight','inference','pilot','studies','measured','summarize'):
            argv=sys.argv[1:];argv[argv.index('--phase')+1]=phase
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*argv],check=True)
        return
    if a.phase=='preflight':prepare(a,cfg);return
    protocol=get_protocol(a)
    {'inference':inference,'pilot':pilot,'studies':studies,'measured':measured,'summarize':summarize}[a.phase](a,protocol)


if __name__=='__main__':main()
