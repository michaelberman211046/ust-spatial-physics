"""Fixed-checkpoint evaluation for ultrasound travel-time reconstruction."""
from pathlib import Path
import argparse
import copy
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
                    write_csv,timed,hardware,digest)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['preflight','inference','comparator','summarize','all'],required=True)
    p.add_argument('--root',default='Data/new_pipeline_512_20000a')
    p.add_argument('--output',default='Data/aggregate_analysis_results')
    p.add_argument('--geometry-mat',default='Data/dot_mat/Malignancy.mat')
    p.add_argument('--config',default=str(HERE/'validation_config.json'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-test',type=int,default=0,help='0 evaluates every saved test index')
    p.add_argument('--batch-size',type=int,default=0)
    p.add_argument('--hash-checkpoints',action='store_true')
    return p.parse_args()


def paths(a):
    root=Path(a.root);train=root/'training_v28'
    return dict(data=train/'ultrasound_data_20000_pairs.pt',splits=train/'ultrasound_data_20000_pairs_splits.pt',
                inverse=train/'reconstruction_model_supervised_roi_tof_best.pth',
                autoencoder=train/'latent_sos_autoencoder_v1.pth',operator=train/'matched_tof_operator_v1.pth',
                latent=train/'adjoint_latent_background_clean.pt',spatial=train/'spatial_physics_reconstructor_clean.pt')


def load(path):
    from checkpoint_reader import load_checkpoint
    return load_checkpoint(path)


def prepare(a,cfg):
    pp=paths(a)
    for key,p in pp.items():
        if key.startswith('measured_'):continue
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
    if not len(indices):
        raise ValueError('No test indices selected')
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
                  inputs={key:file_identity(p,a.hash_checkpoints or key in ('data','splits')) for key,p in pp.items() if not key.startswith('measured_')},
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
    protocol['classical_cohort_indices']=indices[:min(len(indices),int(cfg['classical_cohort_samples']))].tolist()
    protocol['available_test_size']=len(arrays['test_idx'])
    protocol['evaluation_scope']='Complete saved test split' if len(indices)==len(arrays['test_idx']) else 'Technical subset of saved test split'
    protocol['input_representations']={'learned':'processed synthetic tof.npy as expected by retained checkpoints',
                                       'classical':'raw synthetic tof_raw_eikonal.npy; physical Eikonal inversion'}
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
        if key.startswith('measured_'):continue
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


















def comparator(a,protocol):
    """Evaluate a prespecified test-index prefix with no learned initialization."""
    from classical import invert
    from refine_synthetic_eikonal_coarse import ring_mask
    folder=Path(a.output)/'classical';folder.mkdir(exist_ok=True)
    cfg=protocol['configuration'];meta=protocol['model_metadata'];device=torch.device('cpu')
    indices=protocol['classical_cohort_indices'];elapsed=[]
    save_json(Path(a.output)/'classical_cohort.json',dict(indices=indices,
        selection='First N entries of the fixed seeded test permutation, declared before inference',
        samples=len(indices),refinement_enabled=False,
        parameter_selection='Fixed prespecified settings; iterate selected using held-out angular channels, never reference images'))
    for position,index in enumerate(indices):
        dest=folder/f'{index:06d}.npz';record=dest.with_suffix('.json')
        if dest.exists() and record.exists():
            with np.load(dest,allow_pickle=False) as saved:
                if int(saved['sample_index'])!=index or not np.isfinite(saved['classical_mps']).all():
                    raise RuntimeError(f'Invalid completed comparator file: {dest}')
            print(f'[CLASSICAL {position+1}/{len(indices)}] sample={index}: already complete',flush=True)
            continue
        data,observed,valid=synthetic_case(a,protocol,index)
        shape=data['frozen_mps'].shape;sensors,px,py=geometry(meta,shape)
        support=ring_mask(*shape,px,py,.1091)
        print(f'[CLASSICAL {position+1}/{len(indices)}] sample={index}: starting',flush=True)
        result,seconds=timed(lambda:invert(observed,valid,sensors,shape,support,px,py,
            float(meta.get('sos_water',1500)),float(meta['sos_min']),float(meta['sos_max']),
            cfg['classical'],engine_args(cfg,meta,True)),device)
        temp=dest.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,sample_index=index,split='test',classical_mps=result[0])
        os.replace(temp,dest)
        save_json(record,dict(sample_index=index,seconds=seconds,history=result[1],
                              learned_image_used=False,reference_used_in_optimization=False))
        elapsed.append(seconds)
        eta=np.mean(elapsed)*(len(indices)-position-1)/60
        print(f'[CLASSICAL {position+1}/{len(indices)}] {seconds:.1f}s; estimated remaining {eta:.1f} min',flush=True)


def summarize(a,protocol):
    from refine_synthetic_eikonal_coarse import ring_mask
    out=Path(a.output);meta=protocol['model_metadata'];rows=[];pairs=[]
    region=ring_mask(int(meta['nx']),int(meta['ny']),float(meta['phys_x']),float(meta['phys_y']),.1091)
    available=[i for i in protocol['test_indices'] if (out/'frozen'/f'{i:06d}.npz').is_file()]
    full=(len(available)==protocol['available_test_size'] and len(protocol['test_indices'])==protocol['available_test_size'])
    study='synthetic_full_test' if full else 'synthetic_test_subset'
    completed=0
    for index in available:
        with np.load(out/'frozen'/f'{index:06d}.npz',allow_pickle=False) as f:
            data={k:f[k] for k in f.files}
        if int(data['sample_index'])!=index or str(data['split'])!='test':
            raise RuntimeError(f'Invalid frozen provenance for {index}')
        target=data['target_mps'];metrics_by_name={}
        for name,key in (('supervised','supervised_mps'),('latent','latent_mps'),('frozen','frozen_mps')):
            metrics_by_name[name]=image_metrics(data[key],target,region)
            row=dict(study=study,condition='clean',method=name,sample_index=index,**metrics_by_name[name])
            if name=='frozen':row['seconds']=float(data['inference_seconds'])
            rows.append(row)
        p=out/'classical'/f'{index:06d}.npz'
        if p.exists() and p.with_suffix('.json').exists():
            record=read_json(p.with_suffix('.json'))
            with np.load(p,allow_pickle=False) as f:
                if int(f['sample_index'])!=index:raise RuntimeError('Classical provenance mismatch')
                classical_metrics=image_metrics(f['classical_mps'],target,region)
            completed+=1
            for name in ('supervised','latent','frozen','classical'):
                m=classical_metrics if name=='classical' else metrics_by_name[name]
                row=dict(study='synthetic_classical_matched_cohort',condition='clean',method=name,sample_index=index,**m)
                if name=='classical':row['seconds']=record['seconds']
                if name=='frozen':row['seconds']=float(data['inference_seconds'])
                rows.append(row)
            paired=dict(sample_index=index)
            for metric in ('rmse_mps','nrmse_percent','mad_mps','correlation'):
                f=metrics_by_name['frozen'][metric];c=classical_metrics[metric]
                paired[metric+'_classical_minus_frozen']=None if c is None or f is None else c-f
            pairs.append(paired)
    summary=dict(scope=protocol['evaluation_scope'],full_test_complete=full,
                 saved_test_population=protocol['available_test_size'],planned_inference=len(protocol['test_indices']),
                 completed_inference=len(available),planned_classical=len(protocol['classical_cohort_indices']),
                 completed_classical=completed,refinement_run=False,
                 comparison='Methods in the classical matched cohort use the same samples and fixed evaluation support.',
                 metric_convention='NRMSE is percent of per-sample reference range; correlation is Pearson r.',
                 timing='Warm stack includes feature construction and output transfer; excludes startup, dataset I/O and file writing.',
                 paired_comparison={k:describe([p[k] for p in pairs]) for k in pairs[0] if k!='sample_index'} if pairs else {})
    save_json(out/'aggregate_metrics.json',summarize_rows(rows))
    save_json(out/'evaluation_summary.json',summary)
    write_csv(out/'per_sample_metrics.csv',rows);write_csv(out/'paired_classical_vs_frozen.csv',pairs)
    print(f'[SUMMARY] inference {len(available)}/{protocol["available_test_size"]}; classical {completed}/{len(protocol["classical_cohort_indices"])}; full_test_complete={full}',flush=True)


def main():
    a=arguments();Path(a.output).mkdir(parents=True,exist_ok=True)
    from settings import set_output_folder
    set_output_folder(str(Path(a.output)/'logs'))
    os.environ['UST_GEOMETRY_MAT']=str(Path(a.geometry_mat).resolve())
    cfg=read_json(a.config)
    if a.phase=='all':
        for phase in ('preflight','inference','comparator','summarize'):
            argv=sys.argv[1:];argv[argv.index('--phase')+1]=phase
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*argv],check=True)
        return
    if a.phase=='preflight':prepare(a,cfg);return
    protocol=get_protocol(a)
    {'inference':inference,'comparator':comparator,'summarize':summarize}[a.phase](a,protocol)


if __name__=='__main__':main()
