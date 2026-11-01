#!/usr/bin/env python3
"""Matched-budget spatial feature-removal retraining; existing inputs are read-only."""
from pathlib import Path
import argparse, atexit, copy, csv, gc, hashlib, json, math, os, random, sys, time
from types import SimpleNamespace
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
METRICS = ('rmse_mps', 'nrmse_percent', 'mad_mps', 'correlation')
VARIANTS = ('full', 'without_residual_maps', 'without_directional_maps')


def within(path, parent):
    return path.resolve().is_relative_to(parent.resolve())


def install_write_guard(output):
    """Constrain Python-level filesystem mutations; inputs are also hashed."""
    output = output.resolve()
    def check(path):
        if isinstance(path, int) or path is None: return
        if os.fsdecode(path).upper() in ('NUL', '/DEV/NULL'): return
        if not within(Path(os.fsdecode(path)), output):
            raise PermissionError(f'Ablation write outside new output directory: {path}')
    def hook(event, args):
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                check(path)
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
            check(args[0])
        elif event in ('os.rename', 'os.replace'):
            check(args[0]); check(args[1])
        elif event in ('os.symlink', 'os.link', 'subprocess.Popen'):
            raise PermissionError(f'{event} is not used during ablation phases')
    sys.addaudithook(hook)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    if hasattr(x, 'tolist'): return jsonable(x.tolist())
    if hasattr(x, 'item'): return jsonable(x.item())
    return x


def save_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def feature_layout(tof_channels, emitter_sectors, receiver_sectors, sector_highpass):
    # [prior, highpass(prior), base condition, residual condition, sectors].
    base_end = 2 + int(tof_channels) + 7
    residual = list(range(base_end, base_end+5)) + [base_end+6, base_end+7]
    sector_start = base_end+8
    count = (int(emitter_sectors)+int(receiver_sectors))*(2 if sector_highpass else 1)
    directional = list(range(sector_start, sector_start+count)); n = sector_start+count
    removal = {'full':[], 'without_residual_maps':residual+directional,
               'without_directional_maps':directional}
    return {'channels':n, 'direct_residual_channels':residual,
            'directional_channels':directional, 'keep':{
                key:[i for i in range(n) if i not in set(value)] for key,value in removal.items()},
            'retained':'ToF representation (including water referencing), images, support and validity',
            'scope':'Explicit learned-operator residual/adjoint spatial inputs only; upstream physics remains'}


def initial_states(model_class, layout, channels, dropout, limit, seed, variants):
    """Fresh initialization with shared weights copied to the reduced models."""
    torch.manual_seed(seed)
    full = model_class(layout['channels'], channels, dropout, limit)
    state = {k:v.detach().clone() for k,v in full.state_dict().items()}; result = {}
    for variant in variants:
        reduced = {k:v.clone() for k,v in state.items()}
        reduced['stem.0.weight'] = state['stem.0.weight'][:,layout['keep'][variant]].clone()
        result[variant] = reduced
    return result


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['preflight','benchmark','cache','train','evaluate','summarize','all'], required=True)
    p.add_argument('--root', default='Data/new_pipeline_512_20000a')
    p.add_argument('--output', required=True)
    p.add_argument('--geometry-mat', default='Data/dot_mat/Malignancy.mat')
    p.add_argument('--config', default=str(HERE/'ablation_config.json'))
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def prepare(a):
    global torch, np, spatial, utils
    a.root = Path(a.root).resolve(); a.output = Path(a.output).resolve()
    a.geometry_mat = Path(a.geometry_mat).resolve(); a.config = Path(a.config).resolve()
    if within(a.output,a.root) or within(a.root,a.output) or within(a.output,HERE) or within(HERE,a.output):
        raise ValueError('Output must be separate from the pipeline root and this code package')
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    for key in ('train_samples','validation_samples','epochs','batch_size','benchmark_steps','progress_every','bootstrap_replicates'):
        if int(cfg[key]) <= 0: raise ValueError(f'{key} must be positive')
    if int(cfg['test_samples'])<0:raise ValueError('test_samples must be nonnegative; zero means all')
    if cfg['cache_dtype'] != 'float32': raise ValueError('Cache features must remain float32')
    if 'full' not in cfg['variants'] or len(cfg['variants'])<2 or len(set(cfg['variants']))!=len(cfg['variants']):
        raise ValueError('Select full plus at least one distinct feature-removal variant')
    if not set(cfg['variants']).issubset(VARIANTS): raise ValueError('Unknown variant')
    if not cfg['training_seeds'] or len(set(cfg['training_seeds']))!=len(cfg['training_seeds']):
        raise ValueError('Training seeds must be nonempty and distinct')
    train = a.root/'training_v28'
    paths = dict(data=train/'ultrasound_data_20000_pairs.pt',
        splits=train/'ultrasound_data_20000_pairs_splits.pt',
        inverse=train/'reconstruction_model_supervised_roi_tof_best.pth',
        autoencoder=train/'latent_sos_autoencoder_v1.pth',
        operator=train/'matched_tof_operator_v1.pth',
        latent=train/'adjoint_latent_background_clean.pt',
        spatial=train/'spatial_physics_reconstructor_clean.pt', geometry=a.geometry_mat)
    for name,path in paths.items():
        if not path.is_file(): raise FileNotFoundError(f'{name}: {path}')
        if within(path,a.output): raise ValueError('Output encloses an existing input')
    marker = a.output/'ablation_protocol.json'
    ownerfile=a.output/'ablation_owner.json'
    owner={'kind':'separate_spatial_conditioning_ablation','input_root':str(a.root),'package':str(HERE)}
    if a.output.exists():
        if not ownerfile.is_file() or json.loads(ownerfile.read_text(encoding='utf-8'))!=owner:
            raise RuntimeError('Existing output is not this ablation directory. Choose a new --output.')
    a.output.mkdir(parents=True, exist_ok=True)
    if not ownerfile.exists():save_json(ownerfile,owner)
    runtime = a.output/'runtime'; runtime.mkdir(exist_ok=True)
    for name in ['tmp','mpl','torch','cuda']: (runtime/name).mkdir(exist_ok=True)
    os.environ.update(TEMP=str(runtime/'tmp'), TMP=str(runtime/'tmp'), TMPDIR=str(runtime/'tmp'),
        MPLCONFIGDIR=str(runtime/'mpl'), TORCH_HOME=str(runtime/'torch'),
        CUDA_CACHE_PATH=str(runtime/'cuda'), XDG_CACHE_HOME=str(runtime),
        UST_GEOMETRY_MAT=str(a.geometry_mat), MPLBACKEND='Agg')
    import tempfile
    tempfile.tempdir = str(runtime/'tmp'); os.chdir(a.output)
    install_write_guard(a.output); sys.path.insert(0,str(HERE/'engine'))
    import torch as _torch, numpy as _np
    torch,np = _torch,_np
    import settings
    settings.set_output_folder(str(a.output/'logs'))
    from checkpoint_reader import load_checkpoint
    import train_spatial_physics as _spatial
    import report_utils as _utils
    spatial,utils = _spatial,_utils
    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no automatic CPU run')
    device = torch.device(a.device); torch.backends.cudnn.benchmark = False
    source_hashes = {str(p.relative_to(HERE)):digest(p) for p in HERE.rglob('*.py') if '__pycache__' not in p.parts}
    print('[INPUT CHECK] Hashing original checkpoints and split/index files...', flush=True)
    identities = {k:{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
                    'sha256':digest(p) if k!='geometry' else None} for k,p in paths.items()}
    checked = {'done':False};locals_context={}
    def final_integrity():
        if checked['done']: return
        result = {}
        for key,path in paths.items():
            before = identities[key]
            unchanged = path.stat().st_size==before['bytes'] and path.stat().st_mtime_ns==before['mtime_ns']
            if before['sha256'] is not None: unchanged &= digest(path)==before['sha256']
            result[key] = bool(unchanged)
        save_json(a.output/f'input_integrity_{a.phase}.json',result); checked['done']=True
        if 'chunk_files' in locals_context:
            for path,identity in locals_context['chunk_files'].items():
                stat=Path(path).stat()
                if stat.st_size!=identity['bytes'] or stat.st_mtime_ns!=identity['mtime_ns']:
                    raise RuntimeError(f'Dataset chunk changed: {path}')
        if not all(result.values()): raise RuntimeError('An existing input changed during the phase')
    atexit.register(final_integrity)
    dataset_index = load_checkpoint(paths['data'])
    meta = dict(dataset_index['metadata']); n=int(meta.get('N',dataset_index.get('N',0)))
    if not meta.get('chunked'): raise ValueError('The saved chunked dataset index is required')
    split = load_checkpoint(paths['splits'])
    indices = {k:np.asarray(split[k],dtype=np.int64).reshape(-1) for k in ['train_idx','val_idx','test_idx']}
    for key,ids in indices.items():
        if len(ids)!=len(np.unique(ids)) or np.any(ids<0) or np.any(ids>=n): raise ValueError(f'Invalid {key}')
    for left,right in [('train_idx','val_idx'),('train_idx','test_idx'),('val_idx','test_idx')]:
        if np.intersect1d(indices[left],indices[right]).size: raise ValueError('Saved splits overlap')
    if sum(map(len,indices.values()))!=n or int(split['N'])!=n: raise ValueError('Split population mismatch')
    rng=np.random.default_rng(int(cfg['selection_seed'])); selected={}
    for role,key,count in [('train','train_idx',cfg['train_samples']),('validation','val_idx',cfg['validation_samples']),('test','test_idx',cfg['test_samples'])]:
        ids=indices[key].copy()
        if role!='test' or int(count)>0: rng.shuffle(ids)
        if int(count)>len(ids): raise ValueError(f'Too many {role} samples')
        selected[role]=ids[:int(count)].tolist() if int(count)>0 else ids.tolist()
    chunk_root=paths['data'].parent/str(meta['chunk_root_name'])
    chunk_files={}
    all_ids=selected['train']+selected['validation']+selected['test']
    for chunk in sorted(set(int(i)//int(meta['chunk_size']) for i in all_ids)):
        for field in ['sos','tof']:
            path=chunk_root/f'chunk_{chunk:04d}'/(field+'.npy')
            if not path.is_file():raise FileNotFoundError(f'Missing selected dataset chunk: {path}')
            chunk_files[str(path)]={'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
    locals_context['chunk_files']=chunk_files
    from measured_geometry import require_geometry
    _,_,_,geometry_hash=require_geometry(meta['n_emitters'],meta['n_receivers'])
    ck=load_checkpoint(paths['spatial'])
    if ck.get('kind')!='spatial_physics_reconstructor': raise ValueError('Unexpected spatial checkpoint kind')
    old_argv=sys.argv
    try:
        sys.argv=['ablation','--data_path',str(paths['data']),'--base_model_path',str(paths['inverse']),
            '--ae_path',str(paths['autoencoder']),'--operator_path',str(paths['operator']),
            '--background_reconstructor_path',str(paths['latent']),'--model_path',str(paths['spatial'])]
        args=spatial.parse_args()
    finally: sys.argv=old_argv
    for k,v in ck.get('config',{}).items():
        if hasattr(args,k) and not k.endswith('_path') and k not in ['output_dir','device','resume']: setattr(args,k,v)
    for k in ['nx','ny','n_emitters','n_receivers','phys_x','phys_y','radius','sos_water']:
        if k in meta: setattr(args,k,meta[k])
    for k in ['sos_min','sos_max']:
        if k not in ck['config']: raise ValueError(f'Spatial checkpoint lacks {k}')
    args.output_dir=str(a.output/'logs');args.device=str(device);args.resume=False
    args.model_path=str(a.output/'models'/'unused_pipeline_output.pt')
    args.splits_path=str(paths['splits'])
    args.epochs=int(cfg['epochs']);args.batch_size=int(cfg['batch_size'])
    args.max_train_samples=int(cfg['train_samples']);args.max_val_samples=int(cfg['validation_samples'])
    args.lambda_view_consistency=0.;args.lambda_stripe_sensitivity=0.
    args.water_norm=(args.sos_water-args.sos_min)/(args.sos_max-args.sos_min)
    if not args.use_circular_mask: raise ValueError('A fixed circular evaluation region is required')
    if cfg.get('tof_loss_weight') is not None: args.lambda_tof=float(cfg['tof_loss_weight'])
    if args.lambda_tof<0: raise ValueError('ToF loss weight must be nonnegative')
    tof_ch=4 if args.use_tof_mask_channel and args.tof_feature_mode=='residual_stack' else (2 if args.use_tof_mask_channel else 1)
    layout=feature_layout(tof_ch,args.num_emitter_sectors,args.num_receiver_sectors,args.sector_highpass)
    if layout['channels']!=int(ck['in_ch']): raise ValueError('Checkpoint feature layout mismatch')
    for variant in cfg['variants']:
        if variant!='full' and len(layout['keep'][variant])==layout['channels']: raise ValueError(f'{variant} removes no inputs')
    protocol={'schema':1,'configuration':cfg,'inputs':identities,'source_sha256':source_hashes,
        'geometry_sha256':geometry_hash,'indices':selected,'dataset_metadata':meta,'chunk_files':chunk_files,
        'feature_layout':layout,'effective_arguments':vars(args),
        'design':cfg.get('protocol_design','Frozen upstream; fresh spatial retraining; clean cached inputs; common image/ToF losses; validation-MSE selection; no post-inference refinement'),
        'augmentation_note':'Focused clean-input comparison, not the original nuisance/consistency augmentation schedule'}
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8'))!=jsonable(protocol):
            raise RuntimeError('Configuration, inputs or code changed. Use a NEW output directory.')
    else: save_json(marker,protocol)
    save_json(a.output/f'hardware_{a.phase}.json',utils.hardware())
    print(f'[PROTOCOL] train={len(selected["train"])} validation={len(selected["validation"])} test={len(selected["test"])}; {layout["channels"]} channels',flush=True)
    return SimpleNamespace(cli=a,cfg=cfg,paths=paths,meta=meta,args=args,device=device,layout=layout,
        indices=selected,original_checkpoint=ck,verify=final_integrity)


class Indexed:
    def __init__(self,ds,indices): self.ds,self.indices=ds,indices
    def __len__(self): return len(self.indices)
    def __getitem__(self,pos):
        i=int(self.indices[pos]);sample=self.ds[i]
        return sample[0],sample[1],sample[2],torch.tensor(i,dtype=torch.long)


def load_upstream(ctx, ids):
    from dataset import PairsCacheDataset
    from torch.utils.data import DataLoader
    from reconstruction_utils import make_circular_support_mask,make_homogeneous_water_tof_norm
    from frozen_stack import load_frozen_reconstruction_stack
    args=ctx.args;dev=ctx.device
    ds=PairsCacheDataset(str(ctx.paths['data']),sos_min=args.sos_min,sos_max=args.sos_max,return_tof_mask=True)
    args.tof_norm=ds.tof_norm
    mask=make_circular_support_mask(args.nx,args.ny,args.phys_x,args.phys_y,args.mask_radius,device=dev)
    water=make_homogeneous_water_tof_norm(args.nx,args.ny,args.phys_x,args.phys_y,args.radius,
        args.n_emitters,args.n_receivers,args.sos_water,args.tof_norm,dev) if args.tof_feature_mode=='residual_stack' else None
    loader=DataLoader(Indexed(ds,ids),batch_size=1,shuffle=False,num_workers=0)
    with torch.no_grad(): stack=load_frozen_reconstruction_stack(args,loader,dev,mask,water)
    for module in stack:
        module.eval()
        for p in module.parameters(): p.requires_grad_(False)
    return ds,loader,mask,water,stack


def features(ctx,batch,stack,mask,water):
    with torch.no_grad():
        target,tof,valid,idx,initial,prior,feat,_=spatial.make_spatial_physics_features(
            batch,stack,ctx.device,mask,water,ctx.args)
    if feat.shape[1]!=ctx.layout['channels']: raise RuntimeError('Feature schema changed')
    return {'target':target,'tof':tof,'valid':valid,'indices':idx,'prior':prior,'features':feat,'initial':initial}


def prediction_loss(ctx,model,batch,mask,op):
    from latent_operator_models import masked_l1,masked_mse,highpass,support_apply,tv_l1
    args=ctx.args;delta=model(batch['features']);prior=batch['prior'];target=batch['target']
    pred=support_apply(prior+delta,mask,args.water_norm);td=target-prior
    m=mask
    while m.ndim<4:m=m.unsqueeze(0)
    m=m.expand(len(target),1,*mask.shape[-2:])
    loss=(float(args.lambda_res_l1)*masked_l1(delta,td,mask)
        +float(args.lambda_res_detail)*masked_l1(highpass(delta,args.detail_kernel),highpass(td,args.detail_kernel),mask)
        +float(args.lambda_final_mse)*masked_mse(pred,target,mask)
        +float(args.lambda_final_l1)*masked_l1(pred,target,mask)
        +float(args.lambda_tv)*tv_l1(delta,m))
    if float(args.lambda_tof)>0:
        loss=loss+float(args.lambda_tof)*masked_mse(op(pred),batch['tof'],batch['valid'])
    return loss,pred


def cache_phase(ctx):
    pending=[]
    for role in ['train','validation']:
        folder=ctx.cli.output/'cache'/role;folder.mkdir(parents=True,exist_ok=True)
        pending.extend((role,i) for i in ctx.indices[role] if not (folder/f'{i:06d}.npz').is_file())
    if not pending: print('[CACHE] All train/validation features saved',flush=True);return
    ds,loader,mask,water,stack=load_upstream(ctx,[i for _,i in pending]);start=time.perf_counter()
    import shutil
    for number,((role,index),batch) in enumerate(zip(pending,loader),1):
        item=features(ctx,batch,stack,mask,water)
        arrays={k:v[0].detach().cpu().numpy().astype(np.float32) for k,v in item.items() if k not in ['indices','initial']}
        if number==1:
            required=sum(x.nbytes for x in arrays.values())*len(pending)*1.15;free=shutil.disk_usage(ctx.cli.output).free
            if required>free: raise OSError(f'Cache needs ~{required/2**30:.1f} GiB; free {free/2**30:.1f} GiB')
            save_json(ctx.cli.output/'cache_storage_estimate.json',{'pending_samples':len(pending),'estimated_bytes':required})
        dest=ctx.cli.output/'cache'/role/f'{index:06d}.npz';temp=dest.with_suffix('.tmp')
        with temp.open('wb') as stream:np.savez(stream,**arrays,sample_index=np.int64(index),split=np.asarray(role))
        os.replace(temp,dest)
        if number==1 or number%ctx.cfg['progress_every']==0 or number==len(pending):
            elapsed=time.perf_counter()-start;eta=elapsed/number*(len(pending)-number)
            print(f'[CACHE] {number}/{len(pending)} {elapsed/60:.1f} min elapsed; ~{eta/60:.1f} min remaining',flush=True)
    del stack,ds,loader;gc.collect()
    if ctx.device.type=='cuda':torch.cuda.empty_cache()


class Cached:
    def __init__(self,ctx,role,indices,keep): self.folder=ctx.cli.output/'cache'/role;self.role=role;self.indices=indices;self.keep=keep
    def __len__(self):return len(self.indices)
    def __getitem__(self,pos):
        i=int(self.indices[pos]);path=self.folder/f'{i:06d}.npz'
        with np.load(path,allow_pickle=False) as d:
            if int(d['sample_index'])!=i or str(d['split'])!=self.role:raise RuntimeError('Cache index/split mismatch')
            out={k:torch.from_numpy(d[k].copy()) for k in ['target','prior','tof','valid']}
            out['features']=torch.from_numpy(d['features'][self.keep].copy())
        return out


def train_support(ctx):
    from reconstruction_utils import make_circular_support_mask
    from latent_operator_models import load_matched_operator_checkpoint
    a=ctx.args;mask=make_circular_support_mask(a.nx,a.ny,a.phys_x,a.phys_y,a.mask_radius,device=ctx.device);op=None
    if float(a.lambda_tof)>0:
        op,_,ck=load_matched_operator_checkpoint(str(ctx.paths['operator']),ctx.device)
        if 'best_operator_state_dict' in ck:op.load_state_dict(ck['best_operator_state_dict'])
        del ck;op.eval()
        for p in op.parameters():p.requires_grad_(False)
    return mask,op


def batch_to(batch,device):return {k:v.to(device=device,dtype=torch.float32) for k,v in batch.items()}


def benchmark(ctx):
    dest=ctx.cli.output/'benchmark.json'
    if dest.exists():print('[BENCHMARK] Saved timing estimate exists',flush=True);return
    ds,loader,mask,water,stack=load_upstream(ctx,ctx.indices['train'][:1]);batch=next(iter(loader))
    sync(ctx.device);start=time.perf_counter();item=features(ctx,batch,stack,mask,water);sync(ctx.device)
    feature_seconds=time.perf_counter()-start;op=stack[2]
    states=initial_states(spatial.SpatialPhysicsResidualNet,ctx.layout,ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit,int(ctx.cfg['training_seeds'][0]),ctx.cfg['variants'])
    estimates={}
    for variant in ctx.cfg['variants']:
        model=spatial.SpatialPhysicsResidualNet(len(ctx.layout['keep'][variant]),ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit).to(ctx.device)
        model.load_state_dict(states[variant]);model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=ctx.args.lr,weight_decay=ctx.args.weight_decay)
        mini={k:v.repeat(ctx.cfg['batch_size'],*([1]*(v.ndim-1))) for k,v in item.items() if k in ['target','prior','tof','valid','features']}
        mini['features']=mini['features'][:,ctx.layout['keep'][variant]];measurements=[]
        for step in range(ctx.cfg['benchmark_steps']+2):
            sync(ctx.device);start=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss,_=prediction_loss(ctx,model,mini,mask,op);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sync(ctx.device)
            if step>=2:measurements.append(time.perf_counter()-start)
        seconds=float(np.median(measurements));batches=math.ceil(len(ctx.indices['train'])/ctx.cfg['batch_size'])
        estimates[variant]={'training_seconds_per_batch':seconds,'training_only_minutes_per_seed':seconds*batches*ctx.cfg['epochs']/60}
        del model,opt,mini
    result={'feature_seconds_per_sample':feature_seconds,'cache_minutes_estimate':feature_seconds*(len(ctx.indices['train'])+len(ctx.indices['validation']))/60,
        'variants':estimates,'note':'Timing only, on a repeated training example. Excludes validation, test, I/O, hashing and startup. Benchmark models are discarded.'}
    save_json(dest,result);print(json.dumps(result,indent=2),flush=True)


def save_checkpoint(path,payload):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)


def train_phase(ctx):
    from torch.utils.data import DataLoader
    from latent_operator_models import masked_mse,support_apply
    if (ctx.cli.output/'test_metrics.csv').exists():raise RuntimeError('Test evaluation exists. Use a new protocol rather than retraining after test inspection.')
    for role in ['train','validation']:
        for i in ctx.indices[role]:
            if not (ctx.cli.output/'cache'/role/f'{i:06d}.npz').is_file():raise RuntimeError('Run cache first')
    mask,op=train_support(ctx)
    for seed in ctx.cfg['training_seeds']:
        states=initial_states(spatial.SpatialPhysicsResidualNet,ctx.layout,ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit,int(seed),ctx.cfg['variants'])
        for variant in ctx.cfg['variants']:
            folder=ctx.cli.output/'models'/f'seed_{seed}'/variant;folder.mkdir(parents=True,exist_ok=True)
            last=folder/'last.pt';bestpath=folder/'best.pt'
            model=spatial.SpatialPhysicsResidualNet(len(ctx.layout['keep'][variant]),ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit).to(ctx.device)
            model.load_state_dict(states[variant]);optimizer=torch.optim.AdamW(model.parameters(),lr=ctx.args.lr,weight_decay=ctx.args.weight_decay)
            start_epoch=1;best=float('inf');history=[]
            if last.exists():
                ck=torch.load(last,map_location=ctx.device,weights_only=False)
                model.load_state_dict(ck['state']);optimizer.load_state_dict(ck['optimizer'])
                start_epoch=ck['epoch']+1;best=ck['best_validation_mse'];history=ck['history'];del ck
            validation=DataLoader(Cached(ctx,'validation',ctx.indices['validation'],ctx.layout['keep'][variant]),batch_size=ctx.cfg['batch_size'],shuffle=False,num_workers=0)
            for epoch in range(start_epoch,ctx.cfg['epochs']+1):
                seed_all(int(seed)+100003*epoch)
                order=np.random.default_rng(int(seed)+epoch).permutation(ctx.indices['train']).tolist()
                training=DataLoader(Cached(ctx,'train',order,ctx.layout['keep'][variant]),batch_size=ctx.cfg['batch_size'],shuffle=False,num_workers=0)
                model.train();total=0.;seen=0;sync(ctx.device);epoch_start=time.perf_counter()
                for bi,batch in enumerate(training,1):
                    mini=batch_to(batch,ctx.device);optimizer.zero_grad(set_to_none=True)
                    loss,_=prediction_loss(ctx,model,mini,mask,op)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                    count=len(mini['target']);total+=float(loss.detach())*count;seen+=count
                    if bi==1 or bi%ctx.cfg['progress_every']==0 or bi==len(training):
                        print(f'[TRAIN] seed={seed} {variant} epoch={epoch}/{ctx.cfg["epochs"]} batch={bi}/{len(training)} elapsed={(time.perf_counter()-epoch_start)/60:.1f} min',flush=True)
                model.eval();valsum=0.;n=0
                with torch.no_grad():
                    for batch in validation:
                        mini=batch_to(batch,ctx.device);pred=support_apply(mini['prior']+model(mini['features']),mask,ctx.args.water_norm)
                        count=len(pred);valsum+=float(masked_mse(pred,mini['target'],mask))*count;n+=count
                score=valsum/n;sync(ctx.device)
                history.append({'epoch':epoch,'train_loss':total/seen,'validation_mse':score,'seconds':time.perf_counter()-epoch_start})
                if score<best:
                    best=score;save_checkpoint(bestpath,{'state':model.state_dict(),'epoch':epoch,'validation_mse':score,'variant':variant,'seed':seed,'keep_channels':ctx.layout['keep'][variant]})
                save_checkpoint(last,{'state':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best_validation_mse':best,'history':history})
                save_json(folder/'history.json',history)
                print(f'[VALIDATION] {variant} epoch={epoch} mse={score:.7g} best={best:.7g}; {history[-1]["seconds"]:.1f}s',flush=True)
            save_json(folder/'training_summary.json',{'seed':seed,'variant':variant,'epochs':ctx.cfg['epochs'],
                'parameters':sum(p.numel() for p in model.parameters()),'best_validation_mse':best,
                'seconds':sum(r['seconds'] for r in history),'selection':'minimum validation image MSE after each matched-budget epoch'})
            del model,optimizer
    if op is not None:assert all(not p.requires_grad and p.grad is None for p in op.parameters())


def evaluate_phase(ctx):
    from latent_operator_models import support_apply
    from checkpoint_reader import load_checkpoint
    if (ctx.cli.output/'test_evaluation_complete.json').exists():
        print('[TEST] Complete evaluation exists; summarize reads it.',flush=True);return
    models={};best_epochs={}
    for seed in ctx.cfg['training_seeds']:
        for variant in ctx.cfg['variants']:
            path=ctx.cli.output/'models'/f'seed_{seed}'/variant/'best.pt';last=path.with_name('last.pt')
            if not path.is_file() or not last.is_file():raise RuntimeError('Complete matched training before testing')
            record=load_checkpoint(last)
            if int(record['epoch'])!=ctx.cfg['epochs']:raise RuntimeError('Incomplete fixed training budget')
            del record;ck=load_checkpoint(path)
            model=spatial.SpatialPhysicsResidualNet(len(ctx.layout['keep'][variant]),ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit).to(ctx.device)
            model.load_state_dict(ck['state']);model.eval();models[(int(seed),variant)]=model
            best_epochs[f'{seed}:{variant}']=int(ck['epoch'])
    context_model=spatial.SpatialPhysicsResidualNet(ctx.layout['channels'],ctx.args.channels,ctx.args.dropout,ctx.args.delta_limit).to(ctx.device)
    ck=ctx.original_checkpoint;context_model.load_state_dict(ck.get('best_model_state_dict',ck['model_state_dict']));context_model.eval()
    ds,loader,mask,water,stack=load_upstream(ctx,ctx.indices['test']);rows=[];dest=ctx.cli.output/'test_metrics.csv'
    if dest.exists():
        with dest.open(newline='',encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                row['sample_index']=int(row['sample_index']);row['seed']=int(row['seed'])
                for metric in METRICS:row[metric]=float(row[metric]) if row.get(metric) else None
                rows.append(row)
    done={(r['sample_index'],r['seed'],r['method']) for r in rows}
    scale=ctx.args.sos_max-ctx.args.sos_min;region=mask.detach().cpu().numpy().squeeze()>0.5
    expected_per_case=2+len(models)
    expected_methods={(-1,'latent_prior'),(-1,'original_full_context')}|{(s,v) for s,v in models}
    started=time.perf_counter()
    for number,batch in enumerate(loader,1):
        index=int(batch[-1][0])
        if all((index,seed,method) in done for seed,method in expected_methods):continue
        item=features(ctx,batch,stack,mask,water)
        with torch.no_grad():
            outputs={(-1,'latent_prior'):item['prior'],(-1,'original_full_context'):support_apply(item['prior']+context_model(item['features']),mask,ctx.args.water_norm)}
            for (seed,variant),model in models.items():
                outputs[(seed,variant)]=support_apply(item['prior']+model(item['features'][:,ctx.layout['keep'][variant]]),mask,ctx.args.water_norm)
        target=item['target'][0,0].cpu().numpy()*scale+ctx.args.sos_min
        for (seed,method),pred in outputs.items():
            key=(index,seed,method)
            if key in done:continue
            field=pred[0,0].cpu().numpy()*scale+ctx.args.sos_min
            rows.append({'sample_index':index,'seed':seed,'method':method,**utils.image_metrics(field,target,region)});done.add(key)
        if number==1 or number%ctx.cfg['progress_every']==0 or number==len(loader):
            utils.write_csv(dest,rows)
            print(f'[TEST] {number}/{len(loader)} cases; {(time.perf_counter()-started)/60:.1f} min elapsed',flush=True)
    if len(rows)!=len(ctx.indices['test'])*expected_per_case:raise RuntimeError('Incomplete or mixed test rows')
    utils.write_csv(dest,rows)
    save_json(ctx.cli.output/'test_evaluation_complete.json',{'samples':len(ctx.indices['test']),'rows':len(rows),'best_epochs':best_epochs,'seconds':time.perf_counter()-started,
        'note':'Identical test examples. Original full model and prior are context, not matched retraining controls.'})


def summarize(ctx):
    path=ctx.cli.output/'test_metrics.csv'
    if not (ctx.cli.output/'test_evaluation_complete.json').is_file():raise RuntimeError('Complete evaluation before summarizing')
    with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    summary={};paired={}
    for row in rows:
        key=f'{row["seed"]}:{row["method"]}';summary.setdefault(key,{m:[] for m in METRICS})
        for metric in METRICS:
            if row[metric]!='':summary[key][metric].append(float(row[metric]))
    summary={k:{m:utils.describe(v) for m,v in d.items()} for k,d in summary.items()}
    lookup={(int(r['seed']),r['method'],int(r['sample_index'])):r for r in rows}
    for seed in ctx.cfg['training_seeds']:
        for variant in ctx.cfg['variants']:
            if variant=='full':continue
            result={}
            for metric in METRICS:
                diffs=[]
                for i in ctx.indices['test']:
                    left=lookup[(int(seed),variant,i)][metric];right=lookup[(int(seed),'full',i)][metric]
                    if left!='' and right!='':diffs.append(float(left)-float(right))
                d=np.asarray(diffs,dtype=float);rng=np.random.default_rng(ctx.cfg['selection_seed'])
                means=[float(rng.choice(d,size=len(d),replace=True).mean()) for _ in range(ctx.cfg['bootstrap_replicates'])] if len(d) else []
                result[metric]={**utils.describe(d.tolist()),'bootstrap_mean_ci95':np.quantile(means,[0.025,0.975]).tolist() if means else None,
                    'positive_difference':'favors full' if metric!='correlation' else 'favors removed-feature variant'}
            paired[f'{seed}:{variant}_minus_full']=result
    save_json(ctx.cli.output/'aggregate_metrics.json',summary)
    save_json(ctx.cli.output/'paired_differences.json',{'differences':paired,'uncertainty_scope':'Paired test-sample bootstrap conditional on each seed; not variation across retraining seeds.'})
    text=['MATCHED SPATIAL CONDITIONING ABLATION','',f'Train/validation/test: {len(ctx.indices["train"])}/{len(ctx.indices["validation"])}/{len(ctx.indices["test"])}',
        f'Epochs per variant: {ctx.cfg["epochs"]}; seeds: {ctx.cfg["training_seeds"]}',
        'Upstream fixed. Spatial models trained afresh under the same shortened budget.',
        'Explicit learned-operator residual/adjoint inputs removed; upstream physics and common ToF inputs/loss retained.',
        'Clean-input comparison; original nuisance/consistency augmentation is not reproduced.',
        'original_full_context is a contextual baseline, not the matched retraining control.','']
    for key,d in summary.items():
        text.append(key+': '+', '.join(f'{m}={d[m].get("mean",float("nan")):.6g}' for m in METRICS))
    (ctx.cli.output/'RESULTS_README.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)


def main():
    a=arguments();ctx=None
    try:
        ctx=prepare(a)
        if a.phase=='preflight':print('[PREFLIGHT] Passed. No inference or training run.',flush=True)
        elif a.phase=='benchmark':benchmark(ctx)
        elif a.phase=='cache':cache_phase(ctx)
        elif a.phase=='train':train_phase(ctx)
        elif a.phase=='evaluate':evaluate_phase(ctx)
        elif a.phase=='summarize':summarize(ctx)
        elif (a.output/'test_evaluation_complete.json').exists():summarize(ctx)
        elif (a.output/'test_metrics.csv').exists():evaluate_phase(ctx);summarize(ctx)
        else:cache_phase(ctx);train_phase(ctx);evaluate_phase(ctx);summarize(ctx)
    finally:
        if ctx is not None:ctx.verify()


if __name__=='__main__':main()
