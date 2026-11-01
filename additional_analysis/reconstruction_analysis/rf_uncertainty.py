"""RF-picker stability using the retained finite-window/template algorithm."""
from pathlib import Path
import heapq,json,time
import numpy as np
from scipy.signal import hilbert,butter,filtfilt
import run_ablation as ab


def preprocess(raw,time_axis,spec):
    x=np.asarray(raw,dtype=np.float32).copy();t=np.asarray(time_axis,float)
    i0=int(np.clip(np.searchsorted(t,float(spec['t_skip'])),0,len(t)-1))
    if spec.get('apply_gaussian_window',True) and i0>0:
        duration=float(t[i0]-t[0]);sigma=duration*float(spec.get('gaussian_scale',.5))
        if sigma>0:
            window=np.ones(len(t),np.float32)
            window[:i0]=np.exp(-.5*((t[:i0]-t[i0])/sigma)**2).astype(np.float32)
            x*=window[None]
    low,high=spec.get('bandpass_low'),spec.get('bandpass_high')
    if low is not None and high is not None:
        fs=1/np.median(np.diff(t));low=float(low);high=float(high)
        if low<1e3:low*=1e6
        if high<1e3:high*=1e6
        b,a=butter(4,[max(low,.01*fs/2)/(fs/2),min(high,.99*fs/2)/(fs/2)],btype='bandpass')
        x=filtfilt(b,a,x,axis=-1)
    env=np.abs(hilbert(x,axis=-1)).astype(np.float32)
    width=max(1,int(spec['env_smooth']))
    if width>1:
        kernel=np.ones(width,np.float32)/width;pad=width//2
        envp=np.pad(env,((0,0),(pad,pad)),mode='edge')
        env=np.apply_along_axis(lambda v:np.convolve(v,kernel,mode='valid'),1,envp)
    return x,env,i0


def preliminary(env,t,centers,spec,max_geom):
    from extract_measured_tof import _windowed_first_threshold_pick
    return _windowed_first_threshold_pick(env,t,np.asarray(centers),
        twinpre=float(spec['peak_window_pre_frac'])*max_geom,
        twinpost=float(spec['peak_window_post_frac'])*max_geom,thresh_rel=float(spec['thresh_rel']))


def make_template(segments):
    template=np.median(np.asarray(segments,dtype=np.float64),axis=0)
    template-=template.mean();norm=np.linalg.norm(template)
    if norm<=1e-12 or not np.isfinite(norm):raise ValueError('Degenerate matched template')
    return template/norm


def pick(raw,t,centers,spec,max_geom,template):
    x,env,_=preprocess(raw,t,spec);onset=preliminary(env,t,centers,spec,max_geom)
    pre=int(spec['matched_template_pre_samples']);post=int(spec['matched_template_post_samples'])
    length=pre+post+1;lag=int(spec['matched_template_max_lag_samples'])
    times=np.full(len(x),np.nan);status=np.full(len(x),'missing',dtype='<U16')
    for j,ip in enumerate(onset):
        if ip<0:continue
        times[j]=t[ip];status[j]='fallback'
        first=max(0,int(ip)-pre-lag);last=min(len(t)-length,int(ip)-pre+lag)
        if last<first:continue
        trace=np.asarray(x[j,first:last+length],np.float64);trace-=trace.mean()
        corr=np.correlate(trace,template,mode='valid')
        energy=np.sqrt(np.convolve(trace*trace,np.ones(length),mode='valid'))
        score=np.abs(corr/np.maximum(energy,1e-12));q=int(np.argmax(score));best=float(score[q])
        if not np.isfinite(best) or best<float(spec['matched_template_min_correlation']):continue
        if q==0 or q==len(score)-1:continue
        ym,y0,yp=map(float,score[q-1:q+2]);denom=ym-2*y0+yp
        delta=float(np.clip(.5*(ym-yp)/denom,-.5,.5)) if abs(denom)>1e-15 else 0.
        pick_index=first+q+pre+delta
        if abs(pick_index-ip)>lag+.5:continue
        times[j]=np.interp(pick_index,np.arange(len(t)),t);status[j]='refined'
    return times,status


def estimate_noise(raw,onset,pre,i0,geometric_center,max_geom,time_axis):
    """Report availability; zero/absent baseline variability is not imputed."""
    high=int(onset)-pre if onset>=0 else int(np.searchsorted(time_axis,geometric_center-.05*max_geom))
    high=int(np.clip(high,0,len(raw)));low=int(np.clip(max(i0,high-3*pre),0,len(raw)))
    noise=np.asarray(raw[low:high],float)
    if len(noise)<8:return np.nan,'insufficient_prearrival_samples',low,high
    if not np.isfinite(noise).all():return np.nan,'nonfinite_prearrival_samples',low,high
    sigma=float(np.median(np.abs(noise-np.median(noise)))/.6744897501960817)
    if not np.isfinite(sigma) or sigma<=0:return np.nan,'zero_prearrival_MAD',low,high
    return sigma,'usable',low,high


def rf_cache(ctx,folder):
    import h5py
    from checkpoint_reader import load_checkpoint
    dest=folder/'rf_subset.npz'
    raw_package=load_checkpoint(ctx.extra_paths['measured_raw'])
    spec=dict(raw_package['metadata']['sample_spec'])
    if spec.get('subtract_min_per_emitter',False):raise ValueError('RF study expects preserved absolute row times')
    if spec.get('tof_variant_saved')!='geom_window_matched_template':raise ValueError('Saved RF picker differs from this study')
    if dest.exists():
        with np.load(dest,allow_pickle=False) as d:cached={k:d[k].copy() for k in d.files}
        cached['spec']=spec;return cached
    tx=np.asarray(spec['tx_ids'],int);rx=np.asarray(spec['rx_ids'],int)
    with h5py.File(ctx.cli.geometry_mat,'r') as f:
        xy=np.asarray(f['transducerPositionsXY'],float).squeeze()
        if xy.shape[0]==2:xy=xy.T
        t=np.asarray(f['time']).reshape(-1).astype(float);data=f['full_dataset']
        if data.shape[:2]!=(len(xy),len(xy)) or len(t)!=data.shape[-1]:raise ValueError('Unexpected RF/time dimensions')
        points_tx=xy[tx];points_rx=xy[rx]
        chord=np.linalg.norm(points_tx[:,None,:]-points_rx[None,:,:],axis=-1)
        geom=chord/float(spec['c_geom']);max_geom=float(geom.max())
        from extract_measured_tof import _make_exclusion_mask_for_selection
        eligible=~_make_exclusion_mask_for_selection(tx,rx,len(xy),float(spec['exclude_frac']))
        channel_pairs=np.argwhere(eligible)
        rng=np.random.default_rng(ctx.cfg['selection_seed'])
        chosen=channel_pairs[rng.permutation(len(channel_pairs))[:ctx.study['rf_channels']]]
        selected={tuple(pair):i for i,pair in enumerate(chosen)}
        waveforms=np.empty((len(chosen),len(t)),np.float32);sigmas=np.full(len(chosen),np.nan)
        reasons=np.full(len(chosen),'not_read',dtype='<U40');windows=np.zeros((len(chosen),2),dtype=np.int64)
        centers=geom[chosen[:,0],chosen[:,1]];heap=[];ordinal=0
        target=max(32,int(spec['matched_template_count_requested']))
        pre=int(spec['matched_template_pre_samples']);post=int(spec['matched_template_post_samples'])
        rx_sort=np.argsort(rx);inverse=np.argsort(rx_sort);start=time.perf_counter()
        for e,raw_tx in enumerate(tx):
            raw=np.asarray(data[int(raw_tx),rx[rx_sort],:],np.float32)[inverse]
            x,env,i0=preprocess(raw,t,spec);onset=preliminary(env,t,geom[e],spec,max_geom)
            for r,ip in enumerate(onset):
                selected_row=selected.get((e,r))
                if selected_row is not None:
                    waveforms[selected_row]=raw[r]
                    # Estimate perturbation amplitude from the raw pre-arrival waveform.
                    sigma,reason,low,high=estimate_noise(raw[r],ip,pre,i0,geom[e,r],max_geom,t)
                    sigmas[selected_row]=sigma;reasons[selected_row]=reason;windows[selected_row]=[low,high]
                if not eligible[e,r] or ip<pre or ip+post>=len(t):continue
                seg=np.asarray(x[r,ip-pre:ip+post+1],np.float64);seg-=seg.mean();norm=np.linalg.norm(seg)
                if norm<=1e-12 or not np.isfinite(norm):continue
                seg/=norm
                if seg[np.argmax(np.abs(seg))]<0:seg=-seg
                lo=max(i0,int(ip)-4*pre);hi=max(lo+1,int(ip)-pre)
                noise=float(np.median(env[r,lo:hi])) if hi>lo else 0.
                signal=float(np.max(env[r,ip:ip+post+1]));quality=signal/max(noise,1e-12)
                entry=(quality,-ordinal,seg.astype(np.float32));ordinal+=1
                if len(heap)<target:heapq.heappush(heap,entry)
                elif entry[:2]>heap[0][:2]:heapq.heapreplace(heap,entry)
            if e==0 or (e+1)%32==0 or e+1==len(tx):
                print(f'[RF TEMPLATE] {e+1}/{len(tx)} transmitters, {(time.perf_counter()-start)/60:.1f} min',flush=True)
        if len(heap)<32:raise ValueError('Too few valid template segments')
        heap.sort(key=lambda v:(-v[0],-v[1]));segments=np.stack([h[2] for h in heap])
    template=make_template(segments)
    saved=np.asarray(raw_package['tof']);valid=np.asarray(raw_package['tof_mask'])
    while saved.ndim>2:saved=saved[0]
    while valid.ndim>2:valid=valid[0]
    saved_selected=saved[chosen[:,0],chosen[:,1]]
    valid_selected=valid[chosen[:,0],chosen[:,1]]>0.5
    usable_noise=np.isfinite(sigmas)&(sigmas>0)
    cached=dict(raw=waveforms,time=t,centers=centers,max_geom=np.asarray(max_geom),segments=segments,
        template=template,noise_sigma=sigmas,noise_usable=usable_noise,noise_reason=reasons,noise_windows=windows,channel_pairs=chosen,saved_times=saved_selected,saved_valid=valid_selected)
    with dest.open('wb') as stream:np.savez(stream,**cached)
    cached['spec']=spec;return cached


def run(ctx):
    folder=ctx.cli.output/'rf_picker';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():print('[RF] Complete results exist',flush=True);return
    c=rf_cache(ctx,folder);spec=c['spec'];t=c['time'];max_geom=float(c['max_geom'])
    baseline,status=pick(c['raw'],t,c['centers'],spec,max_geom,c['template'])
    use=c['saved_valid']&np.isfinite(baseline)&np.isfinite(c['saved_times'])
    if not use.any():raise ValueError('No comparable saved RF picks')
    error=(baseline[use]-c['saved_times'][use])*1e6
    fraction=float(np.mean(np.abs(error)<=ctx.study['rf_baseline_tolerance_us']))
    ab.save_json(folder/'baseline_validation.json',{'channels':len(baseline),'compared':int(use.sum()),
        'matching_fraction':fraction,'maximum_abs_difference_us':float(np.max(np.abs(error))),
        'tolerance_us':ctx.study['rf_baseline_tolerance_us'],'note':'Rebuilt full-acquisition median template checked against saved raw picks.'})
    if fraction<ctx.study['rf_baseline_min_match_fraction'] or int(use.sum())<ctx.study['rf_baseline_min_match_fraction']*int(c['saved_valid'].sum()):
        raise RuntimeError('Subset picker does not reproduce the saved baseline. Inspect RF baseline_validation.json; no uncertainty result is accepted.')
    usable=np.isfinite(c['noise_sigma'])&(c['noise_sigma']>0)
    noise_rows=[{'selected_row':i,'transmitter_index':int(pair[0]),'receiver_index':int(pair[1]),
        'noise_usable':bool(usable[i]),'noise_sigma':float(c['noise_sigma'][i]) if usable[i] else None,
        'reason':str(c['noise_reason'][i]),'window_low':int(c['noise_windows'][i,0]),'window_high':int(c['noise_windows'][i,1])}
        for i,pair in enumerate(c['channel_pairs'])]
    ctx.utils.write_csv(folder/'noise_availability.csv',noise_rows)
    labels,counts=np.unique(c['noise_reason'],return_counts=True)
    availability={'predetermined_channels':len(usable),'usable_noise_channels':int(usable.sum()),
        'unavailable_noise_channels':int((~usable).sum()),'reason_counts':dict(zip(labels.tolist(),counts.tolist())),
        'handling':'All selected channels remain in baseline, template-only and known-delay tests. Added-noise tests use only channels with finite positive pre-arrival MAD; no replacement channels or imputed amplitudes.'}
    ab.save_json(folder/'noise_availability.json',availability)
    print('[RF NOISE AVAILABILITY]',availability,flush=True)
    summaries=[];noise_arrays={};seed=int(ctx.cfg['selection_seed'])
    for scale in ctx.study['rf_noise_scales']:
        select=np.ones(len(usable),bool) if float(scale)==0 else usable
        count=int(select.sum());base_selected=baseline[select]
        row={'kind':'template_only_repeatability' if float(scale)==0 else 'noise_template_repeatability',
             'noise_scale':scale,'predetermined_channels':len(usable),'channels_tested':count,
             'noise_unavailable_channels':int((~usable).sum()),'available':count>0}
        if not count:
            row.update(repeats=0,median_channel_sd_us=None,p95_channel_sd_us=None,rms_change_us=None,
                       mean_change_us=None,missing_fraction=None,fallback_fraction=None)
            summaries.append(row);print('[RF]',row,flush=True);continue
        label=f'noise_{scale:g}';path=folder/(label+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as d:times=d['times'];states=d['status']
        else:
            values=[];states=[];rng=np.random.default_rng(seed+int(round(scale*1000)))
            for repeat in range(ctx.study['rf_repeats']):
                template=make_template(c['segments'][rng.integers(len(c['segments']),size=len(c['segments']))]) if ctx.study['rf_bootstrap_template'] else c['template']
                noisy=c['raw'][select].copy()
                if float(scale)!=0:
                    noisy=noisy+float(scale)*c['noise_sigma'][select,None]*rng.normal(size=noisy.shape)
                picked,flags=pick(noisy,t,c['centers'][select],spec,max_geom,template);values.append(picked);states.append(flags)
            times=np.asarray(values);states=np.asarray(states)
            np.savez(path,times=times,status=states,baseline=base_selected,channel_pairs=c['channel_pairs'][select],selected_rows=np.flatnonzero(select))
        changes=(times-base_selected[None])*1e6
        std=np.array([np.std(col[np.isfinite(col)],ddof=1) if np.isfinite(col).sum()>1 else np.nan for col in (times*1e6).T])
        finite=std[np.isfinite(std)];noise_arrays[scale]=std
        finite_changes=changes[np.isfinite(changes)]
        row.update(repeats=len(times),channels_with_repeatability_estimate=len(finite),
            median_channel_sd_us=float(np.median(finite)) if len(finite) else None,
            p95_channel_sd_us=float(np.quantile(finite,.95)) if len(finite) else None,
            rms_change_us=float(np.sqrt(np.mean(finite_changes**2))) if len(finite_changes) else None,
            mean_change_us=float(np.mean(finite_changes)) if len(finite_changes) else None,
            missing_fraction=float(np.mean(~np.isfinite(times))),fallback_fraction=float(np.mean(states=='fallback')))
        summaries.append(row);print('[RF]',row,flush=True)
    delay_rows=[]
    for delay in ctx.study['rf_delays_us']:
        shifted=np.stack([np.interp(t-delay*1e-6,t,row,left=0,right=0) for row in c['raw']])
        picked,flags=pick(shifted,t,c['centers'],spec,max_geom,c['template'])
        use=np.isfinite(picked)&np.isfinite(baseline);err=(picked[use]-baseline[use])*1e6-float(delay)
        delay_rows.append({'kind':'known_delay_recovery','delay_us':delay,'valid_pairs':int(use.sum()),
            'bias_us':float(np.mean(err)) if len(err) else None,'rmse_us':float(np.sqrt(np.mean(err*err))) if len(err) else None,
            'p95_abs_error_us':float(np.quantile(np.abs(err),.95)) if len(err) else None,'missing_fraction':float(np.mean(~np.isfinite(picked)))})
    ctx.utils.write_csv(folder/'noise_summary.csv',summaries);ctx.utils.write_csv(folder/'delay_recovery.csv',delay_rows)
    sd=noise_arrays.get(1.,np.array([]));sd=sd[np.isfinite(sd)]
    estimated=float(np.median(sd)) if len(sd) else None
    ab.save_json(folder/'complete.json',{'noise_rows':summaries,'delay_rows':delay_rows,'estimated_median_tof_sd_us':estimated,
        'scope':'RF noise/template-conditional picking repeatability and known-delay increment recovery. No independent measured first-arrival ground truth; these are not absolute in-vivo timing errors.',
        'template':'Rebuilt once from the full acquisition; selected channels predetermined by acquisition validity, before waveform inspection.',
        'baseline_matching_fraction':fraction,'noise_availability':availability})
