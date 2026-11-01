"""Numerical summaries, reproducible manifests, and bounded output utilities."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path, full_hash=False):
    p = Path(path).resolve()
    result = dict(path=str(p), bytes=p.stat().st_size, modified_ns=p.stat().st_mtime_ns)
    if full_hash:
        result['sha256'] = digest(p)
    return result


def image_metrics(image, reference, region):
    use = np.asarray(region) > .5
    a = np.asarray(image, dtype=np.float64)[use]
    b = np.asarray(reference, dtype=np.float64)[use]
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Metric inputs must have a nonempty finite evaluation region')
    error = a - b
    rmse = float(np.sqrt(np.mean(error ** 2)))
    scale = float(np.ptp(b))
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return dict(rmse_mps=rmse, nrmse_percent=100 * rmse / scale if scale > 0 else None,
                mad_mps=float(np.mean(np.abs(error))), correlation=correlation,
                evaluation_pixels=len(a), reference_range_mps=scale)


def describe(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a)>1 else None,
                median=float(np.median(a)), q25=float(np.quantile(a,.25)), q75=float(np.quantile(a,.75)),
                minimum=float(a.min()), maximum=float(a.max()))


def summarize_rows(rows):
    groups = {}
    for row in rows:
        key = (row.get('study', ''), row.get('condition', 'clean'), row['method'])
        groups.setdefault(key, []).append(row)
    output = []
    for (study, condition, method), items in groups.items():
        result = dict(study=study, condition=condition, method=method, samples=len(items))
        for key in ('rmse_mps','nrmse_percent','mad_mps','correlation','seconds','accepted_iterations'):
            if any(key in item for item in items):
                result[key] = describe([item.get(key) for item in items])
        output.append(result)
    return output


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(jsonable(v)) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def sync(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed(call, device):
    sync(device)
    start = time.perf_counter()
    result = call()
    sync(device)
    return result, time.perf_counter() - start


def perturb_tof(tof, mask, kind, amplitude, seed):
    """Return times in seconds; timing amplitudes are in microseconds."""
    rng = np.random.default_rng(seed)
    out = np.asarray(tof, float).copy()
    valid = np.asarray(mask, bool).copy()
    if kind == 'jitter':
        out += rng.normal(0, amplitude*1e-6, out.shape)
    elif kind == 'common':
        out += amplitude * 1e-6
    elif kind == 'element':
        out += (rng.normal(0,amplitude,(out.shape[0],1)) + rng.normal(0,amplitude,(1,out.shape[1]))) * 1e-6
    elif kind == 'dropout':
        valid &= rng.random(valid.shape) >= amplitude
    elif kind == 'outliers':
        select = valid & (rng.random(valid.shape) < amplitude)
        out[select] += rng.choice([-1,1],size=int(select.sum())) * 1e-6
    elif kind != 'clean':
        raise ValueError(kind)
    return out, valid


def hardware():
    import torch
    import importlib.metadata as metadata
    versions = {}
    for name in ('torch','numpy','scipy','scikit-fmm','h5py','matplotlib'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    gpus=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        gpus.append(dict(index=i,name=p.name,memory_bytes=p.total_memory))
    try:
        import psutil
        ram=psutil.virtual_memory().total
    except ImportError:
        ram=None
    return dict(platform=platform.platform(),python=sys.version,processor=platform.processor(),
                cpu_count=os.cpu_count(),ram_bytes=ram,packages=versions,gpus=gpus,
                torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
