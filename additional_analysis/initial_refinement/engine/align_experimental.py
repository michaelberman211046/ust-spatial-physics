"""publication GT-free calibration in the immutable measured channel geometry."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from measured_geometry import require_geometry
from dataset import PairsCacheDataset
from logger import log_message
from settings import set_output_folder

VERSION = "measured-geometry-single-gated-additive-calibration-v1"


def med(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else 0.0


def synthetic_stats(path, samples, sos_min, sos_max):
    ds = PairsCacheDataset(path, sos_min=sos_min, sos_max=sos_max, return_tof_mask=True)
    ids = np.linspace(0, len(ds)-1, min(samples, len(ds)), dtype=int)
    total = squared = count = None
    for index in ids:
        item = ds[int(index)]
        value = item[1].numpy().astype(np.float64)
        valid = item[2].numpy() > .5
        if total is None:
            total = np.zeros_like(value)
            squared = np.zeros_like(value)
            count = np.zeros_like(value)
        total[valid] += value[valid]
        squared[valid] += value[valid]**2
        count[valid] += 1
    mean = total/np.maximum(count, 1)
    std = np.sqrt(np.maximum(squared/np.maximum(count, 1)-mean**2, 0))
    return ds, mean, std, count


def fit_additive(residual, use, limit=.75, iters=8):
    ne, nr = residual.shape
    global_delay = med(residual[use])
    emitter = np.zeros(ne)
    receiver = np.zeros(nr)
    bound = limit*1e-6
    for _ in range(iters):
        for i in range(ne):
            if use[i].any():
                emitter[i] = med((residual[i]-global_delay-receiver)[use[i]])
        emitter = np.clip(emitter-np.median(emitter), -bound, bound)
        for j in range(nr):
            if use[:, j].any():
                receiver[j] = med((residual[:, j]-global_delay-emitter)[use[:, j]])
        receiver = np.clip(receiver-np.median(receiver), -bound, bound)
        global_delay = med((residual-emitter[:, None]-receiver[None, :])[use])
    return global_delay, emitter, receiver, global_delay+emitter[:, None]+receiver[None, :]


def main():
    p = argparse.ArgumentParser()
    for name in ("raw_pt", "synthetic_data_path", "out_pt", "report_json", "output_dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--synthetic_samples", type=int, default=192)
    p.add_argument("--outer_quantile", type=float, default=.20)
    p.add_argument("--component_limit_us", type=float, default=.75)
    p.add_argument("--calibration_min_gain_us", type=float, default=.02)
    p.add_argument("--sos_min", type=float, default=1400.)
    p.add_argument("--sos_max", type=float, default=1650.)
    a = p.parse_args()
    set_output_folder(a.output_dir)
    log_message("[CMD] "+" ".join(sys.argv))
    log_message("[ARGS] "+repr(vars(a)))
    raw = torch.load(a.raw_pt, map_location="cpu", weights_only=False)
    x = np.asarray(raw["tof"])[0].astype(np.float64)
    mask = np.asarray(raw["tof_mask"])
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask > .5
    points, order, center, digest = require_geometry(*x.shape)
    dataset, synthetic_mean, synthetic_std, counts = synthetic_stats(
        a.synthetic_data_path, a.synthetic_samples, a.sos_min, a.sos_max)
    norm = dataset.tof_norm
    scale = float(norm["std"])+float(norm.get("eps", 1e-6))
    synthetic_seconds = synthetic_mean*scale+float(norm["mean"])
    usable = mask & (counts > 0)
    if usable.sum() < 10000:
        raise RuntimeError("Insufficient shared valid channels for publication calibration")
    length = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=-1)
    threshold = float(np.quantile(length[usable], a.outer_quantile))
    outer = usable & (length <= threshold)
    ii, jj = np.indices(x.shape)
    train = outer & (((ii+3*jj) % 5) != 0)
    holdout = outer & ~train
    residual = x-synthetic_seconds
    g0, e0, r0, additive = fit_additive(residual, train, limit=a.component_limit_us)
    additive_error = med(abs((residual-additive)[holdout]))*1e6
    raw_error = med(abs(residual[holdout]))*1e6
    raw_z = med(abs(((x-float(norm["mean"]))/scale-synthetic_mean)[usable]
                    / np.maximum(synthetic_std[usable], .03)))
    candidate_z = med(abs(((x-additive-float(norm["mean"]))/scale-synthetic_mean)[usable]
                          / np.maximum(synthetic_std[usable], .03)))
    accept_additive = (additive_error+a.calibration_min_gain_us < raw_error
                       and candidate_z <= raw_z)
    correction = additive if accept_additive else np.zeros_like(x)
    corrected = x-correction
    out = copy.deepcopy(raw)
    out["tof"] = torch.as_tensor(np.where(mask, corrected, float(norm["mean"])), dtype=torch.float32)[None]
    out["tof_mask"] = mask.astype(np.float32)
    out["tof_weight"] = mask.astype(np.float32)[None]
    report = {"version": VERSION, "geometry_sha256": digest,
              "raw_channel_permutation": "none_after_measured_angular_order",
              "receiver_interpolation": False, "uses_gt": False,
              "measured_center_m": center.tolist(), "measured_receiver_order": order.tolist(),
              "outer_train": int(train.sum()), "outer_holdout": int(holdout.sum()),
              "raw_holdout_median_abs_us": raw_error,
              "additive_holdout_median_abs_us": additive_error,
              "additive_calibration_accepted": bool(accept_additive),
              "pre_channel_z_median_abs": raw_z,
              "candidate_channel_z_median_abs": candidate_z,
              "post_channel_z_median_abs": med(abs(((corrected-float(norm["mean"]))/scale-synthetic_mean)[usable]
                                               / np.maximum(synthetic_std[usable], .03)))}
    metadata = copy.deepcopy(dict(out.get("metadata", {}) or {}))
    metadata["experimental_alignment"] = report
    out["metadata"] = metadata
    Path(a.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out_pt)
    Path(a.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_message(f"[alignment] outer holdout raw={raw_error:.5g} us, additive={additive_error:.5g} us, accepted={accept_additive}")


if __name__ == "__main__":
    main()










