import argparse
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pprint
import sys
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from dataset import PairsCacheDataset, load_splits, make_cache_paths
from latent_operator_models import (
    SCRIPT_FAMILY,
    MatchedToFOperator,
    batch_progress,
    denormalize_tof_seconds,
    format_duration,
    masked_l1,
    masked_mse,
    support_apply,
)
from logger import log_image, log_message
from settings import set_output_folder
from reconstruction_utils import make_circular_support_mask


SCRIPT_VERSION = "2026-06-29-train-matched-tof-operator-v1"


def split_dataset(ds, data_path, splits_path):
    if splits_path is None:
        splits_path = make_cache_paths(data_path)["splits_path"]
    if not os.path.exists(splits_path):
        n = len(ds)
        return Subset(ds, list(range(int(0.8 * n)))), Subset(ds, list(range(int(0.8 * n), n)))
    splits = load_splits(splits_path)
    return Subset(ds, list(map(int, splits["train_idx"]))), Subset(ds, list(map(int, splits["val_idx"])))


def tof_loss(pred, target, mask, args):
    if mask is None:
        return args.lambda_l1 * torch.nn.functional.l1_loss(pred, target) + args.lambda_mse * torch.nn.functional.mse_loss(pred, target)
    return args.lambda_l1 * masked_l1(pred, target, mask) + args.lambda_mse * masked_mse(pred, target, mask)


def tof_metrics(pred, target, mask, tof_norm):
    mse = float(masked_mse(pred, target, mask).detach().item())
    pred_s = denormalize_tof_seconds(pred, tof_norm)
    target_s = denormalize_tof_seconds(target, tof_norm)
    err_us = (pred_s - target_s) / 1e-6
    if mask is not None:
        m = mask.to(device=err_us.device, dtype=err_us.dtype)
        mae = float((err_us.abs() * m).sum().detach().item() / m.sum().clamp_min(1.0).detach().item())
        rmse = float(torch.sqrt((err_us.square() * m).sum() / m.sum().clamp_min(1.0)).detach().item())
    else:
        mae = float(err_us.abs().mean().detach().item())
        rmse = float(torch.sqrt(err_us.square().mean()).detach().item())
    return mse, mae, rmse


@torch.no_grad()
def evaluate(model, loader, device, support_mask, args):
    model.eval()
    sums = {"mse": 0.0, "mae_us": 0.0, "rmse_us": 0.0}
    n = 0
    for batch in loader:
        sos = batch[0].to(device).float().unsqueeze(1)
        tof = batch[1].to(device).float()
        tof_mask = batch[2].to(device).float() if len(batch) > 2 and torch.is_tensor(batch[2]) and batch[2].shape == batch[1].shape else torch.ones_like(tof)
        sos = support_apply(sos, support_mask, args.water_norm)
        pred = model(sos)
        mse, mae, rmse = tof_metrics(pred, tof, tof_mask, args.tof_norm)
        sums["mse"] += mse
        sums["mae_us"] += mae
        sums["rmse_us"] += rmse
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def plot_preview(model, loader, device, support_mask, args, title):
    model.eval()
    batch = next(iter(loader))
    sos = batch[0].to(device).float().unsqueeze(1)
    tof = batch[1].to(device).float()
    tof_mask = batch[2].to(device).float() if len(batch) > 2 and torch.is_tensor(batch[2]) and batch[2].shape == batch[1].shape else torch.ones_like(tof)
    sos = support_apply(sos, support_mask, args.water_norm)
    pred = model(sos)
    n = min(4, sos.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = axes[None, :]
    vmax_res = float(args.residual_plot_clip_us)
    for i in range(n):
        target_us = denormalize_tof_seconds(tof[i], args.tof_norm).detach().cpu() / 1e-6
        pred_us = denormalize_tof_seconds(pred[i], args.tof_norm).detach().cpu() / 1e-6
        mask = tof_mask[i].detach().cpu() > 0.5
        target_us = target_us.masked_fill(~mask, float("nan"))
        pred_us = pred_us.masked_fill(~mask, float("nan"))
        residual = pred_us - target_us
        for ax, img, name, cmap, lo, hi in [
            (axes[i, 0], target_us, "Stored ToF [us]", "viridis", None, None),
            (axes[i, 1], pred_us, "Operator ToF [us]", "viridis", None, None),
            (axes[i, 2], residual, "Residual [us]", "coolwarm", -vmax_res, vmax_res),
        ]:
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="upper", aspect="equal")
            ax.set_title(name)
            ax.set_xlabel("Receiver index")
            ax.set_ylabel("Emitter index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    log_image(fig)
    plt.close(fig)


def save_ckpt(path, args, model, best_state, best_val, best_epoch, epoch, hist, opt=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "script_family": SCRIPT_FAMILY,
            "script_version": SCRIPT_VERSION,
            "kind": "matched_tof_operator",
            "config": vars(args),
            "operator_state_dict": model.state_dict(),
            "best_operator_state_dict": best_state,
            "best_val_mse": float(best_val),
            "best_epoch": int(best_epoch),
            "last_epoch": int(epoch),
            "history": hist,
            "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        },
        path,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--splits_path", default=None)
    p.add_argument("--operator_path", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--operator_channels", type=int, default=56)
    p.add_argument("--operator_latent_dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=0.4)
    p.add_argument("--early_stop_patience", type=int, default=14)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--preview_every", type=int, default=5)
    p.add_argument("--batch_log_every", type=int, default=25)
    p.add_argument("--residual_plot_clip_us", type=float, default=8.0)
    p.add_argument("--use_circular_mask", action="store_true")
    p.add_argument("--mask_radius", type=float, default=0.1091)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--phys_x", type=float, default=0.24)
    p.add_argument("--phys_y", type=float, default=0.24)
    p.add_argument("--n_emitters", type=int, default=64)
    p.add_argument("--n_receivers", type=int, default=64)
    p.add_argument("--sos_min", type=float, default=1380.0)
    p.add_argument("--sos_max", type=float, default=1660.0)
    p.add_argument("--sos_water", type=float, default=1500.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_dir:
        set_output_folder(args.output_dir)
    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message(f"[matched_operator] script_version={SCRIPT_VERSION}")

    requested = str(args.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        msg = "[matched_operator] CUDA requested but unavailable. Actual device will be CPU."
        if args.require_cuda:
            raise RuntimeError(msg)
        log_message(msg)
        requested = "cpu"
    device = torch.device(requested)
    log_message(f"[matched_operator] Actual device used: {device}")

    ds = PairsCacheDataset(args.data_path, sos_min=args.sos_min, sos_max=args.sos_max, return_tof_mask=True)
    args.tof_norm = ds.tof_norm
    meta = dict(getattr(ds, "metadata", {}) or {})
    args.nx = int(meta.get("nx", args.nx))
    args.ny = int(meta.get("ny", args.ny))
    args.phys_x = float(meta.get("phys_x", args.phys_x))
    args.phys_y = float(meta.get("phys_y", args.phys_y))
    args.n_emitters = int(meta.get("n_emitters", args.n_emitters))
    args.n_receivers = int(meta.get("n_receivers", args.n_receivers))
    args.sos_water = float(meta.get("sos_water", args.sos_water))
    args.water_norm = float(max(0.0, min(1.0, (args.sos_water - args.sos_min) / (args.sos_max - args.sos_min))))
    train_ds, val_ds = split_dataset(ds, args.data_path, args.splits_path)
    if args.max_train_samples > 0:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples > 0:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    log_message(f"[matched_operator] Dataset: train={len(train_ds)}, val={len(val_ds)}, tof_norm={args.tof_norm}")

    support_mask = None
    if args.use_circular_mask:
        support_mask = make_circular_support_mask(args.nx, args.ny, args.phys_x, args.phys_y, args.mask_radius, device=device)

    model = MatchedToFOperator(args.n_emitters, args.n_receivers, args.operator_channels, args.operator_latent_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = {"train": [], "val_mse": [], "val_mae_us": [], "val_rmse_us": []}
    best_val, best_epoch, stale = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    start_epoch = 1
    if args.resume and os.path.exists(args.operator_path):
        ckpt = torch.load(args.operator_path, map_location=device, weights_only=False)
        if ckpt.get("kind") != "matched_tof_operator":
            raise ValueError(f"--resume found incompatible checkpoint: {args.operator_path}")
        state = ckpt.get("operator_state_dict")
        if state is None:
            raise ValueError(f"Checkpoint missing operator_state_dict: {args.operator_path}")
        model.load_state_dict(state)
        best_state = copy.deepcopy(ckpt.get("best_operator_state_dict", state))
        best_val = float(ckpt.get("best_val_mse", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        hist = copy.deepcopy(ckpt.get("history", hist))
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        start_epoch = int(ckpt.get("last_epoch", len(hist.get("val_mse", [])))) + 1
        start_epoch = max(1, min(start_epoch, int(args.epochs) + 1))
        log_message(
            f"[matched_operator] Resuming from {args.operator_path}: start_epoch={start_epoch}, "
            f"best={best_val:.6g}@{best_epoch}, completed_val_epochs={len(hist.get('val_mse', []))}."
        )
    elif args.resume:
        log_message(f"[matched_operator] --resume requested but checkpoint does not exist: {args.operator_path}. Starting from scratch.")
    phase_t0 = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_t0 = time.time()
            running = 0.0
            for bi, batch in enumerate(train_loader, start=1):
                sos = batch[0].to(device).float().unsqueeze(1)
                tof = batch[1].to(device).float()
                tof_mask = batch[2].to(device).float() if len(batch) > 2 and torch.is_tensor(batch[2]) and batch[2].shape == batch[1].shape else torch.ones_like(tof)
                sos = support_apply(sos, support_mask, args.water_norm)
                pred = model(sos)
                loss = tof_loss(pred, tof, tof_mask, args)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += float(loss.detach().item())
                if args.batch_log_every > 0 and (bi == 1 or bi % args.batch_log_every == 0 or bi == len(train_loader)):
                    log_message(batch_progress("[matched_operator]", epoch, args.epochs, bi, len(train_loader), epoch_t0, phase_t0) + f" loss={float(loss.detach().item()):.6g}")
            val = evaluate(model, val_loader, device, support_mask, args)
            train_loss = running / max(1, len(train_loader))
            hist["train"].append(train_loss)
            hist["val_mse"].append(val["mse"])
            hist["val_mae_us"].append(val["mae_us"])
            hist["val_rmse_us"].append(val["rmse_us"])
            if val["mse"] < best_val - args.early_stop_min_delta:
                best_val = float(val["mse"])
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                save_ckpt(args.operator_path, args, model, best_state, best_val, best_epoch, epoch, hist, opt)
                log_message(f"[matched_operator] New best: epoch={epoch}, val_mse={best_val:.6g}, val_mae_us={val['mae_us']:.4g}")
            else:
                stale += 1
            elapsed = time.time() - phase_t0
            eta = elapsed / max(1, epoch) * max(0, args.epochs - epoch)
            log_message(f"[matched_operator] epoch {epoch:04d}: train={train_loss:.6g} val_mse={val['mse']:.6g} val_mae_us={val['mae_us']:.4g} val_rmse_us={val['rmse_us']:.4g} best={best_val:.6g}@{best_epoch} stale={stale}/{args.early_stop_patience} ETA={format_duration(eta)}")
            save_ckpt(args.operator_path, args, model, best_state, best_val, best_epoch, epoch, hist, opt)
            if args.preview_every > 0 and (epoch == 1 or epoch % args.preview_every == 0):
                plot_preview(model, val_loader, device, support_mask, args, f"Matched ToF operator epoch {epoch}")
            if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                log_message(f"[matched_operator] Early stopping at epoch {epoch}. Best epoch={best_epoch}, val_mse={best_val:.6g}.")
                break
    except KeyboardInterrupt:
        log_message("[matched_operator] Ctrl-C received. Saving current and best operator checkpoint.")
        save_ckpt(args.operator_path, args, model, best_state, best_val, best_epoch, len(hist["val_mse"]), hist, opt)
        return

    model.load_state_dict(best_state)
    plot_preview(model, val_loader, device, support_mask, args, f"Matched ToF operator best epoch {best_epoch}")
    save_ckpt(args.operator_path, args, model, best_state, best_val, best_epoch, len(hist["val_mse"]), hist, opt)
    log_message(f"[matched_operator] Done. Best epoch={best_epoch}, best val_mse={best_val:.6g}. Saved: {args.operator_path}")


if __name__ == "__main__":
    main()










