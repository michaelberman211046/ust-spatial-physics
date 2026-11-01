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
    ShapeAutoencoder,
    batch_progress,
    format_duration,
    grad_l1,
    masked_l1,
    masked_mse,
    support_apply,
    tv_l1,
)
from logger import log_image, log_message
from settings import set_output_folder
from reconstruction_utils import make_circular_support_mask


SCRIPT_VERSION = "2026-06-29-train-latent-sos-autoencoder-v1"


def split_dataset(ds, data_path, splits_path):
    if splits_path is None:
        splits_path = make_cache_paths(data_path)["splits_path"]
    if not os.path.exists(splits_path):
        n = len(ds)
        return Subset(ds, list(range(int(0.8 * n)))), Subset(ds, list(range(int(0.8 * n), n)))
    splits = load_splits(splits_path)
    return Subset(ds, list(map(int, splits["train_idx"]))), Subset(ds, list(map(int, splits["val_idx"])))


def evaluate(model, loader, device, support_mask, args):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            target = batch[0].to(device).float().unsqueeze(1)
            target = support_apply(target, support_mask, args.water_norm)
            pred = support_apply(model(target), support_mask, args.water_norm)
            total += float(masked_mse(pred, target, support_mask).item())
            n += 1
    return total / max(1, n)


def plot_preview(model, loader, device, support_mask, args, title):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        target = batch[0].to(device).float().unsqueeze(1)
        target = support_apply(target, support_mask, args.water_norm)
        pred = support_apply(model(target), support_mask, args.water_norm)
    b = min(4, target.shape[0])
    vmin, vmax = float(args.sos_min), float(args.sos_max)
    extent = [0.0, float(args.phys_x), float(args.phys_y), 0.0]
    fig, axes = plt.subplots(b, 3, figsize=(12, 4 * b))
    if b == 1:
        axes = axes[None, :]
    for i in range(b):
        gt = target[i, 0].cpu() * (vmax - vmin) + vmin
        pr = pred[i, 0].cpu() * (vmax - vmin) + vmin
        er = (pred[i, 0] - target[i, 0]).abs().cpu() * (vmax - vmin)
        for ax, img, name, cmap, lo, hi in [
            (axes[i, 0], gt, "GT SoS", "gray", vmin, vmax),
            (axes[i, 1], pr, "AE reconstruction", "gray", vmin, vmax),
            (axes[i, 2], er, "|error| [m/s]", "magma", 0.0, None),
        ]:
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=extent, origin="upper", aspect="equal")
            ax.set_title(name)
            ax.set_xlabel("Lateral [m]")
            ax.set_ylabel("Axial [m]")
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
            "kind": "latent_sos_autoencoder",
            "config": vars(args),
            "autoencoder_state_dict": model.state_dict(),
            "best_autoencoder_state_dict": best_state,
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
    p.add_argument("--ae_path", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--latent_ch", type=int, default=64)
    p.add_argument("--latent_grid", type=int, default=32)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=0.7)
    p.add_argument("--lambda_grad", type=float, default=0.5)
    p.add_argument("--lambda_tv", type=float, default=0.002)
    p.add_argument("--early_stop_patience", type=int, default=14)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--preview_every", type=int, default=5)
    p.add_argument("--batch_log_every", type=int, default=25)
    p.add_argument("--use_circular_mask", action="store_true")
    p.add_argument("--mask_radius", type=float, default=0.1091)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--phys_x", type=float, default=0.24)
    p.add_argument("--phys_y", type=float, default=0.24)
    p.add_argument("--sos_min", type=float, default=1380.0)
    p.add_argument("--sos_max", type=float, default=1660.0)
    p.add_argument("--sos_water", type=float, default=1500.0)
    return p.parse_args()


def main():
    global args
    args = parse_args()
    if args.output_dir:
        set_output_folder(args.output_dir)
    log_message("...................")
    log_message(f"[CMD] {' '.join(sys.argv)}")
    log_message(f"[ARGS]\n{pprint.pformat(vars(args))}")
    log_message(f"[latent_ae] script_version={SCRIPT_VERSION}")

    requested = str(args.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        msg = "[latent_ae] CUDA requested but unavailable. Actual device will be CPU."
        if args.require_cuda:
            raise RuntimeError(msg)
        log_message(msg)
        requested = "cpu"
    device = torch.device(requested)
    log_message(f"[latent_ae] Actual device used: {device}")

    ds = PairsCacheDataset(args.data_path, sos_min=args.sos_min, sos_max=args.sos_max, return_tof_mask=True)
    meta = dict(getattr(ds, "metadata", {}) or {})
    args.nx = int(meta.get("nx", args.nx))
    args.ny = int(meta.get("ny", args.ny))
    args.phys_x = float(meta.get("phys_x", args.phys_x))
    args.phys_y = float(meta.get("phys_y", args.phys_y))
    args.sos_water = float(meta.get("sos_water", args.sos_water))
    args.water_norm = float(max(0.0, min(1.0, (args.sos_water - args.sos_min) / (args.sos_max - args.sos_min))))
    train_ds, val_ds = split_dataset(ds, args.data_path, args.splits_path)
    if args.max_train_samples > 0:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples > 0:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    log_message(f"[latent_ae] Dataset: train={len(train_ds)}, val={len(val_ds)}, batch_size={args.batch_size}")

    support_mask = None
    if args.use_circular_mask:
        support_mask = make_circular_support_mask(args.nx, args.ny, args.phys_x, args.phys_y, args.mask_radius, device=device)
        log_message(f"[latent_ae] Circular support enabled: radius={args.mask_radius}")

    model = ShapeAutoencoder(args.latent_ch, args.latent_grid, args.channels, (args.nx, args.ny), args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = {"train": [], "val": []}
    best_val, best_epoch, stale = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    start_epoch = 1
    if args.resume and os.path.exists(args.ae_path):
        ckpt = torch.load(args.ae_path, map_location=device, weights_only=False)
        if ckpt.get("kind") != "latent_sos_autoencoder":
            raise ValueError(f"--resume found incompatible checkpoint: {args.ae_path}")
        state = ckpt.get("autoencoder_state_dict")
        if state is None:
            raise ValueError(f"Checkpoint missing autoencoder_state_dict: {args.ae_path}")
        model.load_state_dict(state)
        best_state = copy.deepcopy(ckpt.get("best_autoencoder_state_dict", state))
        best_val = float(ckpt.get("best_val_mse", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        hist = copy.deepcopy(ckpt.get("history", hist))
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        start_epoch = int(ckpt.get("last_epoch", len(hist.get("val", [])))) + 1
        start_epoch = max(1, min(start_epoch, int(args.epochs) + 1))
        log_message(
            f"[latent_ae] Resuming from {args.ae_path}: start_epoch={start_epoch}, "
            f"best={best_val:.6g}@{best_epoch}, completed_val_epochs={len(hist.get('val', []))}."
        )
    elif args.resume:
        log_message(f"[latent_ae] --resume requested but checkpoint does not exist: {args.ae_path}. Starting from scratch.")
    phase_t0 = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_t0 = time.time()
            running = 0.0
            for bi, batch in enumerate(train_loader, start=1):
                target = batch[0].to(device).float().unsqueeze(1)
                target = support_apply(target, support_mask, args.water_norm)
                pred = support_apply(model(target), support_mask, args.water_norm)
                loss = (
                    args.lambda_l1 * masked_l1(pred, target, support_mask)
                    + args.lambda_mse * masked_mse(pred, target, support_mask)
                    + args.lambda_grad * grad_l1(pred, target, support_mask)
                    + args.lambda_tv * tv_l1(pred, support_mask)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += float(loss.detach().item())
                if args.batch_log_every > 0 and (bi == 1 or bi % args.batch_log_every == 0 or bi == len(train_loader)):
                    log_message(batch_progress("[latent_ae]", epoch, args.epochs, bi, len(train_loader), epoch_t0, phase_t0) + f" loss={float(loss.detach().item()):.6g}")
            train_loss = running / max(1, len(train_loader))
            val_mse = evaluate(model, val_loader, device, support_mask, args)
            hist["train"].append(train_loss)
            hist["val"].append(val_mse)
            if val_mse < best_val - args.early_stop_min_delta:
                best_val = float(val_mse)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                save_ckpt(args.ae_path, args, model, best_state, best_val, best_epoch, epoch, hist, opt)
                log_message(f"[latent_ae] New best: epoch={epoch}, val_mse={best_val:.6g}")
            else:
                stale += 1
            elapsed = time.time() - phase_t0
            eta = elapsed / max(1, epoch) * max(0, args.epochs - epoch)
            log_message(f"[latent_ae] epoch {epoch:04d}: train={train_loss:.6g} val_mse={val_mse:.6g} best={best_val:.6g}@{best_epoch} stale={stale}/{args.early_stop_patience} epoch_time={format_duration(time.time() - epoch_t0)} ETA={format_duration(eta)}")
            save_ckpt(args.ae_path, args, model, best_state, best_val, best_epoch, epoch, hist, opt)
            if args.preview_every > 0 and (epoch == 1 or epoch % args.preview_every == 0):
                plot_preview(model, val_loader, device, support_mask, args, f"Latent AE epoch {epoch}")
            if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                log_message(f"[latent_ae] Early stopping at epoch {epoch}. Best epoch={best_epoch}, val_mse={best_val:.6g}.")
                break
    except KeyboardInterrupt:
        log_message("[latent_ae] Ctrl-C received. Saving current and best AE checkpoint.")
        save_ckpt(args.ae_path, args, model, best_state, best_val, best_epoch, len(hist["val"]), hist, opt)
        return

    model.load_state_dict(best_state)
    plot_preview(model, val_loader, device, support_mask, args, f"Latent AE best epoch {best_epoch}")
    save_ckpt(args.ae_path, args, model, best_state, best_val, best_epoch, len(hist["val"]), hist, opt)
    log_message(f"[latent_ae] Done. Best epoch={best_epoch}, best val_mse={best_val:.6g}. Saved: {args.ae_path}")


if __name__ == "__main__":
    main()










