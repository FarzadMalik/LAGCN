#!/usr/bin/env python
"""Training script for the Sig2Vec 1D-CNN baseline on MSDS-ChS.

Usage
-----
    python -m sig2vec_pp.experiments.train_sig2vec \\
        --config sig2vec_pp/configs/sig2vec_baseline.yaml \\
        --device 0

Or override individual parameters on the CLI::

    python -m sig2vec_pp.experiments.train_sig2vec \\
        --config sig2vec_pp/configs/sig2vec_baseline.yaml \\
        --epochs 100 --lr 0.0005
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sig2vec_pp.configs.config import load_config, merge_cli_args
from sig2vec_pp.data.msds_dataset import (
    MSDSOnlineDataset,
    TaskBatchSampler,
    collate_online,
)
from sig2vec_pp.models.sig2vec import Sig2Vec
from sig2vec_pp.utils.logging import setup_logger, log_metrics
from sig2vec_pp.utils.metrics import compute_eer


# ─── CLI ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Sig2Vec baseline")
    p.add_argument("--config", type=str, default="sig2vec_pp/configs/sig2vec_baseline.yaml")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    return p


# ─── main ────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_args(cfg, args)

    seed = cfg.get("seed", 111)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cfg.get("deterministic", True):
        cudnn.benchmark = False
        cudnn.deterministic = True

    logger = setup_logger("sig2vec_pp")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── dataset ──────────────────────────────────────────────────────
    ds_cfg = cfg["dataset"]
    split_cfg = cfg["split"]
    all_users = list(range(ds_cfg.get("num_users", 402)))
    rng = np.random.RandomState(split_cfg.get("seed", 42))
    rng.shuffle(all_users)

    n_train = split_cfg["train_users"]
    n_val = split_cfg["val_users"]
    train_users = all_users[:n_train]
    val_users = all_users[n_train : n_train + n_val]

    feature_names = ds_cfg.get("features", ["x", "y", "p"])
    max_seq_len = ds_cfg.get("max_seq_len", 800)

    train_ds = MSDSOnlineDataset(
        root=ds_cfg["root"],
        sessions=ds_cfg.get("sessions", [1, 2]),
        user_ids=train_users,
        feature_names=feature_names,
        max_seq_len=max_seq_len,
    )
    logger.info("Computing global stats on training set ...")
    global_mean, global_std = train_ds.compute_global_stats()

    val_ds = MSDSOnlineDataset(
        root=ds_cfg["root"],
        sessions=ds_cfg.get("sessions", [1, 2]),
        user_ids=val_users,
        feature_names=feature_names,
        max_seq_len=max_seq_len,
        global_stats=(global_mean, global_std),
    )

    train_cfg = cfg["training"]
    n_tasks = train_cfg.get("batch_tasks", 4)
    n_gen = train_cfg.get("shots_genuine", 5)
    n_forg = train_cfg.get("shots_forgery", 10)

    sampler = TaskBatchSampler(
        train_ds, n_tasks=n_tasks, n_genuine=n_gen, n_forgery=n_forg, seed=seed
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate_online, num_workers=0
    )
    logger.info("Train: %d samples, %d batches/epoch", len(train_ds), len(sampler))

    # ── model ────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = Sig2Vec(
        n_in=model_cfg.get("n_in", len(feature_names)),
        n_classes=len(train_users),
        n_task=n_tasks,
        n_shot_g=n_gen,
        n_shot_f=n_forg,
        ap_alpha=train_cfg.get("ap_loss_alpha", 6.0),
    ).to(device)

    opt_cfg = train_cfg["optimizer"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt_cfg.get("lr", 0.001),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
        nesterov=opt_cfg.get("nesterov", True),
    )

    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=sched_cfg.get("step_size", 150),
        gamma=sched_cfg.get("gamma", 0.1),
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

    ckpt_dir = Path(cfg["logging"].get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_interval = cfg["logging"].get("log_interval", 50)
    save_interval = cfg["logging"].get("save_interval", 25)
    eps = train_cfg.get("label_smoothing", 0.1)
    use_wandb = cfg["logging"].get("use_wandb", False)

    if use_wandb:
        try:
            import wandb

            wandb.init(project=cfg["logging"].get("wandb_project", "sig2vec_pp"), config=cfg)
        except ImportError:
            logger.warning("wandb not installed — disabling remote logging")
            use_wandb = False

    # ── training loop ────────────────────────────────────────────────
    epochs = train_cfg.get("epochs", cfg.get("epochs", 200))
    if args.epochs is not None:
        epochs = args.epochs

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_map = 0.0
        n_logged = 0

        for batch_idx, (sigs, lens, labels) in enumerate(train_loader):
            mask = model.get_output_mask(lens)

            sigs_t = torch.from_numpy(sigs).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            lens_t = torch.from_numpy(lens).to(device)
            labels_t = torch.from_numpy(labels).to(device)

            optimizer.zero_grad()

            embeddings, logits = model(sigs_t, mask_t, lens_t)

            loss_ce = model.smooth_ce_loss(logits, labels_t, eps=eps)
            loss_ap, mAP = model.ap_loss_dlm(embeddings)
            loss = loss_ap + loss_ce
            loss.backward()

            optimizer.step()

            running_loss += loss_ap.item()
            running_ce += loss_ce.item()
            running_map += mAP
            n_logged += 1

            if (batch_idx + 1) % log_interval == 0:
                avg = {
                    "ap_loss": running_loss / n_logged,
                    "ce_loss": running_ce / n_logged,
                    "mAP": running_map / n_logged,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                global_step = epoch * len(sampler) + batch_idx
                log_metrics(avg, step=global_step, logger=logger, use_wandb=use_wandb)
                running_loss = running_ce = running_map = 0.0
                n_logged = 0

        scheduler.step()

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            path = ckpt_dir / f"sig2vec_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                },
                path,
            )
            logger.info("Checkpoint saved: %s", path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
