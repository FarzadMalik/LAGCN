#!/usr/bin/env python
"""Evaluate a trained Sig2Vec model on the MSDS-ChS test split.

Extracts embeddings, computes cosine-similarity scores for all genuine-
vs-genuine and genuine-vs-forgery pairs per user, and reports EER and
FNMR@FMR metrics.

Usage
-----
    python -m sig2vec_pp.experiments.evaluate \\
        --config sig2vec_pp/configs/sig2vec_baseline.yaml \\
        --weights checkpoints/sig2vec_epoch0199.pt \\
        --device 0
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sig2vec_pp.configs.config import load_config
from sig2vec_pp.data.msds_dataset import MSDSOnlineDataset
from sig2vec_pp.models.sig2vec import Sig2Vec
from sig2vec_pp.utils.logging import setup_logger
from sig2vec_pp.utils.metrics import compute_eer, compute_fnmr_at_fmr


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Sig2Vec")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--device", type=int, default=0)
    return p


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Return dict mapping (user, flag, index) → embedding vector."""
    model.eval()
    embeddings: dict = {}
    for sigs, lens, labels, users in dataloader:
        if isinstance(sigs, np.ndarray):
            sigs = torch.from_numpy(sigs)
        sigs = sigs.to(device)
        mask_np = model.get_output_mask(lens if isinstance(lens, np.ndarray) else lens.numpy())
        mask = torch.from_numpy(mask_np).to(device)
        lens_t = torch.from_numpy(lens if isinstance(lens, np.ndarray) else lens.numpy()).to(device)

        emb, _ = model(sigs, mask, lens_t)
        emb = emb.cpu().numpy()
        for i in range(len(emb)):
            embeddings[(int(users[i]), int(labels[i]), i)] = emb[i]
    return embeddings


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("sig2vec_eval")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ds_cfg = cfg["dataset"]
    split_cfg = cfg["split"]
    all_users = list(range(ds_cfg.get("num_users", 402)))
    rng = np.random.RandomState(split_cfg.get("seed", 42))
    rng.shuffle(all_users)
    n_train = split_cfg["train_users"]
    n_val = split_cfg["val_users"]
    test_users = all_users[n_train + n_val:]

    ckpt = torch.load(args.weights, map_location=device)
    model_cfg = cfg["model"]

    model = Sig2Vec(
        n_in=model_cfg.get("n_in", 3),
        n_classes=ckpt["model"]["cls.weight"].shape[0],
        n_task=1,
        n_shot_g=1,
        n_shot_f=1,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    feature_names = ds_cfg.get("features", ["x", "y", "p"])
    test_ds = MSDSOnlineDataset(
        root=ds_cfg["root"],
        sessions=ds_cfg.get("sessions", [1, 2]),
        user_ids=test_users,
        feature_names=feature_names,
        max_seq_len=ds_cfg.get("max_seq_len", 800),
    )

    def _collate(batch):
        sigs, lens, labels, users = zip(*batch)
        sigs = np.stack([s.numpy() for s in sigs]).astype(np.float32)
        lens = np.array(lens, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)
        users = np.array(users, dtype=np.int64)
        return sigs, lens, labels, users

    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=_collate)

    logger.info("Extracting embeddings for %d test samples ...", len(test_ds))
    all_emb = extract_embeddings(model, loader, device)

    user_genuine = defaultdict(list)
    user_forgery = defaultdict(list)
    for (uid, label, _), emb in all_emb.items():
        if label == 1:
            user_genuine[uid].append(emb)
        else:
            user_forgery[uid].append(emb)

    gen_scores = []
    imp_scores = []

    for uid in user_genuine:
        g = np.array(user_genuine[uid])
        f = np.array(user_forgery.get(uid, []))
        g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)

        for i in range(len(g_norm)):
            for j in range(i + 1, len(g_norm)):
                gen_scores.append(float(g_norm[i] @ g_norm[j]))

        if len(f) > 0:
            f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            for i in range(len(g_norm)):
                for j in range(len(f_norm)):
                    imp_scores.append(float(g_norm[i] @ f_norm[j]))

    gen_scores = np.array(gen_scores)
    imp_scores = np.array(imp_scores)

    eer = compute_eer(gen_scores, imp_scores)
    fnmr_001 = compute_fnmr_at_fmr(gen_scores, imp_scores, target_fmr=0.01)
    fnmr_0001 = compute_fnmr_at_fmr(gen_scores, imp_scores, target_fmr=0.001)

    logger.info("EER:            %.4f", eer)
    logger.info("FNMR@FMR=0.01:  %.4f", fnmr_001)
    logger.info("FNMR@FMR=0.001: %.4f", fnmr_0001)


if __name__ == "__main__":
    main()
