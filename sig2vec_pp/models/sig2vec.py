"""Sig2Vec 1D-CNN model — faithful re-implementation of the original
architecture from Lai et al. (IEEE TPAMI 2021).

Architecture summary
--------------------
Three 1D-CNN blocks (conv → pool → SELU → conv → SELU), followed by an
FPN-style upsample + skip connection between block-2 and block-3,
multi-head selective pooling on two feature maps, BN, and a linear
classifier head.

The forward pass returns:
  * ``embedding`` — L2-normalisable fixed-length vector (dim = 1024)
  * ``logits``    — class logits for auxiliary cross-entropy training
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_pooling import SelectivePooling


class Sig2Vec(nn.Module):
    """1D-CNN signature embedding network.

    Parameters
    ----------
    n_in : int
        Number of input features per time-step (e.g. 3 for X,Y,P).
    n_classes : int
        Number of writer identities (used for auxiliary CE head).
    n_task : int
        Number of episodic tasks per batch.
    n_shot_g : int
        Genuine reference shots per task.
    n_shot_f : int
        Forgery reference shots per task.
    ap_alpha : float
        Scaling factor for the AP-DLM loss.
    """

    def __init__(
        self,
        n_in: int = 3,
        n_classes: int = 300,
        n_task: int = 4,
        n_shot_g: int = 5,
        n_shot_f: int = 10,
        ap_alpha: float = 6.0,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_task = n_task
        self.n_shot_g = n_shot_g
        self.n_shot_f = n_shot_f
        self.ap_alpha = ap_alpha
        self.epsilon = 1.0

        # ── 1-D ConvNet blocks ───────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv1d(n_in, 64, kernel_size=7, padding=3),
            nn.MaxPool1d(2, 2, ceil_mode=True),
            nn.SELU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(2, 2, ceil_mode=True),
            nn.SELU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool1d(2, 2, ceil_mode=True),
            nn.SELU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
        )

        # ── FPN-style skip connection ─────────────────────────────────
        self.upsample = nn.Upsample(scale_factor=2)
        self.block2_proj = nn.Conv1d(128, 256, kernel_size=1)

        # ── Selective pooling heads ───────────────────────────────────
        self.sp2 = SelectivePooling(256, head_dim=32, num_heads=16)  # → 512
        self.sp3 = SelectivePooling(256, head_dim=32, num_heads=16)  # → 512

        # ── Classifier ────────────────────────────────────────────────
        self.bn = nn.BatchNorm1d(1024, affine=False, momentum=0.001)
        self.cls = nn.Linear(1024, n_classes, bias=False)

        self._init_weights()

    # ── weight initialisation (matches original) ─────────────────────

    def _init_weights(self):
        for m in [
            self.block1[0], self.block1[3],
            self.block2[0], self.block2[3],
            self.block3[0], self.block3[3],
        ]:
            nn.init.kaiming_normal_(m.weight, a=1)
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.block2_proj.weight, a=0)
        nn.init.zeros_(self.block2_proj.bias)
        nn.init.kaiming_normal_(self.cls.weight, a=1)

    # ── helpers ──────────────────────────────────────────────────────

    def get_output_mask(self, lengths: np.ndarray) -> np.ndarray:
        """Build a float mask for the temporal positions *after* two
        pooling operations (each halves the length, rounding up)."""
        lens = np.asarray(lengths, dtype=np.int32)
        for _ in range(2):
            lens = (lens + 1) // 2
        N = len(lens)
        D = int(lens.max())
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, : lens[i]] = 1.0
        return mask

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (N, T, D) — padded time-series batch.
        mask : (N, T_pool2) — valid-position mask at block-2 resolution.
        lengths : (N,) — original un-padded lengths.

        Returns
        -------
        embedding : (N, 1024)
        logits : (N, n_classes)
        """
        x = x.transpose(1, 2)  # (N, D, T)

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        fused2 = F.selu(
            self.upsample(out3)[:, :, : out2.shape[2]] + self.block2_proj(out2),
            inplace=True,
        )

        mask2 = mask
        feat2 = self.sp2(fused2, mask2)

        mask3 = F.max_pool1d(mask2.unsqueeze(1), 2, ceil_mode=True).squeeze(1)
        feat3 = self.sp3(out3, mask3)

        embedding = torch.cat([feat2, feat3], dim=1)
        embedding = self.bn(embedding)

        logits = self.cls(embedding)
        return embedding, logits

    # ── loss functions ───────────────────────────────────────────────

    def smooth_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """Label-smoothed cross-entropy."""
        n_cls = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, targets.view(-1, 1), 1)
        smooth = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_cls - 1)
        log_prb = F.log_softmax(logits, dim=1)
        return -(smooth * log_prb).sum(dim=1).mean()

    def ap_loss_dlm(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Direct-Loss-Minimisation Average-Precision loss.

        Operates on episodic batches structured as
        ``[anchor, g_1, …, g_K, f_1, …, f_M]`` repeated *n_task* times.
        """
        step = 1 + self.n_shot_g + self.n_shot_f
        total_score_std = 0.0
        total_score_aug = 0.0
        total_map = 0.0

        for i in range(self.n_task):
            anchor = embeddings[i * step]
            pos = embeddings[i * step + 1 : i * step + 1 + self.n_shot_g]
            neg = embeddings[i * step + 1 + self.n_shot_g : (i + 1) * step]

            la = anchor.norm() + 1e-8
            lp = pos.norm(dim=1) + 1e-8
            ln = neg.norm(dim=1) + 1e-8

            score_ap = (anchor.unsqueeze(0) * pos).sum(dim=1) / (la * lp)
            score_an = (anchor.unsqueeze(0) * neg).sum(dim=1) / (la * ln)

            score_ap, _ = score_ap.sort(descending=True)
            score_an, _ = score_an.sort(descending=True)

            diff = score_ap.unsqueeze(1) - score_an.unsqueeze(0)
            y_std = torch.sign(diff)
            total_score_std += (y_std * diff).mean()

            direction = self._loss_augmented_inference(score_ap, score_an)
            total_map += self._compute_ap(direction)
            y_aug = -1 * direction[1:, 1:]
            total_score_aug += (y_aug.to(embeddings.device) * diff).mean()

        score_std = total_score_std / self.n_task
        score_aug = total_score_aug / self.n_task
        mAP = total_map / self.n_task

        loss = (1.0 / self.epsilon) * (self.ap_alpha * score_aug - score_std)
        return loss, mAP

    # ── AP loss helpers ──────────────────────────────────────────────

    @staticmethod
    def _loss_augmented_inference(
        score_pos: torch.Tensor,
        score_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy loss-augmented inference for AP surrogate.

        Returns a direction matrix of shape (1+K, 1+M) on CPU.
        """
        K = len(score_pos)
        M = len(score_neg)
        sp = score_pos.detach().cpu().numpy()
        sn = score_neg.detach().cpu().numpy()

        direction = np.zeros((K + 1, M + 1), dtype=np.float32)
        i = j = 0
        tp = 0
        while i < K and j < M:
            if sp[i] >= sn[j]:
                tp += 1
                i += 1
            else:
                rank = tp + j + 1
                delta = tp / rank if tp > 0 else 0.0
                direction[i, j + 1] = 1.0
                for k in range(i):
                    direction[k + 1, j + 1] = -delta
                j += 1
        while j < M:
            rank = tp + j + 1
            delta = tp / rank if tp > 0 else 0.0
            direction[i, j + 1] = 1.0
            for k in range(i):
                direction[k + 1, j + 1] = -delta
            j += 1
        return torch.from_numpy(direction)

    @staticmethod
    def _compute_ap(direction: torch.Tensor) -> float:
        """Compute AP from the direction matrix."""
        K = direction.shape[0] - 1
        M = direction.shape[1] - 1
        if K == 0:
            return 0.0
        tp = 0
        ap = 0.0
        i = j = 0
        while i < K or j < M:
            if i < K and (j >= M or direction[i + 1, j + 1 if j + 1 <= M else M].item() <= 0):
                tp += 1
                ap += tp / (tp + j)
                i += 1
            elif j < M:
                j += 1
            else:
                break
        return ap / K
