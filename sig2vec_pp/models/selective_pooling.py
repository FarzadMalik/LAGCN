"""Multi-head selective pooling layer from Sig2Vec.

Implements an attention-based temporal pooling mechanism where each head
independently learns a query (key) vector and produces a weighted average
over the temporal dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectivePooling(nn.Module):
    """Attention pooling with *num_heads* independent heads.

    For each head the module learns a key vector.  A 1×1 convolution projects
    the input to queries/values, attention weights are computed via scaled dot
    product, and the output is the weighted sum over time.

    Parameters
    ----------
    in_dim : int
        Number of input channels.
    head_dim : int
        Dimensionality of each attention head.
    num_heads : int
        Number of independent attention heads.
    """

    def __init__(self, in_dim: int, head_dim: int = 32, num_heads: int = 16):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1.0 / head_dim ** 0.5

        self.keys = nn.Parameter(torch.empty(num_heads, head_dim))
        self.w_q = nn.Conv1d(in_dim, head_dim * num_heads, kernel_size=1)

        nn.init.orthogonal_(self.keys, gain=1)
        nn.init.kaiming_normal_(self.w_q.weight, a=1)
        nn.init.zeros_(self.w_q.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C, T)
        mask : (N, T)  — 1.0 for valid positions, 0.0 for padding.

        Returns
        -------
        pooled : (N, num_heads * head_dim)
        """
        N, _, T = x.shape
        qv = self.w_q(x).transpose(1, 2).view(N, T, self.num_heads, self.head_dim)
        scores = (qv * self.keys).sum(dim=-1) * self.scale  # (N, T, H)
        scores = scores - (1.0 - mask).unsqueeze(2) * 1e3
        attn = F.softmax(scores, dim=1)  # (N, T, H)
        pooled = (qv * attn.unsqueeze(3)).sum(dim=1).reshape(N, -1)
        return pooled

    def ortho_reg(self) -> torch.Tensor:
        """Orthogonality regularisation on the key vectors."""
        keys_normed = F.normalize(self.keys, dim=1)
        corr = keys_normed @ keys_normed.T
        return torch.triu(corr, diagonal=1).abs().sum()
