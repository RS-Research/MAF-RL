# maf_rl/recbole_ext/models/fusion.py
# -*- coding: utf-8 -*-
"""
Multi-Modal Fusion Modules for MAF-RL (RecBole Extension)

This module provides a gated-attention fusion mechanism that combines:
- base behavioral sequence embeddings (from item id embedding)
with optional modality embeddings such as:
- text, image, graph, and context (each projected to the same hidden size)

Design goals
------------
- RecBole-friendly (pure PyTorch, no external deps)
- Robust to missing modalities (can accept any subset)
- Lightweight: suitable for large-scale sequential recommendation training

Usage
-----
from maf_rl.recbole_ext.models.fusion import GatedAttentionFusion

fuser = GatedAttentionFusion(hidden_size=H, dropout=0.1)
fused_seq = fuser(base_seq_emb, {"text": text_emb, "image": image_emb})
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class GatedAttentionFusion(nn.Module):
    """
    Gated-attention fusion at the token (sequence position) level.

    Inputs
    ------
    base_seq : torch.Tensor
        Shape [B, L, H]. Behavioral embedding sequence (item-id embedding backbone).
    modal_seq : Dict[str, torch.Tensor]
        Dict of modality embeddings. Each tensor should be [B, L, H].

    Output
    ------
    fused_seq : torch.Tensor
        Shape [B, L, H]. Fused sequence embeddings.

    Mechanism (high level)
    ----------------------
    1) For each token position (b, t), build a list:
         [base, m1, m2, ...]
    2) Compute attention weights over sources using a small scorer network
       conditioned on (base + modality).
    3) Weighted sum of sources produces fused embedding.
    4) Apply gated residual update:
         fused = gate * fused + (1 - gate) * base
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = int(hidden_size)

        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.hidden_size)

        # Score each source embedding for attention weighting.
        # Uses a simple MLP on the concatenation [base, source, base*source, base-source]
        self.score_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, 1),
        )

        # Gate controls how much to deviate from the base embedding.
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )

        # Optional post-fusion projection (kept as identity-ish by default)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(float(dropout)),
        )

    def forward(self, base_seq: torch.Tensor, modal_seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        base_seq : Tensor[B, L, H]
        modal_seq : dict[str, Tensor[B, L, H]]

        Returns
        -------
        Tensor[B, L, H]
        """
        if not modal_seq:
            return base_seq

        # Ensure consistent shapes and device
        B, L, H = base_seq.shape
        sources = [base_seq]
        for k, v in modal_seq.items():
            if v is None:
                continue
            if v.shape != base_seq.shape:
                raise ValueError(
                    f"Modality '{k}' must match base_seq shape [B,L,H]={tuple(base_seq.shape)}, got {tuple(v.shape)}"
                )
            sources.append(v)

        if len(sources) == 1:
            return base_seq

        # Stack sources: [B, L, S, H]
        src = torch.stack(sources, dim=2)

        # Attention scoring: compare each source to base
        # base expand: [B, L, S, H]
        base_exp = base_seq.unsqueeze(2).expand_as(src)

        # Features for scoring: [base, src, base*src, base-src] -> [B,L,S,4H]
        score_in = torch.cat([base_exp, src, base_exp * src, base_exp - src], dim=-1)

        # Compute unnormalized scores: [B,L,S,1] -> [B,L,S]
        logits = self.score_mlp(self.dropout(score_in)).squeeze(-1)

        # Normalize over sources: [B,L,S]
        alpha = torch.softmax(logits, dim=2)

        # Weighted sum: sum over S -> [B,L,H]
        fused = torch.sum(alpha.unsqueeze(-1) * src, dim=2)

        # Gated residual update: gate in [0,1] per dimension
        g = self.gate(torch.cat([base_seq, fused], dim=-1))  # [B,L,H]
        out = g * fused + (1.0 - g) * base_seq

        # Post projection + norm
        out = self.out_proj(out)
        out = self.norm(out)
        return out
