# maf_rl/recbole_ext/utils/candidates.py
# -*- coding: utf-8 -*-
"""
Candidate Set Construction Utilities for MAF-RL (RecBole Extension)

In offline sequential recommendation with PPO, we commonly compute policy
probabilities over a *candidate set* rather than the full item vocabulary,
especially for large datasets.

This module provides:
- build_candidate_set: {positive item} ∪ {sampled negatives}

Design
------
- Ensures candidates[:, 0] is always the positive (ground-truth) item.
- Samples negatives uniformly by default, with optional filtering:
  - avoid padding id 0
  - avoid positive item id
  - optionally avoid items appearing in the user's recent history

You can later replace the sampler with:
- popularity-based sampling
- in-batch negatives
- approximate nearest-neighbor candidate generation
"""

from __future__ import annotations

from typing import Optional

import torch
from recbole.data.interaction import Interaction


def build_candidate_set(
    interaction: Interaction,
    pos_item: torch.Tensor,
    n_items: int,
    candidate_size: int = 100,
    device: Optional[torch.device] = None,
    item_id_field: str = "item_id",
    item_seq_field: str = "item_id_list",
    avoid_seen: bool = True,
    max_resample: int = 10,
) -> torch.Tensor:
    """
    Build per-sample candidate set.

    Parameters
    ----------
    interaction : Interaction
        RecBole Interaction batch.
    pos_item : torch.Tensor
        Positive/ground-truth next item ids, shape [B].
    n_items : int
        Total number of items in RecBole internal id space (includes padding=0).
    candidate_size : int
        Total candidates per sample (including the positive). Must be >= 2.
    device : torch.device, optional
        Device for tensors. If None, inferred from pos_item.
    item_seq_field : str
        Name of the sequence field in Interaction (defaults to "item_id_list").
    avoid_seen : bool
        If True, attempt to avoid sampling items that appear in the user's history.
    max_resample : int
        Max resampling iterations for filtering.

    Returns
    -------
    candidates : torch.Tensor
        Shape [B, C] where candidates[:, 0] == pos_item and C == candidate_size.
    """
    if candidate_size < 2:
        raise ValueError("candidate_size must be >= 2")

    if device is None:
        device = pos_item.device

    B = pos_item.size(0)
    C = candidate_size

    # Optionally build a "seen" mask set per row from history
    seen = None
    if avoid_seen and item_seq_field in interaction:
        # item history sequence: [B, L]
        seq = interaction[item_seq_field].to(device)
        seen = seq  # store ids; we'll filter via comparisons

    # Sample negatives uniformly in [1, n_items-1] (avoid padding=0)
    neg = torch.randint(1, n_items, (B, C - 1), device=device)

    # Filtering loop (avoid pos, and optionally avoid seen)
    for _ in range(max_resample):
        # avoid sampling the positive item
        bad = neg.eq(pos_item.unsqueeze(1))

        if seen is not None:
            # mark negatives that appear in history as bad
            # vectorized membership check:
            # neg [B, C-1], seen [B, L] -> compare [B, C-1, L] then any over L
            bad = bad | neg.unsqueeze(-1).eq(seen.unsqueeze(1)).any(dim=-1)

        if not bad.any():
            break

        # resample only bad positions
        resample = torch.randint(1, n_items, (int(bad.sum().item()),), device=device)
        neg[bad] = resample

    candidates = torch.cat([pos_item.unsqueeze(1), neg], dim=1)
    return candidates
