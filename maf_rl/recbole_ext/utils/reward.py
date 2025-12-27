# maf_rl/recbole_ext/utils/reward.py
# -*- coding: utf-8 -*-
"""
Reward Computation for MAF-RL (RecBole Extension)

This module defines a pluggable reward function for offline RL training in
sequential recommendation, suitable for PPOTrainer.

Because training is offline (logged interactions), the reward is computed from:
- correctness proxy (ground-truth next item)
- optional novelty proxy (inverse popularity)
- optional diversity proxy (intra-list or candidate-set dispersion)
- optional long-term proxy (e.g., time-based or session-based heuristics)

Default Behavior
----------------
If you do not provide additional statistics/features, the reward reduces to an
accuracy-only constant (1.0) for logged positives.

Extensibility
-------------
You can extend this function to match exactly the reward design in your paper,
e.g., weighted multi-objective rewards:
  r = α * r_acc + β * r_novelty + δ * r_diversity + η * r_longterm

Inputs
------
interaction: RecBole Interaction batch
pos_item: ground-truth next item ids [B]
candidates: candidate ids [B, C]

config: RecBole config (dict-like)
model: model instance (optional use for embeddings)

Author: MAF-RL
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from recbole.data.interaction import Interaction


def compute_reward(
    interaction: Interaction,
    pos_item: torch.Tensor,
    candidates: torch.Tensor,
    config: Any,
    model: Optional[Any] = None,
) -> torch.Tensor:
    """
    Compute per-sample scalar reward.

    Returns
    -------
    reward : torch.Tensor
        Shape [B], float32
    """
    device = pos_item.device
    B = pos_item.size(0)

    # -------------------------
    # Reward weights (defaults)
    # -------------------------
    rw = config.get("reward_weights", None)
    if rw is None:
        # allow nested config: reward: {weights: {...}}
        reward_cfg = config.get("reward", {}) or {}
        rw = reward_cfg.get("weights", None)

    # default weights: accuracy only
    alpha = float(rw.get("acc", 1.0)) if isinstance(rw, dict) else 1.0
    beta = float(rw.get("novelty", 0.0)) if isinstance(rw, dict) else 0.0
    delta = float(rw.get("diversity", 0.0)) if isinstance(rw, dict) else 0.0
    eta = float(rw.get("longterm", 0.0)) if isinstance(rw, dict) else 0.0

    # -------------------------
    # 1) Accuracy proxy
    # -------------------------
    # In offline RL with logged positives, correctness for the logged action is 1.
    r_acc = torch.ones(B, device=device, dtype=torch.float32)

    # -------------------------
    # 2) Novelty proxy (optional)
    # -------------------------
    # If popularity statistics are available, novelty can be inverse popularity.
    # Expect a tensor/list mapping item_id -> popularity count or probability.
    # You can provide:
    #   config["item_popularity_path"] -> load separately in your pipeline, or
    #   config["item_popularity"] -> dict or list-like.
    r_novelty = torch.zeros(B, device=device, dtype=torch.float32)
    pop = config.get("item_popularity", None)
    if beta != 0.0 and pop is not None:
        # pop can be a dict {item_id: count} or list/torch tensor indexed by item_id
        r_novelty = _novelty_from_popularity(pos_item, pop, device=device)

    # -------------------------
    # 3) Diversity proxy (optional)
    # -------------------------
    # For 1-step offline RL, "diversity" can be approximated relative to user's history:
    # encourage items dissimilar to recent history embeddings (if model embeddings exist).
    r_div = torch.zeros(B, device=device, dtype=torch.float32)
    if delta != 0.0 and model is not None and hasattr(model, "item_embedding"):
        # Use cosine distance between action embedding and mean of history embeddings
        # Requires item history sequence to be in interaction
        item_seq_field = getattr(model, "ITEM_SEQ", "item_id_list")
        if item_seq_field in interaction:
            r_div = _diversity_against_history(interaction, pos_item, model, item_seq_field)

    # -------------------------
    # 4) Long-term proxy (optional)
    # -------------------------
    # Offline data makes true long-term reward difficult. A simple proxy is:
    # - reward higher if interaction occurs after longer gaps (retention-like)
    # - or use available session/return signals if present
    r_long = torch.zeros(B, device=device, dtype=torch.float32)
    if eta != 0.0:
        # Example: timestamp gap proxy if "timestamp" exists
        time_field = getattr(model, "TIME_FIELD", "timestamp") if model is not None else "timestamp"
        if time_field in interaction:
            r_long = _longterm_time_gap_proxy(interaction, time_field)

    # -------------------------
    # Combine (weighted sum)
    # -------------------------
    reward = alpha * r_acc + beta * r_novelty + delta * r_div + eta * r_long

    # Optional clipping for stability
    clip_min = float(config.get("reward_clip_min", -10.0))
    clip_max = float(config.get("reward_clip_max", 10.0))
    reward = torch.clamp(reward, min=clip_min, max=clip_max)

    return reward


def _novelty_from_popularity(pos_item: torch.Tensor, pop: Any, device: torch.device) -> torch.Tensor:
    """
    Convert popularity into novelty = 1 / log(1 + pop) (common heuristic).
    """
    if isinstance(pop, torch.Tensor):
        pop_t = pop.to(device)
        p = pop_t[pos_item].float()
    elif isinstance(pop, (list, tuple)):
        pop_t = torch.tensor(pop, device=device, dtype=torch.float32)
        p = pop_t[pos_item].float()
    elif isinstance(pop, dict):
        # dict lookup per item id (slower; acceptable for small batches)
        p = torch.tensor([float(pop.get(int(i.item()), 0.0)) for i in pos_item], device=device)
    else:
        # unsupported
        return torch.zeros_like(pos_item, dtype=torch.float32, device=device)

    novelty = 1.0 / torch.log1p(p + 1.0)
    return novelty


def _diversity_against_history(interaction: Interaction, pos_item: torch.Tensor, model: Any, item_seq_field: str) -> torch.Tensor:
    """
    Diversity proxy: cosine distance between action embedding and mean history embedding.

    Returns [B] in [0, 2] approximately (since cosine distance = 1 - cosine_sim).
    """
    device = pos_item.device
    seq = interaction[item_seq_field].to(device)  # [B, L]
    hist_emb = model.item_embedding(seq)          # [B, L, H]
    # mean over non-padding tokens
    mask = seq.ne(0).float().unsqueeze(-1)        # [B, L, 1]
    denom = mask.sum(dim=1).clamp_min(1.0)
    hist_mean = (hist_emb * mask).sum(dim=1) / denom  # [B, H]

    act_emb = model.item_embedding(pos_item)      # [B, H]
    cos = torch.nn.functional.cosine_similarity(act_emb, hist_mean, dim=-1).clamp(-1.0, 1.0)
    diversity = 1.0 - cos
    return diversity.float()


def _longterm_time_gap_proxy(interaction: Interaction, time_field: str) -> torch.Tensor:
    """
    Simple proxy: scale reward by normalized timestamp gaps if available.

    If interaction contains:
    - current timestamp (t)
    and also contains a sequence of timestamps, you can compute gaps.
    Here we implement a conservative fallback: zero (unless you extend).

    Recommended extension:
    - store previous timestamp in Interaction, or
    - store time sequence field and compute last gap.
    """
    # Default: no long-term info available
    t = interaction[time_field]
    # If time is present but no gap fields, return zeros.
    return torch.zeros_like(t, dtype=torch.float32, device=t.device)
