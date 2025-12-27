# maf_rl/recbole_ext/dataloader/rl_dataloader.py
# -*- coding: utf-8 -*-
"""
RL DataLoader Utilities for MAF-RL (RecBole Extension)

This module provides optional utilities to support trajectory-style batching
for offline reinforcement learning in sequential recommendation.

Context
-------
RecBole already provides dataloaders for sequential recommendation that yield
batches of `Interaction` objects containing:
  - ITEM_SEQ (history), ITEM_SEQ_LEN, and target ITEM_ID (next item)

For PPO-style training, you may want to:
  - treat each training sample as a 1-step transition (default, simplest), or
  - build short trajectories (multi-step) for GAE and long-horizon objectives.

This file implements:
  1) A simple wrapper to transform RecBole batches into "transition batches"
  2) An optional trajectory buffer collector (n-step), if you decide to extend.

Important
---------
- You can use PPOTrainer without this file (1-step offline RL).
- Keep it lightweight and pure-PyTorch/RecBole.

Author: MAF-RL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

from recbole.data.interaction import Interaction


@dataclass
class TransitionBatch:
    """
    Minimal transition representation for offline RL in sequential recommendation.

    Fields (common):
    - state:      user history sequence fields (from Interaction)
    - action:     recommended item id (typically the logged next item in offline RL)
    - reward:     scalar reward
    - done:       episode termination flag (optional)
    - next_state: next user history (optional for 1-step offline RL)
    """
    state: Interaction
    action: torch.Tensor
    reward: torch.Tensor
    done: Optional[torch.Tensor] = None
    next_state: Optional[Interaction] = None


class RLTransitionDataLoader:
    """
    Wrap a RecBole DataLoader and yield TransitionBatch objects.

    By default, this treats each RecBole sample as a 1-step transition:
      state := (ITEM_SEQ, ITEM_SEQ_LEN, ...)
      action := ITEM_ID (ground-truth next item)
      reward := placeholder (computed later in PPOTrainer) or precomputed

    You can pass `reward_fn` to compute reward on-the-fly if desired.
    """

    def __init__(
        self,
        recbole_dataloader,
        item_id_field: str = "item_id",
        reward_fn=None,
        device: Optional[torch.device] = None,
    ):
        self.dataloader = recbole_dataloader
        self.item_id_field = item_id_field
        self.reward_fn = reward_fn
        self.device = device

    def __iter__(self) -> Iterator[TransitionBatch]:
        for interaction in self.dataloader:
            if self.device is not None:
                interaction = interaction.to(self.device)

            action = interaction[self.item_id_field]

            if self.reward_fn is None:
                # Reward can be computed inside the PPO trainer using:
                # - action correctness
                # - novelty/diversity proxies
                # - multi-objective weights from config
                reward = torch.ones_like(action, dtype=torch.float32)
            else:
                reward = self.reward_fn(interaction)

            if self.device is not None:
                reward = reward.to(self.device)

            yield TransitionBatch(
                state=interaction,
                action=action,
                reward=reward,
                done=None,
                next_state=None,
            )

    def __len__(self) -> int:
        return len(self.dataloader)


class TrajectoryBuffer:
    """
    Optional trajectory buffer for n-step offline RL.

    This is a lightweight container you can use if you later decide to collect
    multi-step trajectories (e.g., for longer-horizon GAE). In typical offline
    recommender RL, many works use 1-step transitions; multi-step requires careful
    handling of session boundaries and logged trajectories.

    This class stores tensors; you can add any additional fields as needed.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.reset()

    def reset(self):
        self.storage: Dict[str, list] = {
            "state_emb": [],
            "action": [],
            "logp": [],
            "reward": [],
            "value": [],
            "done": [],
        }

    def add(
        self,
        state_emb: torch.Tensor,
        action: torch.Tensor,
        logp: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: Optional[torch.Tensor] = None,
    ):
        if self.device is not None:
            state_emb = state_emb.to(self.device)
            action = action.to(self.device)
            logp = logp.to(self.device)
            reward = reward.to(self.device)
            value = value.to(self.device)
            if done is not None:
                done = done.to(self.device)

        self.storage["state_emb"].append(state_emb)
        self.storage["action"].append(action)
        self.storage["logp"].append(logp)
        self.storage["reward"].append(reward)
        self.storage["value"].append(value)
        self.storage["done"].append(done if done is not None else torch.zeros_like(reward))

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Stack stored lists into tensors.
        """
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.storage.items():
            if len(v) == 0:
                out[k] = torch.empty(0, device=self.device)
            else:
                out[k] = torch.cat(v, dim=0)
        out["size"] = torch.tensor(int(out["action"].size(0)), device=self.device)
        return out
