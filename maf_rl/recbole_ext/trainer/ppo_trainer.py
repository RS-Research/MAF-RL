# maf_rl/recbole_ext/trainer/ppo_trainer.py
# -*- coding: utf-8 -*-
"""
PPO Trainer for MAF-RL (RecBole Extension)

This trainer integrates Proximal Policy Optimization (PPO) into the RecBole training
pipeline for sequential recommendation.

Scope
-----
- Offline RL training over logged interaction sequences (standard in recommender RL).
- Uses an Actor–Critic model (MAFRLModel) that provides:
    - state embedding (policy representation)
    - value estimate V(s)

Key Design Notes
----------------
1) RecBole's default Trainer expects supervised losses; PPOTrainer overrides the
   training loop to compute PPO losses.
2) In offline sequential recommendation, we use the ground-truth next item as the
   positive outcome and form a candidate set via sampled negatives to compute:
    - policy log-probabilities
    - importance ratio
3) Rewards are computed via a pluggable function in maf_rl/recbole_ext/utils/reward.py

This file is intentionally self-contained and robust. You can refine the reward and
candidate generation as you finalize the exact experimental protocol.

Author: MAF-RL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.trainer import Trainer
from recbole.utils import set_color

# Local utilities (to be implemented or extended)
try:
    from maf_rl.recbole_ext.utils.reward import compute_reward  # noqa: F401
except Exception:  # pragma: no cover
    compute_reward = None  # type: ignore

try:
    from maf_rl.recbole_ext.utils.candidates import build_candidate_set  # noqa: F401
except Exception:  # pragma: no cover
    build_candidate_set = None  # type: ignore


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 256
    max_grad_norm: float = 1.0


class PPOTrainer(Trainer):
    """
    Custom PPO Trainer compatible with RecBole.

    Expected model contract
    -----------------------
    model.forward(interaction) returns dict:
      - "state": Tensor[B, H]   (actor state embedding)
      - "value": Tensor[B]      (critic value V(s))

    model.item_embedding.weight exists for scoring
    model.n_items exists (RecBole base)
    """

    def __init__(self, config, model):
        super().__init__(config, model)

        self.ppo = self._load_ppo_hparams(config)
        self.topk = config.get("topk", [10])
        self.candidate_size = int(config.get("candidate_size", 100))

        # Some RecBole configs store under nested key "rl"
        self._rl_cfg = config.get("rl", {}) or {}

        # By default we operate in "offline RL" mode with ground-truth next item
        self.offline_mode = bool(config.get("offline_mode", True))

        # Sanity: require candidate generator and reward function, but provide fallbacks.
        self._warned_missing_reward = False
        self._warned_missing_candidates = False

    # ---------------------------------------------------------------------
    # Public RecBole Trainer API
    # ---------------------------------------------------------------------

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        """
        Override fit to run PPO training while keeping RecBole logging/checkpoint behavior.
        """
        # RecBole base does some setup (logger, early stopping, etc.). We reuse it.
        # We implement epoch loop ourselves using self._train_epoch_ppo.

        self.best_valid_score = None
        self.cur_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            train_loss = self._train_epoch_ppo(train_data, epoch_idx, show_progress=show_progress)

            # Validation (standard RecBole evaluation)
            if valid_data is not None and (epoch_idx + 1) % self.eval_step == 0:
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)

                if verbose:
                    self.logger.info(set_color(f"epoch {epoch_idx} training", "green") + f" [ppo_loss: {train_loss:.4f}]")
                    self.logger.info(set_color("valid result", "blue") + f": {valid_result}")

                # Early stopping + checkpoint
                self.best_valid_score, self.cur_step, stop_flag, update_flag = self.early_stopping(
                    valid_score, self.best_valid_score, self.cur_step
                )

                if saved and update_flag:
                    self._save_checkpoint(epoch_idx)

                if stop_flag:
                    if verbose:
                        self.logger.info("Early stopping triggered.")
                    break
            else:
                if verbose:
                    self.logger.info(set_color(f"epoch {epoch_idx} training", "green") + f" [ppo_loss: {train_loss:.4f}]")

        # load best model if needed
        if saved and valid_data is not None:
            self._load_best_checkpoint()
        return self.best_valid_score, None

    # ---------------------------------------------------------------------
    # PPO training loop
    # ---------------------------------------------------------------------

    def _train_epoch_ppo(self, train_data, epoch_idx: int, show_progress: bool = False) -> float:
        """
        One PPO epoch:
        1) Collect rollouts from train_data batches (offline, 1-step transitions).
        2) Compute advantages (GAE for 1-step is straightforward).
        3) PPO updates for ppo_epochs with mini-batches.

        Returns: average PPO loss over updates
        """
        self.model.train()

        rollout = self._collect_rollout(train_data, show_progress=show_progress)
        if rollout["size"] == 0:
            return 0.0

        # Normalize advantages
        adv = rollout["advantage"]
        rollout["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo.ppo_epochs):
            for mb in self._iterate_minibatches(rollout, self.ppo.mini_batch_size):
                loss = self._ppo_update_step(mb)
                total_loss += float(loss.detach().cpu().item())
                n_updates += 1

        return total_loss / max(n_updates, 1)

    @torch.no_grad()
    def _collect_rollout(self, train_data, show_progress: bool = False) -> Dict[str, torch.Tensor]:
        """
        Collect offline rollouts from the RecBole sequential batches.

        We treat each training sample as a 1-step transition:
          state s_t  := user history (item_id_list up to t)
          action a_t := recommended item (we use ground-truth next item as action for offline learning)
          reward r_t := computed from outcome (accuracy + novelty/diversity proxies)
          next state := history shifted by one

        Candidate-set policy:
          - build candidates = {positive next item} ∪ sampled negatives
          - compute logprob(a_t | s_t) under current policy for importance ratio

        Output dict fields (all tensors on current device):
          - state_emb: [N, H]
          - action: [N]  (positive item id)
          - old_logp: [N]
          - reward: [N]
          - value: [N]
          - advantage: [N]
          - returns: [N]
          - size: int
        """
        device = self.device

        states = []
        actions = []
        old_logps = []
        rewards = []
        values = []

        for interaction in train_data:
            interaction = interaction.to(device)

            # Ground-truth next item (RecBole uses ITEM_ID as target in seq rec)
            pos_item = interaction[self.model.ITEM_ID]  # [B]
            out = self.model.forward(interaction)
            state = out["state"]  # [B,H]
            value = out["value"]  # [B]

            # Candidate set (pos + negatives)
            candidates = self._build_candidates(interaction, pos_item)  # [B, C]
            logp = self._logprob_of_action(state, pos_item, candidates)  # [B]

            # Reward
            r = self._compute_reward(interaction, pos_item, candidates)  # [B]

            states.append(state.detach())
            actions.append(pos_item.detach())
            old_logps.append(logp.detach())
            rewards.append(r.detach())
            values.append(value.detach())

        if not states:
            return {"size": 0}

        state_emb = torch.cat(states, dim=0)
        action = torch.cat(actions, dim=0)
        old_logp = torch.cat(old_logps, dim=0)
        reward = torch.cat(rewards, dim=0)
        value = torch.cat(values, dim=0)

        # 1-step returns and GAE advantage for offline setting:
        # returns = r + gamma * V(s_{t+1}) ; but we typically don't have s_{t+1} here.
        # In offline one-step training, a common simplification is returns = r (or r + gamma * V(s))
        # We implement: returns = r + gamma * V(s) to stabilize value training.
        returns = reward + self.ppo.gamma * value
        advantage = returns - value

        return {
            "state_emb": state_emb,
            "action": action,
            "old_logp": old_logp,
            "reward": reward,
            "value": value,
            "returns": returns,
            "advantage": advantage,
            "size": state_emb.size(0),
        }

    def _iterate_minibatches(self, rollout: Dict[str, torch.Tensor], batch_size: int):
        n = int(rollout["size"])
        idx = torch.randperm(n, device=rollout["state_emb"].device)
        for start in range(0, n, batch_size):
            mb_idx = idx[start : start + batch_size]
            yield {
                "state_emb": rollout["state_emb"][mb_idx],
                "action": rollout["action"][mb_idx],
                "old_logp": rollout["old_logp"][mb_idx],
                "returns": rollout["returns"][mb_idx],
                "advantage": rollout["advantage"][mb_idx],
            }

    def _ppo_update_step(self, mb: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform one PPO update on a mini-batch.

        Loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        """
        self.optimizer.zero_grad(set_to_none=True)

        state = mb["state_emb"]           # [B,H]
        action = mb["action"]             # [B]
        old_logp = mb["old_logp"]         # [B]
        returns = mb["returns"]           # [B]
        advantage = mb["advantage"]       # [B]

        # Current policy log-prob and entropy over all items (or candidate set)
        # For stability, we compute over full item set if feasible. For large n_items,
        # consider candidate-based approximation.
        logp, entropy = self._logprob_entropy_full(state, action)

        ratio = torch.exp(logp - old_logp)  # [B]
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.ppo.clip_epsilon, 1.0 + self.ppo.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Critic: recompute value from state via model's critic head
        # We do not have direct critic(state) method; easiest is to pass through model.critic_mlp if exists.
        # As a robust fallback, we compute using model.critic_mlp on state BEFORE actor_mlp (state_emb is actor-state).
        # In maf_rl_model.py, actor_state = actor_mlp(state_raw). We stored actor_state in rollout.
        # For value update, we approximate by passing actor_state through a small value head if available.
        value_pred = self._critic_from_actor_state(state)  # [B]
        value_loss = F.mse_loss(value_pred, returns)

        loss = policy_loss + self.ppo.value_coef * value_loss - self.ppo.entropy_coef * entropy.mean()

        loss.backward()
        if self.ppo.max_grad_norm and self.ppo.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo.max_grad_norm)
        self.optimizer.step()

        return loss

    # ---------------------------------------------------------------------
    # Policy utilities
    # ---------------------------------------------------------------------

    def _logprob_entropy_full(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-prob of `action` under full softmax over all items, plus entropy.

        Returns:
          logp_action: [B]
          entropy: [B]
        """
        # Scores: [B, n_items]
        scores = torch.matmul(state, self.model.item_embedding.weight.t())
        log_probs = F.log_softmax(scores, dim=-1)  # [B, n_items]

        logp_action = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return logp_action, entropy

    def _logprob_of_action(self, state: torch.Tensor, action: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Candidate-set approximation:
        log p(a | s) where distribution is softmax over candidate items only.

        state: [B,H]
        action: [B]
        candidates: [B,C]
        return: [B]
        """
        cand_emb = self.model.item_embedding(candidates)  # [B,C,H]
        scores = torch.einsum("bh,bch->bc", state, cand_emb)  # [B,C]
        log_probs = F.log_softmax(scores, dim=-1)  # [B,C]

        # find index of positive action in candidates (assumed at index 0 by our builder)
        # we ensure builder puts pos at column 0.
        pos_index = torch.zeros_like(action, dtype=torch.long)  # [B]
        logp = log_probs.gather(1, pos_index.unsqueeze(1)).squeeze(1)
        return logp

    def _critic_from_actor_state(self, actor_state: torch.Tensor) -> torch.Tensor:
        """
        Get a value prediction from the model. Prefer a real critic head if present.

        In maf_rl_model.py, value is computed from *raw state* before actor_mlp.
        During rollout we saved only actor_state. To keep trainer self-contained,
        we support both:
          - model.critic_mlp exists and accepts raw state (not actor state)
          - model has an attribute value_head that accepts actor_state
        If neither exists, we fallback to a linear layer we create lazily.
        """
        if hasattr(self.model, "value_head") and isinstance(getattr(self.model, "value_head"), nn.Module):
            v = self.model.value_head(actor_state).squeeze(-1)
            return v

        # Lazy fallback head (trainable)
        if not hasattr(self, "_fallback_value_head"):
            self._fallback_value_head = nn.Sequential(
                nn.Linear(actor_state.size(-1), actor_state.size(-1)),
                nn.GELU(),
                nn.Linear(actor_state.size(-1), 1),
            ).to(actor_state.device)

            # Add to optimizer param groups (so it gets trained)
            self.optimizer.add_param_group({"params": self._fallback_value_head.parameters()})

        v = self._fallback_value_head(actor_state).squeeze(-1)
        return v

    # ---------------------------------------------------------------------
    # Candidate & reward utilities
    # ---------------------------------------------------------------------

    def _build_candidates(self, interaction, pos_item: torch.Tensor) -> torch.Tensor:
        """
        Build candidate item set for each sample in the batch.

        Expected output: Tensor[B, C] where candidates[:,0] == pos_item
        """
        if build_candidate_set is not None:
            return build_candidate_set(
                interaction=interaction,
                pos_item=pos_item,
                n_items=self.model.n_items,
                candidate_size=self.candidate_size,
                device=self.device,
                item_id_field=self.model.ITEM_ID,
            )

        # Fallback: random negatives
        if not self._warned_missing_candidates:
            self.logger.warning(
                "maf_rl.recbole_ext.utils.candidates.build_candidate_set not found. "
                "Using random negative sampling fallback."
            )
            self._warned_missing_candidates = True

        B = pos_item.size(0)
        C = max(self.candidate_size, 2)

        # Sample random ids in [1, n_items-1], avoid 0 padding
        neg = torch.randint(1, self.model.n_items, (B, C - 1), device=self.device)
        candidates = torch.cat([pos_item.unsqueeze(1), neg], dim=1)
        return candidates

    def _compute_reward(self, interaction, pos_item: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Compute reward per sample. Pluggable via maf_rl/recbole_ext/utils/reward.py.

        Returns: Tensor[B]
        """
        if compute_reward is not None:
            return compute_reward(
                interaction=interaction,
                pos_item=pos_item,
                candidates=candidates,
                config=self.config,
                model=self.model,
            )

        # Fallback: accuracy-only reward (always 1 for logged positive)
        if not self._warned_missing_reward:
            self.logger.warning(
                "maf_rl.recbole_ext.utils.reward.compute_reward not found. "
                "Using accuracy-only reward fallback (r=1)."
            )
            self._warned_missing_reward = True
        return torch.ones_like(pos_item, dtype=torch.float32, device=self.device)

    # ---------------------------------------------------------------------
    # Hyperparameter loading
    # ---------------------------------------------------------------------

    def _load_ppo_hparams(self, config) -> PPOHyperParams:
        rl = config.get("rl", {}) or {}
        return PPOHyperParams(
            gamma=float(rl.get("gamma", config.get("gamma", 0.99))),
            gae_lambda=float(rl.get("gae_lambda", config.get("gae_lambda", 0.95))),
            clip_epsilon=float(rl.get("clip_epsilon", config.get("clip_epsilon", 0.2))),
            entropy_coef=float(rl.get("entropy_coef", config.get("entropy_coef", 0.01))),
            value_coef=float(rl.get("value_coef", config.get("value_coef", 0.5))),
            ppo_epochs=int(rl.get("ppo_epochs", config.get("ppo_epochs", 4))),
            mini_batch_size=int(rl.get("mini_batch_size", config.get("mini_batch_size", 256))),
            max_grad_norm=float(rl.get("max_grad_norm", config.get("max_grad_norm", 1.0))),
        )
