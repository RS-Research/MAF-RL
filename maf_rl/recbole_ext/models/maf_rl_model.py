# maf_rl/recbole_ext/models/maf_rl_model.py
# -*- coding: utf-8 -*-
"""
MAF-RL (RecBole Extension): Actor–Critic + Multi-Modal Fusion Model

This file defines a RecBole-compatible model wrapper that:
- Encodes a user's sequential state (item history)
- Optionally fuses multi-modal item signals (text/image/graph/context features)
- Produces:
  - Actor outputs: logits/scores over candidate items (or full item set)
  - Critic outputs: scalar V(s) for PPO training

Notes
-----
1) This model is RecBole-evaluation compatible via `full_sort_predict`.
2) PPO training is typically implemented in a custom Trainer (see ppo_trainer.py),
   which will call the Actor/Critic heads and compute PPO losses.
3) Multi-modal features are assumed to be precomputed and loaded as matrices
   indexed by internal item ids (RecBole token ids).

Author: MAF-RL
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import os
import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.utils import InputType

# Local imports (your repo)
# These modules can be implemented next; for now, the model works with internal defaults.
try:
    from maf_rl.recbole_ext.models.fusion import GatedAttentionFusion
except Exception:  # pragma: no cover
    GatedAttentionFusion = None  # type: ignore

try:
    from maf_rl.recbole_ext.models.encoders import (
        TextFeatureEncoder,
        ImageFeatureEncoder,
        GraphFeatureEncoder,
        ContextFeatureEncoder,
    )
except Exception:  # pragma: no cover
    TextFeatureEncoder = ImageFeatureEncoder = GraphFeatureEncoder = ContextFeatureEncoder = None  # type: ignore


class MAFRLModel(SequentialRecommender):
    """
    RecBole-compatible model for MAF-RL.

    InputType.POINTWISE is used as a safe default for RecBole; evaluation uses full_sort_predict.
    PPO training will typically use the Actor/Critic heads directly.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---------- Core dimensions ----------
        self.hidden_size = int(config.get("hidden_size", 128))
        self.dropout_prob = float(config.get("dropout_prob", 0.1))
        self.max_seq_length = int(config.get("MAX_ITEM_LIST_LENGTH", 50))

        # ---------- Item embedding (behavioral backbone) ----------
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # ---------- Sequential encoder (Transformer) ----------
        n_layers = int(config.get("n_layers", 2))
        n_heads = int(config.get("n_heads", 2))
        inner_size = int(config.get("inner_size", 256))
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.transformer_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=self.hidden_size,
            inner_size=inner_size,
            hidden_dropout_prob=self.dropout_prob,
            attn_dropout_prob=self.dropout_prob,
            hidden_act=config.get("hidden_act", "gelu"),
            layer_norm_eps=1e-12,
        )

        # ---------- Optional: Multi-modal feature matrices ----------
        # Expected shapes: [n_items, feat_dim]
        # These are not trainable by default; they can be projected to hidden_size.
        self.text_feat = self._maybe_load_item_feature_matrix(config, "text_feat_path")
        self.image_feat = self._maybe_load_item_feature_matrix(config, "image_feat_path")
        self.graph_feat = self._maybe_load_item_feature_matrix(config, "graph_feat_path")
        self.context_feat = self._maybe_load_item_feature_matrix(config, "context_feat_path")

        # ---------- Feature encoders (project external features -> hidden_size) ----------
        self.text_encoder = self._build_feature_encoder(self.text_feat, TextFeatureEncoder, "text")
        self.image_encoder = self._build_feature_encoder(self.image_feat, ImageFeatureEncoder, "image")
        self.graph_encoder = self._build_feature_encoder(self.graph_feat, GraphFeatureEncoder, "graph")
        self.context_encoder = self._build_feature_encoder(self.context_feat, ContextFeatureEncoder, "context")

        # ---------- Fusion module ----------
        fusion_type = (config.get("fusion", {}) or {}).get("type", config.get("fusion_type", "gated_attention"))
        self.use_fusion = any(x is not None for x in [self.text_feat, self.image_feat, self.graph_feat, self.context_feat])

        if self.use_fusion and fusion_type in {"gated_attention", "gated_attn"} and GatedAttentionFusion is not None:
            fusion_dropout = float((config.get("fusion", {}) or {}).get("dropout", self.dropout_prob))
            self.fusion = GatedAttentionFusion(hidden_size=self.hidden_size, dropout=fusion_dropout)
        else:
            self.fusion = None

        # ---------- Actor & Critic heads ----------
        # Actor: produce a state embedding; item scoring is via dot product with item embeddings.
        # Critic: scalar value V(s)
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
        )
        self.critic_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, 1),
        )

        # Initialize parameters
        self.apply(self._init_weights)

    # -------------------------------------------------------------------------
    # Public API expected by RecBole
    # -------------------------------------------------------------------------

    def forward(self, interaction) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning state representation and critic value.

        Returns a dict to make PPO trainer integration straightforward:
        {
            "state": [B, H],
            "value": [B],
        }
        """
        state = self.encode_state(interaction)  # [B, H]
        actor_state = self.actor_mlp(state)     # [B, H]
        value = self.critic_mlp(state).squeeze(-1)  # [B]
        return {"state": actor_state, "value": value}

    def full_sort_predict(self, interaction) -> torch.Tensor:
        """
        RecBole evaluation hook: scores all items for each user state.

        Output shape: [B, n_items]
        """
        out = self.forward(interaction)
        state = out["state"]  # [B, H]

        # Dot-product scoring against all item embeddings
        all_item_emb = self.item_embedding.weight  # [n_items, H]
        scores = torch.matmul(state, all_item_emb.t())  # [B, n_items]

        return scores

    def predict(self, interaction) -> torch.Tensor:
        """
        Pointwise scoring for a specific target item in `interaction[self.ITEM_ID]`.
        Output shape: [B]
        """
        out = self.forward(interaction)
        state = out["state"]  # [B, H]
        item_ids = interaction[self.ITEM_ID]  # [B]
        item_emb = self.item_embedding(item_ids)  # [B, H]
        scores = torch.sum(state * item_emb, dim=-1)  # [B]
        return scores

    def calculate_loss(self, interaction) -> torch.Tensor:
        """
        RecBole training loop expects a supervised loss. For PPO training, you will
        typically use a custom Trainer and bypass this.

        We provide a safe fallback: BPR-style pairwise loss if NEG_ITEM_ID is present,
        otherwise a simple cross-entropy over full-sort logits with the next item.

        Recommended: Use PPOTrainer (maf_rl/recbole_ext/trainer/ppo_trainer.py).
        """
        if self.NEG_ITEM_ID in interaction:
            pos_scores = self.predict(interaction)
            neg_item_ids = interaction[self.NEG_ITEM_ID]
            out = self.forward(interaction)
            state = out["state"]
            neg_emb = self.item_embedding(neg_item_ids)
            neg_scores = torch.sum(state * neg_emb, dim=-1)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()
            return loss

        # Fallback: next-item classification using full_sort_predict
        scores = self.full_sort_predict(interaction)
        target = interaction[self.ITEM_ID]
        loss = nn.CrossEntropyLoss()(scores, target)
        return loss

    # -------------------------------------------------------------------------
    # State Encoding: sequence backbone + optional multimodal fusion
    # -------------------------------------------------------------------------

    def encode_state(self, interaction) -> torch.Tensor:
        """
        Build the user state representation from:
        - sequential item history (Transformer over item embeddings)
        - optional multi-modal fusion (text/image/graph/context) at item level

        Returns:
            state: [B, H]
        """
        item_seq = interaction[self.ITEM_SEQ]  # [B, L]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]

        # Base item embeddings for the sequence: [B, L, H]
        seq_item_emb = self.item_embedding(item_seq)

        # Optional: fuse external modalities into the sequence embedding
        if self.use_fusion and self.fusion is not None:
            modal_embs = self._get_modal_sequence_embeddings(item_seq)  # dict[str, Tensor[B,L,H]]
            seq_item_emb = self.fusion(seq_item_emb, modal_embs)  # [B, L, H]

        # Transformer encoding
        seq_item_emb = self.layer_norm(seq_item_emb)
        seq_item_emb = self.dropout(seq_item_emb)

        # attention mask for transformer
        attn_mask = self.get_attention_mask(item_seq)  # [B, 1, L, L]
        output = self.transformer_encoder(seq_item_emb, attn_mask, output_all_encoded_layers=False)
        encoded_layers = output[-1]  # [B, L, H]

        # Gather last valid position as state (standard sequential rec)
        state = self.gather_indexes(encoded_layers, item_seq_len - 1)  # [B, H]
        return state

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_modal_sequence_embeddings(self, item_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns projected embeddings for each available modality at the sequence level.
        Each modality tensor is [B, L, H].
        """
        modal: Dict[str, torch.Tensor] = {}

        if self.text_feat is not None and self.text_encoder is not None:
            modal["text"] = self.text_encoder(item_seq)  # [B,L,H]
        if self.image_feat is not None and self.image_encoder is not None:
            modal["image"] = self.image_encoder(item_seq)  # [B,L,H]
        if self.graph_feat is not None and self.graph_encoder is not None:
            modal["graph"] = self.graph_encoder(item_seq)  # [B,L,H]
        if self.context_feat is not None and self.context_encoder is not None:
            modal["context"] = self.context_encoder(item_seq)  # [B,L,H]

        return modal

    def _build_feature_encoder(self, feat: Optional[torch.Tensor], encoder_cls, name: str):
        """
        If a feature matrix exists, create a small encoder module that:
        - looks up item ids -> feature vectors
        - projects to hidden_size
        """
        if feat is None:
            return None

        feat_dim = int(feat.shape[1])
        if encoder_cls is None:
            # Minimal default encoder if your encoders.py is not implemented yet
            return _DefaultItemFeatureEncoder(feat_matrix=feat, out_dim=self.hidden_size)
        return encoder_cls(feat_matrix=feat, out_dim=self.hidden_size, name=name)

    def _maybe_load_item_feature_matrix(self, config, key: str) -> Optional[torch.Tensor]:
        """
        Load a precomputed item feature matrix from config[key] if provided.

        Supported formats:
        - .npy (NumPy array)
        - .pt / .pth (torch tensor)

        Expected:
        - First dimension aligns with RecBole internal item id space: n_items
          (including padding at index 0). If your feature file does not include
          padding, we will prepend a zero vector.
        """
        path = config.get(key, None)
        if not path:
            return None

        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature path not found for {key}: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
            tensor = torch.tensor(arr, dtype=torch.float32)
        elif ext in {".pt", ".pth"}:
            tensor = torch.load(path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{key} must be a torch.Tensor when loading from .pt/.pth: {path}")
            tensor = tensor.float()
        else:
            raise ValueError(f"Unsupported feature file extension for {key}: {ext}")

        # Ensure [N, D]
        if tensor.dim() != 2:
            raise ValueError(f"{key} feature matrix must be 2D [n_items, feat_dim], got shape {tuple(tensor.shape)}")

        # Align with RecBole n_items (includes padding id=0)
        if tensor.size(0) == self.n_items - 1:
            pad = torch.zeros(1, tensor.size(1), dtype=torch.float32)
            tensor = torch.cat([pad, tensor], dim=0)
        elif tensor.size(0) != self.n_items:
            raise ValueError(
                f"{key} first dimension must be n_items ({self.n_items}) or n_items-1 ({self.n_items-1}); "
                f"got {tensor.size(0)}"
            )

        return tensor

    @staticmethod
    def _init_weights(module):
        """Standard initialization; compatible with RecBole defaults."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class _DefaultItemFeatureEncoder(nn.Module):
    """
    Minimal feature encoder:
    - Stores a fixed feature matrix (not trainable)
    - Projects features to out_dim with a trainable Linear layer
    """

    def __init__(self, feat_matrix: torch.Tensor, out_dim: int):
        super().__init__()
        # Register as buffer so it moves with .to(device) and is saved in checkpoints.
        self.register_buffer("feat_matrix", feat_matrix, persistent=True)
        self.proj = nn.Linear(int(feat_matrix.shape[1]), int(out_dim))

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [B, L] or [B]
        returns:  [B, L, out_dim] or [B, out_dim]
        """
        feat = self.feat_matrix[item_ids]  # broadcast indexing
        return self.proj(feat)
