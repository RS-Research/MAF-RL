# maf_rl/recbole_ext/models/encoders.py
# -*- coding: utf-8 -*-
"""
Feature Encoders for MAF-RL (RecBole Extension)

These encoders adapt precomputed item-side features (text/image/graph/context)
into the model hidden space (H) used by the sequential backbone.

Assumptions
-----------
- Features are precomputed offline and stored as a matrix aligned with RecBole
  internal item token ids, including padding id=0:
    feat_matrix.shape == [n_items, feat_dim]
- The encoders perform:
  item_ids -> feat lookup -> projection -> dropout -> layer norm

This design keeps training efficient and reproducible, and it decouples feature
extraction (e.g., BERT/ResNet/graph embedding) from the RL training pipeline.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BaseItemFeatureEncoder(nn.Module):
    """
    Base class for item feature encoders.

    Parameters
    ----------
    feat_matrix : torch.Tensor
        Fixed feature lookup table, shape [n_items, feat_dim].
        Must be aligned with RecBole internal item ids (including padding=0).
    out_dim : int
        Output projection dimension (usually model hidden size).
    name : str
        Optional name for logging/debugging.

    Notes
    -----
    - feat_matrix is stored as a buffer (not trainable by default).
    - Projection layer is trainable.
    """

    def __init__(
        self,
        feat_matrix: torch.Tensor,
        out_dim: int,
        name: str = "feature",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if not isinstance(feat_matrix, torch.Tensor):
            raise TypeError("feat_matrix must be a torch.Tensor")
        if feat_matrix.dim() != 2:
            raise ValueError(f"feat_matrix must be 2D [n_items, feat_dim], got {tuple(feat_matrix.shape)}")

        self.name = str(name)
        self.out_dim = int(out_dim)
        self.feat_dim = int(feat_matrix.shape[1])

        # Stored as buffer so it moves with .to(device) and is saved in checkpoints.
        self.register_buffer("feat_matrix", feat_matrix.float(), persistent=True)

        self.proj = nn.Linear(self.feat_dim, self.out_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.out_dim) if use_layernorm else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        item_ids : torch.Tensor
            Shape [B, L] or [B]. RecBole internal item ids.

        Returns
        -------
        torch.Tensor
            Shape [B, L, out_dim] or [B, out_dim].
        """
        # Lookup: broadcasts for [B,L] indexing
        feats = self.feat_matrix[item_ids]  # [B,L,D] or [B,D]
        out = self.proj(feats)
        out = self.dropout(out)
        out = self.norm(out)
        return out


class TextFeatureEncoder(BaseItemFeatureEncoder):
    """
    Encoder for textual item features (e.g., BERT embeddings).

    Typical inputs:
    - item title/description embeddings from a Transformer model.
    """

    def __init__(
        self,
        feat_matrix: torch.Tensor,
        out_dim: int,
        name: str = "text",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__(
            feat_matrix=feat_matrix,
            out_dim=out_dim,
            name=name,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )


class ImageFeatureEncoder(BaseItemFeatureEncoder):
    """
    Encoder for visual item features (e.g., ResNet / ViT embeddings).

    Typical inputs:
    - precomputed CNN/Vision Transformer features for product images.
    """

    def __init__(
        self,
        feat_matrix: torch.Tensor,
        out_dim: int,
        name: str = "image",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__(
            feat_matrix=feat_matrix,
            out_dim=out_dim,
            name=name,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )


class GraphFeatureEncoder(BaseItemFeatureEncoder):
    """
    Encoder for graph-derived item features (e.g., LightGCN embeddings, item-item graph embeddings).

    Typical inputs:
    - embeddings computed from collaborative filtering graphs or knowledge graphs.
    """

    def __init__(
        self,
        feat_matrix: torch.Tensor,
        out_dim: int,
        name: str = "graph",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__(
            feat_matrix=feat_matrix,
            out_dim=out_dim,
            name=name,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )


class ContextFeatureEncoder(BaseItemFeatureEncoder):
    """
    Encoder for contextual item-side features.

    Typical inputs:
    - category vectors, price bins, location embeddings, time/context bucket embeddings, etc.
    """

    def __init__(
        self,
        feat_matrix: torch.Tensor,
        out_dim: int,
        name: str = "context",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__(
            feat_matrix=feat_matrix,
            out_dim=out_dim,
            name=name,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )


def build_feature_encoder(
    feat_matrix: Optional[torch.Tensor],
    out_dim: int,
    kind: str,
    dropout: float = 0.1,
    use_layernorm: bool = True,
) -> Optional[nn.Module]:
    """
    Convenience factory for constructing encoders by kind.

    Parameters
    ----------
    feat_matrix : Optional[torch.Tensor]
        Feature matrix aligned with item ids, or None.
    out_dim : int
        Output dim (hidden size).
    kind : str
        One of: {"text","image","graph","context"}.
    """
    if feat_matrix is None:
        return None

    kind = kind.lower().strip()
    if kind == "text":
        return TextFeatureEncoder(feat_matrix, out_dim, dropout=dropout, use_layernorm=use_layernorm)
    if kind == "image":
        return ImageFeatureEncoder(feat_matrix, out_dim, dropout=dropout, use_layernorm=use_layernorm)
    if kind == "graph":
        return GraphFeatureEncoder(feat_matrix, out_dim, dropout=dropout, use_layernorm=use_layernorm)
    if kind == "context":
        return ContextFeatureEncoder(feat_matrix, out_dim, dropout=dropout, use_layernorm=use_layernorm)

    raise ValueError(f"Unknown encoder kind: {kind}. Expected one of: text, image, graph, context.")
