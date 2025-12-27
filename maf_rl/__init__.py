"""
MAF-RL: Multi-Modal Actor–Critic Reinforcement Learning for Sequential Recommendation.

This package extends RecBole with:
- A custom model (Actor–Critic + multi-modal fusion)
- A PPO-based trainer
- Optional RL-specific dataloading utilities
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "get_version",
]

# Keep version here so tools (pip, docs, CI) can read it reliably.
# Update when you tag releases.
__version__ = "0.1.0"


def get_version() -> str:
    """Return the installed package version."""
    return __version__
