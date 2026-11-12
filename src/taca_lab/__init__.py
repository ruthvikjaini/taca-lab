"""taca-lab: Transition-Aware Credit Assignment in agentic RL.

A CPU-only, NumPy reproduction of Lu et al., arXiv:2611.11330 (TACA),
on a multi-turn calculator testbed: Advantage Projection, Decision
Contrast, and Supervised Replay on top of multi-turn GRPO.
"""

from .env import make_problem, prompt_tokens
from .grpo import ReplayBuffer, grpo_train, grpo_update, rollout_batch
from .model import Transformer

__all__ = [
    "Transformer",
    "make_problem",
    "prompt_tokens",
    "rollout_batch",
    "grpo_update",
    "grpo_train",
    "ReplayBuffer",
]
