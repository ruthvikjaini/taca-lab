"""Quick demo: SFT a tiny base, run a few TACA updates, show the
diagnostics. Fast enough for CI (<60 s)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from taca_lab.env import make_problem, make_sft_dataset  # noqa: E402
from taca_lab.evaluate import (  # noqa: E402
    diagnose_credit, diagnose_retries, greedy_evaluate)
from taca_lab.grpo import ReplayBuffer, grpo_train, rollout_batch  # noqa: E402
from taca_lab.model import Transformer  # noqa: E402
from taca_lab.sft import sft_train  # noqa: E402
from taca_lab.vocab import VOCAB_SIZE  # noqa: E402


def main():
    model = Transformer(VOCAB_SIZE, max_len=160, d=64, seed=0)
    data = make_sft_dataset(np.random.default_rng(1), n=6000,
                            tier_p=(0.35, 0.35, 0.30), direct_p=0.5)
    losses = sft_train(model, data, epochs=4, batch=64, lr=3e-3,
                       rng=np.random.default_rng(2), label_smoothing=0.05)
    print(f"SFT loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    rng = np.random.default_rng(123)
    held = [make_problem(rng, tier=t) for t in (1, 2, 3) for _ in range(12)]
    acc, per_tier, _ = greedy_evaluate(model, held, rng=np.random.default_rng(5))
    print(f"base greedy accuracy {acc:.3f} {per_tier}")
    train = [make_problem(np.random.default_rng(0)) for _ in range(64)]
    stats = grpo_train(model, train, updates=5, batch=8, group=8, lr=3e-4,
                       rng=np.random.default_rng(42), log_every=5,
                       buffer=ReplayBuffer())
    print(f"after 5 TACA updates: reward "
          f"{stats[0]['mean_reward']:.3f} -> {stats[-1]['mean_reward']:.3f}, "
          f"tool {stats[-1]['tool_use_rate']:.2f}, "
          f"exec {stats[-1]['exec_success']:.2f}")
    trajs = rollout_batch(model, held, group=4, temperature=1.0,
                          rng=np.random.default_rng(9))
    print("credit:", {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in diagnose_credit(trajs).items()})
    print("retries:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in diagnose_retries(trajs).items()})


if __name__ == "__main__":
    main()
