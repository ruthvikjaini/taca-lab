"""E5: out-of-distribution generalisation to 4-step problems.

The main grid trains and evaluates on tiers 1-3. Here MT-GRPO and
TACA (seed 0) are retrained with their models saved, then evaluated
greedy + sampled on tier-4 problems (4-step chains, never trained on).
If TACA's sustained tool use and replay supervision preserve the
compositional skill better than plain MT-GRPO, it should show here
even when in-distribution accuracy saturates.

Writes results/ood.json. Models are saved under checkpoints/ (not
part of the shipped repo).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import CKPT, held_out_problems, save_json, sft_base, train_problems  # noqa: E402
from taca_lab.env import make_problem  # noqa: E402
from taca_lab.evaluate import greedy_evaluate, sampled_evaluate  # noqa: E402
from taca_lab.grpo import ReplayBuffer, grpo_train  # noqa: E402

UPDATES = 100
BATCH = 16
GROUP = 8
LR = 3e-4


def tier4_problems(n=80, seed=999):
    rng = np.random.default_rng(seed)
    return [make_problem(rng, tier=4) for _ in range(n)]


def main():
    held = held_out_problems()
    train = train_problems(seed=0)
    base = sft_base(seed=0)
    t4 = tier4_problems()
    out = {}
    for arm, kw in {
        "mt-grpo": dict(use_ap=False, use_dc=False, use_replay=False),
        "taca": dict(use_ap=True, use_dc=True, use_replay=True,
                     lambda_dc=1.0, lambda_rep=0.5, eta=4.0),
    }.items():
        print(f"=== E5 {arm} ===", flush=True)
        model = base.copy()
        t0 = time.time()
        grpo_train(model, train, updates=UPDATES, batch=BATCH, group=GROUP,
                   lr=LR, rng=np.random.default_rng(5000),
                   log_every=50, buffer=ReplayBuffer(capacity=512), **kw)
        model.save(os.path.join(CKPT, f"ood_{arm}_seed0.npz"))
        acc4, per4, _ = greedy_evaluate(model, t4, rng=np.random.default_rng(7))
        samp4 = sampled_evaluate(model, t4, group=8, rng=np.random.default_rng(8))
        acc, _, _ = greedy_evaluate(model, held, rng=np.random.default_rng(7))
        samp = sampled_evaluate(model, held, group=8, rng=np.random.default_rng(8))
        out[arm] = {
            "seconds": round(time.time() - t0, 1),
            "tier4_greedy": acc4,
            "tier4_sampled": samp4,
            "tier123_greedy": acc,
            "tier123_sampled": samp,
        }
        print(f"  tier4 greedy {acc4:.3f} sampled {samp4:.3f} | "
              f"tier1-3 greedy {acc:.3f} sampled {samp:.3f}", flush=True)
        save_json(out, "ood.json")
    save_json(out, "ood.json")
    print("ood done", flush=True)


if __name__ == "__main__":
    main()
