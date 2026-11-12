"""Extensions (seed 0, 100 updates, batch 16 x G=8, lr 3e-4).

E1  coefficient sensitivity: lambda_dc in {0.5, 2.0}, lambda_rep in
    {0.1, 1.0}, eta in {0, 16}, one at a time from the TACA defaults.
E2  replay capacity / staleness: capacity 64 (high turnover, recent
    pairs only) vs 4096 (long memory) vs default 512.
E3  aggregation (paper Sec. 3.3): trajectory-mean instead of token-mean
    for MT-GRPO and for TACA.
E4  retry cap (paper Sec. 3.4): max_errors=1 (episode ends on the
    second failed call) for MT-GRPO and for TACA.

Writes results/extensions.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import held_out_problems, save_json, sft_base, train_problems  # noqa: E402
from run_grid import run_arm  # noqa: E402
from taca_lab.grpo import ReplayBuffer, grpo_train  # noqa: E402
from taca_lab.evaluate import (  # noqa: E402
    diagnose_credit, diagnose_retries, greedy_evaluate, sampled_evaluate)

UPDATES = 100
BATCH = 16
GROUP = 8
LR = 3e-4


def run_custom(model, train, held, seed=0, **kw):
    """A single arm with arbitrary TACA kwargs; returns the same record
    shape as run_grid.run_arm."""
    buf = ReplayBuffer(capacity=kw.pop("replay_capacity", 512))
    t0 = time.time()
    stats = grpo_train(model, train, updates=UPDATES, batch=BATCH, group=GROUP,
                       lr=LR, rng=np.random.default_rng(5000 + seed),
                       log_every=50, buffer=buf, **kw)
    acc, per_tier, gstats = greedy_evaluate(model, held,
                                            rng=np.random.default_rng(7))
    sampled = sampled_evaluate(model, held, group=8,
                               rng=np.random.default_rng(8))
    from taca_lab.grpo import rollout_batch
    trajs = rollout_batch(model, held, group=8, temperature=1.0,
                          rng=np.random.default_rng(9))
    return {
        "seconds": round(time.time() - t0, 1),
        "train_reward_first10": float(np.mean([s["mean_reward"] for s in stats[:10]])),
        "train_reward_last10": float(np.mean([s["mean_reward"] for s in stats[-10:]])),
        "tool_use_first10": float(np.mean([s["tool_use_rate"] for s in stats[:10]])),
        "tool_use_last10": float(np.mean([s["tool_use_rate"] for s in stats[-10:]])),
        "exec_first10": float(np.mean([s["exec_success"] for s in stats[:10]])),
        "exec_last10": float(np.mean([s["exec_success"] for s in stats[-10:]])),
        "mean_projection_b": float(np.mean([s["mean_projection_b"] for s in stats])),
        "greedy_accuracy": acc,
        "greedy_per_tier": {str(k): v for k, v in per_tier.items()},
        "sampled_accuracy": sampled,
        "greedy_stats": gstats,
        "final_credit": diagnose_credit(trajs),
        "final_retries": diagnose_retries(trajs),
    }


def main():
    held = held_out_problems()
    train = train_problems(seed=0)
    base = sft_base(seed=0)
    # Resume: keep variants already finished (the VM can restart mid-run).
    results_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                "extensions.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            out = json.load(f)
    else:
        out = {}
    variants = {
        # E1: one-at-a-time coefficient changes from the TACA defaults
        "E1-lambda_dc-0.5": dict(use_ap=True, use_dc=True, use_replay=True,
                                lambda_dc=0.5, lambda_rep=0.5, eta=4.0),
        "E1-lambda_dc-2.0": dict(use_ap=True, use_dc=True, use_replay=True,
                                lambda_dc=2.0, lambda_rep=0.5, eta=4.0),
        "E1-lambda_rep-0.1": dict(use_ap=True, use_dc=True, use_replay=True,
                                 lambda_dc=1.0, lambda_rep=0.1, eta=4.0),
        "E1-lambda_rep-1.0": dict(use_ap=True, use_dc=True, use_replay=True,
                                 lambda_dc=1.0, lambda_rep=1.0, eta=4.0),
        "E1-eta-0": dict(use_ap=True, use_dc=True, use_replay=True,
                         lambda_dc=1.0, lambda_rep=0.5, eta=0.0),
        "E1-eta-16": dict(use_ap=True, use_dc=True, use_replay=True,
                          lambda_dc=1.0, lambda_rep=0.5, eta=16.0),
        # E2: replay capacity (staleness)
        "E2-capacity-64": dict(use_ap=True, use_dc=True, use_replay=True,
                              lambda_dc=1.0, lambda_rep=0.5, eta=4.0,
                              replay_capacity=64),
        "E2-capacity-4096": dict(use_ap=True, use_dc=True, use_replay=True,
                                lambda_dc=1.0, lambda_rep=0.5, eta=4.0,
                                replay_capacity=4096),
        # E3: trajectory-mean aggregation
        "E3-mtgrpo-trajmean": dict(use_ap=False, use_dc=False, use_replay=False,
                                  aggregation="trajectory"),
        "E3-taca-trajmean": dict(use_ap=True, use_dc=True, use_replay=True,
                                lambda_dc=1.0, lambda_rep=0.5, eta=4.0,
                                aggregation="trajectory"),
        # E4: retry cap
        "E4-mtgrpo-maxerr1": dict(use_ap=False, use_dc=False, use_replay=False,
                                 max_errors=1),
        "E4-taca-maxerr1": dict(use_ap=True, use_dc=True, use_replay=True,
                               lambda_dc=1.0, lambda_rep=0.5, eta=4.0,
                               max_errors=1),
    }
    for name, kw in variants.items():
        if name in out:
            print(f"=== {name} already done, skipping ===", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        model = base.copy()
        out[name] = run_custom(model, train, held, **kw)
        r = out[name]
        print(f"  sampled {r['sampled_accuracy']:.3f} greedy {r['greedy_accuracy']:.3f} "
              f"reward {r['train_reward_first10']:.3f} -> {r['train_reward_last10']:.3f} "
              f"tool {r['tool_use_first10']:.2f} -> {r['tool_use_last10']:.2f} "
              f"({r['seconds']}s)", flush=True)
        save_json(out, "extensions.json")
    save_json(out, "extensions.json")
    print("extensions done", flush=True)


if __name__ == "__main__":
    main()
