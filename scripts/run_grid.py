"""Main grid: MT-GRPO vs TACA vs one-component-out ablations (paper
Table 3), 3 seeds, 100 updates each.

Arms:
  mt-grpo    plain multi-turn GRPO (Eqs. 1-2, dual clip, beta=0)
  taca       full TACA (AP + DC + replay)
  taca-no-ap TACA without Advantage Projection
  taca-no-dc TACA without Decision Contrast
  taca-no-sr TACA without Supervised Replay

Batch 16 prompts x G=8, lr 3e-4 (calibrated in the pilot: 20 steps at
3e-4 lift sampled reward 0.41 -> 0.70 with tool use and execution
success both rising; 1e-4 learns half as fast). Held-out eval is
sampled pass@1 (G=8, temp 1.0) on 240 problems plus greedy accuracy,
because the SFT base is already at greedy 1.0 on easy held-out sets.

Writes results/grid.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import held_out_problems, save_json, sft_base, train_problems  # noqa: E402
from taca_lab.evaluate import (  # noqa: E402
    diagnose_credit, diagnose_retries, greedy_evaluate, sampled_evaluate)
from taca_lab.grpo import ReplayBuffer, grpo_train  # noqa: E402

ARMS = {
    "mt-grpo": dict(use_ap=False, use_dc=False, use_replay=False),
    "taca": dict(use_ap=True, use_dc=True, use_replay=True),
    "taca-no-ap": dict(use_ap=False, use_dc=True, use_replay=True),
    "taca-no-dc": dict(use_ap=True, use_dc=False, use_replay=True),
    "taca-no-sr": dict(use_ap=True, use_dc=True, use_replay=False),
}
UPDATES = 100
BATCH = 16
GROUP = 8
LR = 3e-4


def run_arm(model, arm, train, held, seed):
    kw = dict(ARMS[arm])
    kw.update(lambda_dc=1.0, lambda_rep=0.5, eta=4.0)
    buf = ReplayBuffer(capacity=512)
    t0 = time.time()
    stats = grpo_train(model, train, updates=UPDATES, batch=BATCH, group=GROUP,
                       lr=LR, rng=np.random.default_rng(5000 + seed),
                       log_every=25, buffer=buf, **kw)
    # final evals
    acc, per_tier, gstats = greedy_evaluate(model, held,
                                            rng=np.random.default_rng(7))
    sampled = sampled_evaluate(model, held, group=8,
                               rng=np.random.default_rng(8))
    from taca_lab.grpo import rollout_batch
    trajs = rollout_batch(model, held, group=8, temperature=1.0,
                          rng=np.random.default_rng(9))
    credit = diagnose_credit(trajs)
    retries = diagnose_retries(trajs)
    return {
        "arm": arm,
        "seed": seed,
        "seconds": round(time.time() - t0, 1),
        "train_reward_first10": float(np.mean([s["mean_reward"] for s in stats[:10]])),
        "train_reward_last10": float(np.mean([s["mean_reward"] for s in stats[-10:]])),
        "tool_use_first10": float(np.mean([s["tool_use_rate"] for s in stats[:10]])),
        "tool_use_last10": float(np.mean([s["tool_use_rate"] for s in stats[-10:]])),
        "exec_first10": float(np.mean([s["exec_success"] for s in stats[:10]])),
        "exec_last10": float(np.mean([s["exec_success"] for s in stats[-10:]])),
        "mean_projection_b": float(np.mean([s["mean_projection_b"] for s in stats])),
        "mean_abs_contrast_u": float(np.mean([s["mean_abs_contrast_u"] for s in stats])),
        "greedy_accuracy": acc,
        "greedy_per_tier": {str(k): v for k, v in per_tier.items()},
        "sampled_accuracy": sampled,
        "greedy_stats": gstats,
        "final_credit": credit,
        "final_retries": retries,
        "curve": [
            {"update": s["update"], "reward": s["mean_reward"],
             "tool": s["tool_use_rate"], "exec": s["exec_success"],
             "b": s["mean_projection_b"], "signal": s["signal_frac"]}
            for s in stats
        ],
    }


def main():
    held = held_out_problems()
    # Resume: keep arms already finished (the VM can restart mid-grid).
    results_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                "grid.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {"arms": {}, "config": {"updates": UPDATES, "batch": BATCH,
                                           "group": GROUP, "lr": LR,
                                           "lambda_dc": 1.0, "lambda_rep": 0.5,
                                           "eta": 4.0, "replay_capacity": 512,
                                           "replay_per_update": 32}}
    for seed in (0, 1, 2):
        base = sft_base(seed=seed)
        train = train_problems(seed=seed)
        for arm in ARMS:
            key = f"{arm}/seed{seed}"
            if key in results["arms"]:
                print(f"=== seed {seed} arm {arm} already done, skipping ===",
                      flush=True)
                continue
            print(f"=== seed {seed} arm {arm} ===", flush=True)
            model = base.copy()
            results["arms"][key] = run_arm(model, arm, train, held, seed)
            r = results["arms"][key]
            print(f"  sampled {r['sampled_accuracy']:.3f} greedy {r['greedy_accuracy']:.3f} "
                  f"reward {r['train_reward_first10']:.3f} -> {r['train_reward_last10']:.3f} "
                  f"tool {r['tool_use_first10']:.2f} -> {r['tool_use_last10']:.2f} "
                  f"exec {r['exec_first10']:.2f} -> {r['exec_last10']:.2f} "
                  f"({r['seconds']}s)", flush=True)
            save_json(results, "grid.json")
    save_json(results, "grid.json")
    print("grid done", flush=True)


if __name__ == "__main__":
    main()
