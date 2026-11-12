"""Paper Sec. 3 diagnostics on the SFT warm-start (before any RL).

D1 (Prop. 3.1): per-group tool-token gap and aggregate tool credit.
D2 (Sec. 3.3): token-mean vs trajectory-mean credit; the fraction of
groups where token-mean aggregation flips positive credit negative.
D3 (Sec. 3.4): failure-driven retries (call rate after an error vs
after a success; retries in correct vs incorrect trajectories).

Writes results/diagnostics.json.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import held_out_problems, save_json, sft_base  # noqa: E402
from taca_lab.evaluate import diagnose_credit, diagnose_retries  # noqa: E402
from taca_lab.grpo import rollout_batch  # noqa: E402


def main():
    model = sft_base(seed=0)
    problems = held_out_problems()
    out = {"per_tier": {}, "overall": {}}
    all_trajs = []
    for tier in (1, 2, 3):
        probs = [p for p in problems if len(p[1]) == tier]
        trajs = rollout_batch(model, probs, group=8, temperature=1.0,
                              rng=np.random.default_rng(100 + tier))
        all_trajs.extend(trajs)
        credit = diagnose_credit(trajs)
        retries = diagnose_retries(trajs)
        out["per_tier"][str(tier)] = {
            "sampled_reward": float(np.mean([t.reward for t in trajs])),
            "tool_use_rate": float(np.mean([t.used_tool for t in trajs])),
            "mean_calls": float(np.mean([t.n_calls for t in trajs])),
            "mean_errors": float(np.mean([t.n_errors for t in trajs])),
            "exec_success": float(1.0 - sum(t.n_errors for t in trajs)
                                  / max(sum(t.n_calls for t in trajs), 1)),
            "credit": credit,
            "retries": retries,
        }
        print(f"tier {tier}: reward {out['per_tier'][str(tier)]['sampled_reward']:.3f} "
              f"C_token {credit['mean_c_token']:+.4f} "
              f"frac_neg_gap {credit['frac_negative_gap']:.2f} "
              f"flip {credit['frac_flip']:.2f} "
              f"P(call|err) {retries['p_call_after_error']:.2f} "
              f"P(call|ok) {retries['p_call_after_success']:.2f}", flush=True)
    out["overall"]["credit"] = diagnose_credit(all_trajs)
    out["overall"]["retries"] = diagnose_retries(all_trajs)
    print("overall credit:", out["overall"]["credit"], flush=True)
    print("overall retries:", out["overall"]["retries"], flush=True)
    save_json(out, "diagnostics.json")


if __name__ == "__main__":
    main()
