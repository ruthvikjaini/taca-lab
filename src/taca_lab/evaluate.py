"""Greedy / sampled evaluation and paper Sec. 3 diagnostics.

Diagnostics (run on a rollout batch from any policy):
- D1 (Prop. 3.1): tool-token gap m_bar+ - m_bar- per prompt group and
  the aggregate tool credit C_tool = (1/N) sum_i m_i A_i^turn; the
  proposition says C_tool < 0 iff the gap is negative.
- D2 (Sec. 3.3, Eq. 4-5): token-mean vs trajectory-mean aggregate tool
  credit per batch, and the fraction of batches where token-mean
  aggregation flips positive credit negative (paper: 73.6%).
- D3 (Sec. 3.4): P(call next turn | failed execution) vs P(call next
  turn | successful execution), and retry rates in correct vs
  incorrect trajectories.
"""

import numpy as np

from .env import MAX_TURNS, execute_call, fold, observation_tokens, parse_turn, prompt_tokens
from .grpo import _padded, rollout_batch, turn_level_advantages
from .vocab import PAD


def greedy_evaluate(model, problems, rng=None):
    """Greedy rollouts; returns (accuracy, per-tier accuracy, stats)."""
    rng = rng or np.random.default_rng(0)
    trajs = rollout_batch(model, problems, group=1, temperature=1.0,
                          rng=rng, greedy=True)
    acc = float(np.mean([t.reward for t in trajs]))
    tiers = {}
    for (operands, ops), tr in zip(problems, trajs):
        tier = len(ops)
        tiers.setdefault(tier, []).append(tr.reward)
    per_tier = {t: float(np.mean(v)) for t, v in sorted(tiers.items())}
    stats = {
        "accuracy": acc,
        "per_tier": per_tier,
        "tool_use_rate": float(np.mean([t.used_tool for t in trajs])),
        "mean_calls": float(np.mean([t.n_calls for t in trajs])),
        "mean_errors": float(np.mean([t.n_errors for t in trajs])),
        "exec_success": float(1.0 - sum(t.n_errors for t in trajs)
                              / max(sum(t.n_calls for t in trajs), 1)),
        "no_tool_failure_rate": float(np.mean([
            1.0 if (not t.used_tool and t.reward == 0.0) else 0.0
            for t in trajs])),
        "mean_retries": float(np.mean([t.n_retries for t in trajs])),
    }
    return acc, per_tier, stats


def sampled_evaluate(model, problems, group=8, rng=None):
    """Sampled pass@1 (temperature 1.0) on held-out problems."""
    rng = rng or np.random.default_rng(0)
    trajs = rollout_batch(model, problems, group=group, temperature=1.0, rng=rng)
    return float(np.mean([t.reward for t in trajs]))


def diagnose_credit(trajs):
    """D1 + D2 on one rollout batch (list of Trajectory, group = problem).

    Returns a dict with the per-group tool-token gap, C_tool (token-mean
    and trajectory-mean), and whether the Prop. 3.1 sign condition holds.
    """
    tokens, resp, tool, dec, dir_m, rewards, turns, groups = _padded(trajs)
    adv = turn_level_advantages(rewards, turns, groups)
    n_tok = resp.sum(axis=1)
    m = tool.sum(axis=1)  # tool tokens per trajectory
    gaps, c_token, c_traj = [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        s = turns[idx]
        m_turn = m[idx] / np.maximum(s, 1)
        pos = rewards[idx] > 0.5
        neg = ~pos
        gap = (m_turn[pos].mean() if pos.any() else 0.0) - \
              (m_turn[neg].mean() if neg.any() else 0.0)
        gaps.append(gap)
        # C_tool under token-mean aggregation (Prop. 3.1, Eq. 3)
        n = resp[idx].sum()
        c_token.append(float((m[idx] * adv[idx]).sum() / max(n, 1)))
        # C_tool under trajectory-mean aggregation (Eq. 4)
        c_traj.append(float(np.mean((m[idx] / np.maximum(n_tok[idx], 1)) * adv[idx])))
    return {
        "mean_gap": float(np.mean(gaps)),
        "frac_negative_gap": float(np.mean([g < 0 for g in gaps])),
        "mean_c_token": float(np.mean(c_token)),
        "mean_c_traj": float(np.mean(c_traj)),
        "frac_c_token_negative": float(np.mean([c < 0 for c in c_token])),
        "frac_flip": float(np.mean([
            ct > 0 and ck < 0 for ct, ck in zip(c_traj, c_token)])),
        "n_groups": len(gaps),
    }


def diagnose_retries(trajs):
    """D3: retry behaviour from one rollout batch, using the per-turn
    outcome counters recorded during rollout.

    A retry is a tool call in the turn immediately after a failed
    execution. Returns P(call | error), P(call | success), and mean
    retries split by final outcome.
    """
    tae = sum(t.turns_after_error for t in trajs)
    cae = sum(t.calls_after_error for t in trajs)
    tas = sum(t.turns_after_success for t in trajs)
    cas = sum(t.calls_after_success for t in trajs)
    rc = [t.n_retries for t in trajs if t.reward > 0.5]
    ri = [t.n_retries for t in trajs if t.reward <= 0.5]
    return {
        "p_call_after_error": (cae / tae if tae else float("nan")),
        "p_call_after_success": (cas / tas if tas else float("nan")),
        "n_after_error": tae,
        "n_after_success": tas,
        "mean_retries_correct": float(np.mean(rc)) if rc else 0.0,
        "mean_retries_incorrect": float(np.mean(ri)) if ri else 0.0,
    }
