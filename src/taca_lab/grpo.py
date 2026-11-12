"""Multi-turn GRPO with Transition-Aware Credit Assignment (TACA).

Implements Lu et al., arXiv:2611.11330 (8 Nov 2026), Eqs. (1)-(10), on
the calculator testbed:

- Turn-level reward normalization (Eq. 1): each trajectory reward is
  replicated across its S_i turns before the group mean/variance.
- Token-mean PPO objective (Eq. 2) with dual clipping (c=3.0, App. I)
  and KL weight beta=0 (App. I, Table 7). One policy epoch per rollout
  batch, so the importance ratio is 1 on-policy and the clip is
  vacuous; the clip machinery is kept so the objective is exactly the
  paper's.
- Advantage Projection (Eq. 6): a uniform offset b = max(-mean advantage
  over tool-call tokens, 0) added to tool-call token advantages only.
- Decision Contrast (Eq. 7): U_x from the mean-reward gap between
  tool-using and direct-answer trajectories, shrunk by n/(n+eta), added
  with +lambda_dc at tool-call opening markers and -lambda_dc at
  answer-opening markers of direct trajectories (Eq. 9).
- Supervised Replay (Eq. 8): a FIFO buffer of (context, tool-calling
  turn) pairs from correct trajectories; mean NLL over the tool-call
  tokens of a sampled minibatch, weighted by lambda_rep (Eq. 10).

Observations are context only and are never scored.
"""

import numpy as np

from .env import (
    MAX_TURNS, MAX_TURN_TOKENS, execute_call, fold, observation_tokens,
    parse_turn, prompt_tokens,
)
from .vocab import CALL, PAD

DUAL_CLIP = 3.0  # App. I: negative-advantage dual-clipping parameter


# ---------------------------------------------------------------- rollout
class Trajectory:
    __slots__ = ("seq", "resp", "tool", "dec", "dir", "reward", "turns",
                 "used_tool", "n_calls", "n_errors", "n_retries", "group",
                 "replay_pairs", "turns_after_error", "calls_after_error",
                 "turns_after_success", "calls_after_success")

    def __init__(self, prompt, group):
        self.seq = list(prompt)
        self.resp = [False] * len(prompt)
        self.tool = [False] * len(prompt)
        self.dec = [False] * len(prompt)
        self.dir = [False] * len(prompt)
        self.reward = 0.0
        self.turns = 0
        self.used_tool = False
        self.n_calls = 0
        self.n_errors = 0
        self.n_retries = 0
        self.group = group
        self.replay_pairs = []  # (context, response_turn) for correct trajs
        # Sec. 3.4 diagnostics: what the policy does in the turn after
        # an execution outcome.
        self.turns_after_error = 0
        self.calls_after_error = 0
        self.turns_after_success = 0
        self.calls_after_success = 0


def rollout_batch(model, problems, group=8, temperature=1.0, rng=None,
                  max_errors=None, greedy=False):
    """Sample `group` multi-turn trajectories per problem.

    Returns a list of Trajectory objects (len = len(problems) * group).
    max_errors caps failure-driven retries: the episode ends (reward 0)
    on the (max_errors+1)-th failed call. None means unlimited.
    """
    rng = rng or np.random.default_rng(0)
    trajs = []
    for gi, (operands, ops) in enumerate(problems):
        prompt = prompt_tokens(operands, ops)
        for _ in range(group):
            trajs.append(Trajectory(prompt, gi))
    truth = [fold(o, p) for o, p in problems for _ in range(group)]
    active = list(range(len(trajs)))
    last_value = [None] * len(trajs)
    last_error = [False] * len(trajs)

    for _turn in range(MAX_TURNS):
        if not active:
            break
        seqs = [trajs[i].seq for i in active]
        gens = model.generate_turn(seqs, MAX_TURN_TOKENS, temperature=temperature,
                                   greedy=greedy, rng=rng)
        nxt_active = []
        for k, i in enumerate(active):
            tr = trajs[i]
            turn = gens[k]
            start = len(tr.seq)
            # Sec. 3.4: what happens in the turn after an outcome
            if last_error[i]:
                tr.turns_after_error += 1
                if turn and turn[0] == CALL:
                    tr.calls_after_error += 1
            elif tr.n_calls > 0:
                tr.turns_after_success += 1
                if turn and turn[0] == CALL:
                    tr.calls_after_success += 1
            tr.seq += turn
            tr.resp += [True] * len(turn)
            tr.tool += [False] * len(turn)
            tr.dec += [False] * len(turn)
            tr.dir += [False] * len(turn)
            tr.turns += 1
            kind, payload = parse_turn(turn)
            if kind == "call":
                tr.used_tool = True
                tr.n_calls += 1
                if last_error[i]:
                    tr.n_retries += 1
                # the whole <call>...</call> span is tool-call tokens
                for j in range(start, start + len(turn)):
                    tr.tool[j] = True
                tr.dec[start] = True  # opening marker: decision token
                value, ok = execute_call(payload, last_value[i])
                if ok:
                    last_value[i] = value
                    last_error[i] = False
                    tr.replay_pairs.append((list(tr.seq[:start]), list(turn)))
                else:
                    last_error[i] = True
                    tr.n_errors += 1
                    if max_errors is not None and tr.n_errors > max_errors:
                        continue  # episode ends on the excess error
                obs = observation_tokens(value if ok else None, ok)
                tr.seq += obs
                tr.resp += [False] * len(obs)
                tr.tool += [False] * len(obs)
                tr.dec += [False] * len(obs)
                tr.dir += [False] * len(obs)
                nxt_active.append(i)
            elif kind == "answer":
                if payload is not None and payload == truth[i]:
                    tr.reward = 1.0
                # a direct answer marks the decision token (Z_dir)
                if not tr.used_tool:
                    tr.dir[start] = True
            else:  # garbage turn: episode ends, reward 0
                pass
        active = nxt_active
    # only correct trajectories contribute their tool-calling turns
    for tr in trajs:
        if tr.reward != 1.0:
            tr.replay_pairs = []
    return trajs


def _padded(trajs):
    """Pad trajectories to arrays in the shifted frame.

    Returns tokens [B, W], resp/tool/dec/dir masks [B, W-1] (mask[b, j]
    marks token j+1), plus per-trajectory reward/turns/group arrays.
    """
    B = len(trajs)
    W = max(len(t.seq) for t in trajs)
    tokens = np.full((B, W), PAD, dtype=int)
    resp = np.zeros((B, W - 1), dtype=float)
    tool = np.zeros((B, W - 1), dtype=float)
    dec = np.zeros((B, W - 1), dtype=float)
    dir_m = np.zeros((B, W - 1), dtype=float)
    for b, tr in enumerate(trajs):
        L = len(tr.seq)
        tokens[b, :L] = tr.seq
        for j in range(L - 1):
            if tr.resp[j + 1]:
                resp[b, j] = 1.0
            if tr.tool[j + 1]:
                tool[b, j] = 1.0
            if tr.dec[j + 1]:
                dec[b, j] = 1.0
            if tr.dir[j + 1]:
                dir_m[b, j] = 1.0
    rewards = np.array([t.reward for t in trajs])
    turns = np.array([t.turns for t in trajs], dtype=float)
    groups = np.array([t.group for t in trajs])
    return tokens, resp, tool, dec, dir_m, rewards, turns, groups


# ---------------------------------------------------------------- advantages
def turn_level_advantages(rewards, turns, groups):
    """Eq. (1): replicate each reward across its turns, then standardize
    within each prompt group. Returns per-trajectory advantages."""
    adv = np.zeros_like(rewards)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        r, s = rewards[idx], turns[idx]
        m = s.sum()
        mu = (s * r).sum() / m if m > 0 else 0.0
        var = (s * (r - mu) ** 2).sum() / m if m > 0 else 0.0
        sd = np.sqrt(var)
        if sd > 1e-8:
            adv[idx] = (r - mu) / sd
    return adv


def taca_advantages(adv_traj, tool, dec, dir_m, used_tool, groups,
                    lambda_dc=1.0, eta=4.0, rewards=None,
                    use_ap=True, use_dc=True):
    """Eqs. (6), (7), (9): per-token advantages with Advantage Projection
    and Decision Contrast. Returns (A_tok [B, T-1], diagnostics dict)."""
    B = adv_traj.shape[0]
    A = np.repeat(adv_traj[:, None], tool.shape[1], axis=1).astype(float)
    b_mean, u_mean, u_abs = 0.0, 0.0, 0.0
    n_groups = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        n_groups += 1
        if use_ap:
            # Advantage Projection (Eq. 6) over tool-call tokens
            z = tool[idx] > 0
            if z.sum() > 0:
                mean_z = float(A[idx][z].mean())
                b = max(-mean_z, 0.0)
                b_mean += b
                if b > 0:
                    A[idx] += b * tool[idx]
        if use_dc and rewards is not None:
            # Decision Contrast (Eq. 7)
            t_idx = [i for i in idx if used_tool[i]]
            n_idx = [i for i in idx if not used_tool[i]]
            if t_idx and n_idx:
                r_t = float(rewards[t_idx].mean())
                r_n = float(rewards[n_idx].mean())
                n = min(len(t_idx), len(n_idx))
                u = n / (n + eta) * (r_t - r_n)
                u_mean += u
                u_abs += abs(u)
                for i in t_idx:
                    A[i] += lambda_dc * u * dec[i]
                for i in n_idx:
                    A[i] -= lambda_dc * u * dir_m[i]
    diag = {
        "mean_projection_b": b_mean / max(n_groups, 1),
        "mean_contrast_u": u_mean / max(n_groups, 1),
        "mean_abs_contrast_u": u_abs / max(n_groups, 1),
    }
    return A, diag


# ---------------------------------------------------------------- replay
class ReplayBuffer:
    """FIFO buffer of (context, tool-calling turn) pairs from correct
    trajectories (paper Sec. 4.3)."""

    def __init__(self, capacity=512):
        self.capacity = capacity
        self.items = []

    def add(self, pairs):
        self.items.extend(pairs)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    def sample(self, n, rng):
        if not self.items:
            return []
        idx = rng.integers(0, len(self.items), size=min(n, len(self.items)))
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)


# ---------------------------------------------------------------- update
def surrogate_loss_and_coef(lp, A, resp, ratio, clip=0.2, aggregation="token"):
    """PPO clipped surrogate (dual clip c=3.0 for negative advantages).

    loss = -(1/N) sum phi(ratio, A) over scored tokens, where phi is the
    clipped surrogate. Returns (loss, dcoef): d loss / d logits =
    dcoef * (probs - onehot), because d lp / d logits = onehot - probs.
    On-policy with one epoch per rollout batch the ratio is 1 and the
    clip is vacuous; the formula is kept so the objective is the
    paper's (Eq. 2 + App. I dual clip).
    """
    B = lp.shape[0]
    a_tok = A * resp
    active = np.where(a_tok > 0, ratio <= 1 + clip,
                      np.where(a_tok < 0,
                               (ratio >= 1 - clip) & (ratio <= DUAL_CLIP),
                               False))
    w = np.where(active, a_tok * ratio, 0.0)
    if aggregation == "token":
        n_resp = resp.sum()
        scale = 1.0 / max(n_resp, 1)
        loss = -(w * lp * resp).sum() * scale
        dcoef = w * resp * scale
    elif aggregation == "trajectory":
        n_tok = resp.sum(axis=1, keepdims=True)
        per_traj = (w * lp * resp).sum(axis=1) / np.maximum(n_tok[:, 0], 1)
        loss = -per_traj.mean()
        dcoef = w * resp / np.maximum(n_tok, 1) / B
    else:
        raise ValueError(aggregation)
    return float(loss), dcoef


def grpo_update(model, problems, group=8, lr=1e-4, clip=0.2, rng=None,
                buffer=None, lambda_dc=1.0, lambda_rep=0.5, eta=4.0,
                use_ap=True, use_dc=True, use_replay=True,
                aggregation="token", max_errors=None):
    """One TACA / MT-GRPO update on a batch of problems.

    With use_ap/use_dc/use_replay all False this is plain MT-GRPO.
    Returns a stats dict.
    """
    rng = rng or np.random.default_rng(0)
    trajs = rollout_batch(model, problems, group=group, temperature=1.0,
                          rng=rng, max_errors=max_errors)
    tokens, resp, tool, dec, dir_m, rewards, turns, groups = _padded(trajs)
    used_tool = np.array([t.used_tool for t in trajs])
    adv_traj = turn_level_advantages(rewards, turns, groups)
    signal = float(np.mean([rewards[groups == g].std() > 1e-8
                            for g in np.unique(groups)]))
    A, diag = taca_advantages(adv_traj, tool, dec, dir_m, used_tool, groups,
                              lambda_dc=lambda_dc, eta=eta, rewards=rewards,
                              use_ap=use_ap, use_dc=use_dc)

    # --- PPO surrogate gradient (token-mean, dual clip for A < 0) ---
    logits, cache = model.forward(tokens)
    pred = logits[:, :-1, :]
    mx = pred.max(axis=-1, keepdims=True)
    e = np.exp(pred - mx)
    probs = e / e.sum(axis=-1, keepdims=True)
    tgt = tokens[:, 1:]
    B, T1 = tgt.shape
    lp_new = np.log(probs[np.arange(B)[:, None], np.arange(T1)[None, :], tgt] + 1e-12)
    # On-policy with one epoch per rollout batch: ratio is exactly 1.
    ratio = np.ones_like(lp_new)
    loss_pg, dcoef = surrogate_loss_and_coef(lp_new, A, resp, ratio,
                                             clip=clip, aggregation=aggregation)

    d = probs.copy()
    d[np.arange(B)[:, None], np.arange(T1)[None, :], tgt] -= 1.0
    d *= dcoef[:, :, None]
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1, :] = d
    grads = model._backward(dlogits, cache)

    # --- Supervised Replay (Eq. 8) ---
    loss_rep = 0.0
    n_rep_pairs = 0
    if use_replay and buffer is not None and len(buffer) > 0:
        pairs = buffer.sample(32, rng)
        n_rep_pairs = len(pairs)
        if pairs:
            rt, rm = _replay_arrays(pairs)
            l_rep, g_rep, _ = model.loss_and_grads(rt, rm)
            loss_rep = float(l_rep)
            for k in grads:
                grads[k] += lambda_rep * g_rep[k]

    model.adam_step(grads, lr)

    # refill the buffer with this batch's correct tool-calling turns
    if buffer is not None:
        pairs = []
        for tr in trajs:
            pairs.extend(tr.replay_pairs)
        buffer.add(pairs)

    total_calls = sum(t.n_calls for t in trajs)
    total_errors = sum(t.n_errors for t in trajs)
    stats = {
        "mean_reward": float(rewards.mean()),
        "signal_frac": signal,
        "loss_pg": float(loss_pg),
        "loss_rep": loss_rep,
        "n_rep_pairs": n_rep_pairs,
        "buffer_size": len(buffer) if buffer is not None else 0,
        "tool_use_rate": float(used_tool.mean()),
        "mean_calls": float(np.mean([t.n_calls for t in trajs])),
        "mean_errors": float(np.mean([t.n_errors for t in trajs])),
        "mean_retries": float(np.mean([t.n_retries for t in trajs])),
        "exec_success": float(1.0 - total_errors / max(total_calls, 1)),
        "mean_projection_b": diag["mean_projection_b"],
        "mean_contrast_u": diag["mean_contrast_u"],
        "mean_abs_contrast_u": diag["mean_abs_contrast_u"],
    }
    return stats


def _replay_arrays(pairs):
    """Pad (context, response) pairs; mask marks response tokens."""
    W = max(len(c) + len(r) for c, r in pairs)
    tokens = np.full((len(pairs), W), PAD, dtype=int)
    mask = np.zeros((len(pairs), W), dtype=float)
    for i, (c, r) in enumerate(pairs):
        tokens[i, :len(c)] = c
        tokens[i, len(c):len(c) + len(r)] = r
        mask[i, len(c):len(c) + len(r)] = 1.0
    return tokens, mask


def grpo_train(model, train_problems, updates=150, batch=24, group=8,
               lr=1e-4, clip=0.2, rng=None, log_every=25, buffer=None,
               **taca_kwargs):
    """Run GRPO/TACA for `updates` steps. Returns per-update stats."""
    rng = rng or np.random.default_rng(0)
    model.reset_optimizer()
    if buffer is None:
        buffer = ReplayBuffer()
    stats = []
    for u in range(updates):
        idx = rng.integers(0, len(train_problems), size=batch)
        problems = [train_problems[i] for i in idx]
        st = grpo_update(model, problems, group=group, lr=lr, clip=clip,
                         rng=rng, buffer=buffer, **taca_kwargs)
        st["update"] = u
        stats.append(st)
        if log_every and (u + 1) % log_every == 0:
            print(f"  update {u+1}/{updates} reward {st['mean_reward']:.3f} "
                  f"tool {st['tool_use_rate']:.2f} "
                  f"exec {st['exec_success']:.2f} "
                  f"b {st['mean_projection_b']:.3f} "
                  f"buf {st['buffer_size']}", flush=True)
    return stats
