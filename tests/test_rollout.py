"""Integration tests: rollouts, GRPO update direction, replay wiring."""

import numpy as np
import pytest

from taca_lab.env import make_problem, make_sft_dataset
from taca_lab.grpo import ReplayBuffer, _padded, grpo_update, rollout_batch
from taca_lab.model import Transformer
from taca_lab.sft import sft_train
from taca_lab.vocab import VOCAB_SIZE


@pytest.fixture(scope="module")
def base_model():
    """A moderately SFT-trained base policy (built once per module)."""
    model = Transformer(VOCAB_SIZE, max_len=160, d=64, seed=0)
    data = make_sft_dataset(np.random.default_rng(101), n=2000,
                            tier_p=(0.35, 0.35, 0.30), direct_p=0.5)
    sft_train(model, data, epochs=2, batch=64, lr=3e-3,
              rng=np.random.default_rng(202), label_smoothing=0.05)
    return model


def test_rollout_invariants(base_model):
    rng = np.random.default_rng(0)
    problems = [make_problem(rng, tier=t) for t in (1, 2, 3)] * 4
    trajs = rollout_batch(base_model, problems, group=4, temperature=1.0,
                          rng=np.random.default_rng(7))
    assert len(trajs) == len(problems) * 4
    for tr in trajs:
        assert tr.turns <= 6
        assert len(tr.seq) == len(tr.resp) == len(tr.tool) == len(tr.dec)
        assert tr.reward in (0.0, 1.0)
        assert tr.n_errors <= tr.n_calls
        assert tr.n_retries <= tr.n_calls
        # replay pairs only from correct trajectories, and only tool turns
        if tr.reward == 1.0:
            for ctx, turn in tr.replay_pairs:
                assert turn[0] == 14  # <call>
        else:
            assert tr.replay_pairs == []
    assert any(t.used_tool for t in trajs)
    assert any(t.reward == 1.0 for t in trajs)


def test_rollout_max_errors_cap(base_model):
    rng = np.random.default_rng(0)
    problems = [make_problem(rng, tier=2) for _ in range(4)]
    trajs = rollout_batch(base_model, problems, group=4, temperature=1.0,
                          rng=np.random.default_rng(3), max_errors=0)
    assert all(t.n_errors <= 1 for t in trajs)


def _logprob_gap(model, trajs):
    """Mean response logprob of correct trajectories minus incorrect ones."""
    tokens, resp, *_ = _padded(trajs)
    lps = model.logprobs(tokens)
    vals = []
    for b, tr in enumerate(trajs):
        m = resp[b] > 0
        vals.append((tr.reward, float(lps[b][m].mean()) if m.any() else 0.0))
    good = [v for r, v in vals if r == 1.0]
    bad = [v for r, v in vals if r == 0.0]
    return float(np.mean(good)) - float(np.mean(bad)), len(good), len(bad)


def test_grpo_update_raises_rewarded_logprob(base_model):
    """One update must raise the mean logprob of correct trajectories
    relative to incorrect ones (the Day 17 sign regression test).

    Measured on the update batch's own trajectories (rolled out here
    with the same rng the update will use, so they are identical —
    guarded by the mean_reward check below). A fresh-sample gap is
    not used: with a weak base only 2-3 of 96 sampled trajectories
    are correct, so the fresh-sample gap is dominated by which
    trajectories happen to be sampled, and it flips sign across
    measurement seeds even on a strong base. The update batch is
    the direct sign check: the PPO surrogate gradient raises
    correct-trajectory logprob and lowers incorrect-trajectory
    logprob on exactly these trajectories.
    """
    model = base_model.copy()
    rng = np.random.default_rng(0)
    problems = [make_problem(rng, tier=t) for t in (1, 2, 3)] * 4

    # The exact batch grpo_update will train on (same rng, same model).
    trajs = rollout_batch(model, problems, group=8, temperature=1.0,
                          rng=np.random.default_rng(5))
    before, n_good, n_bad = _logprob_gap(model, trajs)
    assert n_good > 0 and n_bad > 0
    st = grpo_update(model, problems, group=8, lr=3e-4, rng=np.random.default_rng(5),
                     buffer=ReplayBuffer(), use_replay=False)
    assert st["signal_frac"] > 0
    # Guard: the update really trained on the trajectories we measured.
    assert abs(float(np.mean([t.reward for t in trajs])) - st["mean_reward"]) < 1e-9
    after, _, _ = _logprob_gap(model, trajs)
    assert after > before, (before, after)


def test_replay_buffer_fills_from_correct_trajectories(base_model):
    model = base_model.copy()
    rng = np.random.default_rng(0)
    problems = [make_problem(rng, tier=2) for _ in range(8)]
    buf = ReplayBuffer(capacity=512)
    grpo_update(model, problems, group=8, lr=1e-4, rng=np.random.default_rng(4),
                buffer=buf, use_replay=True)
    # buffer holds (context, turn) pairs from correct trajectories only
    for ctx, turn in buf.items:
        assert turn[0] == 14
        assert len(ctx) > 0
