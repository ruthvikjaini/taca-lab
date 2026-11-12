"""Tests for the paper's credit-assignment machinery.

Prop. 3.1 (negative tool-credit condition), Eq. (5) (length-weighting
shift), Advantage Projection (Eq. 6), Decision Contrast (Eq. 7), the
dual-clipped surrogate, and the replay buffer.
"""

import numpy as np

from taca_lab.grpo import (
    DUAL_CLIP, ReplayBuffer, surrogate_loss_and_coef, taca_advantages,
    turn_level_advantages,
)


def test_turn_level_advantages():
    # all-equal rewards -> zero advantage everywhere
    adv = turn_level_advantages(np.array([1.0, 1.0]), np.array([2.0, 3.0]),
                                np.array([0, 0]))
    assert np.allclose(adv, 0.0)
    # mixed group: turn-weighted mean zero
    adv = turn_level_advantages(np.array([1.0, 0.0, 1.0, 0.0]),
                                np.array([2.0, 3.0, 1.0, 4.0]),
                                np.array([0, 0, 0, 0]))
    s = np.array([2.0, 3.0, 1.0, 4.0])
    assert abs((s * adv).sum()) < 1e-12
    # correct trajectories get positive advantage
    assert adv[0] > 0 and adv[1] < 0


def test_prop_3_1_identity_and_sign():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = 8
        rewards = (rng.random(g) > 0.5).astype(float)
        if rewards.std() == 0:
            continue
        turns = rng.integers(1, 7, size=g).astype(float)
        m = rng.integers(0, 12, size=g).astype(float)  # tool tokens
        lengths = rng.integers(10, 60, size=g).astype(float)
        adv = turn_level_advantages(rewards, turns, np.zeros(g, dtype=int))
        n = lengths.sum()
        c_tool = float((m * adv).sum() / n)
        # gap formula: (M * sigma / N) * (m_bar+ - m_bar-)
        mu = (turns * rewards).sum() / turns.sum()
        sigma = np.sqrt((turns * (rewards - mu) ** 2).sum() / turns.sum())
        pos = rewards > 0.5
        neg = ~pos
        m_plus = m[pos].sum() / turns[pos].sum() if pos.any() else 0.0
        m_minus = m[neg].sum() / turns[neg].sum() if neg.any() else 0.0
        rhs = (turns.sum() * sigma / n) * (m_plus - m_minus)
        assert abs(c_tool - rhs) < 1e-9, (c_tool, rhs)
        # sign condition: C_tool < 0 iff the gap is negative
        assert (c_tool < -1e-12) == ((m_plus - m_minus) < -1e-12)


def test_eq5_length_weighting_shift():
    rng = np.random.default_rng(1)
    for _ in range(200):
        g = 12
        rewards = (rng.random(g) > 0.5).astype(float)
        turns = rng.integers(1, 7, size=g).astype(float)
        m = rng.integers(0, 15, size=g).astype(float)
        lengths = rng.integers(10, 80, size=g).astype(float)
        adv = turn_level_advantages(rewards, turns, np.zeros(g, dtype=int))
        per_tok = (m / lengths) * adv
        c_token = float((lengths * per_tok).sum() / lengths.sum())
        c_traj = float(per_tok.mean())
        cov = float(np.mean((lengths - lengths.mean()) * (per_tok - per_tok.mean())))
        rhs = cov / lengths.mean()
        assert abs((c_token - c_traj) - rhs) < 1e-9


def _fake_masks():
    # 2 prompt groups x 4 trajectories; tool spans are 3 tokens each
    B, T = 8, 10
    tool = np.zeros((B, T))
    dec = np.zeros((B, T))
    dir_m = np.zeros((B, T))
    used = np.array([True, True, False, False, True, False, False, False])
    # group 0: trajs 0,1 tool; 2,3 direct. group 1: traj 4 tool; 5,6,7 direct
    for b in (0, 1, 4):
        tool[b, 2:5] = 1.0
        dec[b, 2] = 1.0
    for b in (2, 3, 5, 6, 7):
        dir_m[b, 2] = 1.0
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return tool, dec, dir_m, used, groups


def test_advantage_projection_properties():
    tool, dec, dir_m, used, groups = _fake_masks()
    # group 0: correct = traj 0 (tool), incorrect = traj 1 (tool) + 2,3
    rewards = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    turns = np.array([3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0])
    adv = turn_level_advantages(rewards, turns, groups)
    A, diag = taca_advantages(adv, tool, dec, dir_m, used, groups,
                              lambda_dc=0.0, rewards=rewards)
    # mean advantage over tool tokens in each group is >= 0 after projection
    for g in (0, 1):
        idx = np.where(groups == g)[0]
        z = tool[idx] > 0
        if z.sum():
            assert A[idx][z].mean() >= -1e-12
    # projection is a uniform shift: pairwise differences preserved
    a0 = A[0, 2:5].mean()
    a1 = A[1, 2:5].mean()
    assert abs((a0 - a1) - (adv[0] - adv[1])) < 1e-12
    # non-tool tokens unchanged
    assert np.allclose(A[:, 0], adv)
    # no-op when the mean tool advantage is already positive
    rewards2 = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    adv2 = turn_level_advantages(rewards2, turns, groups)
    A2, diag2 = taca_advantages(adv2, tool, dec, dir_m, used, groups,
                                lambda_dc=0.0, rewards=rewards2)
    assert diag2["mean_projection_b"] == 0.0
    assert np.allclose(A2, np.repeat(adv2[:, None], 10, axis=1))


def test_decision_contrast_signs_and_shrinkage():
    tool, dec, dir_m, used, groups = _fake_masks()
    turns = np.array([3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
    # tool-using trajectories win in group 0, lose in group 1
    rewards = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    adv = turn_level_advantages(rewards, turns, groups)
    A, diag = taca_advantages(adv, tool, dec, dir_m, used, groups,
                              lambda_dc=1.0, eta=4.0, rewards=rewards,
                              use_ap=False)
    # group 0: U > 0 -> tool decision tokens up, direct decision tokens down
    assert A[0, 2] > adv[0] and A[2, 2] < adv[2]
    # group 1: U < 0 -> reversed
    assert A[4, 2] < adv[4] and A[5, 2] > adv[5]
    # shrinkage: larger eta -> smaller |U|
    _, d0 = taca_advantages(adv, tool, dec, dir_m, used, groups,
                           lambda_dc=1.0, eta=0.0, rewards=rewards,
                           use_ap=False)
    _, d16 = taca_advantages(adv, tool, dec, dir_m, used, groups,
                            lambda_dc=1.0, eta=16.0, rewards=rewards,
                            use_ap=False)
    assert d16["mean_abs_contrast_u"] < d0["mean_abs_contrast_u"]
    # empty mode group -> no contrast
    rewards_all_tool = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    used_all = np.ones(8, dtype=bool)
    A3, _ = taca_advantages(adv, tool, dec, dir_m, used_all, groups,
                           lambda_dc=1.0, eta=4.0, rewards=rewards_all_tool,
                           use_ap=False)
    assert np.allclose(A3, np.repeat(adv[:, None], 10, axis=1))


def test_surrogate_dual_clip_active_set():
    rng = np.random.default_rng(0)
    B, T = 4, 6
    lp = rng.normal(0, 0.1, (B, T))
    resp = np.ones((B, T))
    A = np.array([[1.0] * T, [-1.0] * T, [1.0] * T, [-1.0] * T])
    # ratio 1: everything active
    loss, dcoef = surrogate_loss_and_coef(lp, A, resp, np.ones((B, T)))
    assert np.allclose(dcoef, A * resp / resp.sum())
    # A>0, ratio above 1+eps: clipped (zero gradient)
    ratio = np.ones((B, T))
    ratio[0] = 1.5
    _, dcoef = surrogate_loss_and_coef(lp, A, resp, ratio)
    assert dcoef[0].sum() == 0.0 and dcoef[2].sum() != 0.0
    # A<0, ratio below 1-eps: clipped; above dual clip 3.0: clipped
    ratio = np.ones((B, T))
    ratio[1] = 0.5
    ratio[3] = 3.5
    _, dcoef = surrogate_loss_and_coef(lp, A, resp, ratio)
    assert dcoef[1].sum() == 0.0 and dcoef[3].sum() == 0.0
    # A<0, ratio inside [1-eps, 3]: active
    ratio = np.ones((B, T))
    ratio[1] = 2.0
    _, dcoef = surrogate_loss_and_coef(lp, A, resp, ratio)
    assert np.allclose(dcoef[1], -2.0 * resp[1] / resp.sum())


def test_surrogate_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    B, T = 3, 5
    lp = rng.normal(-1.0, 0.3, (B, T))
    resp = (rng.random((B, T)) > 0.3).astype(float)
    A = rng.normal(0, 1, (B, T)) * resp
    ratio = np.ones((B, T))
    loss, dcoef = surrogate_loss_and_coef(lp, A, resp, ratio)
    # d loss / d lp = -dcoef (w * resp * scale, with sign from -phi)
    eps = 1e-6
    for b in range(B):
        for t in range(T):
            if resp[b, t] == 0:
                continue
            lp2 = lp.copy()
            lp2[b, t] += eps
            loss2, _ = surrogate_loss_and_coef(lp2, A, resp, ratio)
            fd = (loss2 - loss) / eps
            assert abs(fd - (-dcoef[b, t])) < 1e-4, (b, t, fd, -dcoef[b, t])


def test_surrogate_trajectory_mean_aggregation():
    rng = np.random.default_rng(3)
    B, T = 4, 6
    lp = rng.normal(-1.0, 0.3, (B, T))
    resp = np.ones((B, T))
    resp[0, 4:] = 0  # trajectory 0 is shorter: equal weight per traj
    A = np.ones((B, T))
    loss_tok, _ = surrogate_loss_and_coef(lp, A, resp, np.ones((B, T)),
                                         aggregation="token")
    loss_traj, _ = surrogate_loss_and_coef(lp, A, resp, np.ones((B, T)),
                                           aggregation="trajectory")
    # trajectory-mean weights the short trajectory equally, so it differs
    assert loss_tok != loss_traj
    # with equal lengths the two aggregations coincide
    resp2 = np.ones((B, T))
    l1, _ = surrogate_loss_and_coef(lp, A, resp2, np.ones((B, T)),
                                    aggregation="token")
    l2, _ = surrogate_loss_and_coef(lp, A, resp2, np.ones((B, T)),
                                    aggregation="trajectory")
    assert abs(l1 - l2) < 1e-12


def test_replay_buffer_fifo_cap():
    buf = ReplayBuffer(capacity=3)
    buf.add([("c1", "r1"), ("c2", "r2")])
    assert len(buf) == 2
    buf.add([("c3", "r3"), ("c4", "r4")])
    assert len(buf) == 3
    assert buf.items[0] == ("c2", "r2")  # oldest evicted
    rng = np.random.default_rng(0)
    assert len(buf.sample(2, rng)) == 2
    assert buf.sample(10, rng) and len(buf.sample(10, rng)) == 3
    assert ReplayBuffer().sample(5, rng) == []
