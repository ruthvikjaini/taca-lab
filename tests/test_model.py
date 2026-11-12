"""Tests for the transformer: backprop vs finite differences and
generation behaviour."""

import numpy as np

from taca_lab.model import Transformer
from taca_lab.vocab import ANS, ANS_END, CALL, CALL_END, EOS, PAD, VOCAB_SIZE


def _small_model():
    return Transformer(VOCAB_SIZE, max_len=32, d=16, seed=0)


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = _small_model()
    # shrink to d=16: rebuild params already done in __init__ with d=16
    tokens = rng.integers(0, VOCAB_SIZE, size=(2, 6))
    mask = np.zeros((2, 6))
    mask[:, 3:] = 1.0
    loss, grads, _ = model.loss_and_grads(tokens, mask)
    for key in ("Wte", "Wq", "Wo", "W1", "Wh", "ln1_g"):
        g = grads[key]
        idx = tuple(rng.integers(0, s) for s in g.shape)
        orig = model.params[key][idx]
        eps = 1e-5
        model.params[key][idx] = orig + eps
        lp, _, _ = model.loss_and_grads(tokens, mask)
        model.params[key][idx] = orig - eps
        lm, _, _ = model.loss_and_grads(tokens, mask)
        model.params[key][idx] = orig
        fd = (lp - lm) / (2 * eps)
        assert abs(fd - g[idx]) < 1e-4, (key, idx, fd, g[idx])


def test_loss_decreases_on_repeated_data():
    rng = np.random.default_rng(1)
    model = _small_model()
    tokens = rng.integers(0, VOCAB_SIZE, size=(4, 8))
    mask = np.ones((4, 8))
    mask[:, :2] = 0
    l0, _, _ = model.loss_and_grads(tokens, mask)
    for _ in range(30):
        loss, grads, _ = model.loss_and_grads(tokens, mask)
        model.adam_step(grads, 1e-2)
    l1, _, _ = model.loss_and_grads(tokens, mask)
    assert l1 < l0


def test_generate_turn_stops_and_pads():
    model = _small_model()
    seqs = [[CALL, 3, 10], [ANS, 4]]
    rng = np.random.default_rng(0)
    out = model.generate_turn(seqs, 6, temperature=1.0, greedy=True, rng=rng)
    assert len(out) == 2
    for row in out:
        assert len(row) <= 6
        # stops at the first stop id
        stops = [i for i, t in enumerate(row) if t in (CALL_END, ANS_END, EOS)]
        if stops:
            assert len(row) == stops[0] + 1


def test_generate_turn_padding_invariance():
    """A row's greedy continuation must not depend on batch companions."""
    model = _small_model()
    rng = np.random.default_rng(0)
    row = [CALL, 3, 10, 4]
    alone = model.generate_turn([row], 4, greedy=True, rng=rng)
    rng2 = np.random.default_rng(0)
    batch = model.generate_turn(
        [row, [ANS, 1, 2, 3, 4, 5, 6, 7], [3, 10, 4]], 4, greedy=True, rng=rng2)
    assert batch[0] == alone[0]


def test_logprobs_shape_and_shift():
    model = _small_model()
    tokens = np.array([[3, 10, 4, CALL_END, PAD, PAD],
                       [ANS, 1, 4, ANS_END, EOS, PAD]])
    lp = model.logprobs(tokens)
    assert lp.shape == (2, 5)
    assert np.all(lp < 0)
