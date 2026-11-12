"""SFT warm-start training on expert calculator traces."""

import numpy as np

from .vocab import PAD


def _batch_arrays(examples, idxs):
    seqs, masks = [], []
    for i in idxs:
        tokens, mask = examples[i]
        seqs.append(tokens)
        masks.append(mask)
    width = max(len(s) for s in seqs)
    tokens = np.full((len(seqs), width), PAD, dtype=int)
    mask = np.zeros((len(seqs), width), dtype=float)
    for j, (s, m) in enumerate(zip(seqs, masks)):
        tokens[j, :len(s)] = s
        mask[j, :len(m)] = m
    return tokens, mask


def sft_train(model, examples, epochs=2, batch=64, lr=3e-3, rng=None,
              label_smoothing=0.05, log_every=0):
    """Adam SFT on (tokens, response_mask) examples. Returns batch losses."""
    rng = rng or np.random.default_rng(0)
    n = len(examples)
    losses = []
    for _ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idxs = order[s:s + batch]
            tokens, mask = _batch_arrays(examples, idxs)
            loss, grads, _ = model.loss_and_grads(
                tokens, mask, label_smoothing=label_smoothing)
            model.adam_step(grads, lr)
            losses.append(float(loss))
            if log_every and len(losses) % log_every == 0:
                print(f"  sft step {len(losses)} loss {loss:.4f}", flush=True)
    return losses
