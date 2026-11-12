"""Tiny causal transformer (1 layer, d=64, 4 heads) with manual backprop.

NumPy only, CPU only. Token + positional embeddings, one pre-norm
transformer block (causal multi-head self-attention + ReLU MLP), final
LayerNorm, linear head. Cross-entropy loss on masked positions only.
Backprop is verified against finite differences in tests/test_model.py.
"""

import numpy as np

from .vocab import EOS, PAD

D = 64
N_HEADS = 4
DK = D // N_HEADS
D_FF = 4 * D


def _layernorm_forward(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mean) / std
    out = gamma * xhat + beta
    cache = (x, xhat, mean, std, gamma)
    return out, cache


def _layernorm_backward(dout, cache):
    x, xhat, mean, std, gamma = cache
    dgamma = (dout * xhat).sum(axis=tuple(range(x.ndim - 1)))
    dbeta = dout.sum(axis=tuple(range(x.ndim - 1)))
    dxhat = dout * gamma
    dx = (dxhat - dxhat.mean(axis=-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / std
    return dx, dgamma, dbeta


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class Transformer:
    def __init__(self, vocab_size, max_len=160, d=D, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d = d
        self.dk = d // N_HEADS
        self.d_ff = 4 * d
        s = 0.02
        self.params = {
            "Wte": rng.normal(0, s, (vocab_size, d)),
            "Wpe": rng.normal(0, s, (max_len, d)),
            "ln1_g": np.ones(d), "ln1_b": np.zeros(d),
            "Wq": rng.normal(0, s, (d, d)), "bq": np.zeros(d),
            "Wk": rng.normal(0, s, (d, d)), "bk": np.zeros(d),
            "Wv": rng.normal(0, s, (d, d)), "bv": np.zeros(d),
            "Wo": rng.normal(0, s, (d, d)), "bo": np.zeros(d),
            "ln2_g": np.ones(d), "ln2_b": np.zeros(d),
            "W1": rng.normal(0, s, (d, 4 * d)), "b1": np.zeros(4 * d),
            "W2": rng.normal(0, s, (4 * d, d)), "b2": np.zeros(d),
            "lnf_g": np.ones(d), "lnf_b": np.zeros(d),
            "Wh": rng.normal(0, s, (vocab_size, d)), "bh": np.zeros(vocab_size),
        }
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t = 0

    # -- forward ---------------------------------------------------------
    def forward(self, tokens):
        """tokens: [B, T] int. Returns logits [B, T, V] and a cache."""
        p = self.params
        B, T = tokens.shape
        pos = np.arange(T)
        x = p["Wte"][tokens] + p["Wpe"][pos][None, :, :]  # [B,T,d]
        cache = {"tokens": tokens, "x0": x}

        h, c1 = _layernorm_forward(x, p["ln1_g"], p["ln1_b"])
        cache["ln1"] = c1
        Q = h @ p["Wq"] + p["bq"]
        K = h @ p["Wk"] + p["bk"]
        V = h @ p["Wv"] + p["bv"]
        Qh = Q.reshape(B, T, N_HEADS, self.dk).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, N_HEADS, self.dk).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, N_HEADS, self.dk).transpose(0, 2, 1, 3)
        scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(self.dk)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask[None, None, :, :], -1e9, scores)
        P = _softmax(scores, axis=-1)
        Oh = P @ Vh  # [B,H,T,dk]
        O = Oh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        attn_out = O @ p["Wo"] + p["bo"]
        cache.update({"h": h, "Q": Q, "K": K, "V": V, "Qh": Qh, "Kh": Kh,
                      "Vh": Vh, "P": P, "O": O, "attn_out": attn_out})
        x = x + attn_out

        h2, c2 = _layernorm_forward(x, p["ln2_g"], p["ln2_b"])
        cache["ln2"] = c2
        ff = np.maximum(h2 @ p["W1"] + p["b1"], 0.0)
        mlp_out = ff @ p["W2"] + p["b2"]
        cache.update({"h2": h2, "ff": ff, "mlp_out": mlp_out})
        x = x + mlp_out

        y, cf = _layernorm_forward(x, p["lnf_g"], p["lnf_b"])
        cache["lnf"] = cf
        logits = y @ p["Wh"].T + p["bh"]
        cache["y"] = y
        return logits, cache

    def loss_and_grads(self, tokens, mask, label_smoothing=0.0):
        """Mean CE over masked positions; returns (loss, grads, n_tokens).

        `tokens` is the full sequence; `mask` marks response positions in
        the full sequence. Logits at position t predict token t+1, so the
        loss is computed on the shifted frame (logits[:, :-1] vs
        tokens[:, 1:], mask[:, 1:]). With label_smoothing > 0 the target
        is (1-ls) * onehot + ls / V, which keeps the SFT base from
        driving never-taken actions to probability ~0.
        """
        logits, cache = self.forward(tokens)
        B, T, V = logits.shape
        pred = logits[:, :-1, :]
        tgt = tokens[:, 1:]
        m = mask[:, 1:]
        mx = pred.max(axis=-1, keepdims=True)
        e = np.exp(pred - mx)
        probs = e / e.sum(axis=-1, keepdims=True)
        n = m.sum()
        logp = np.log(probs[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] + 1e-12)
        loss = -(logp * m).sum() / max(n, 1)
        d = probs.copy()
        if label_smoothing > 0:
            d -= label_smoothing / V
            d[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] -= (1.0 - label_smoothing)
        else:
            d[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] -= 1.0
        d *= (m / max(n, 1))[:, :, None]
        dlogits = np.zeros_like(logits)
        dlogits[:, :-1, :] = d
        grads = self._backward(dlogits, cache)
        return loss, grads, int(n)

    def _backward(self, dlogits, cache):
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        B, T, V = dlogits.shape
        y = cache["y"]
        dWh = dlogits.transpose(0, 1, 2).reshape(-1, V).T @ y.reshape(-1, self.d)
        grads["Wh"] = dWh
        grads["bh"] = dlogits.sum(axis=(0, 1))
        dy = dlogits @ p["Wh"]
        dx, dg, db = _layernorm_backward(dy, cache["lnf"])
        grads["lnf_g"] = dg
        grads["lnf_b"] = db
        dmlp = dx
        h2, ff = cache["h2"], cache["ff"]
        dW2 = ff.reshape(-1, self.d_ff).T @ dmlp.reshape(-1, self.d)
        grads["W2"] = dW2
        grads["b2"] = dmlp.sum(axis=(0, 1))
        dff = dmlp @ p["W2"].T
        dff = dff * (ff > 0)
        dW1 = h2.reshape(-1, self.d).T @ dff.reshape(-1, self.d_ff)
        grads["W1"] = dW1
        grads["b1"] = dff.sum(axis=(0, 1))
        dh2 = dff @ p["W1"].T
        dx2, dg2, db2 = _layernorm_backward(dh2, cache["ln2"])
        grads["ln2_g"] = dg2
        grads["ln2_b"] = db2
        dx = dx + dx2
        dx_resid = dx
        dattn = dx
        O = cache["O"]
        dWo = O.reshape(-1, self.d).T @ dattn.reshape(-1, self.d)
        grads["Wo"] = dWo
        grads["bo"] = dattn.sum(axis=(0, 1))
        dO = dattn @ p["Wo"].T
        dOh = dO.reshape(B, T, N_HEADS, self.dk).transpose(0, 2, 1, 3)
        P, Vh = cache["P"], cache["Vh"]
        dVh = P.transpose(0, 1, 3, 2) @ dOh
        dP = dOh @ Vh.transpose(0, 1, 3, 2)
        s = (dP * P).sum(axis=-1, keepdims=True)
        dscores = P * (dP - s)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores = np.where(mask[None, None, :, :], 0.0, dscores)
        dQh = (dscores @ cache["Kh"]) / np.sqrt(self.dk)
        dKh = (dscores.transpose(0, 1, 3, 2) @ cache["Qh"]) / np.sqrt(self.dk)
        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        h = cache["h"]
        grads["Wq"] = h.reshape(-1, self.d).T @ dQ.reshape(-1, self.d)
        grads["bq"] = dQ.sum(axis=(0, 1))
        grads["Wk"] = h.reshape(-1, self.d).T @ dK.reshape(-1, self.d)
        grads["bk"] = dK.sum(axis=(0, 1))
        grads["Wv"] = h.reshape(-1, self.d).T @ dV.reshape(-1, self.d)
        grads["bv"] = dV.sum(axis=(0, 1))
        dh = dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
        dx1, dg1, db1 = _layernorm_backward(dh, cache["ln1"])
        grads["ln1_g"] = dg1
        grads["ln1_b"] = db1
        dx = dx1 + dx_resid
        tokens = cache["tokens"]
        dWte = np.zeros_like(p["Wte"])
        np.add.at(dWte, tokens.ravel(), dx.reshape(-1, self.d))
        grads["Wte"] = dWte
        dWpe = np.zeros_like(p["Wpe"])
        dWpe[: tokens.shape[1]] = dx.sum(axis=0)
        grads["Wpe"] = dWpe
        return grads

    # -- optimizer -------------------------------------------------------
    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        t = self._adam_t
        for k in self.params:
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * g * g
            mhat = self._adam_m[k] / (1 - beta1 ** t)
            vhat = self._adam_v[k] / (1 - beta2 ** t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def reset_optimizer(self):
        """Fresh optimizer state (new training phase: SFT moments are stale)."""
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t = 0

    # -- persistence -----------------------------------------------------
    def save(self, path):
        np.savez(path, **{k: v for k, v in self.params.items()})

    @classmethod
    def load(cls, path, vocab_size, max_len=160, d=D):
        m = cls(vocab_size, max_len=max_len, d=d, seed=0)
        z = np.load(path)
        for k in m.params:
            m.params[k] = z[k]
        return m

    def copy(self):
        m = Transformer(self.vocab_size, max_len=self.max_len, d=self.d, seed=0)
        for k in self.params:
            m.params[k] = self.params[k].copy()
        return m

    # -- inference -------------------------------------------------------
    def logprobs(self, tokens):
        """Shifted log-probs: lp[:, j] = log p(tokens[:, j+1] | tokens[:, :j+1])."""
        logits, _ = self.forward(tokens)
        pred = logits[:, :-1, :]
        tgt = tokens[:, 1:]
        m = pred.max(axis=-1, keepdims=True)
        e = np.exp(pred - m)
        probs = e / e.sum(axis=-1, keepdims=True)
        B, T1 = tgt.shape
        lp = np.log(probs[np.arange(B)[:, None], np.arange(T1)[None, :], tgt] + 1e-12)
        return lp

    def generate_turn(self, seqs, max_new, temperature=1.0, greedy=False, rng=None,
                      stop_ids=(15, 17, 22)):
        """Generate one turn for each row of `seqs` (list of token lists).

        Generation stops per row at a stop id (</call>, </ans>, EOS) or
        after max_new tokens. Returns a list of generated token lists
        (stop id included). Rows are re-padded each step and the next
        token is read from each row's last real position, so padding
        never enters a row's conditioning context.
        """
        rows = [list(s) for s in seqs]
        B = len(rows)
        out = [[] for _ in range(B)]
        done = np.zeros(B, dtype=bool)
        for _ in range(max_new):
            width = max(len(r) for r in rows)
            seq = np.full((B, width), PAD, dtype=int)
            last = np.zeros(B, dtype=int)
            for i, r in enumerate(rows):
                seq[i, :len(r)] = r
                last[i] = len(r) - 1
            logits, _ = self.forward(seq)
            nxt_logits = logits[np.arange(B), last, :]
            if greedy:
                nxt = nxt_logits.argmax(axis=-1)
            else:
                t = max(temperature, 1e-6)
                z = nxt_logits / t
                z = z - z.max(axis=-1, keepdims=True)
                e = np.exp(z)
                pr = e / e.sum(axis=-1, keepdims=True)
                u = rng.random((B, 1))
                nxt = (np.cumsum(pr, axis=-1) < u).sum(axis=-1)
                nxt = np.clip(nxt, 0, pr.shape[-1] - 1)
            for i in range(B):
                if not done[i]:
                    tok = int(nxt[i])
                    out[i].append(tok)
                    rows[i].append(tok)
                    if tok in stop_ids:
                        done[i] = True
            if done.all():
                break
        return out
