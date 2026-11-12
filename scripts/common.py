"""Shared setup for the experiment scripts: SFT base + problem sets.

The SFT warm-start is the same for every arm within a seed (the paper's
post-training starts from one instruct model per backbone). Bases are
cached under checkpoints/ so the grid and the extensions reuse them.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from taca_lab.env import make_problem, make_sft_dataset  # noqa: E402
from taca_lab.model import Transformer  # noqa: E402
from taca_lab.sft import sft_train  # noqa: E402
from taca_lab.vocab import VOCAB_SIZE  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CKPT = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")

SFT_CONFIG = dict(n=6000, tier_p=(0.35, 0.35, 0.30), direct_p=0.5)
SFT_EPOCHS = 4
SFT_LR = 3e-3
N_TRAIN_PROBLEMS = 400
N_HELD_PER_TIER = 80


def held_out_problems(seed=123):
    rng = np.random.default_rng(seed)
    return [make_problem(rng, tier=t) for t in (1, 2, 3)
            for _ in range(N_HELD_PER_TIER)]


def train_problems(seed=0):
    rng = np.random.default_rng(seed)
    return [make_problem(rng) for _ in range(N_TRAIN_PROBLEMS)]


def sft_base(seed=0, force=False):
    """Train (or load the cached) SFT warm-start for `seed`."""
    os.makedirs(CKPT, exist_ok=True)
    path = os.path.join(CKPT, f"sft_seed{seed}.npz")
    if os.path.exists(path) and not force:
        return Transformer.load(path, VOCAB_SIZE, max_len=160)
    model = Transformer(VOCAB_SIZE, max_len=160, d=64, seed=seed)
    data = make_sft_dataset(np.random.default_rng(1000 + seed), **SFT_CONFIG)
    losses = sft_train(model, data, epochs=SFT_EPOCHS, batch=64, lr=SFT_LR,
                       rng=np.random.default_rng(2000 + seed),
                       label_smoothing=0.05)
    print(f"SFT seed {seed}: loss {losses[0]:.3f} -> {losses[-1]:.3f}",
          flush=True)
    model.save(path)
    return model


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
