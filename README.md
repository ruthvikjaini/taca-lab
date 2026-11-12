# taca-lab

Transition-Aware Credit Assignment in multi-turn agentic reinforcement
learning — a CPU-only, NumPy reproduction of Lu et al.,
arXiv:2611.11330 (8 Nov 2026), on a multi-turn calculator testbed.

## What this is

Multi-turn GRPO with verifiable rewards trains a language model through
tool interactions, but reasoning performance and tool-calling ability
can decline during training. Lu et al. trace the degradation to three
coupled mechanisms:

1. **Tool-token imbalance.** When incorrect trajectories contain more
   tool tokens per generated turn than correct ones, terminal-reward
   supervision assigns *negative aggregate credit* to tool calls
   (Prop. 3.1), even when correct trajectories are the majority.
2. **Aggregation amplification.** Token-mean aggregation gives longer
   incorrect trajectories greater weight, shifting credit further
   down (Eq. 5) — the paper reports length weighting flips positive
   credit negative in 73.6% of batches.
3. **Failure-driven retries.** Models call tools again after execution
   errors, and incorrect trajectories retry more, extending negative
   credit to the recovery attempts.

TACA counters with three components: **Advantage Projection** (a
uniform offset that enforces non-negative mean advantage on tool-call
tokens, Eq. 6), **Decision Contrast** (a reward contrast between
tool-using and direct-answer trajectories for the same prompt, applied
at the decision tokens, Eq. 7), and **Supervised Replay** (an auxiliary
NLL loss on tool calls from previously sampled correct trajectories,
Eq. 8).

This lab implements all three on a tiny causal transformer (1 layer,
d=64, NumPy, manual backprop) solving multi-step arithmetic with a
small expression-evaluator tool — the paper's Python code execution in
miniature. Problems are left-to-right chains of single-digit operations
(1/2/3 steps). The policy can solve a problem one-shot (one fully
parenthesised call), step-by-step (one binary op per call), or answer
directly. Malformed calls error; a missing paren silently changes the
value. Rewards are binary terminal answer correctness.

## Results

### Main grid (paper Table 3 ablations)

100 updates, batch 16 prompts x G=8, lr 3e-4, 3 seeds. Held-out eval on
240 problems (80/tier): sampled pass@1 (G=8, temp 1.0) and greedy
accuracy. Values are 3-seed means.

| arm | sampled | greedy |
| --- | --- | --- |
| mt-grpo sampled 0.905 | mt-grpo greedy 1.000 |
| taca sampled 0.910 | taca greedy 1.000 |
| taca-no-ap sampled 0.912 | taca-no-ap greedy 0.999 |
| taca-no-dc sampled 0.914 | taca-no-dc greedy 1.000 |
| taca-no-sr sampled 0.910 | taca-no-sr greedy 0.999 |

TACA minus MT-GRPO sampled 0.006.

Training dynamics (3-seed means, first 10 vs last 10 updates):
MT-GRPO reward 0.451 -> 0.889, tool use 0.819 -> 0.992, execution
success 0.788 -> 0.969. TACA reward 0.473 -> 0.893, tool use 0.845 ->
0.992, execution success 0.784 -> 0.974.

The headline is a null result, and it is the honest one: every arm
converges to the same sampled accuracy (~0.91) and the same learning
curve. The paper's +4.6pp does not reproduce here — on this testbed
MT-GRPO does not degrade (tool use rises 0.82 -> 0.99, execution
success 0.78 -> 0.97), so TACA's corrections have little to fix.
Advantage Projection does activate (mean offset b ~0.04), and the
Sec. 3 mechanisms are real and verified in the diagnostics below, but
they are a minority phenomenon on this testbed (negative tool credit
in 30% of prompt groups, not most batches), and none of the three
components moves final accuracy beyond seed noise (+/-0.01).

### Paper Sec. 3 diagnostics on the SFT base (before RL)

240 held-out problems x G=8, temperature 1.0. Overall (80 prompt groups): diagnostics overall mean C_token 0.041, diagnostics overall frac flip 0.275, diagnostics P(call|error) 0.666 (vs P(call|success) 0.411), mean retries 0.294 in incorrect vs 0.091 in correct trajectories. Per tier:

| tier | sampled reward | mean C_token | frac groups with negative gap | flip fraction | P(call\|error) | P(call\|success) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.309 | +0.118 | 0.09 | 0.11 | 0.69 | 0.03 |
| 2 | 0.491 | +0.014 | 0.30 | 0.31 | 0.73 | 0.39 |
| 3 | 0.328 | +0.054 | 0.19 | 0.07 | 0.60 | 0.53 |

Negative aggregate tool credit is a substantial minority phenomenon
here (30% of groups), not the dominant regime of the paper — see
Limitations.

### Extensions (seed 0)

| variant | sampled |
| --- | --- |
| E1-lambda_dc-0.5 sampled 0.914 |
| E1-lambda_dc-2.0 sampled 0.898 |
| E1-lambda_rep-0.1 sampled 0.909 |
| E1-lambda_rep-1.0 sampled 0.894 |
| E1-eta-0 sampled 0.900 |
| E1-eta-16 sampled 0.897 |
| E2-capacity-64 sampled 0.907 |
| E2-capacity-4096 sampled 0.892 |
| E3-mtgrpo-trajmean sampled 0.878 |
| E3-taca-trajmean sampled 0.899 |
| E4-mtgrpo-maxerr1 sampled 0.904 |
| E4-taca-maxerr1 sampled 0.896 |

Baselines (seed 0): MT-GRPO 0.897, TACA 0.897. All extensions are
within seed noise (+/-0.01) of their baseline except trajectory-mean
aggregation, which is the one clear deviation: E3-mtgrpo-trajmean
sampled 0.878, 1.9pp below token-mean MT-GRPO, and E3-taca-trajmean
sampled 0.899 vs 0.897 for token-mean TACA. The paper's Eq. 5
suggests trajectory-mean aggregation should mitigate the
length-weighting of negative credit; here it hurts MT-GRPO instead —
in this regime the token-mean weighting of longer (mostly incorrect)
trajectories is doing useful work, not just amplifying a bias. Single
seed, so read it as suggestive, not conclusive. Capping retries at
one error (E4) changes nothing (0.904 / 0.896 vs 0.897 / 0.897), and
the TACA coefficients are flat across the tested range: lambda_dc
0.5/2.0, lambda_rep 0.1/1.0, eta 0/16, replay capacity 64/4096 all
land within 0.89-0.91.

### E5: OOD generalisation to 4-step problems (seed 0)

Models retrained and evaluated on tier-4 problems (4-step chains,
never trained on), greedy and sampled accuracy:

| arm | tier-4 greedy | tier-4 sampled | tier-1-3 sampled |
| --- | --- | --- | --- |
| mt-grpo | E5-mtgrpo tier4 greedy 0.013 | E5-mtgrpo tier4 sampled 0.009 | E5-mtgrpo tier123 sampled 0.910 |
| taca | E5-taca tier4 greedy 0.025 | E5-taca tier4 sampled 0.020 | E5-taca tier123 sampled 0.904 |

Both arms collapse: greedy 1.000 in-distribution, 0.013-0.025 on
4-step problems. Inspecting greedy tier-4 rollouts shows why. Every
trajectory opens with an empty call `<call> </call>` (ERR) — the
10-token prompt is one token longer than anything in training, and
the model's first turn misfires at the shifted position. When it
recovers into step-by-step mode it solves a shorter, wrong chain
(e.g. on `6 + 2 + 5 + 6 + 3` it computes 3+5, #+5, #+6 and answers
19; on `1 + 7 + 9 + 9 * 7` it computes 7+9, #+9, #+9 and answers 34,
dropping the first operand and the final `* 7`). TACA's 0.025 vs
0.013 is 2 vs 1 problems out of 80 — noise. The lab's models have no
compositional generalisation to a longer chain, and TACA does not
buy any: the credit-assignment fixes operate within the trained
distribution and do not extend the model's effective reasoning
depth. That is the honest boundary of this reproduction — and the
most interesting result in it.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest tests -q        # 31 tests
python scripts/demo.py           # SFT + 5 TACA updates + diagnostics
python scripts/run_diagnostics.py
python scripts/run_grid.py       # main grid (5 arms x 3 seeds x 100 steps)
python scripts/run_extensions.py
python scripts/crossfoot.py      # README numbers vs results/*.json
```

## Layout

- `src/taca_lab/vocab.py` — 27-token vocabulary and codecs.
- `src/taca_lab/env.py` — problem generation, expression tool,
  expert traces (one-shot / step-by-step / direct / recovery), SFT data.
- `src/taca_lab/model.py` — 1-layer causal transformer, manual backprop.
- `src/taca_lab/grpo.py` — multi-turn rollout, turn-level advantages
  (Eq. 1), Advantage Projection (Eq. 6), Decision Contrast (Eq. 7),
  Supervised Replay (Eq. 8), dual-clipped token-mean objective (Eq. 10).
- `src/taca_lab/evaluate.py` — greedy/sampled eval and Sec. 3 diagnostics.
- `src/taca_lab/sft.py` — SFT warm-start (label smoothing 0.05).
- `scripts/` — diagnostics, grid, extensions, crossfoot, demo, common.
- `docs/RESEARCH.md` — trending check, paper verification, design
  decisions, calibration pilots, bugs fixed.
- `results/` — `diagnostics.json`, `grid.json`, `extensions.json`.

## Paper numbers (kept separate)

Paper Table 1 (Qwen2.5-1.5B, six math benchmarks, mean accuracy):
MT-GRPO 28.63%, TACA 33.24% (+4.61pp); 3B: 39.90% -> 43.51%
(+3.61pp). TACO unit-test score 41.98 -> 53.70; WebShop success
56.8% -> 67.2%. Ablations (Table 3): w/o Advantage Projection 30.27%,
w/o Decision Contrast 30.58%, w/o Supervised Replay 30.21%.
Training diagnostics (Table 9, steps 1-100): TACA tool-use rate
75.76% vs ReTool 1.49%; execution success 86.28% vs 79.45%;
tool-using correct trajectories 42.00% of all trajectories.

## Limitations

- Small-model regime: a 1-layer d=64 transformer on synthetic
  arithmetic, not a 1.5B instruct model on MATH. Magnitudes differ from
  the paper; orderings and mechanisms are what transfer.
- The SFT base is already at greedy accuracy 1.0 on the held-out set,
  so the headline eval is sampled pass@1, not greedy accuracy.
- The direct-answer mode is weak (the tiny model cannot do single-digit
  arithmetic reliably in its head; sampled direct-mode accuracy on
  tier 1 is ~0.18), so Decision Contrast mostly pushes toward tool use.
- Negative aggregate tool credit is present but not dominant: on the
  SFT base it is negative in 30% of prompt groups (mean C_token
  +0.041), vs "most batches" in the paper. The first tool design (one
  binary op per call) made correct trajectories structurally
  tool-heavy; the expression tool (one-shot vs step-by-step) was needed
  to create the regime at all (see `docs/RESEARCH.md`).
- The exact Table 7 TACA coefficients (lambda_dc, lambda_rep, eta,
  replay capacity) did not render in the arXiv HTML text extraction;
  this lab calibrates them (1.0 / 0.5 / 4.0 / 512) and ablates them
  in E1/E2 rather than asserting the paper's values.
- One policy epoch per rollout batch, so the importance ratio is 1
  on-policy and the PPO clip is vacuous; the clip machinery is kept so
  the objective is exactly the paper's.
- 3 seeds, 100 updates; seed variance is reported via the per-seed
  values in `results/grid.json`.
