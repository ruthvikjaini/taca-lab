# Research notes — taca-lab (Day 19, 2026-11-12)

## Trending-first check (2026-11-12, before any paper work)

GitHub trending (gitdiscover daily picks Nov 11, findarepo papers-with-code
Nov 11, whatstrending.ai weekly Nov 12, topopensource.ai Nov 12):
- Daily/weekly trending again dominated by skill/prompt packs
  (janestreet/agent_skills, mattpocock/skills ~322K, anthropics/skills —
  accelerating 362/day → 988/day, claude-mem) and agent
  frameworks/platforms (morluto/rea ~453K, +2,946/day; meta/astryx
  24.9K, +2,175/day; deepseek-harness ~279K; stablyai/orca;
  vercel-labs/agent-desktop; homeassistant-ai/ha-mcp).
- Papers-with-code: google-research/timesfm 35K (+241/7d — needs the
  pretrained checkpoint), LearningCircuit/local-deep-research 9.6K (needs
  a real LLM), chenxiachan/textfold 995 (a folding UI, no benchmarkable
  training mechanism), Companion-Inc/feynman (a research agent).
- No trending repo with a genuine, measurable, one-night CPU weakness:
  the code repos need multi-hundred-GB checkpoints, a real LLM, NIM API
  keys, or a GPU fleet. Paper fallback per the standing rule.

## Paper chosen

Lu, Zhou, Ning, Cao, Yao & Han (HKBU / SJTU), "When Credit Turns
Against Tools: Transition-Aware Credit Assignment in Agentic
Reinforcement Learning" (TACA), arXiv:2611.11330, submitted 8 Nov 2026.
Verified on the arXiv abstract page and the full HTML (v1) on
2026-11-12. Held as the strong backup since Day 17 (2026-11-10).
Official code: github.com/tmlr-group/TACA (not read or used;
implementation is from the paper's equations only).

Core claims (paper Table 1, Qwen2.5-1.5B, six math benchmarks):
MT-GRPO 28.63% mean accuracy → TACA 33.24% (+4.61pp); 3B: 39.90% →
43.51% (+3.61pp). TACO 41.98 → 53.70; WebShop success 56.8% → 67.2%.
Ablations (Table 3, 1.5B): w/o Advantage Projection 30.27% (-2.97),
w/o Decision Contrast 30.58% (-2.66), w/o Supervised Replay 30.21%
(-3.03). Training config (App. I, Table 7): 32 prompts/batch, G=8,
temp 1.0, top-p 1.0, AdamW lr 1e-6 constant, 1 epoch/batch, clip 0.2,
dual clip 3.0 for negative advantages, KL beta=0, entropy 0, max 6
turns, 2,048 response tokens/turn, 3 s tool timeout. Replay: up to 32
pairs per update, sampled uniformly from a fixed-capacity buffer.
(The exact Table 7 values of lambda_dc, lambda_rep, eta and the replay
capacity did not render in the arXiv HTML text extraction, so this lab
calibrates them and ablates them in E1/E2 instead of asserting the
paper's values.)

Mechanisms (Sec. 3, verified exactly in tests/test_credit.py):
- Prop. 3.1: C_tool = (M sigma_turn / N)(m_bar+_turn - m_bar-_turn);
  negative aggregate tool credit iff incorrect trajectories carry more
  tool tokens per generated turn than correct ones.
- Sec. 3.3, Eq. (5): token-mean aggregation shifts credit by
  Cov(L, (m/L)A) / E[L]; the paper reports length weighting flips
  positive credit negative in 73.6% of batches.
- Sec. 3.4: retries are more likely after failed than successful
  executions, and incorrect trajectories retry more; each retry extends
  negative credit to the recovery attempt.

Method (Sec. 4): Advantage Projection (Eq. 6, uniform offset
b = max(-mean tool advantage, 0)); Decision Contrast (Eq. 7, U_x =
n/(n+eta) (R_bar_T - R_bar_N), applied +lambda_dc at <python_code>
openers and -lambda_dc at <answer> openers); Supervised Replay (Eq. 8,
mean NLL over tool-call tokens of sampled correct-turn pairs);
objective Eq. (10), token-mean PPO surrogate + KL - lambda_rep L_rep.

## Rejected / held

- Liu et al., "No Free Lunch in RLVR" (arXiv:2610.39295): held a ninth
  night. Same RLVR cluster as Days 1/7/10; TACA is the fresher,
  agentic-RL-native choice.
- Reward Budgeting (arXiv:2611.05942) and KAPPO (arXiv:2611.03146):
  held; both are small GRPO-variant papers, less mechanistic than TACA.
- ZoomGain-4B (arXiv:2611.14076, 10 Nov 2026): vision-language RL,
  needs a 4B checkpoint and a vision testbed — not CPU-reproducible
  in one night.
- LARC latent reasoning (2 Nov 2026): continuous-space reasoning for
  image generation, needs the BAGEL base model.
- ICRL self-critique RL: needs Qwen3-4B/8B backbones and ALFWorld /
  WebShop simulators — too heavy for one CPU night.

## Testbed design (deviations from the paper, all documented)

The paper trains Qwen2.5-1.5B/3B/7B and Llama-3.2-1B with a Python
code-execution tool on MATH. This lab uses a 1-layer d=64 causal
transformer (NumPy, manual backprop, same harness lineage as Days
10/12/15/17) on a multi-turn calculator task: left-to-right chains of
single-digit binary operations (1/2/3 steps). The tool is a small
expression evaluator with standard precedence, parentheses and '#'
(last result) — the paper's Python tool in miniature. A problem can be
solved one-shot (one fully parenthesised call), step-by-step (one
binary op per call), or answered directly (derivation format). Terminal
reward is binary answer correctness; observations are context only.

Why the one-shot option exists: the first tool design evaluated only
one binary op per call, which made correct trajectories structurally
tool-heavy (a correct tier-2/3 solution *requires* 2-3 calls) — the
measured tool-token gap was +0.49..+1.12 per tier and aggregate tool
credit was positive in 65-90% of groups, so the paper's negative-credit
regime barely existed. The expression tool makes tool use a genuine
strategic choice (one efficient but error-prone call vs several safe
calls vs a direct answer), which is what lets incorrect trajectories
accumulate more tool tokens per turn through errors and retries.

## Calibration pilots (all numbers from real runs)

- P1 (binary-op tool, 6000 SFT, 2 epochs): greedy acc 0.767, tier-1
  direct answers 0.325 — the model cannot do single-digit arithmetic
  in its head; sampled exec success 1.0 (no errors greedily).
- P2 (binary-op tool, 6000 SFT, 4 epochs, tier mix 35/35/30, direct
  0.5): greedy 0.742, tier-1 0.225; sampled reward 0.405, exec 0.772;
  mean tool-token gap +0.75, frac negative gap 0.16, mean C_tool
  +0.064, flip fraction 0.117. Retries: P(call|error) 0.654 vs
  P(call|success) 0.567; incorrect retry 0.33 vs correct 0.09.
- Per-tier gap analysis on P2: even tool-only trajectories had gaps
  near zero (+0.05..+0.10) because correct trajectories need the full
  call sequence while incorrect ones stop early (mean calls correct
  2.49 vs incorrect 2.12). This motivated the expression-tool redesign.
- Direct-answer derivation format (<ans> a op b = v </ans>) was tried:
  direct accuracy 0.179 vs 0.136 without it — the 1-layer model cannot
  do single-digit arithmetic reliably in direct mode either way, so the
  format was kept (it is the honest direct mode) and direct accuracy is
  reported as-is. The first derivation pilot was invalid: parse_turn
  did not handle the derivation format, so all derivation answers
  scored 0; fixed by extracting the value after the last '='.
- P3 (expression tool, 6000 SFT, 4 epochs): greedy acc 0.975 (tier 1
  1.0, tier 2 1.0, tier 3 0.925), tool use 1.0; sampled reward
  0.35-0.41, exec success 0.63-0.69, mean errors 0.34-0.67/traj.
  Credit: tier-2 mean C_token -0.010 (negative on average), frac
  negative gap 0.375, frac C_token negative 0.475, flip fraction
  0.45; retries P(call|error) 0.77-0.86 vs P(call|success) 0.14-0.54,
  incorrect retries 0.4-0.74 vs correct 0.05-0.10. The paper's
  mechanisms now bite.
- LR pilot (20 TACA steps, batch 24x8): lr 3e-4 lifts sampled reward
  0.406 -> 0.698, tool 0.84 -> 0.93, exec 0.76 -> 0.83; lr 1e-4 reaches
  only 0.547. Grid uses lr 3e-4, batch 16x8, 100 steps (~4 s/step at
  batch 24; ~2.7 s/step at batch 16). Greedy accuracy is saturated at
  1.0 on the held-out set, so the headline eval is sampled pass@1
  (G=8, temp 1.0), as in the paper's 32-sample eval.

## Bugs / design fixes during the night

1. `execute_call` first took `call[1:]` (including `</call>`) in
   expert_trace — fixed to `call[1:-1]`.
2. `generate_turn` first read next-token logits from the last padded
   column, letting PAD tokens enter the conditioning context — fixed
   to read each row's last real position (padding-invariance test).
3. Model hardcoded global DK/D_FF, breaking d != 64 — made them
   instance attributes (dk, d_ff).
4. `parse_turn` did not handle the derivation answer format — fixed
   (value after the last '=').
5. Rollout observation bookkeeping first appended a wrong-length resp
   mask — rewritten to append the observation once, with matching
   masks, in a single place.
6. `diagnose_retries` first reconstructed turns from the token stream
   with `seq.index` (first occurrence — wrong) — replaced with
   counters recorded during rollout.
7. The ablation path in `grpo_update` was tangled (recomputing
   advantages three times with contradictory flags) — replaced with
   explicit `use_ap`/`use_dc` flags in `taca_advantages`.
8. SFT direct-mode traces initially had no derivation; direct accuracy
   was 0.14 — derivation format kept, reported honestly (see above).

## Final results (2026-11-12)

Main grid, 5 arms x 3 seeds x 100 updates, batch 16 prompts x G=8,
lr 3e-4, lambda_dc=1.0, lambda_rep=0.5, eta=4.0, replay capacity 512.
Sampled pass@1 (G=8, temp 1.0) on 240 held-out problems, 3-seed means:
MT-GRPO 0.905, TACA 0.910, TACA-no-AP 0.912, TACA-no-DC 0.914,
TACA-no-SR 0.910; greedy 0.999-1.000 for all arms. TACA minus
MT-GRPO = +0.006, within seed noise (sd 0.004-0.012). Learning
curves are indistinguishable: reward 0.45 -> 0.89, tool use 0.82 ->
0.99, exec success 0.78 -> 0.97 for every arm. The paper's +4.6pp
does not reproduce here — MT-GRPO does not degrade on this testbed,
so the corrections have little to fix. Advantage Projection does
activate (mean offset b ~0.04).

Extensions (seed 0, 12 runs): all 0.878-0.914. The only clear
deviation is trajectory-mean aggregation: E3-mtgrpo-trajmean 0.878
(-1.9pp vs token-mean MT-GRPO seed 0), E3-taca-trajmean 0.899 — the
paper's Eq. 5 intuition (trajectory-mean mitigates length weighting)
does not hold here; token-mean does useful work. Replay coefficient
and capacity are flat (lambda_rep 0.1/1.0: 0.909/0.894; capacity
64/4096: 0.907/0.892); eta 0/16: 0.900/0.897; retry cap 1 error:
0.904/0.896 vs 0.897/0.897 baselines.

E5 OOD (seed 0, retrained, tier-4 4-step problems, never trained on):
MT-GRPO tier-4 greedy 0.013, sampled 0.009; TACA 0.025 / 0.020,
vs tier-1-3 greedy 1.000, sampled ~0.91. Both arms collapse. Greedy
rollout inspection: every tier-4 trajectory opens with an empty call
`<call> </call>` (ERR) — the 10-token prompt is one token longer
than anything in training — and recovered step-by-step traces solve
a shorter, wrong chain (wrong operands dropped). TACA buys no
compositional generalisation. MAX_TURN_TOKENS was raised 18 -> 22
for this experiment so a one-shot tier-4 call (19 tokens) is
representable; the failure is not the cap.

Runtime notes: the VM restarted twice during the night (once during
the grid, once during extensions); both scripts have resume logic
(load existing results/*.json, skip finished arms/variants) and logs
live in ~/workspace/github-daily-projects/ (not /tmp, which is wiped
on restart). Grid 15 arms ~65 min total; extensions 12 runs ~60 min;
OOD 2 runs ~11 min.

9. The Day 17 sign test (`test_grpo_update_raises_rewarded_logprob`)
   measured the correct-vs-incorrect logprob gap on fresh samples.
   Raising MAX_TURN_TOKENS 18 -> 22 for the OOD experiment changed
   the sampled trajectories enough to flip it (before 0.582, after
   0.381). Root cause: the fresh-sample gap is noise-dominated — the
   weak test base yields only 2-3 correct trajectories out of 96,
   and even on the full SFT base (36-42 correct) the gap flips sign
   across measurement seeds (seed 99: 0.810 -> 0.894; seed 100:
   0.969 -> 0.810), because the trajectory composition changes.
   Fixed by measuring on the update batch's own trajectories
   (rolled out with the same rng the update uses, guarded by a
   mean_reward identity check): the gap then rises reliably
   (+0.07 to +0.09 across three update seeds), with correct-trajectory
   logprob up and incorrect-trajectory logprob down — the direct
   PPO sign check. 31/31 passes three times in a row after the fix.
