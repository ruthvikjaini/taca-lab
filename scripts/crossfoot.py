"""Cross-foot the README against results/*.json.

Recomputes every headline number from the raw run files and compares
it with the value printed in README.md (tolerance 0.005, or 0.01 for
percentages quoted to one decimal). Exits non-zero on any mismatch.
"""

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def readme():
    with open(os.path.join(ROOT, "README.md")) as f:
        return f.read()


def find_number(text, label):
    """Find `label ... <number>` in the README (first match)."""
    m = re.search(re.escape(label) + r"[^0-9\-+]*([-+]?\d+\.\d+)", text)
    if not m:
        raise AssertionError(f"README number not found for label: {label}")
    return float(m.group(1))


def check(label, actual, expected, tol=0.005):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: README {actual} != recomputed {expected}")
    print(f"  ok {label}: {actual} == {expected}")


def arm_means(grid):
    out = {}
    for arm in ("mt-grpo", "taca", "taca-no-ap", "taca-no-dc", "taca-no-sr"):
        recs = [v for k, v in grid["arms"].items() if k.startswith(arm + "/")]
        assert len(recs) == 3, (arm, len(recs))
        out[arm] = {
            "sampled": float(np.mean([r["sampled_accuracy"] for r in recs])),
            "greedy": float(np.mean([r["greedy_accuracy"] for r in recs])),
            "reward_last10": float(np.mean([r["train_reward_last10"] for r in recs])),
            "tool_last10": float(np.mean([r["tool_use_last10"] for r in recs])),
            "exec_last10": float(np.mean([r["exec_last10"] for r in recs])),
            "reward_first10": float(np.mean([r["train_reward_first10"] for r in recs])),
        }
    return out


def main():
    grid = load("grid.json")
    ext = load("extensions.json")
    diag = load("diagnostics.json")
    text = readme()
    print("cross-footing README against results/*.json")
    means = arm_means(grid)

    # headline grid table (3-seed means, sampled pass@1)
    for arm in means:
        check(f"{arm} sampled", find_number(text, f"{arm} sampled"),
              means[arm]["sampled"])
        check(f"{arm} greedy", find_number(text, f"{arm} greedy"),
              means[arm]["greedy"])
    # TACA - MT-GRPO headline gap
    gap = means["taca"]["sampled"] - means["mt-grpo"]["sampled"]
    check("TACA-MT-GRPO sampled gap", find_number(text, "TACA minus MT-GRPO sampled"),
          gap)
    # per-run curve consistency: first10/last10 recompute from the curve
    for key, rec in grid["arms"].items():
        curve = rec["curve"]
        assert len(curve) == grid["config"]["updates"], key
        r_first = float(np.mean([c["reward"] for c in curve[:10]]))
        r_last = float(np.mean([c["reward"] for c in curve[-10:]]))
        assert abs(r_first - rec["train_reward_first10"]) < 1e-9, key
        assert abs(r_last - rec["train_reward_last10"]) < 1e-9, key
    print("  ok per-run curves recompute from stored first/last-10 means")
    # diagnostics: fractions in [0, 1], groups counted
    for tier, d in diag["per_tier"].items():
        c = d["credit"]
        assert 0.0 <= c["frac_negative_gap"] <= 1.0
        assert 0.0 <= c["frac_c_token_negative"] <= 1.0
        assert 0.0 <= c["frac_flip"] <= 1.0
        assert c["n_groups"] == 80, (tier, c["n_groups"])
    print("  ok diagnostics fractions in range, 80 groups per tier")
    # diagnostics headline numbers in the README
    check("diag overall mean C_token",
          find_number(text, "diagnostics overall mean C_token"),
          diag["overall"]["credit"]["mean_c_token"])
    check("diag overall frac flip",
          find_number(text, "diagnostics overall frac flip"),
          diag["overall"]["credit"]["frac_flip"])
    check("diag P(call|error)",
          find_number(text, "diagnostics P(call|error)"),
          diag["overall"]["retries"]["p_call_after_error"])
    # extensions: sampled accuracy of each variant recomputes from file
    for name, rec in ext.items():
        check(f"extension {name} sampled",
              find_number(text, f"{name} sampled"), rec["sampled_accuracy"])
    # OOD: tier-4 generalisation (seed 0, retrained models)
    ood = load("ood.json")
    for arm in ("mt-grpo", "taca"):
        slug = arm.replace("-", "")
        check(f"ood {arm} tier4 greedy",
              find_number(text, f"E5-{slug} tier4 greedy"), ood[arm]["tier4_greedy"])
        check(f"ood {arm} tier4 sampled",
              find_number(text, f"E5-{slug} tier4 sampled"), ood[arm]["tier4_sampled"])
        check(f"ood {arm} tier123 sampled",
              find_number(text, f"E5-{slug} tier123 sampled"), ood[arm]["tier123_sampled"])
    print("CROSSFOOT PASS")


if __name__ == "__main__":
    main()
