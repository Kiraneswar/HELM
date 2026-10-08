"""
build_summary_v2.py  (Phase 2 + Phase 3 / L8)
---------------------------------------------
Builds the fixed-environment comparison table AND the statistical-power analysis:
mean +/- std over 20 seeds, plus, for the key FMARL-vs-CRL contrast, a paired
t-test and a 95% bootstrap CI on the per-seed energy and Gini GAPS.
Reads results_v2/raw_results. Writes results_v2/summary/.
"""
import os, glob
import numpy as np
import pandas as pd
from scipy import stats

RAW = "results_v2/raw_results"
SUM = "results_v2/summary"
os.makedirs(SUM, exist_ok=True)

METHODS = ["federated_marl", "random", "centralized_rl", "full", "oort"]
LABELS = {"federated_marl": "Federated MARL (Ours)", "random": "Random",
          "centralized_rl": "Centralized RL", "full": "Full Participation",
          "oort": "Oort (Heuristic)"}

def per_seed(ue, method):
    rows = []
    for c in sorted(glob.glob(f"{RAW}/update_every_{ue}/{method}_seed*.csv")):
        d = pd.read_csv(c)
        rows.append((d["accuracy"].iloc[-1]*100, d["energy_cost"].sum(),
                     d["gini"].iloc[-1],
                     d["num_stragglers"].mean() if "num_stragglers" in d else np.nan))
    return np.array(rows)  # (n_seeds, 4): acc, energy, gini, stragglers/round

# --- comparison table ---
recs = []
for ue in (1, 3):
    for m in METHODS:
        a = per_seed(ue, m)
        if len(a) == 0:
            continue
        recs.append({
            "update_every": ue, "method": m, "label": LABELS[m], "n_seeds": len(a),
            "acc_mean": a[:,0].mean(), "acc_std": a[:,0].std(ddof=1),
            "energy_mean": a[:,1].mean(), "energy_std": a[:,1].std(ddof=1),
            "gini_mean": a[:,2].mean(), "gini_std": a[:,2].std(ddof=1),
            "stragglers_per_round": np.nanmean(a[:,3]),
        })
out = pd.DataFrame(recs).round(4)
out.to_csv(f"{SUM}/comparison_table_v2.csv", index=False)

# --- L8: paired stats for FMARL vs CRL at ue=1 (seeds aligned) ---
def bootstrap_ci(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)

stat_lines = []
for ue in (1, 3):
    f = per_seed(ue, "federated_marl")
    c = per_seed(ue, "centralized_rl")
    n = min(len(f), len(c))
    if n < 2:
        continue
    f, c = f[:n], c[:n]
    for col, name, better in [(0,"accuracy","higher"), (1,"energy","lower"), (2,"gini","lower")]:
        # gap defined as CRL - FMARL for energy/gini (positive = FMARL better),
        # FMARL - CRL for accuracy (positive = FMARL better)
        diff = (c[:,col]-f[:,col]) if name != "accuracy" else (f[:,col]-c[:,col])
        t, p = stats.ttest_rel(f[:,col], c[:,col])
        lo, hi = bootstrap_ci(diff)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        stat_lines.append(
            f"ue={ue} {name:8} FMARL-vs-CRL: mean gap(FMARL better)={diff.mean():+.4f} "
            f"95%CI[{lo:+.4f},{hi:+.4f}] paired-t p={p:.4g} ({sig}, n={n})")

with open(f"{SUM}/significance_v2.txt", "w") as fh:
    fh.write("\n".join(stat_lines) + "\n")

# pretty print
for ue in (1, 3):
    print(f"\n=== update_every={ue} (fixed env, 20 seeds) ===")
    sub = out[out.update_every == ue]
    print(f"{'Method':<24}{'Acc%':<16}{'Energy(J)':<18}{'Gini':<16}{'Strag/rnd'}")
    for _, r in sub.iterrows():
        print(f"{r['label']:<24}{r['acc_mean']:.2f}±{r['acc_std']:.2f}     "
              f"{r['energy_mean']:.2f}±{r['energy_std']:.2f}    "
              f"{r['gini_mean']:.3f}±{r['gini_std']:.3f}    {r['stragglers_per_round']:.2f}")
print("\n--- Significance (FMARL vs CRL) ---")
print("\n".join(stat_lines))
print(f"\nSaved {SUM}/comparison_table_v2.csv and significance_v2.txt")
