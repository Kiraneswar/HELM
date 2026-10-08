"""
cross_check_paper_numbers.py
----------------------------
Recomputes every headline number straight from the raw per-seed CSVs and checks
it against (a) the summary CSVs and (b) the exact values hard-written into
results_bundle/PAPER_NUMBERS.md (2 dp). Fails loudly on any mismatch.
"""
import glob, sys
import numpy as np
import pandas as pd

RAW = "results_bundle/raw_results"
ABL = "results_bundle/ablations_ue1"

def r2(x): return round(float(x) + 1e-12, 2)

def recompute_comparison(ue, method):
    csvs = sorted(glob.glob(f"{RAW}/update_every_{ue}/{method}_seed*.csv"))
    acc = [pd.read_csv(c)["accuracy"].iloc[-1] * 100 for c in csvs]
    en  = [pd.read_csv(c)["energy_cost"].sum() for c in csvs]
    gi  = [pd.read_csv(c)["gini"].iloc[-1] for c in csvs]
    return (np.mean(acc), np.std(acc, ddof=1),
            np.mean(en), np.std(en, ddof=1),
            np.mean(gi), np.std(gi, ddof=1))

def recompute_ablation(sweep_dir, prefix, val):
    d = f"{ABL}/{sweep_dir}/{prefix}_{val}"
    csvs = sorted(glob.glob(f"{d}/federated_marl_seed*.csv"))
    acc = [pd.read_csv(c)["accuracy"].iloc[-1] * 100 for c in csvs]
    en  = [pd.read_csv(c)["energy_cost"].sum() for c in csvs]
    gi  = [pd.read_csv(c)["gini"].iloc[-1] for c in csvs]
    sim = []
    for c in csvs:
        s = pd.read_csv(c)["policy_similarity"].dropna()
        sim.append(s.iloc[-1] if len(s) else np.nan)
    return (np.mean(acc), np.std(acc, ddof=1), np.mean(en), np.mean(gi), np.nanmean(sim))

# Values as written in PAPER_NUMBERS.md (2dp): (acc, acc_std, energy, energy_std, gini, gini_std)
PAPER_COMP = {
    (1,"federated_marl"): (63.31,2.19,19.61,0.83,0.49,0.02),
    (1,"random"):         (64.00,1.57,19.80,0.87,0.49,0.02),
    (1,"centralized_rl"): (63.30,2.63,20.99,0.80,0.58,0.01),
    (1,"full"):           (63.16,2.64,22.54,1.11,0.58,0.01),
    (1,"oort"):           (61.45,3.01,19.74,0.99,0.58,0.03),
    (3,"federated_marl"): (63.32,1.44,19.68,0.92,0.49,0.02),
    (3,"centralized_rl"): (63.42,2.58,21.00,0.77,0.58,0.02),
}
# ablation: (acc, acc_std, energy, gini, policy_sim) at 2dp as in the doc
PAPER_ABL = {
    ("alpha","alpha","0.0"): (48.96,2.32,9.55,0.38,1.00),
    ("alpha","alpha","0.3"): (49.29,2.80,9.56,0.38,1.00),
    ("alpha","alpha","0.5"): (49.25,2.83,9.56,0.38,1.00),
    ("alpha","alpha","0.7"): (49.25,2.71,9.52,0.38,1.00),
    ("alpha","alpha","1.0"): (50.08,4.17,9.54,0.38,0.33),
    ("cadence","T_RL","1"):  (49.75,1.35,9.44,0.38,1.00),
    ("cadence","T_RL","3"):  (49.50,2.20,9.55,0.38,1.00),
    ("cadence","T_RL","5"):  (49.25,2.83,9.56,0.38,1.00),
    ("cadence","T_RL","10"): (49.42,2.80,9.55,0.38,1.00),
}

fails = []
print("=== Comparison: raw-recomputed (2dp) vs PAPER_NUMBERS.md ===")
for key, paper in PAPER_COMP.items():
    ue, m = key
    a,as_,e,es,g,gs = recompute_comparison(ue,m)
    got = (r2(a),r2(as_),r2(e),r2(es),r2(g),r2(gs))
    ok = got == paper
    if not ok: fails.append((key, got, paper))
    print(f"  ue={ue} {m:<16} raw={got}  paper={paper}  {'OK' if ok else 'MISMATCH'}")

print("\n=== Ablation: raw-recomputed (2dp) vs PAPER_NUMBERS.md ===")
for key, paper in PAPER_ABL.items():
    sweep_dir, prefix, val = key
    a,as_,e,g,s = recompute_ablation(sweep_dir, prefix, val)
    got = (r2(a),r2(as_),r2(e),r2(g),r2(s))
    ok = got == paper
    if not ok: fails.append((key, got, paper))
    print(f"  {prefix}={val:<4} raw={got}  paper={paper}  {'OK' if ok else 'MISMATCH'}")

# also confirm raw-recomputed matches the summary CSVs (4dp)
print("\n=== Raw-recomputed vs summary CSVs (4dp sanity) ===")
comp = pd.read_csv("results_bundle/summary/comparison_table.csv")
csv_fail = 0
for _, row in comp.iterrows():
    a,as_,e,es,g,gs = recompute_comparison(int(row.update_every), row.method)
    for got, col in [(a,"final_acc_mean"),(e,"total_energy_mean"),(g,"final_gini_mean")]:
        if abs(got - row[col]) > 1e-3:
            csv_fail += 1
            print(f"  MISMATCH {row.method} ue={row.update_every} {col}: raw={got} csv={row[col]}")
print(f"  comparison summary CSV: {'all match' if csv_fail==0 else str(csv_fail)+' mismatches'}")

print("\n" + "="*56)
if fails or csv_fail:
    print(f"RESULT: {len(fails)} PAPER_NUMBERS mismatches, {csv_fail} summary-CSV mismatches")
    sys.exit(1)
print("RESULT: PAPER_NUMBERS.md is fully consistent with the raw CSVs.")
sys.exit(0)
