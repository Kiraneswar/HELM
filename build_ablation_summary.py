"""
build_ablation_summary.py
-------------------------
Builds results_bundle/summary/ablation_results.csv from the phase-5 sweeps.

Per (sweep, value): mean +/- std across 3 seeds of
  - final_accuracy (%)   (last round)
  - total_energy         (cumulative sum of per-round energy_cost)
  - final_gini           (last round)
  - final_policy_sim     (policy_similarity at the last federation round;
                          measures how converged the cluster policies are)
Also prints a flatness assessment: the spread (max-min of the per-value means)
relative to the pooled seed std, for accuracy / gini / policy_sim.
"""
import os, glob
import numpy as np
import pandas as pd

ABL = "results_bundle/ablations_ue1"
SUM = "results_bundle/summary"
os.makedirs(SUM, exist_ok=True)

def summarize(sweep_dir, prefix, cast):
    rows = []
    for d in sorted(glob.glob(os.path.join(sweep_dir, f"{prefix}_*")),
                    key=lambda p: cast(os.path.basename(p).replace(prefix + "_", ""))):
        val = cast(os.path.basename(d).replace(prefix + "_", ""))
        accs, energies, ginis, sims = [], [], [], []
        for c in sorted(glob.glob(os.path.join(d, "federated_marl_seed*.csv"))):
            df = pd.read_csv(c)
            accs.append(df["accuracy"].iloc[-1] * 100)
            energies.append(df["energy_cost"].sum())
            ginis.append(df["gini"].iloc[-1])
            sim_series = df["policy_similarity"].dropna()
            sims.append(sim_series.iloc[-1] if len(sim_series) else np.nan)
        rows.append({
            "sweep": prefix, "value": val, "n_seeds": len(accs),
            "final_acc_mean": np.mean(accs), "final_acc_std": np.std(accs, ddof=1),
            "total_energy_mean": np.mean(energies), "total_energy_std": np.std(energies, ddof=1),
            "final_gini_mean": np.mean(ginis), "final_gini_std": np.std(ginis, ddof=1),
            "final_policy_sim_mean": np.nanmean(sims), "final_policy_sim_std": np.nanstd(sims, ddof=1),
        })
    return rows

rows = []
rows += summarize(os.path.join(ABL, "alpha"), "alpha", float)
rows += summarize(os.path.join(ABL, "cadence"), "T_RL", int)
out = pd.DataFrame(rows)
for col in out.columns:
    if out[col].dtype == float:
        out[col] = out[col].round(4)
out.to_csv(os.path.join(SUM, "ablation_results.csv"), index=False)

def flatness(sub, col, label):
    means = sub[f"{col}_mean"].values
    spread = means.max() - means.min()
    pooled_std = np.sqrt(np.mean(sub[f"{col}_std"].values ** 2))
    ratio = spread / pooled_std if pooled_std > 0 else float("inf")
    verdict = "FLAT (spread < 1 seed-std)" if ratio < 1 else \
              ("mild" if ratio < 2 else "REAL pattern (spread > 2 seed-std)")
    print(f"    {label:<16} spread={spread:.3f}  pooled_seed_std={pooled_std:.3f}  "
          f"spread/std={ratio:.2f}  -> {verdict}")

for sweep, cast, xlabel in [("alpha", float, "alpha"), ("T_RL", int, "T_RL")]:
    sub = out[out["sweep"] == sweep].copy()
    print(f"\n=== {sweep} sweep (federated_marl, update_every=1, 3 seeds, 50 rounds) ===")
    print(f"{xlabel:<8}{'FinalAcc%':<18}{'TotEnergy':<16}{'FinalGini':<16}{'PolicySim(last fed)'}")
    for _, r in sub.iterrows():
        v = int(r['value']) if sweep == 'T_RL' else r['value']
        print(f"{str(v):<8}{r['final_acc_mean']:.2f} ± {r['final_acc_std']:.2f}      "
              f"{r['total_energy_mean']:.2f} ± {r['total_energy_std']:.2f}    "
              f"{r['final_gini_mean']:.3f} ± {r['final_gini_std']:.3f}    "
              f"{r['final_policy_sim_mean']:.4f} ± {r['final_policy_sim_std']:.4f}")
    print("  flatness (spread of per-value means vs pooled seed std):")
    flatness(sub, "final_acc", "accuracy")
    flatness(sub, "final_gini", "gini")
    flatness(sub, "final_policy_sim", "policy_sim")

print("\nSaved:", os.path.join(SUM, "ablation_results.csv"))
