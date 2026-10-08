"""
build_summary.py
----------------
Builds results_bundle/summary/comparison_table.csv from the robustness raw CSVs.

For each (update_every, method): mean +/- std across seeds of
  - final_accuracy  (last round accuracy)
  - total_energy    (sum of per-round energy_cost -- cumulative)
  - final_gini      (last round gini)
Also reports mean per-round energy for reference.
"""
import os, glob
import numpy as np
import pandas as pd

RAW = "results_bundle/raw_results"
SUM = "results_bundle/summary"
os.makedirs(SUM, exist_ok=True)

METHOD_ORDER = ["federated_marl", "random", "centralized_rl", "full", "oort"]
LABELS = {
    "federated_marl": "Federated MARL (Ours)",
    "random": "Random",
    "centralized_rl": "Centralized RL",
    "full": "Full Participation",
    "oort": "Oort (Heuristic)",
}

rows = []
for ue_dir in sorted(glob.glob(os.path.join(RAW, "update_every_*"))):
    u = int(os.path.basename(ue_dir).split("_")[-1])
    for method in METHOD_ORDER:
        csvs = sorted(glob.glob(os.path.join(ue_dir, f"{method}_seed*.csv")))
        if not csvs:
            continue
        fin_acc, tot_energy, fin_gini = [], [], []
        for c in csvs:
            df = pd.read_csv(c)
            fin_acc.append(df["accuracy"].iloc[-1])
            tot_energy.append(df["energy_cost"].sum())
            fin_gini.append(df["gini"].iloc[-1])
        rows.append({
            "update_every": u,
            "method": method,
            "label": LABELS.get(method, method),
            "n_seeds": len(csvs),
            "final_acc_mean": np.mean(fin_acc) * 100,
            "final_acc_std": np.std(fin_acc, ddof=1) * 100,
            "total_energy_mean": np.mean(tot_energy),
            "total_energy_std": np.std(tot_energy, ddof=1),
            "final_gini_mean": np.mean(fin_gini),
            "final_gini_std": np.std(fin_gini, ddof=1),
        })

out = pd.DataFrame(rows)
for col in ["final_acc_mean", "final_acc_std", "total_energy_mean",
            "total_energy_std", "final_gini_mean", "final_gini_std"]:
    out[col] = out[col].round(4)
out.to_csv(os.path.join(SUM, "comparison_table.csv"), index=False)

# Pretty print
for u in sorted(out["update_every"].unique()):
    print(f"\n=== update_every = {u} ===")
    sub = out[out["update_every"] == u]
    print(f"{'Method':<24}{'FinalAcc%':<18}{'TotEnergy':<16}{'FinalGini'}")
    for _, r in sub.iterrows():
        print(f"{r['label']:<24}"
              f"{r['final_acc_mean']:.3f} ± {r['final_acc_std']:.3f}    "
              f"{r['total_energy_mean']:.2f} ± {r['total_energy_std']:.3f}   "
              f"{r['final_gini_mean']:.3f} ± {r['final_gini_std']:.3f}")
print("\nSaved:", os.path.join(SUM, "comparison_table.csv"))
