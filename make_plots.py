"""
make_plots.py
-------------
Generates every plot for the results bundle:
  Main comparison (from update_every_1 robustness raw CSVs):
    - accuracy_vs_round.png
    - energy_vs_round.png
    - gini_vs_round.png
    - accuracy_energy_tradeoff.png
    - policy_similarity.png   (NaN-gapped on non-federation rounds)
  Ablations (from ablations_ue{U}):
    - alpha_sweep.png
    - cadence_sweep.png
"""
import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results_bundle/raw_results"
PLOTS = "results_bundle/plots"
os.makedirs(PLOTS, exist_ok=True)

ABLATION_UE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAIN_UE_DIR = os.path.join(RAW, "update_every_1")

COLORS = {
    "random": "#9E9E9E", "full": "#4CAF50", "oort": "#FF9800",
    "centralized_rl": "#2196F3", "federated_marl": "#E91E63",
}
LABELS = {
    "random": "Random", "full": "Full Participation", "oort": "Oort (Heuristic)",
    "centralized_rl": "Centralized RL", "federated_marl": "Federated MARL (Ours)",
}
METHODS = ["random", "full", "oort", "centralized_rl", "federated_marl"]


def load_stack(ue_dir, method):
    """Return dataframe stacked across seeds, indexed by fl_round."""
    csvs = sorted(glob.glob(os.path.join(ue_dir, f"{method}_seed*.csv")))
    if not csvs:
        return None
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def per_round(ue_dir, method, col, agg="mean"):
    df = load_stack(ue_dir, method)
    if df is None:
        return None, None, None
    g = df.groupby("fl_round")[col]
    if agg == "nanmean":
        m = g.apply(lambda s: np.nanmean(s) if s.notna().any() else np.nan)
        return m.index.values, m.values, None
    return g.mean().index.values, g.mean().values, g.std().values


# ---- 1. accuracy vs round ----
fig, ax = plt.subplots(figsize=(10, 6))
for m in METHODS:
    r, mean, std = per_round(MAIN_UE_DIR, m, "accuracy")
    if r is None:
        continue
    ax.plot(r, mean * 100, label=LABELS[m], color=COLORS[m], lw=2)
    ax.fill_between(r, (mean - std) * 100, (mean + std) * 100, alpha=0.12, color=COLORS[m])
ax.set_xlabel("FL Round"); ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Global Model Accuracy vs. Round (update_every=1, 5 seeds)", fontweight="bold")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
plt.tight_layout(); plt.savefig(f"{PLOTS}/accuracy_vs_round.png", dpi=150); plt.close()

# ---- 2. energy vs round ----
fig, ax = plt.subplots(figsize=(10, 6))
for m in METHODS:
    r, mean, std = per_round(MAIN_UE_DIR, m, "energy_cost")
    if r is None:
        continue
    ax.plot(r, mean, label=LABELS[m], color=COLORS[m], lw=2)
ax.set_xlabel("FL Round"); ax.set_ylabel("Per-Round Energy Cost")
ax.set_title("Per-Round Energy Cost vs. Round (update_every=1)", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOTS}/energy_vs_round.png", dpi=150); plt.close()

# ---- 3. gini vs round ----
fig, ax = plt.subplots(figsize=(10, 6))
for m in METHODS:
    r, mean, std = per_round(MAIN_UE_DIR, m, "gini")
    if r is None:
        continue
    ax.plot(r, mean, label=LABELS[m], color=COLORS[m], lw=2)
ax.set_xlabel("FL Round"); ax.set_ylabel("Gini (lower = fairer)")
ax.set_title("Participation Fairness (Gini) vs. Round (update_every=1)", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{PLOTS}/gini_vs_round.png", dpi=150); plt.close()

# ---- 4. accuracy vs energy tradeoff ----
fig, ax = plt.subplots(figsize=(8, 6))
for m in METHODS:
    df = load_stack(MAIN_UE_DIR, m)
    if df is None:
        continue
    seeds = df["seed"].unique()
    fin_acc = np.mean([df[df.seed == s]["accuracy"].iloc[-1] for s in seeds]) * 100
    tot_e = np.mean([df[df.seed == s]["energy_cost"].sum() for s in seeds])
    ax.scatter(tot_e, fin_acc, color=COLORS[m], s=140, zorder=5)
    ax.annotate(LABELS[m], (tot_e, fin_acc), textcoords="offset points",
                xytext=(8, 4), fontsize=9)
ax.set_xlabel("Total Cumulative Energy Cost"); ax.set_ylabel("Final Test Accuracy (%)")
ax.set_title("Accuracy vs. Energy Tradeoff (update_every=1)", fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOTS}/accuracy_energy_tradeoff.png", dpi=150); plt.close()

# ---- 5. policy similarity (NaN-gapped) ----
fig, ax = plt.subplots(figsize=(10, 5))
r, mean, _ = per_round(MAIN_UE_DIR, "federated_marl", "policy_similarity", agg="nanmean")
if r is not None:
    ax.plot(r, mean, "o-", color=COLORS["federated_marl"], lw=2, ms=5,
            label="Federated MARL policy cosine similarity")
ax.set_xlabel("FL Round"); ax.set_ylabel("Cosine Similarity (post-federation)")
ax.set_title("RL Policy Convergence at Federation Events (NaN-gapped, T_RL=5)",
             fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOTS}/policy_similarity.png", dpi=150); plt.close()


# ---- ablation helpers ----
def ablation_points(sweep_dir, pshort):
    """Return sorted list of (value, acc_mean, acc_std, energy_mean, gini_mean)."""
    pts = []
    for d in glob.glob(os.path.join(sweep_dir, f"{pshort}_*")):
        val = d.split(f"{pshort}_")[-1]
        try:
            valf = float(val)
        except ValueError:
            continue
        csvs = sorted(glob.glob(os.path.join(d, "federated_marl_seed*.csv")))
        if not csvs:
            continue
        accs, energies, ginis = [], [], []
        for c in csvs:
            df = pd.read_csv(c)
            accs.append(df["accuracy"].iloc[-1] * 100)
            energies.append(df["energy_cost"].sum())
            ginis.append(df["gini"].iloc[-1])
        pts.append((valf, np.mean(accs), np.std(accs, ddof=1),
                    np.mean(energies), np.mean(ginis)))
    return sorted(pts)


def plot_sweep(sweep_dir, pshort, xlabel, title, outfile):
    pts = ablation_points(sweep_dir, pshort)
    if not pts:
        print(f"  [skip] no data in {sweep_dir}")
        return
    x = [p[0] for p in pts]
    acc = [p[1] for p in pts]; accs = [p[2] for p in pts]
    energy = [p[3] for p in pts]; gini = [p[4] for p in pts]
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.errorbar(x, acc, yerr=accs, fmt="o-", color="#E91E63", lw=2,
                 capsize=4, label="Final Accuracy (%)")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("Final Accuracy (%)", color="#E91E63")
    ax1.tick_params(axis="y", labelcolor="#E91E63")
    ax2 = ax1.twinx()
    ax2.plot(x, gini, "s--", color="#2196F3", lw=2, label="Final Gini")
    ax2.set_ylabel("Final Gini", color="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#2196F3")
    ax1.set_title(title, fontweight="bold"); ax1.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()
    print(f"  saved {outfile}")


ABL = f"results_bundle/ablations_ue{ABLATION_UE}"
plot_sweep(os.path.join(ABL, "alpha"), "alpha",
           "Soft-update alpha (0=hard replace, 1=no federation)",
           f"Alpha Sweep (federated_marl, update_every={ABLATION_UE}, 3 seeds)",
           f"{PLOTS}/alpha_sweep.png")
plot_sweep(os.path.join(ABL, "cadence"), "T_RL",
           "T_RL (policy federation cadence, rounds)",
           f"T_RL Cadence Sweep (federated_marl, update_every={ABLATION_UE}, 3 seeds)",
           f"{PLOTS}/cadence_sweep.png")

print("All plots written to", PLOTS)
