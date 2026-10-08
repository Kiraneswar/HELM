"""
make_overlay_plot.py
--------------------
Accuracy-vs-round overlay for FMARL and CRL at BOTH update_every=1 and =3
(4 lines, mean +/- std across 5 seeds), so the accuracy-tie finding is visually
obvious. Written to results_bundle/plots/accuracy_tie_ue1_vs_ue3.png.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results_bundle/raw_results"
PLOTS = "results_bundle/plots"
os.makedirs(PLOTS, exist_ok=True)

# 4 series: (method, ue) -> style
SERIES = [
    ("federated_marl", 1, "#E91E63", "-",  "Federated MARL — update_every=1"),
    ("federated_marl", 3, "#E91E63", "--", "Federated MARL — update_every=3"),
    ("centralized_rl", 1, "#2196F3", "-",  "Centralized RL — update_every=1"),
    ("centralized_rl", 3, "#2196F3", "--", "Centralized RL — update_every=3"),
]

fig, ax = plt.subplots(figsize=(11, 6.5))
for method, ue, color, ls, label in SERIES:
    csvs = sorted(glob.glob(os.path.join(RAW, f"update_every_{ue}", f"{method}_seed*.csv")))
    if not csvs:
        continue
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    g = df.groupby("fl_round")["accuracy"]
    r = g.mean().index.values
    mean = g.mean().values * 100
    std = g.std().values * 100
    ax.plot(r, mean, ls, color=color, lw=2, label=label)
    ax.fill_between(r, mean - std, mean + std, alpha=0.10, color=color)

ax.set_xlabel("FL Round", fontsize=13)
ax.set_ylabel("Test Accuracy (%)", fontsize=13)
ax.set_title("Accuracy is a tie: Federated MARL vs Centralized RL\n"
             "at update_every = 1 and 3 (mean ± std, 5 seeds, 100 rounds)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
out = os.path.join(PLOTS, "accuracy_tie_ue1_vs_ue3.png")
plt.savefig(out, dpi=150)
plt.close()
print("saved", out)

# print the final-round means for reference/cross-check
print("\nFinal-round accuracy (%) mean across seeds:")
for method, ue, *_ in SERIES:
    csvs = sorted(glob.glob(os.path.join(RAW, f"update_every_{ue}", f"{method}_seed*.csv")))
    finals = [pd.read_csv(c)["accuracy"].iloc[-1] * 100 for c in csvs]
    print(f"  {method:<16} ue={ue}: {np.mean(finals):.2f} ± {np.std(finals, ddof=1):.2f}")
