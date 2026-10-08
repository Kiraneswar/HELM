"""
make_v2_plots.py
----------------
Plots for the fixed-environment (v2) results:
  - accuracy_vs_round_v2.png, energy_vs_round_v2.png, gini_vs_round_v2.png
  - accuracy_energy_tradeoff_v2.png
  - learning_curve_v2.png  (mean_reward over rounds, FMARL vs CRL -- L7 evidence)
Reads results_v2/raw_results/update_every_1.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results_v2/raw_results/update_every_1"
PLOTS = "results_v2/plots"
os.makedirs(PLOTS, exist_ok=True)
COL = {"random":"#9E9E9E","full":"#4CAF50","oort":"#FF9800",
       "centralized_rl":"#2196F3","federated_marl":"#E91E63"}
LAB = {"random":"Random","full":"Full Participation","oort":"Oort (Heuristic)",
       "centralized_rl":"Centralized RL","federated_marl":"Federated MARL (Ours)"}
METHODS = list(COL)

def stack(m):
    cs = sorted(glob.glob(f"{RAW}/{m}_seed*.csv"))
    return pd.concat([pd.read_csv(c) for c in cs], ignore_index=True) if cs else None

def curve(col, ylabel, title, out, pct=False, ylim=None):
    fig, ax = plt.subplots(figsize=(10,6))
    for m in METHODS:
        d = stack(m)
        if d is None: continue
        g = d.groupby("fl_round")[col]
        r = g.mean().index.values; mean = g.mean().values*(100 if pct else 1)
        std = g.std().values*(100 if pct else 1)
        ax.plot(r, mean, color=COL[m], lw=2, label=LAB[m])
        ax.fill_between(r, mean-std, mean+std, color=COL[m], alpha=0.10)
    ax.set_xlabel("FL Round"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if ylim: ax.set_ylim(*ylim)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print("saved", out)

curve("accuracy","Test Accuracy (%)","Accuracy vs Round (fixed env, 20 seeds)",
      f"{PLOTS}/accuracy_vs_round_v2.png", pct=True, ylim=(0,None))
curve("energy_cost","Per-Round Energy (J)","Per-Round Energy vs Round (fixed env)",
      f"{PLOTS}/energy_vs_round_v2.png")
curve("gini","Gini (lower=fairer)","Participation Fairness vs Round (fixed env)",
      f"{PLOTS}/gini_vs_round_v2.png", ylim=(0,1))

# tradeoff
fig, ax = plt.subplots(figsize=(8,6))
for m in METHODS:
    d = stack(m)
    if d is None: continue
    seeds = d.seed.unique()
    fa = np.mean([d[d.seed==s]["accuracy"].iloc[-1] for s in seeds])*100
    te = np.mean([d[d.seed==s]["energy_cost"].sum() for s in seeds])
    ax.scatter(te, fa, color=COL[m], s=140, zorder=5)
    ax.annotate(LAB[m], (te,fa), textcoords="offset points", xytext=(8,4), fontsize=9)
ax.set_xlabel("Total Cumulative Energy (J)"); ax.set_ylabel("Final Accuracy (%)")
ax.set_title("Accuracy vs Energy (fixed env, grounded energy model)", fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOTS}/accuracy_energy_tradeoff_v2.png", dpi=150); plt.close()
print("saved tradeoff")

# learning curve (mean_reward) -- L7
fig, ax = plt.subplots(figsize=(10,6))
for m in ["federated_marl","centralized_rl"]:
    d = stack(m)
    if d is None: continue
    g = d.groupby("fl_round")["mean_reward"]
    r = g.mean().index.values; mean = g.mean().values
    # smooth with a rolling window for readability
    s = pd.Series(mean).rolling(5, min_periods=1).mean().values
    ax.plot(r, s, color=COL[m], lw=2, label=LAB[m]+" (reward, 5-round MA)")
ax.set_xlabel("FL Round"); ax.set_ylabel("RL reward (per round)")
ax.set_title("RL Learning Curve: reward over training (fixed env, 20 seeds)", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOTS}/learning_curve_v2.png", dpi=150); plt.close()
print("saved learning curve")
