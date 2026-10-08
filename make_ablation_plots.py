"""
make_ablation_plots.py
----------------------
Two ablation figures, each a 1x2 panel:
  LEFT  panel = task metrics: final accuracy (err bars) + final Gini  -> shows FLAT
  RIGHT panel = mechanism: final policy similarity at last federation event
Written to results_bundle/plots/{alpha_sweep.png, cadence_sweep.png}.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABL = "results_bundle/ablations_ue1"
PLOTS = "results_bundle/plots"
os.makedirs(PLOTS, exist_ok=True)
ACC_C, GINI_C, SIM_C = "#E91E63", "#2196F3", "#4CAF50"


def collect(sweep_dir, prefix, cast):
    pts = []
    for d in glob.glob(os.path.join(sweep_dir, f"{prefix}_*")):
        val = cast(os.path.basename(d).replace(prefix + "_", ""))
        accs, ginis, sims = [], [], []
        for c in glob.glob(os.path.join(d, "federated_marl_seed*.csv")):
            df = pd.read_csv(c)
            accs.append(df["accuracy"].iloc[-1] * 100)
            ginis.append(df["gini"].iloc[-1])
            s = df["policy_similarity"].dropna()
            sims.append(s.iloc[-1] if len(s) else np.nan)
        pts.append((val, np.mean(accs), np.std(accs, ddof=1),
                    np.mean(ginis), np.nanmean(sims)))
    return sorted(pts)


def figure(sweep_dir, prefix, cast, xlabel, title, outfile):
    pts = collect(sweep_dir, prefix, cast)
    x = [p[0] for p in pts]
    acc = [p[1] for p in pts]; accs = [p[2] for p in pts]
    gini = [p[3] for p in pts]; sim = [p[4] for p in pts]
    xt = [str(int(v)) if prefix == "T_RL" else str(v) for v in x]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    # LEFT: task metrics
    axL.errorbar(range(len(x)), acc, yerr=accs, fmt="o-", color=ACC_C, lw=2,
                 capsize=4, label="Final Accuracy (%)")
    axL.set_ylabel("Final Accuracy (%)", color=ACC_C)
    axL.tick_params(axis="y", labelcolor=ACC_C)
    axL.set_xticks(range(len(x))); axL.set_xticklabels(xt)
    axL.set_xlabel(xlabel)
    axLg = axL.twinx()
    axLg.plot(range(len(x)), gini, "s--", color=GINI_C, lw=2, label="Final Gini")
    axLg.set_ylabel("Final Gini", color=GINI_C)
    axLg.tick_params(axis="y", labelcolor=GINI_C)
    axLg.set_ylim(0, 0.6)  # fixed range so a ~0.0002 spread reads honestly as flat
    axL.set_title("Task metrics (flat across sweep)", fontsize=11)
    axL.grid(True, alpha=0.3)
    # RIGHT: mechanism
    axR.plot(range(len(x)), sim, "D-", color=SIM_C, lw=2, ms=7)
    axR.set_xticks(range(len(x))); axR.set_xticklabels(xt)
    axR.set_xlabel(xlabel)
    axR.set_ylabel("Policy cosine similarity (last federation)")
    axR.set_title("Federation mechanism (policy convergence)", fontsize=11)
    axR.grid(True, alpha=0.3)
    axR.set_ylim(0, 1.05)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150); plt.close()
    print("saved", outfile)


figure(os.path.join(ABL, "alpha"), "alpha", float,
       "Soft-update alpha (0=hard replace ... 1=no federation)",
       "Alpha Sweep (federated_marl, update_every=1, 3 seeds, 50 rounds)",
       f"{PLOTS}/alpha_sweep.png")
figure(os.path.join(ABL, "cadence"), "T_RL", int,
       "T_RL (policy federation cadence, rounds)",
       "T_RL Cadence Sweep (federated_marl, update_every=1, 3 seeds, 50 rounds)",
       f"{PLOTS}/cadence_sweep.png")
