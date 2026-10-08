"""
analyze_hard.py  (Phase 5 / L7)
-------------------------------
On the selection-sensitive HARD config (extreme non-IID alpha=0.1, tight budget
k=3), does RL beat Random on accuracy? Reports final acc/energy/Gini per method.
"""
import glob, os
import numpy as np
import pandas as pd

RAW = "results_v2/hard_selection"
METHODS = ["random", "oort", "centralized_rl", "federated_marl"]
LAB = {"random":"Random","oort":"Oort","centralized_rl":"Centralized RL",
       "federated_marl":"Federated MARL (Ours)"}

rows = []
print(f"{'method':<24}{'Acc%':<16}{'Energy(J)':<14}{'Gini'}")
for m in METHODS:
    cs = sorted(glob.glob(f"{RAW}/{m}_seed*.csv"))
    if not cs: continue
    acc = [pd.read_csv(c)["accuracy"].iloc[-1]*100 for c in cs]
    en  = [pd.read_csv(c)["energy_cost"].sum() for c in cs]
    gi  = [pd.read_csv(c)["gini"].iloc[-1] for c in cs]
    rows.append({"method":m,"acc_mean":np.mean(acc),"acc_std":np.std(acc,ddof=1),
                 "energy_mean":np.mean(en),"gini_mean":np.mean(gi),"n":len(cs)})
    print(f"{LAB[m]:<24}{np.mean(acc):.2f}±{np.std(acc,ddof=1):.2f}     "
          f"{np.mean(en):.2f}       {np.mean(gi):.3f}")

df = pd.DataFrame(rows).round(4)
os.makedirs("results_v2/summary", exist_ok=True)
df.to_csv("results_v2/summary/hard_selection.csv", index=False)

if len(df):
    rnd = df[df.method=="random"]["acc_mean"].iloc[0]
    fm  = df[df.method=="federated_marl"]["acc_mean"].iloc[0]
    v = (f"On the hard (alpha=0.1, k=3) config, FMARL acc={fm:.2f}% vs Random={rnd:.2f}% "
         f"-> RL {'beats' if fm-rnd>0.5 else 'ties'} Random by {fm-rnd:+.2f}pp.")
    print("\n"+v)
    open("results_v2/summary/hard_selection_verdict.txt","w").write(v+"\n")
