"""
analyze_reward_ablation.py  (Phase 4 / L6)
------------------------------------------
Answers the circularity critique. For each reward setting, report FMARL vs CRL
on accuracy / energy / Gini. The key row is 'accuracy_only' (energy & fairness
weights = 0): if FMARL STILL uses less energy / is fairer there, the advantage
is structural (from the hierarchical selection), not just because we rewarded it.
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = "results_v2/reward_ablation"
SETTINGS = ["accuracy_only", "default", "energy_heavy", "fairness_heavy"]

def stats(setting, method):
    cs = sorted(glob.glob(f"{ROOT}/{setting}/{method}_seed*.csv"))
    if not cs:
        return None
    acc = [pd.read_csv(c)["accuracy"].iloc[-1]*100 for c in cs]
    en  = [pd.read_csv(c)["energy_cost"].sum() for c in cs]
    gi  = [pd.read_csv(c)["gini"].iloc[-1] for c in cs]
    return np.mean(acc), np.mean(en), np.mean(gi)

rows = []
print(f"{'setting':<15}{'method':<16}{'Acc%':<9}{'Energy(J)':<12}{'Gini'}")
for s in SETTINGS:
    for m in ["federated_marl", "centralized_rl"]:
        r = stats(s, m)
        if r is None: continue
        rows.append({"setting": s, "method": m, "acc": r[0], "energy": r[1], "gini": r[2]})
        print(f"{s:<15}{m:<16}{r[0]:<9.2f}{r[1]:<12.2f}{r[2]:.3f}")

df = pd.DataFrame(rows)
os.makedirs("results_v2/summary", exist_ok=True)
df.round(4).to_csv("results_v2/summary/reward_ablation.csv", index=False)

print("\n--- FMARL vs CRL gap per reward setting (positive = FMARL better) ---")
print(f"{'setting':<15}{'d_energy(CRL-FMARL)':<22}{'d_gini(CRL-FMARL)':<20}{'d_acc(FMARL-CRL)'}")
verdict = []
for s in SETTINGS:
    f = df[(df.setting==s)&(df.method=='federated_marl')]
    c = df[(df.setting==s)&(df.method=='centralized_rl')]
    if len(f)==0 or len(c)==0: continue
    de = c.energy.iloc[0]-f.energy.iloc[0]
    dg = c.gini.iloc[0]-f.gini.iloc[0]
    da = f.acc.iloc[0]-c.acc.iloc[0]
    print(f"{s:<15}{de:<22.2f}{dg:<20.3f}{da:+.2f}")
    if s=="accuracy_only":
        verdict.append(f"Under accuracy-only reward (no energy/fairness terms): "
                       f"FMARL still uses {de:.1f} J less and Gini {dg:.3f} lower than CRL "
                       f"-> the gap is {'STRUCTURAL, not reward-driven' if de>2 else 'reward-driven'}.")
print("\n".join(verdict))
with open("results_v2/summary/reward_ablation_verdict.txt","w") as fh:
    fh.write("\n".join(verdict)+"\n")
