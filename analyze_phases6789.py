"""
analyze_phases6789.py
---------------------
Summaries for Phase 6 (MAPPO), 7 (DP), 8 (personalization), 9 (scale).
Writes CSVs + a text verdict per phase into results_v2/summary/.
"""
import glob, os
import numpy as np
import pandas as pd
os.makedirs("results_v2/summary", exist_ok=True)

def finals(csvs):
    acc=[pd.read_csv(c)["accuracy"].iloc[-1]*100 for c in csvs]
    en =[pd.read_csv(c)["energy_cost"].sum() for c in csvs]
    gi =[pd.read_csv(c)["gini"].iloc[-1] for c in csvs]
    sim=[]
    for c in csvs:
        s=pd.read_csv(c)["policy_similarity"].dropna()
        sim.append(s.iloc[-1] if len(s) else np.nan)
    return acc,en,gi,sim

out=[]

# ---- Phase 6: MAPPO ----
print("=== Phase 6: MAPPO (centralized critic) vs Federated MARL (independent PPO) vs CRL ===")
rows=[]
for m in ["mappo_marl","federated_marl","centralized_rl"]:
    cs=sorted(glob.glob(f"results_v2/mappo/update_every_1/{m}_seed*.csv"))
    if not cs: continue
    acc,en,gi,_=finals(cs)
    rows.append({"method":m,"n":len(cs),"acc_mean":np.mean(acc),"acc_std":np.std(acc,ddof=1),
                 "energy_mean":np.mean(en),"gini_mean":np.mean(gi)})
    print(f"  {m:16} acc={np.mean(acc):.2f}±{np.std(acc,ddof=1):.2f}  E={np.mean(en):.2f}  Gini={np.mean(gi):.3f}")
if rows:
    df=pd.DataFrame(rows); df.round(4).to_csv("results_v2/summary/mappo.csv",index=False)
    d=df.set_index("method")
    if "mappo_marl" in d.index and "federated_marl" in d.index:
        da=d.loc["mappo_marl","acc_mean"]-d.loc["federated_marl","acc_mean"]
        v=(f"MAPPO vs Federated MARL: acc {da:+.2f}pp, energy "
           f"{d.loc['mappo_marl','energy_mean']-d.loc['federated_marl','energy_mean']:+.2f} J. "
           f"{'MAPPO (centralized critic) improves on independent PPO' if da>0.5 else 'MAPPO ties independent PPO here'}.")
        print("  ->",v); open("results_v2/summary/mappo_verdict.txt","w").write(v+"\n")

# ---- Phase 7: DP ----
print("\n=== Phase 7: DP noise on policy federation (privacy/utility) ===")
rows=[]
for d in sorted(glob.glob("results_v2/dp_sweep/noise_*")):
    noise=float(os.path.basename(d).replace("noise_",""))
    cs=sorted(glob.glob(f"{d}/federated_marl_seed*.csv"))
    if not cs: continue
    acc,en,gi,sim=finals(cs)
    rows.append({"dp_noise_std":noise,"acc_mean":np.mean(acc),"acc_std":np.std(acc,ddof=1),
                 "final_policy_sim":np.nanmean(sim),"gini_mean":np.mean(gi)})
    print(f"  noise={noise:<6} acc={np.mean(acc):.2f}±{np.std(acc,ddof=1):.2f}  policy_sim={np.nanmean(sim):.3f}  Gini={np.mean(gi):.3f}")
if rows:
    df=pd.DataFrame(rows).sort_values("dp_noise_std"); df.round(4).to_csv("results_v2/summary/dp_sweep.csv",index=False)
    lo=df.iloc[0]; hi=df.iloc[-1]
    v=(f"DP: raising noise 0 -> {hi.dp_noise_std} changes policy similarity "
       f"{lo.final_policy_sim:.3f} -> {hi.final_policy_sim:.3f} and accuracy "
       f"{lo.acc_mean:.2f} -> {hi.acc_mean:.2f}% (utility cost of privacy).")
    print("  ->",v); open("results_v2/summary/dp_verdict.txt","w").write(v+"\n")

# ---- Phase 8: personalization ----
print("\n=== Phase 8: personalized vs fully-shared vs fully-local federation ===")
rows=[]
for name in ["fully_shared","personalized","fully_local"]:
    cs=sorted(glob.glob(f"results_v2/personalize/{name}/federated_marl_seed*.csv"))
    if not cs: continue
    acc,en,gi,sim=finals(cs)
    rows.append({"variant":name,"acc_mean":np.mean(acc),"acc_std":np.std(acc,ddof=1),
                 "energy_mean":np.mean(en),"gini_mean":np.mean(gi),"final_policy_sim":np.nanmean(sim)})
    print(f"  {name:14} acc={np.mean(acc):.2f}±{np.std(acc,ddof=1):.2f}  E={np.mean(en):.2f}  Gini={np.mean(gi):.3f}  sim={np.nanmean(sim):.3f}")
if rows:
    df=pd.DataFrame(rows); df.round(4).to_csv("results_v2/summary/personalize.csv",index=False)
    d=df.set_index("variant")
    if set(["personalized","fully_shared","fully_local"]).issubset(d.index):
        best=d["acc_mean"].idxmax()
        v=(f"Personalization: acc personalized={d.loc['personalized','acc_mean']:.2f}, "
           f"shared={d.loc['fully_shared','acc_mean']:.2f}, local={d.loc['fully_local','acc_mean']:.2f} "
           f"-> best = {best}.")
        print("  ->",v); open("results_v2/summary/personalize_verdict.txt","w").write(v+"\n")

# ---- Phase 9: scale ----
print("\n=== Phase 9: scale to 200 clients / 8 clusters ===")
rows=[]
for m in ["random","centralized_rl","federated_marl"]:
    cs=sorted(glob.glob(f"results_v2/scale/{m}_seed*.csv"))
    if not cs: continue
    acc,en,gi,_=finals(cs)
    rows.append({"method":m,"n":len(cs),"acc_mean":np.mean(acc),"acc_std":np.std(acc,ddof=1),
                 "energy_mean":np.mean(en),"gini_mean":np.mean(gi)})
    print(f"  {m:16} acc={np.mean(acc):.2f}±{np.std(acc,ddof=1):.2f}  E={np.mean(en):.2f}  Gini={np.mean(gi):.3f}")
if rows:
    df=pd.DataFrame(rows); df.round(4).to_csv("results_v2/summary/scale.csv",index=False)
    d=df.set_index("method")
    if "federated_marl" in d.index and "centralized_rl" in d.index:
        de=(d.loc["centralized_rl","energy_mean"]-d.loc["federated_marl","energy_mean"])/d.loc["centralized_rl","energy_mean"]*100
        dg=(d.loc["centralized_rl","gini_mean"]-d.loc["federated_marl","gini_mean"])/d.loc["centralized_rl","gini_mean"]*100
        v=(f"At 200 clients: FMARL vs CRL energy -{de:.0f}%, Gini -{dg:.0f}% "
           f"-> the energy/fairness advantage {'holds at scale' if de>5 else 'shrinks at scale'}.")
        print("  ->",v); open("results_v2/summary/scale_verdict.txt","w").write(v+"\n")
print("\nDone.")
