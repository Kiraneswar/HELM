"""
verify_ablations.py
-------------------
Independent per-CSV verification of the phase-5 ablation raw results
(update_every=1, 50 rounds, 3 seeds, federated_marl).

Checks, directly from each per-round CSV:
  1. rows == 50
  2. non-zero policy_loss rows == 50   (PPO fires every round at update_every=1)
  3. non-NaN policy_similarity rows == number of federation rounds == floor(50/T_RL)
     - alpha sweep: T_RL fixed at 5  -> 10
     - cadence sweep: T_RL in {1,3,5,10} -> {50,16,10,5}
Exit 0 iff all pass.
"""
import os, glob, sys
import pandas as pd

ABL = "results_bundle/ablations_ue1"
NUM_ROUNDS = 50
UPDATE_EVERY = 1
exp_ploss = NUM_ROUNDS // UPDATE_EVERY

def fed_rounds(t_rl):
    return len([r for r in range(1, NUM_ROUNDS + 1) if r % t_rl == 0])

all_ok = True
lines = [f"{'sweep':<8}{'config':<14}{'seed':<6}{'rows':<6}{'nz_ploss':<10}"
         f"{'exp_pl':<8}{'nonNaN_sim':<12}{'exp_sim':<9}{'status'}", "-" * 90]

# alpha sweep: T_RL fixed = 5
for d in sorted(glob.glob(os.path.join(ABL, "alpha", "alpha_*"))):
    val = os.path.basename(d).replace("alpha_", "")
    exp_sim = fed_rounds(5)
    for csv in sorted(glob.glob(os.path.join(d, "federated_marl_seed*.csv"))):
        df = pd.read_csv(csv)
        seed = os.path.basename(csv).split("seed")[-1].replace(".csv", "")
        rows = len(df)
        nz = int((df["policy_loss"].astype(float) != 0.0).sum())
        sim = int(df["policy_similarity"].notna().sum())
        ok = (rows == NUM_ROUNDS and nz == exp_ploss and sim == exp_sim)
        all_ok &= ok
        lines.append(f"{'alpha':<8}{'alpha='+val:<14}{seed:<6}{rows:<6}{nz:<10}"
                     f"{exp_ploss:<8}{sim:<12}{exp_sim:<9}{'PASS' if ok else 'FAIL'}")

# cadence sweep: T_RL varies
for d in sorted(glob.glob(os.path.join(ABL, "cadence", "T_RL_*"))):
    val = os.path.basename(d).replace("T_RL_", "")
    t_rl = int(val)
    exp_sim = fed_rounds(t_rl)
    for csv in sorted(glob.glob(os.path.join(d, "federated_marl_seed*.csv"))):
        df = pd.read_csv(csv)
        seed = os.path.basename(csv).split("seed")[-1].replace(".csv", "")
        rows = len(df)
        nz = int((df["policy_loss"].astype(float) != 0.0).sum())
        sim = int(df["policy_similarity"].notna().sum())
        ok = (rows == NUM_ROUNDS and nz == exp_ploss and sim == exp_sim)
        all_ok &= ok
        lines.append(f"{'cadence':<8}{'T_RL='+val:<14}{seed:<6}{rows:<6}{nz:<10}"
                     f"{exp_ploss:<8}{sim:<12}{exp_sim:<9}{'PASS' if ok else 'FAIL'}")

report = "\n".join(lines)
print(report)
print("\n" + "=" * 60)
print("VERDICT:", "ALL ABLATION RAW-CSV CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
print("=" * 60)

os.makedirs("results_bundle/logs", exist_ok=True)
with open("results_bundle/logs/verification_ablations.log", "w") as f:
    f.write(report + "\n\nVERDICT: " +
            ("ALL ABLATION RAW-CSV CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + "\n")
sys.exit(0 if all_ok else 1)
