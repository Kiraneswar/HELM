"""
verify_bundle.py
----------------
Independent verification of the raw per-round CSVs (NOT summary claims).

For every update_every_{u} directory it checks, for each PPO method and seed:
  1. non-zero policy_loss row count == expected 100 // u  (PPO fired exactly
     the expected number of times)
  2. for federated_marl: non-NaN policy_similarity row count == 100 // T_RL
     (federation fired on the expected schedule, NaN elsewhere)

Prints a per-file table and a final PASS/FAIL. Exit 0 iff all checks pass.
"""
import os, sys, glob
import numpy as np
import pandas as pd

RAW = "results_bundle/raw_results"
T_RL = 5
NUM_ROUNDS = 100
PPO_METHODS = ["centralized_rl", "federated_marl"]

all_ok = True
lines = []
lines.append(f"{'dir':<16}{'method':<18}{'seed':<6}{'rows':<6}"
             f"{'nz_ploss':<10}{'exp_ploss':<10}{'nonNaN_sim':<12}{'exp_sim':<9}{'status'}")
lines.append("-" * 100)

for ue_dir in sorted(glob.glob(os.path.join(RAW, "update_every_*"))):
    u = int(os.path.basename(ue_dir).split("_")[-1])
    exp_ploss = NUM_ROUNDS // u
    exp_sim = NUM_ROUNDS // T_RL
    for method in PPO_METHODS:
        for csv in sorted(glob.glob(os.path.join(ue_dir, f"{method}_seed*.csv"))):
            df = pd.read_csv(csv)
            seed = os.path.basename(csv).split("seed")[-1].replace(".csv", "")
            rows = len(df)
            nz_ploss = int((df["policy_loss"].astype(float) != 0.0).sum())
            nonnan_sim = int(df["policy_similarity"].notna().sum())
            ok_ploss = (nz_ploss == exp_ploss)
            # policy_similarity check only meaningful for federated_marl
            if method == "federated_marl":
                ok_sim = (nonnan_sim == exp_sim)
                sim_disp = str(nonnan_sim)
                exp_sim_disp = str(exp_sim)
            else:
                ok_sim = True
                sim_disp = "n/a"
                exp_sim_disp = "n/a"
            ok = ok_ploss and ok_sim and rows == NUM_ROUNDS
            all_ok &= ok
            lines.append(f"{os.path.basename(ue_dir):<16}{method:<18}{seed:<6}"
                         f"{rows:<6}{nz_ploss:<10}{exp_ploss:<10}"
                         f"{sim_disp:<12}{exp_sim_disp:<9}{'PASS' if ok else 'FAIL'}")

report = "\n".join(lines)
print(report)
print()
print("=" * 60)
print("VERDICT:", "ALL RAW-CSV CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
print("=" * 60)

os.makedirs("results_bundle/logs", exist_ok=True)
with open("results_bundle/logs/verification_raw_csv.log", "w") as f:
    f.write(report + "\n\nVERDICT: " +
            ("ALL RAW-CSV CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + "\n")

sys.exit(0 if all_ok else 1)
