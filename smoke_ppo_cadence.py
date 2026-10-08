"""
smoke_ppo_cadence.py
--------------------
Self-check: verifies update_every_n_rounds actually controls PPO firing cadence
for BOTH federated_marl AND centralized_rl.

Run BEFORE the full 5-seed experiment. Expected output:
  - Exactly one PPO-fire line per method per update_every_n_rounds rounds
  - Final counts equal for both methods
  - All assertions pass

Usage:
  python -X utf8 smoke_ppo_cadence.py
"""
import sys, os, types, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))

import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("configs/demo.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["num_rounds"] = 10
cfg["num_seeds"]  = 1
cfg["logging"]    = {"save_csv": False, "save_plots": False,
                     "results_dir": "results/smoke_ppo/", "verbose": False}

update_every = cfg.get("rl", {}).get("update_every_n_rounds", cfg.get("T_RL", 5))
T_RL         = cfg.get("T_RL", 5)
num_rounds   = cfg["num_rounds"]
seed         = 42

print("=" * 62)
print("  PPO Cadence Smoke Test")
print("=" * 62)
print(f"  Config:                demo.yaml ({num_rounds} rounds, seed={seed})")
print(f"  update_every_n_rounds: {update_every}")
print(f"  T_RL (federation):     {T_RL}")
print(f"  Expected PPO fires:    {num_rounds} / {update_every} = {num_rounds // update_every}")
print()

from src.simulation.runner import ExperimentRunner

results = {}   # method -> list of rounds where PPO fired

for method in ("centralized_rl", "federated_marl"):
    print(f"  --- {method} ---")
    ppo_fire_rounds = []

    cfg_copy = dict(cfg)
    cfg_copy["methods"] = [method]

    runner = ExperimentRunner(config=cfg_copy, method=method, verbose=False)

    orig_update_rl = runner._update_rl.__func__

    def _instrumented(self, selector_ctx, env, cluster_selections,
                      log_probs, values, acc_before, acc_after,
                      total_energy, all_participation, fl_round,
                      T_RL, update_every):
        result = orig_update_rl(self, selector_ctx, env, cluster_selections,
                                log_probs, values, acc_before, acc_after,
                                total_energy, all_participation, fl_round,
                                T_RL, update_every)
        if self.method == "centralized_rl":
            sel = selector_ctx["selector"]
            if getattr(sel, "last_update_fired", False):
                ppo_fire_rounds.append(fl_round)
                print(f"    Round {fl_round:>3}: PPO UPDATE fired  "
                      f"(update_every={update_every})")
        elif self.method == "federated_marl":
            if fl_round % update_every == 0:
                ppo_fire_rounds.append(fl_round)
                print(f"    Round {fl_round:>3}: PPO UPDATE fired  "
                      f"(update_every={update_every})")
        return result

    runner._update_rl = types.MethodType(_instrumented, runner)

    with contextlib.redirect_stdout(io.StringIO()):
        runner.run(seed=seed)

    results[method] = ppo_fire_rounds
    print(f"    Total PPO updates: {len(ppo_fire_rounds)} over {num_rounds} rounds")
    print(f"    Fired at rounds: {ppo_fire_rounds}")
    print()

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
print("=" * 62)
print("  ASSERTIONS")
print("=" * 62)

expected = num_rounds // update_every
ok = True

for method, fire_rounds in results.items():
    n = len(fire_rounds)
    status = "PASS" if n == expected else "FAIL"
    print(f"  [{status}] {method}: {n} PPO fires (expected {expected})")
    if n != expected:
        ok = False

crl_n   = len(results.get("centralized_rl", []))
fmarl_n = len(results.get("federated_marl", []))
eq_status = "PASS" if crl_n == fmarl_n else "FAIL"
print(f"  [{eq_status}] Equal PPO updates: centralized_rl={crl_n}, federated_marl={fmarl_n}")
if crl_n != fmarl_n:
    ok = False

print()
if ok:
    print("  ALL ASSERTIONS PASSED.")
    print("  Safe to proceed with full 5-seed 100-round run.")
    sys.exit(0)
else:
    print("  ASSERTIONS FAILED — do NOT run full experiment.")
    sys.exit(1)
