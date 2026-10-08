"""
driver_robustness.py
--------------------
Runs the robustness matrix: all methods x 5 seeds x 100 rounds at
update_every_n_rounds = 1 and update_every_n_rounds = 3.

- update_every=1: all 5 methods (random, full, oort, centralized_rl, federated_marl)
- update_every=3: only the PPO methods (centralized_rl, federated_marl) --
  random/full/oort do not use PPO so their update_every=1 CSVs are reused.

CSVs land in:
  results_bundle/raw_results/update_every_1/{method}_seed{S}.csv
  results_bundle/raw_results/update_every_3/{method}_seed{S}.csv
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.runner import ExperimentRunner, load_config

BUNDLE = "results_bundle"
RAW = os.path.join(BUNDLE, "raw_results")

cfg_base = load_config("configs/full_experiment.yaml")
cfg_base["num_seeds"] = 5
cfg_base["num_rounds"] = 100

PLAN = [
    (1, ["random", "full", "oort", "centralized_rl", "federated_marl"]),
    (3, ["centralized_rl", "federated_marl"]),
]

t_start = time.time()
for update_every, methods in PLAN:
    outdir = os.path.join(RAW, f"update_every_{update_every}")
    os.makedirs(outdir, exist_ok=True)
    for method in methods:
        cfg = copy.deepcopy(cfg_base)
        cfg.setdefault("rl", {})["update_every_n_rounds"] = update_every
        cfg.setdefault("logging", {})["results_dir"] = outdir
        t0 = time.time()
        runner = ExperimentRunner(config=cfg, method=method,
                                  results_dir=outdir, verbose=False)
        runner.run_multi_seed(num_seeds=5)
        print(f"[ue={update_every}] {method}: done in {time.time()-t0:.1f}s",
              flush=True)

print(f"ROBUSTNESS COMPLETE in {time.time()-t_start:.1f}s", flush=True)
open(os.path.join(BUNDLE, ".robustness_done"), "w").write("ok")
