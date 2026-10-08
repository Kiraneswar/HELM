"""
driver_rebaseline.py
--------------------
Phase 2+3: re-baseline the full comparison on the FIXED environment (L5 network
filtering + L2 grounded energy), at 20 seeds so the energy/Gini gaps get real
confidence intervals (L8).

- update_every=1: all 5 methods x 20 seeds x 100 rounds
- update_every=3: centralized_rl, federated_marl x 20 seeds (robustness carry-over)

Output: results_v2/raw_results/update_every_{1,3}/{method}_seed{S}.csv
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.runner import ExperimentRunner, load_config

OUT = "results_v2/raw_results"
N_SEEDS = 20

cfg_base = load_config("configs/full_experiment.yaml")
cfg_base["num_seeds"] = N_SEEDS
cfg_base["num_rounds"] = 100

PLAN = [
    (1, ["random", "full", "oort", "centralized_rl", "federated_marl"]),
    (3, ["centralized_rl", "federated_marl"]),
]

t0 = time.time()
for ue, methods in PLAN:
    outdir = os.path.join(OUT, f"update_every_{ue}")
    os.makedirs(outdir, exist_ok=True)
    for m in methods:
        cfg = copy.deepcopy(cfg_base)
        cfg.setdefault("rl", {})["update_every_n_rounds"] = ue
        cfg.setdefault("logging", {})["results_dir"] = outdir
        ts = time.time()
        ExperimentRunner(config=cfg, method=m, results_dir=outdir,
                         verbose=False).run_multi_seed(num_seeds=N_SEEDS)
        print(f"[ue={ue}] {m}: {time.time()-ts:.0f}s", flush=True)
print(f"REBASELINE COMPLETE {time.time()-t0:.0f}s", flush=True)
open("results_v2/.rebaseline_done", "w").write("ok")
