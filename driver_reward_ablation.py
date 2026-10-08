"""
driver_reward_ablation.py  (Phase 4 / L6)
-----------------------------------------
Defuses the "circular reward" critique. The RL reward explicitly values low
energy and low Gini; a reviewer will ask whether the energy/fairness advantage
is just because we rewarded it. This sweep answers it head-on:

We run BOTH federated_marl AND centralized_rl under several reward settings,
including one with ZERO energy/fairness weight (accuracy-only), and report
whether FMARL still beats CRL on energy/Gini in each.

  - accuracy_only : energy=0.0,  fairness=0.0   (the key control)
  - default       : energy=0.1,  fairness=0.05
  - energy_heavy  : energy=0.3,  fairness=0.05
  - fairness_heavy: energy=0.1,  fairness=0.20

Scale: 20 clients / 3 clusters / 50 rounds / 5 seeds (fixed env).
Output: results_v2/reward_ablation/<setting>/<method>_seed*.csv
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.runner import ExperimentRunner, load_config

OUT = "results_v2/reward_ablation"
SETTINGS = {
    "accuracy_only":  {"accuracy": 1.0, "energy": 0.0, "fairness": 0.0},
    "default":        {"accuracy": 1.0, "energy": 0.1, "fairness": 0.05},
    "energy_heavy":   {"accuracy": 1.0, "energy": 0.3, "fairness": 0.05},
    "fairness_heavy": {"accuracy": 1.0, "energy": 0.1, "fairness": 0.20},
}

base = load_config("configs/demo.yaml")   # 20 clients / 3 clusters (fixed env)
base["num_rounds"] = 50
base["num_seeds"] = 5
base.setdefault("rl", {})["update_every_n_rounds"] = 1

t0 = time.time()
for name, rw in SETTINGS.items():
    for method in ("federated_marl", "centralized_rl"):
        cfg = copy.deepcopy(base)
        cfg["rl"]["reward_weights"] = rw
        outdir = os.path.join(OUT, name)
        os.makedirs(outdir, exist_ok=True)
        cfg.setdefault("logging", {})["results_dir"] = outdir
        ts = time.time()
        ExperimentRunner(config=cfg, method=method, results_dir=outdir,
                         verbose=False).run_multi_seed(num_seeds=5)
        print(f"[{name}] {method}: {time.time()-ts:.0f}s", flush=True)
print(f"REWARD ABLATION COMPLETE {time.time()-t0:.0f}s", flush=True)
open("results_v2/.reward_ablation_done", "w").write("ok")
