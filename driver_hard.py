"""
driver_hard.py  (Phase 5 / L7)
------------------------------
Runs all methods on the selection-sensitive "hard" config (extreme non-IID
alpha=0.1, tight budget k=3) to test whether the RL methods actually beat Random
when selection matters more. Output: results_v2/hard_selection/<method>_seed*.csv
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.runner import ExperimentRunner, load_config

cfg = load_config("configs/hard_selection.yaml")
cfg.setdefault("rl", {})["update_every_n_rounds"] = 1
outdir = "results_v2/hard_selection"
os.makedirs(outdir, exist_ok=True)
t0 = time.time()
for m in ["random", "oort", "centralized_rl", "federated_marl"]:
    ts = time.time()
    ExperimentRunner(config=cfg, method=m, results_dir=outdir,
                     verbose=False).run_multi_seed(num_seeds=cfg["num_seeds"])
    print(f"[hard] {m}: {time.time()-ts:.0f}s", flush=True)
print(f"HARD COMPLETE {time.time()-t0:.0f}s", flush=True)
open("results_v2/.hard_done", "w").write("ok")
