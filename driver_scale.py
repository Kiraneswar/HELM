"""Phase 9 / L3: scale to 200 clients / 8 clusters. random vs centralized_rl vs federated_marl."""
import os, time, copy
from src.simulation.runner import ExperimentRunner, load_config
OUT="results_v2/scale"; os.makedirs(OUT, exist_ok=True)
cfg=load_config("configs/scale_large.yaml"); cfg.setdefault("rl",{})["update_every_n_rounds"]=1
t0=time.time()
for m in ["random","centralized_rl","federated_marl"]:
    ts=time.time()
    ExperimentRunner(config=copy.deepcopy(cfg), method=m, results_dir=OUT, verbose=False).run_multi_seed(num_seeds=3)
    print(f"[scale] {m}: {time.time()-ts:.0f}s", flush=True)
print(f"SCALE PHASE DONE {time.time()-t0:.0f}s", flush=True); open("results_v2/.scale_done","w").write("ok")
