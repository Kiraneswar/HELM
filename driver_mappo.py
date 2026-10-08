"""Phase 6 / L4: MAPPO (centralized critic) vs Federated MARL (independent PPO) vs Centralized RL."""
import os, time, copy
from src.simulation.runner import ExperimentRunner, load_config
OUT="results_v2/mappo/update_every_1"; os.makedirs(OUT, exist_ok=True)
cfg=load_config("configs/full_experiment.yaml"); cfg["num_rounds"]=100; cfg["num_seeds"]=8
cfg.setdefault("rl",{})["update_every_n_rounds"]=1
t0=time.time()
for m in ["mappo_marl","federated_marl","centralized_rl"]:
    ts=time.time()
    ExperimentRunner(config=copy.deepcopy(cfg), method=m, results_dir=OUT, verbose=False).run_multi_seed(num_seeds=8)
    print(f"[mappo-phase] {m}: {time.time()-ts:.0f}s", flush=True)
print(f"MAPPO PHASE DONE {time.time()-t0:.0f}s", flush=True); open("results_v2/.mappo_done","w").write("ok")
