"""Phase 7 / L9: differential-privacy noise on policy federation — utility/privacy trade-off."""
import os, time, copy
from src.simulation.runner import ExperimentRunner, load_config
OUT="results_v2/dp_sweep"; 
base=load_config("configs/demo.yaml"); base["num_rounds"]=50; base["num_seeds"]=5
base.setdefault("rl",{})["update_every_n_rounds"]=1
t0=time.time()
for noise in [0.0, 0.005, 0.01, 0.05]:
    cfg=copy.deepcopy(base)
    cfg["federation"]={"alpha":0.5,"mode":"soft","dp_clip":1.0,"dp_noise_std":noise}
    d=os.path.join(OUT, f"noise_{noise}"); os.makedirs(d, exist_ok=True)
    cfg.setdefault("logging",{})["results_dir"]=d
    ts=time.time()
    ExperimentRunner(config=cfg, method="federated_marl", results_dir=d, verbose=False).run_multi_seed(num_seeds=5)
    print(f"[dp] noise={noise}: {time.time()-ts:.0f}s", flush=True)
print(f"DP PHASE DONE {time.time()-t0:.0f}s", flush=True); open("results_v2/.dp_done","w").write("ok")
