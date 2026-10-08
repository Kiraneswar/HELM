import os, copy
from src.simulation.runner import ExperimentRunner, load_config
base=load_config("configs/demo.yaml"); base["num_rounds"]=50; base["num_seeds"]=5
base.setdefault("rl",{})["update_every_n_rounds"]=1
for noise in [0.1, 0.3, 0.5, 1.0]:
    cfg=copy.deepcopy(base); cfg["federation"]={"alpha":0.5,"mode":"soft","dp_clip":1.0,"dp_noise_std":noise}
    d=f"results_v2/dp_sweep/noise_{noise}"; os.makedirs(d, exist_ok=True); cfg.setdefault("logging",{})["results_dir"]=d
    ExperimentRunner(config=cfg, method="federated_marl", results_dir=d, verbose=False).run_multi_seed(num_seeds=5)
    print(f"[dp-ext] noise={noise} done", flush=True)
open("results_v2/.dp_ext_done","w").write("ok")
