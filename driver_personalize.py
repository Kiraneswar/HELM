"""Phase 8 / L10: personalized federation (local head + shared backbone) vs fully-shared vs fully-local."""
import os, time, copy
from src.simulation.runner import ExperimentRunner, load_config
OUT="results_v2/personalize"
base=load_config("configs/demo.yaml"); base["num_rounds"]=50; base["num_seeds"]=5
base.setdefault("rl",{})["update_every_n_rounds"]=1
VARIANTS={
  "fully_shared":   {"alpha":0.5,"mode":"soft","personalize":False},
  "personalized":   {"alpha":0.5,"mode":"soft","personalize":True},
  "fully_local":    {"alpha":1.0,"mode":"soft","personalize":False},   # alpha=1 => no federation
}
t0=time.time()
for name,fed in VARIANTS.items():
    cfg=copy.deepcopy(base); cfg["federation"]=fed
    d=os.path.join(OUT,name); os.makedirs(d, exist_ok=True)
    cfg.setdefault("logging",{})["results_dir"]=d
    ts=time.time()
    ExperimentRunner(config=cfg, method="federated_marl", results_dir=d, verbose=False).run_multi_seed(num_seeds=5)
    print(f"[personalize] {name}: {time.time()-ts:.0f}s", flush=True)
print(f"PERSONALIZE PHASE DONE {time.time()-t0:.0f}s", flush=True); open("results_v2/.personalize_done","w").write("ok")
