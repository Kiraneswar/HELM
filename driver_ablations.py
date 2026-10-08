"""
driver_ablations.py <update_every>
----------------------------------
Re-runs the alpha sweep and T_RL cadence sweep at a chosen (non-stale)
update_every_n_rounds value, overriding the old buggy update_every=5 in the
ablation config files.

alpha sweep:   federation.alpha in {0.0, 0.3, 0.5, 0.7, 1.0}
cadence sweep: T_RL           in {1, 3, 5, 10}
Both: federated_marl only, 20 clients / 3 clusters / 50 rounds / 3 seeds.

CSVs land in:
  results_bundle/ablations_ue{U}/alpha/alpha_{v}/federated_marl_seed{S}.csv
  results_bundle/ablations_ue{U}/cadence/T_RL_{v}/federated_marl_seed{S}.csv
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(__file__))
from src.simulation.runner import ExperimentRunner, load_config

U = int(sys.argv[1]) if len(sys.argv) > 1 else 1
BUNDLE = "results_bundle"
OUT = os.path.join(BUNDLE, f"ablations_ue{U}")


def run_sweep(cfg_path, param, values, subdir):
    cfg0 = load_config(cfg_path)
    cfg0["num_seeds"] = 3
    cfg0.setdefault("rl", {})["update_every_n_rounds"] = U
    for val in values:
        cfg = copy.deepcopy(cfg0)
        # set nested dotted param
        parts = param.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
        pshort = parts[-1]
        outdir = os.path.join(OUT, subdir, f"{pshort}_{val}")
        os.makedirs(outdir, exist_ok=True)
        cfg.setdefault("logging", {})["results_dir"] = outdir
        t0 = time.time()
        runner = ExperimentRunner(config=cfg, method="federated_marl",
                                  results_dir=outdir, verbose=False)
        runner.run_multi_seed(num_seeds=3)
        print(f"[ue={U}] {param}={val}: done in {time.time()-t0:.1f}s", flush=True)


t_start = time.time()
run_sweep("configs/ablations/alpha_sweep.yaml", "federation.alpha",
          [0.0, 0.3, 0.5, 0.7, 1.0], "alpha")
run_sweep("configs/ablations/cadence_sweep.yaml", "T_RL",
          [1, 3, 5, 10], "cadence")
print(f"ABLATIONS (ue={U}) COMPLETE in {time.time()-t_start:.1f}s", flush=True)
open(os.path.join(BUNDLE, f".ablations_ue{U}_done"), "w").write("ok")
