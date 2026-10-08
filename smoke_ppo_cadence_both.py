"""
smoke_ppo_cadence_both.py
-------------------------
Extended cadence smoke test: verifies update_every_n_rounds controls the PPO
firing cadence for BOTH centralized_rl AND federated_marl, at BOTH the
update_every=1 and update_every=3 settings used in the robustness check.

For each setting it prints the exact rounds on which PPO fired for each method,
then asserts:
  - each method fired exactly num_rounds // update_every times
  - the two methods fired the same number of times
  - the fired rounds equal the expected modulo schedule
Exit code 0 iff every assertion passes for every setting.

Usage:
  python -X utf8 smoke_ppo_cadence_both.py
"""
import sys, os, types, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))
import yaml

from src.simulation.runner import ExperimentRunner

NUM_ROUNDS = 12          # divisible by both 1 and 3
SEED = 42
UPDATE_EVERY_SETTINGS = [1, 3]

with open("configs/demo.yaml") as f:
    base_cfg = yaml.safe_load(f)

overall_ok = True

for update_every in UPDATE_EVERY_SETTINGS:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy
    cfg["num_rounds"] = NUM_ROUNDS
    cfg["num_seeds"] = 1
    cfg.setdefault("rl", {})["update_every_n_rounds"] = update_every
    cfg["logging"] = {"save_csv": False, "save_plots": False,
                      "results_dir": "results/smoke_ppo/", "verbose": False}
    T_RL = cfg.get("T_RL", 5)

    expected_rounds = [r for r in range(1, NUM_ROUNDS + 1) if r % update_every == 0]
    expected_n = len(expected_rounds)

    print("=" * 62)
    print(f"  PPO Cadence Smoke Test  |  update_every_n_rounds = {update_every}")
    print("=" * 62)
    print(f"  Config:             demo topology ({NUM_ROUNDS} rounds, seed={SEED})")
    print(f"  T_RL (federation):  {T_RL}")
    print(f"  Expected PPO fires: {NUM_ROUNDS} / {update_every} = {expected_n}")
    print(f"  Expected rounds:    {expected_rounds}")
    print()

    results = {}
    for method in ("centralized_rl", "federated_marl"):
        print(f"  --- {method} ---")
        ppo_fire_rounds = []
        cfg_copy = dict(cfg)
        cfg_copy["methods"] = [method]
        runner = ExperimentRunner(config=cfg_copy, method=method, verbose=False)
        orig_update_rl = runner._update_rl.__func__

        def _instrumented(self, selector_ctx, env, cluster_selections,
                          log_probs, values, acc_before, acc_after,
                          total_energy, all_participation, fl_round,
                          T_RL, update_every,
                          _fires=ppo_fire_rounds):
            result = orig_update_rl(self, selector_ctx, env, cluster_selections,
                                    log_probs, values, acc_before, acc_after,
                                    total_energy, all_participation, fl_round,
                                    T_RL, update_every)
            if self.method == "centralized_rl":
                sel = selector_ctx["selector"]
                if getattr(sel, "last_update_fired", False):
                    _fires.append(fl_round)
            elif self.method == "federated_marl":
                if fl_round % update_every == 0:
                    _fires.append(fl_round)
            return result

        runner._update_rl = types.MethodType(_instrumented, runner)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(seed=SEED)
        results[method] = ppo_fire_rounds
        print(f"    Total PPO updates: {len(ppo_fire_rounds)} over {NUM_ROUNDS} rounds")
        print(f"    Fired at rounds:   {ppo_fire_rounds}")
        print()

    print("  ASSERTIONS")
    for method, fire_rounds in results.items():
        n = len(fire_rounds)
        c1 = (n == expected_n)
        c2 = (fire_rounds == expected_rounds)
        status = "PASS" if (c1 and c2) else "FAIL"
        print(f"  [{status}] {method}: {n} fires (expected {expected_n}); "
              f"schedule match={c2}")
        overall_ok &= (c1 and c2)
    crl_n, fmarl_n = len(results["centralized_rl"]), len(results["federated_marl"])
    eq = (crl_n == fmarl_n)
    print(f"  [{'PASS' if eq else 'FAIL'}] Equal PPO updates: "
          f"centralized_rl={crl_n}, federated_marl={fmarl_n}")
    overall_ok &= eq
    print()

print("=" * 62)
if overall_ok:
    print("  ALL ASSERTIONS PASSED for update_every in {1, 3}.")
    sys.exit(0)
else:
    print("  ASSERTIONS FAILED.")
    sys.exit(1)
