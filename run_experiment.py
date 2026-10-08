"""
run_experiment.py
------------------
CLI ENTRYPOINT for running any experiment configuration.

USAGE EXAMPLES:
    # Run demo config, all methods, 1 seed
    python run_experiment.py --config configs/demo.yaml --method all

    # Run full experiment, specific method
    python run_experiment.py --config configs/full_experiment.yaml --method federated_marl

    # Override seeds and results directory
    python run_experiment.py --config configs/demo.yaml --method random --seeds 3 --results-dir results/my_run/

    # Run ablation sweep (sweeps the parameter defined in ablation.parameter)
    python run_experiment.py --config configs/ablations/alpha_sweep.yaml --sweep

    # Quick sanity check (5 rounds, 1 seed, all methods)
    python run_experiment.py --config configs/demo.yaml --method all --rounds 5 --seeds 1
"""

import argparse
import copy
import sys
import os

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, os.path.dirname(__file__))

from src.simulation.runner import ExperimentRunner, load_config, generate_all_plots


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Federated Multi-Agent RL experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/demo.yaml --method all
  python run_experiment.py --config configs/full_experiment.yaml --method federated_marl
  python run_experiment.py --config configs/ablations/alpha_sweep.yaml --sweep
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/demo.yaml",
        help="Path to the YAML experiment config file (default: configs/demo.yaml)",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="all",
        help=(
            "Client-selection method to run.  One of: random, full, oort, "
            "centralized_rl, federated_marl, all (default: all)"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override the number of seeds in the config.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override num_rounds in the config (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the results directory from the config.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help=(
            "Run an ablation sweep over the parameter defined in config.ablation. "
            "Saves results to separate subdirectories per sweep value."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip plot generation after the experiment.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress per-round progress output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: apply nested config override by dotted key
# ---------------------------------------------------------------------------

def set_nested(config: dict, key_path: str, value) -> None:
    """
    Sets a nested config key using dot-notation path.
    E.g., set_nested(cfg, 'federation.alpha', 0.3) sets cfg['federation']['alpha'] = 0.3
    """
    parts = key_path.split(".")
    d = config
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


# ---------------------------------------------------------------------------
# Single-method run
# ---------------------------------------------------------------------------

def run_method(config: dict, method: str, verbose: bool) -> None:
    """Runs one method for all seeds defined in the config."""
    results_dir = config["logging"]["results_dir"]
    num_seeds   = config.get("num_seeds", 1)

    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()}")
    print(f"  Config: {config.get('num_clients')} clients | "
          f"{config.get('num_clusters')} clusters | "
          f"{config.get('num_rounds')} rounds | "
          f"{num_seeds} seed(s)")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}")

    runner = ExperimentRunner(
        config=config,
        method=method,
        results_dir=results_dir,
        verbose=verbose,
    )
    runner.run_multi_seed(num_seeds=num_seeds)


# ---------------------------------------------------------------------------
# Ablation sweep run
# ---------------------------------------------------------------------------

def run_sweep(config: dict, verbose: bool) -> None:
    """
    Runs the ablation sweep defined in config.ablation.
    Iterates over values in config['ablation']['values'] and overrides
    config['ablation']['parameter'] for each run.
    """
    ablation_cfg = config.get("ablation")
    if not ablation_cfg:
        print("[ERROR] --sweep was specified but config has no 'ablation' section.")
        sys.exit(1)

    param  = ablation_cfg["parameter"]
    values = ablation_cfg["values"]
    methods = config.get("methods", ["federated_marl"])
    base_results_dir = config["logging"]["results_dir"]
    num_seeds = config.get("num_seeds", 1)

    print(f"\nAblation sweep: {param} in {values}")
    print(f"Methods: {methods}")
    print(f"Seeds: {num_seeds}")

    for val in values:
        print(f"\n{'='*60}")
        print(f"  {param} = {val}")
        print("="*60)

        # Deep-copy config and apply this sweep value
        sweep_cfg = copy.deepcopy(config)
        set_nested(sweep_cfg, param, val)

        # Results go in a subdirectory named after the parameter value
        param_short = param.split(".")[-1]
        subdir = os.path.join(base_results_dir, f"{param_short}_{val}")
        sweep_cfg["logging"]["results_dir"] = subdir

        for method in methods:
            runner = ExperimentRunner(
                config=sweep_cfg,
                method=method,
                results_dir=subdir,
                verbose=verbose,
            )
            runner.run_multi_seed(num_seeds=num_seeds)

        # Generate plots for this sweep value
        generate_all_plots(
            results_dir=subdir,
            methods=methods,
            num_seeds=num_seeds,
            plots_dir=subdir,
            run_name=f"{param_short}_{val}",
        )

    print(f"\nSweep complete. Results in: {base_results_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seeds is not None:
        config["num_seeds"] = args.seeds
    if args.rounds is not None:
        config["num_rounds"] = args.rounds
    if args.results_dir is not None:
        config.setdefault("logging", {})["results_dir"] = args.results_dir

    # Ensure results_dir is set
    config.setdefault("logging", {}).setdefault(
        "results_dir",
        f"results/{os.path.splitext(os.path.basename(args.config))[0]}/"
    )

    verbose = not args.quiet

    # Sweep mode
    if args.sweep:
        run_sweep(config, verbose)
        return

    # Determine methods to run
    if args.method == "all":
        methods = config.get("methods", ["random", "full", "oort",
                                          "centralized_rl", "federated_marl"])
    else:
        methods = [args.method]

    # Validate methods
    valid_methods = {"random", "full", "oort", "centralized_rl", "federated_marl", "mappo_marl"}
    for m in methods:
        if m not in valid_methods:
            print(f"[ERROR] Unknown method: {m!r}. Choose from: {valid_methods}")
            sys.exit(1)

    # Run each method
    for method in methods:
        run_method(config, method, verbose)

    # Generate plots
    if not args.no_plots:
        results_dir = config["logging"]["results_dir"]
        num_seeds   = config.get("num_seeds", 1)
        run_name    = os.path.splitext(os.path.basename(args.config))[0]
        print("\nGenerating comparison plots...")
        generate_all_plots(
            results_dir=results_dir,
            methods=methods,
            num_seeds=num_seeds,
            plots_dir=results_dir,
            run_name=run_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
