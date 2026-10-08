"""
demo.py
-------
QUICK DEMO SCRIPT.  Runs the demo config (20 clients / 3 clusters / 30 rounds)
for all 5 methods with 1 seed and opens the Streamlit dashboard automatically.

Usage:
    python demo.py                    # run all methods + launch dashboard
    python demo.py --no-dashboard     # run all methods, skip dashboard launch
    python demo.py --rounds 10        # override to 10 rounds for a faster run
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

from src.simulation.runner import ExperimentRunner, load_config, generate_all_plots


def parse_args():
    p = argparse.ArgumentParser(description="FMARL demo runner")
    p.add_argument("--rounds",        type=int, default=None,
                   help="Override num_rounds (default: from demo.yaml = 30)")
    p.add_argument("--no-dashboard",  action="store_true",
                   help="Skip launching the Streamlit dashboard after the run")
    p.add_argument("--quiet", "-q",   action="store_true",
                   help="Suppress per-round output")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/demo.yaml")

    if args.rounds:
        config["num_rounds"] = args.rounds

    results_dir = config["logging"]["results_dir"]
    methods     = config.get("methods", ["random", "full", "oort",
                                          "centralized_rl", "federated_marl"])
    num_seeds   = config.get("num_seeds", 1)
    verbose     = not args.quiet

    print("\n" + "="*60)
    print("  FMARL DEMO")
    print(f"  {config['num_clients']} clients | {config['num_clusters']} clusters | "
          f"{config['num_rounds']} rounds | {num_seeds} seed")
    print("="*60)

    for method in methods:
        print(f"\n--- {method} ---")
        runner = ExperimentRunner(
            config=config,
            method=method,
            results_dir=results_dir,
            verbose=verbose,
        )
        runner.run_multi_seed(num_seeds=num_seeds)

    print("\nGenerating plots...")
    generate_all_plots(
        results_dir=results_dir,
        methods=methods,
        num_seeds=num_seeds,
        plots_dir=results_dir,
        run_name="demo",
    )
    print(f"Plots saved to: {results_dir}")

    if not args.no_dashboard:
        print("\nLaunching Streamlit dashboard...")
        print("(Press Ctrl+C to stop the dashboard)\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                        "--", "--results-dir", results_dir])


if __name__ == "__main__":
    main()
