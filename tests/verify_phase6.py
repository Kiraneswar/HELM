"""
tests/verify_phase6.py
-----------------------
Phase 6 Verification Script — Dashboard + CLI Entrypoints

PURPOSE:
    1. Confirms dashboard.py imports cleanly and all chart-builder functions
       work on the existing verify_phase5 results (no Streamlit runtime needed)
    2. Confirms run_experiment.py CLI argument parsing and config overrides work
    3. Runs the CLI for a single method (5 rounds) via subprocess and verifies
       the output CSV is produced

RUN FROM REPO ROOT:
    python tests/verify_phase6.py
"""

import sys
import os
import subprocess
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


# -----------------------------------------------------------------------
# Test 1: dashboard.py imports and chart builders
# -----------------------------------------------------------------------
def verify_dashboard_imports():
    separator("1. DASHBOARD IMPORTS + CHART BUILDERS")

    # Import chart builders directly (bypassing Streamlit's runtime)
    import unittest.mock as mock

    # Create a mock streamlit module so dashboard.py can be imported
    st_mock = mock.MagicMock()
    st_mock.set_page_config = mock.MagicMock()
    # Make cache_data a passthrough decorator so load_results is not wrapped
    st_mock.cache_data = lambda *a, **kw: (lambda f: f)
    sys.modules["streamlit"] = st_mock

    import importlib
    import dashboard

    # Test load_results with the phase5 results
    RESULTS_DIR = "results/verify_phase5"
    methods = ["random", "full", "oort", "centralized_rl", "federated_marl"]

    data = dashboard.load_results(RESULTS_DIR, methods)

    print(f"  Available methods in results: {list(data.keys())}")
    assert len(data) > 0, "No data loaded from results/verify_phase5"

    available = list(data.keys())

    # Test each chart builder
    fig = dashboard.build_accuracy_chart(data, available)
    assert fig is not None, "build_accuracy_chart returned None"
    print(f"  build_accuracy_chart:    {len(fig.data)} traces [OK]")

    fig2 = dashboard.build_tradeoff_scatter(data, available)
    assert fig2 is not None
    print(f"  build_tradeoff_scatter:  {len(fig2.data)} traces [OK]")

    fig3 = dashboard.build_fairness_chart(data, available)
    assert fig3 is not None
    print(f"  build_fairness_chart:    {len(fig3.data)} traces [OK]")

    fig4 = dashboard.build_loss_chart(data, available)
    assert fig4 is not None
    print(f"  build_loss_chart:        {len(fig4.data)} traces [OK]")

    fig5 = dashboard.build_energy_chart(data, available)
    assert fig5 is not None
    print(f"  build_energy_chart:      {len(fig5.data)} traces [OK]")

    # RL diagnostics chart (only if federated_marl data available)
    if "federated_marl" in data:
        fig6 = dashboard.build_policy_chart(data)
        assert fig6 is not None, "build_policy_chart returned None"
        print(f"  build_policy_chart:      {len(fig6.data)} traces [OK]")

    # Summary table
    table = dashboard.build_summary_table(data, available)
    assert len(table) == len(available), "Summary table row count mismatch"
    assert "Final Acc (%)" in table.columns
    print(f"  build_summary_table:     {len(table)} rows × {len(table.columns)} cols [OK]")

    # Hex-to-RGB utility
    rgb = dashboard._hex_to_rgb("#E91E63")
    assert rgb == "233,30,99", f"Wrong RGB: {rgb}"
    print(f"  _hex_to_rgb('#E91E63'): {rgb} [OK]")

    print(f"\n  [OK] Dashboard imports and chart builders passed.")


# -----------------------------------------------------------------------
# Test 2: run_experiment.py argument parsing
# -----------------------------------------------------------------------
def verify_cli_parsing():
    separator("2. CLI ARGUMENT PARSING")

    # Import run_experiment and test parse_args via argparse
    import run_experiment as re_mod

    # Simulate: python run_experiment.py --config configs/demo.yaml --method random
    sys.argv = ["run_experiment.py",
                "--config", "configs/demo.yaml",
                "--method", "random",
                "--seeds", "1",
                "--rounds", "5",
                "--quiet"]
    args = re_mod.parse_args()

    assert args.config == "configs/demo.yaml"
    assert args.method == "random"
    assert args.seeds == 1
    assert args.rounds == 5
    assert args.quiet is True
    print(f"  Parsed args: config={args.config}, method={args.method}, "
          f"seeds={args.seeds}, rounds={args.rounds} [OK]")

    # Test set_nested
    cfg = {"federation": {"alpha": 0.5}, "T_RL": 5}
    re_mod.set_nested(cfg, "federation.alpha", 0.3)
    assert cfg["federation"]["alpha"] == 0.3
    re_mod.set_nested(cfg, "T_RL", 10)
    assert cfg["T_RL"] == 10
    print(f"  set_nested dotted override: OK")

    print(f"\n  [OK] CLI argument parsing passed.")


# -----------------------------------------------------------------------
# Test 3: end-to-end CLI run (subprocess, 5 rounds, 1 method)
# -----------------------------------------------------------------------
def verify_cli_e2e():
    separator("3. CLI END-TO-END (subprocess, 5 rounds, random)")

    results_dir = "results/verify_phase6"

    cmd = [
        sys.executable, "run_experiment.py",
        "--config", "configs/demo.yaml",
        "--method", "random",
        "--rounds", "5",
        "--seeds", "1",
        "--results-dir", results_dir,
        "--no-plots",
        "--quiet",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr}")
        print(f"  STDOUT:\n{result.stdout}")
        assert False, f"CLI exited with code {result.returncode}"

    # Check CSV was produced
    pattern = os.path.join(results_dir, "random_seed*.csv")
    files = glob.glob(pattern)
    assert len(files) >= 1, f"No CSV found matching {pattern}"

    import pandas as pd
    df = pd.read_csv(files[0])
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
    assert df["accuracy"].iloc[-1] > 0
    print(f"  CSV produced: {files[0]}")
    print(f"  Rows: {len(df)}, Final accuracy: {df['accuracy'].iloc[-1]:.2%} [OK]")

    print(f"\n  [OK] CLI end-to-end passed.")


# -----------------------------------------------------------------------
# Test 4: demo.py imports cleanly (no crash on import)
# -----------------------------------------------------------------------
def verify_demo_import():
    separator("4. DEMO.PY IMPORT CHECK")
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo", "demo.py")
    mod  = importlib.util.module_from_spec(spec)
    # Don't execute — just check it compiles
    try:
        spec.loader.exec_module(mod)
        print(f"  demo.py imported cleanly [OK]")
    except SystemExit:
        # argparse may call sys.exit(0) on --help
        print(f"  demo.py imported cleanly (SystemExit caught) [OK]")
    except Exception as e:
        assert False, f"demo.py import failed: {e}"

    print(f"\n  [OK] demo.py import check passed.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  PHASE 6 VERIFICATION")
    print("  Federated Multi-Agent RL -- Dashboard + CLI")
    print("="*60)

    verify_dashboard_imports()
    verify_cli_parsing()
    verify_cli_e2e()
    verify_demo_import()

    print("\n" + "="*60)
    print("  PHASE 6 VERIFICATION PASSED -- all checks green")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
