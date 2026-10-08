"""Compute Step 7 final comparison table: federated_marl 5-seed stats + all baselines."""
import sys; sys.path.insert(0, '.')
import pandas as pd, glob, os, numpy as np

RESULTS = {
    'random':         'results/step6/random_seed42.csv',
    'full':           'results/step6/full_seed42.csv',
    'oort':           'results/step6/oort_seed42.csv',
    'centralized_rl': 'results/step6/centralized_rl_seed42.csv',
    'federated_marl': 'results/step7/federated_marl_seed*.csv',
}
LABELS = {
    'random':         'Random',
    'full':           'Full Participation',
    'oort':           'Oort (Heuristic)',
    'centralized_rl': 'Centralized RL',
    'federated_marl': 'Federated MARL (Ours) ★',
}

print("=" * 72)
print("  STEP 7 — FINAL COMPARISON (100 rounds, full_experiment.yaml)")
print("=" * 72)
print()

rows = []
for method, pattern in RESULTS.items():
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no files found for {method}")
        continue
    dfs = [pd.read_csv(f) for f in files]
    n_seeds = len(dfs)

    # Per-seed final-round stats
    final_accs    = [df.iloc[-1]['accuracy']    for df in dfs]
    final_losses  = [df.iloc[-1]['loss']        for df in dfs]
    final_ginis   = [df.iloc[-1]['gini']        for df in dfs]
    total_energies= [df['energy_cost'].sum()    for df in dfs]

    rows.append({
        'method':      method,
        'label':       LABELS[method],
        'n_seeds':     n_seeds,
        'n_rounds':    int(dfs[0].iloc[-1]['fl_round']),
        'acc_mean':    np.mean(final_accs),
        'acc_std':     np.std(final_accs, ddof=1) if n_seeds > 1 else 0.0,
        'loss_mean':   np.mean(final_losses),
        'energy_mean': np.mean(total_energies),
        'energy_std':  np.std(total_energies, ddof=1) if n_seeds > 1 else 0.0,
        'gini_mean':   np.mean(final_ginis),
        'gini_std':    np.std(final_ginis, ddof=1) if n_seeds > 1 else 0.0,
    })

# Sort: federated_marl first, then by accuracy
rows.sort(key=lambda r: (r['method'] != 'federated_marl', -r['acc_mean']))

# Header
hdr = f"{'Method':<30}  {'Seeds':>5}  {'Final Acc':>18}  {'Total Energy':>14}  {'Final Gini':>12}"
print(hdr)
print('-' * len(hdr))
for r in rows:
    acc_str    = f"{r['acc_mean']*100:.1f}% ± {r['acc_std']*100:.1f}%"
    energy_str = f"{r['energy_mean']:.2f} ± {r['energy_std']:.2f}" if r['energy_std'] > 0 else f"{r['energy_mean']:.2f}"
    gini_str   = f"{r['gini_mean']:.3f} ± {r['gini_std']:.3f}" if r['gini_std'] > 0 else f"{r['gini_mean']:.3f}"
    print(f"{r['label']:<30}  {r['n_seeds']:>5}  {acc_str:>18}  {energy_str:>14}  {gini_str:>12}")

print()
print("Random chance baseline (10-class): 10.0%")
print()

# Federated MARL vs baselines
marl = next(r for r in rows if r['method'] == 'federated_marl')
print("Federated MARL advantages over baselines:")
for r in rows:
    if r['method'] == 'federated_marl': continue
    acc_delta    = (marl['acc_mean'] - r['acc_mean']) * 100
    energy_delta = (r['energy_mean'] - marl['energy_mean'])  # positive = baseline uses more
    gini_delta   = (r['gini_mean']   - marl['gini_mean'])    # positive = baseline less fair
    print(f"  vs {r['label']:<25}: acc {acc_delta:+.1f}pp | "
          f"energy saved {energy_delta:+.2f} | gini {gini_delta:+.3f}")

print()
print("std = 0.0 for 1-seed baselines (expected); federated_marl has real ±std from 5 seeds")
print()

# Also show federated_marl per-seed final accuracy
marl_files = sorted(glob.glob('results/step7/federated_marl_seed*.csv'))
print("Federated MARL per-seed final accuracy:")
seed_accs = []
for f in marl_files:
    df = pd.read_csv(f)
    seed = df.iloc[0]['seed']
    acc  = df.iloc[-1]['accuracy']
    seed_accs.append(acc)
    print(f"  seed={int(seed)}: {acc:.2%}")
print(f"  Mean: {np.mean(seed_accs):.2%}  Std: {np.std(seed_accs, ddof=1):.2%}")
