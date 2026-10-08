import sys; sys.path.insert(0, '.')
import pandas as pd, os

results_dir = 'results/demo'
methods = ['random', 'full', 'oort', 'centralized_rl', 'federated_marl']
labels  = {
    'random': 'Random',
    'full': 'Full Participation',
    'oort': 'Oort',
    'centralized_rl': 'Centralized RL',
    'federated_marl': 'Federated MARL (Ours)',
}

print("=== STEP 5: Full comparison results (demo.yaml, 30 rounds, 1 seed) ===\n")

summary = []
for method in methods:
    f  = os.path.join(results_dir, method + '_seed42.csv')
    df = pd.read_csv(f)
    summary.append({
        'label':  labels[method],
        'R1':     df.iloc[0]['accuracy'],
        'R10':    df.iloc[9]['accuracy'],
        'R20':    df.iloc[19]['accuracy'],
        'RF':     df.iloc[-1]['accuracy'],
        'energy': df['energy_cost'].sum(),
        'gini':   df['gini'].iloc[-1],
    })

header = "Method                     R1      R10     R20     R30    Energy   Gini"
print(header)
print('-' * len(header))
for s in summary:
    print(
        "{:<25}  {:>6.1%}  {:>6.1%}  {:>6.1%}  {:>6.1%}  {:>8.3f}  {:>6.3f}".format(
            s['label'], s['R1'], s['R10'], s['R20'], s['RF'], s['energy'], s['gini']
        )
    )

print("\nRandom chance (10-class): 10.0%\n")

# Round-by-round for key methods
for method in ['full', 'federated_marl']:
    df = pd.read_csv(os.path.join(results_dir, method + '_seed42.csv'))
    print(labels[method] + " -- round by round:")
    print("  Rnd     Acc     Loss  Energy    Gini")
    for _, row in df.iterrows():
        print("  {:>3}  {:>6.1%}  {:>6.4f}  {:>6.3f}  {:>6.3f}".format(
            int(row['fl_round']), row['accuracy'], row['loss'],
            row['energy_cost'], row['gini']
        ))
    print()
