"""Quick sanity check for fixed data pipeline."""
import sys; sys.path.insert(0, '.')
import torch
from src.simulation.runner import build_environment, load_config
from src.fl.local_trainer import LocalTrainer

config = load_config('configs/demo.yaml')
config['num_clients'] = 10
config['num_clusters'] = 2
config['clients_per_cluster'] = [5, 5]
config['data']['num_samples_per_client'] = 100

env = build_environment(config, seed=42)
X_test, y_test = env['X_test'], env['y_test']
print(f"Test set shape: {X_test.shape}, labels: {sorted(y_test.unique().tolist())}")

trainer = LocalTrainer(num_features=20, num_classes=10, local_epochs=5, hidden1=128, hidden2=64)
global_w = env['aggregator'].get_global_weights()

print(f"{'Round':>5}  {'Test Acc':>8}  {'Test Loss':>10}")
print("  " + "-"*30)
for r in range(15):
    all_updates = []
    for cid in range(10):
        w, loss, n = trainer.train(global_w, env['datasets'][cid], cid)
        all_updates.append((w, n))
    env['aggregator'].aggregate_fl_updates(all_updates)
    global_w = env['aggregator'].get_global_weights()
    test_loss, acc = env['aggregator'].evaluate_global_model(X_test, y_test)
    print(f"{r+1:>5}  {acc:>8.2%}  {test_loss:>10.4f}")

print()
print("SUCCESS if accuracy steadily rises well above 10%")
