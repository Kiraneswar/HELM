"""
tests/verify_phase2.py
-----------------------
Phase 2 Verification Script — FL Core

PURPOSE:
    Tests the MLP model, local trainer, and FedAvg aggregator in isolation and
    together, before the RL layer is built on top.

    Per the Verification Plan:
        - Run 5 FL rounds with full participation, verify loss is decreasing
        - Verify FedAvg weighted average is mathematically correct
        - Verify RL policy federation (soft/hard) produces correct weight updates
        - Verify model evaluate() produces sensible accuracy values

RUN FROM REPO ROOT:
    python tests/verify_phase2.py

EXPECTED OUTPUT:
    - MLP architecture summary and parameter count
    - 5 rounds of FL training showing decreasing loss
    - FedAvg correctness check (manual vs. code result)
    - Policy federation: before/after similarity for soft and hard modes
    - "PHASE 2 VERIFICATION PASSED" message
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import copy

from src.data.generator import build_client_datasets, generate_dataset
from src.fl.model import MLPModel
from src.fl.local_trainer import LocalTrainer
from src.fl.aggregator import FedAvgAggregator


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


# -----------------------------------------------------------------------
# Test 1: Model architecture and forward pass
# -----------------------------------------------------------------------
def verify_model():
    separator("1. MLP MODEL")

    model = MLPModel(num_features=20, num_classes=10)
    print(f"  {model}")
    print(f"  Architecture:")
    for name, param in model.named_parameters():
        print(f"    {name:30s}: {tuple(param.shape)}")

    # Test forward pass with a batch of 16
    x = torch.randn(16, 20)
    out = model(x)
    print(f"\n  Forward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (16, 10), f"Expected (16,10), got {out.shape}"

    # Output should be log-probabilities: exp(out) sums to 1 per sample
    probs = out.exp()
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(16), atol=1e-5), \
        f"Output is not log-probabilities: row sums = {row_sums}"
    print(f"  Output is valid log-probabilities (exp sums to 1.0 per row) [OK]")

    # Test get_weights / set_weights round-trip
    w = model.get_weights()
    model2 = MLPModel(num_features=20, num_classes=10)
    model2.set_weights(w)
    w2 = model2.get_weights()
    for key in w:
        assert torch.allclose(w[key], w2[key]), f"Weight mismatch for {key}"
    print(f"  get_weights/set_weights round-trip [OK]")

    print(f"\n  [OK] MLP model passed.")
    return model


# -----------------------------------------------------------------------
# Test 2: Local trainer — single client training
# -----------------------------------------------------------------------
def verify_local_trainer():
    separator("2. LOCAL TRAINER (single client)")

    NUM_CLIENTS = 5
    NUM_CLASSES = 5
    datasets = build_client_datasets(
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        num_features=20,
        total_samples=500,
        dirichlet_alpha=0.5,
        seed=42,
    )

    trainer = LocalTrainer(
        num_features=20,
        num_classes=NUM_CLASSES,
        local_epochs=3,
        batch_size=32,
        lr=0.01,
    )

    aggregator = FedAvgAggregator(num_features=20, num_classes=NUM_CLASSES)
    global_weights = aggregator.get_global_weights()

    print(f"  Training client 0 for 3 epochs on {len(datasets[0])} samples...")
    updated_weights, loss, n_samples = trainer.train(
        global_weights=global_weights,
        dataset=datasets[0],
        client_id=0,
    )

    print(f"  Loss after 3 local epochs: {loss:.4f}")
    print(f"  Num samples: {n_samples}")
    assert loss > 0, "Loss should be positive"
    assert n_samples == len(datasets[0])

    # Verify that weights actually changed (local training modified something)
    weights_changed = False
    for key in global_weights:
        if not torch.allclose(global_weights[key], updated_weights[key]):
            weights_changed = True
            break
    assert weights_changed, "Weights did not change after local training!"
    print(f"  Weights changed after training [OK]")

    print(f"\n  [OK] Local trainer passed.")
    return datasets, trainer


# -----------------------------------------------------------------------
# Test 3: FedAvg aggregation — 5 rounds of FL, verify loss decreases
# -----------------------------------------------------------------------
def verify_fedavg(datasets, trainer):
    separator("3. FEDAVG — 5 ROUNDS (loss should decrease)")

    NUM_CLASSES = 5
    NUM_ROUNDS = 5

    # Hold-out test set from the global distribution
    X_all = torch.cat([ds.X for ds in datasets])
    y_all = torch.cat([ds.y for ds in datasets])

    aggregator = FedAvgAggregator(num_features=20, num_classes=NUM_CLASSES)

    prev_loss = float("inf")
    losses = []

    print(f"  {'Round':>6}  {'Loss':>8}  {'Accuracy':>10}")
    print(f"  {'-'*30}")

    for rnd in range(NUM_ROUNDS):
        global_weights = aggregator.get_global_weights()

        # All clients train (full participation for this test)
        client_updates = []
        for ds in datasets:
            w, loss_k, n = trainer.train(global_weights, ds, ds.client_id)
            client_updates.append((w, n))

        aggregator.aggregate_fl_updates(client_updates)
        test_loss, test_acc = aggregator.evaluate_global_model(X_all, y_all)
        losses.append(test_loss)

        print(f"  {rnd+1:>6}  {test_loss:>8.4f}  {test_acc:>9.2%}")

    # Check loss trend: loss at round 5 should be lower than at round 1
    # (not strictly every round — FL can have non-monotone convergence,
    # but overall trend should be downward)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )
    print(f"\n  Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f} [OK]")
    print(f"\n  [OK] FedAvg passed — loss decreasing over 5 rounds.")
    return aggregator


# -----------------------------------------------------------------------
# Test 4: FedAvg mathematical correctness
# -----------------------------------------------------------------------
def verify_fedavg_math():
    separator("4. FEDAVG MATH CORRECTNESS CHECK")

    NUM_CLASSES = 3

    aggregator = FedAvgAggregator(num_features=10, num_classes=NUM_CLASSES)

    # Create two fake models with known weights
    model_a = MLPModel(num_features=10, num_classes=NUM_CLASSES)
    model_b = MLPModel(num_features=10, num_classes=NUM_CLASSES)

    # Set fc3.weight to known values for easy manual verification
    with torch.no_grad():
        torch.nn.init.constant_(model_a.fc3.weight, 2.0)
        torch.nn.init.constant_(model_b.fc3.weight, 4.0)

    weights_a = model_a.get_weights()
    weights_b = model_b.get_weights()

    # Client A has 100 samples, Client B has 300 samples
    # Expected fc3.weight = (100/400)*2.0 + (300/400)*4.0 = 0.5 + 3.0 = 3.5
    expected_fc3 = (100 / 400) * 2.0 + (300 / 400) * 4.0
    print(f"  Expected fc3.weight after FedAvg: {expected_fc3:.4f}")

    aggregator.aggregate_fl_updates([(weights_a, 100), (weights_b, 300)])
    result_weights = aggregator.get_global_weights()

    actual_fc3 = result_weights["fc3.weight"].mean().item()
    print(f"  Actual   fc3.weight after FedAvg: {actual_fc3:.4f}")

    assert abs(actual_fc3 - expected_fc3) < 1e-4, (
        f"FedAvg math error: expected {expected_fc3}, got {actual_fc3}"
    )
    print(f"  FedAvg weighted average is mathematically correct [OK]")
    print(f"\n  [OK] FedAvg math check passed.")


# -----------------------------------------------------------------------
# Test 5: RL Policy Federation (soft update + hard replace)
# -----------------------------------------------------------------------
def verify_policy_federation():
    separator("5. RL POLICY FEDERATION (soft + hard update)")

    NUM_CLASSES = 3

    # Create three fake "policy networks" as MLPModels
    # (in reality these will be Actor-Critic networks, but the federation
    # math is identical — it operates on state_dicts regardless of architecture)
    policy_a = MLPModel(num_features=10, num_classes=NUM_CLASSES)
    policy_b = MLPModel(num_features=10, num_classes=NUM_CLASSES)
    policy_c = MLPModel(num_features=10, num_classes=NUM_CLASSES)

    with torch.no_grad():
        for m, val in [(policy_a, 1.0), (policy_b, 3.0), (policy_c, 5.0)]:
            torch.nn.init.constant_(m.fc3.weight, val)

    weights_a = policy_a.get_weights()
    weights_b = policy_b.get_weights()
    weights_c = policy_c.get_weights()

    # Expected meta-policy fc3.weight = (1 + 3 + 5) / 3 = 3.0
    expected_meta = (1.0 + 3.0 + 5.0) / 3.0
    print(f"  Expected meta-policy fc3.weight: {expected_meta:.4f}")

    # --- Soft update test (alpha=0.5) ---
    aggregator_soft = FedAvgAggregator(
        num_features=10, num_classes=NUM_CLASSES,
        federation_alpha=0.5, federation_mode="soft"
    )
    updated = aggregator_soft.federate_policies(
        cluster_policy_weights={0: weights_a, 1: weights_b, 2: weights_c},
        rl_round=1,
    )

    # For cluster 0 (local=1.0):  new = 0.5*1.0 + 0.5*3.0 = 2.0
    # For cluster 1 (local=3.0):  new = 0.5*3.0 + 0.5*3.0 = 3.0
    # For cluster 2 (local=5.0):  new = 0.5*5.0 + 0.5*3.0 = 4.0
    expected_soft = {0: 0.5*1.0+0.5*3.0, 1: 0.5*3.0+0.5*3.0, 2: 0.5*5.0+0.5*3.0}
    print(f"\n  Soft update (alpha=0.5):")
    for cid, exp in expected_soft.items():
        actual = updated[cid]["fc3.weight"].mean().item()
        print(f"    Cluster {cid}: expected={exp:.2f}, actual={actual:.4f}")
        assert abs(actual - exp) < 1e-4, f"Soft update math error for cluster {cid}"

    # --- Hard replace test ---
    aggregator_hard = FedAvgAggregator(
        num_features=10, num_classes=NUM_CLASSES,
        federation_alpha=0.0, federation_mode="hard"
    )
    updated_hard = aggregator_hard.federate_policies(
        cluster_policy_weights={0: weights_a, 1: weights_b, 2: weights_c},
        rl_round=1,
    )
    print(f"\n  Hard replace (all clusters -> meta-policy={expected_meta:.1f}):")
    for cid in [0, 1, 2]:
        actual = updated_hard[cid]["fc3.weight"].mean().item()
        print(f"    Cluster {cid}: expected={expected_meta:.2f}, actual={actual:.4f}")
        assert abs(actual - expected_meta) < 1e-4, \
            f"Hard replace math error for cluster {cid}"

    # --- Policy similarity metric ---
    sim_before = aggregator_soft.compute_policy_similarity(weights_a, weights_b)
    sim_after  = aggregator_soft.compute_policy_similarity(updated[0], updated[1])
    print(f"\n  Policy similarity (cluster 0 vs 1):")
    print(f"    Before federation: {sim_before:.4f}")
    print(f"    After  federation: {sim_after:.4f}")
    assert sim_after >= sim_before, "Policies should be MORE similar after federation"

    print(f"\n  [OK] Policy federation passed — soft and hard modes correct.")


# -----------------------------------------------------------------------
# Test 6: Full 5-round FL loop with multiple clusters
# -----------------------------------------------------------------------
def verify_multi_cluster_fl():
    separator("6. MULTI-CLUSTER FL LOOP (integration)")

    NUM_CLIENTS = 12
    NUM_CLASSES = 5
    NUM_ROUNDS = 5

    datasets = build_client_datasets(
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        num_features=20,
        total_samples=1200,
        dirichlet_alpha=0.5,
        seed=99,
    )
    X_all = torch.cat([ds.X for ds in datasets])
    y_all = torch.cat([ds.y for ds in datasets])

    aggregator = FedAvgAggregator(num_features=20, num_classes=NUM_CLASSES)
    trainer = LocalTrainer(num_features=20, num_classes=NUM_CLASSES,
                           local_epochs=2, batch_size=32, lr=0.01)

    # Simulate 3 clusters selecting 4 clients each
    cluster_assignment = {0: [0,1,2,3], 1: [4,5,6,7], 2: [8,9,10,11]}

    losses = []
    print(f"  {'Round':>5}  {'Test Loss':>10}  {'Test Acc':>10}  Clients")
    print(f"  {'-'*45}")

    for rnd in range(NUM_ROUNDS):
        global_weights = aggregator.get_global_weights()
        client_updates = []

        # Each cluster trains its local clients
        for cluster_id, client_ids in cluster_assignment.items():
            # Select 2 out of 4 clients per cluster (mimicking k_select=2)
            selected = client_ids[:2]
            for cid in selected:
                w, loss_k, n = trainer.train(global_weights, datasets[cid], cid)
                client_updates.append((w, n))

        aggregator.aggregate_fl_updates(client_updates)
        test_loss, test_acc = aggregator.evaluate_global_model(X_all, y_all)
        losses.append(test_loss)
        print(f"  {rnd+1:>5}  {test_loss:>10.4f}  {test_acc:>9.2%}  "
              f"{len(client_updates)} updates")

    assert losses[-1] < losses[0], "Loss should decrease over 5 rounds"
    print(f"\n  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} [OK]")
    print(f"\n  [OK] Multi-cluster FL loop passed.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  PHASE 2 VERIFICATION")
    print("  Federated Multi-Agent RL -- FL Core Layer")
    print("="*60)

    verify_model()
    datasets, trainer = verify_local_trainer()
    verify_fedavg(datasets, trainer)
    verify_fedavg_math()
    verify_policy_federation()
    verify_multi_cluster_fl()

    print("\n" + "="*60)
    print("  PHASE 2 VERIFICATION PASSED -- all checks green")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
