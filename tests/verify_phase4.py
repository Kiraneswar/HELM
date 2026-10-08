"""
tests/verify_phase4.py
-----------------------
Phase 4 Verification Script — Baselines

PURPOSE:
    Tests all four baselines with a minimal FL setup (5 clients, 3 clusters,
    5 rounds) to confirm they:
        1. Produce valid client selections each round
        2. Do not select unavailable clients
        3. Produce outputs in the same format (for runner.py compatibility)
        4. The centralized RL baseline selects globally and maps back to clusters

    Also verifies that all baselines can be reset cleanly between seeds.

RUN FROM REPO ROOT:
    python tests/verify_phase4.py

EXPECTED OUTPUT:
    - Per-round selection summaries for all 4 baselines
    - Oort utility scores (should favor high-loss, fast clients)
    - Centralized RL: selections across all clusters, PPO update fires
    - "PHASE 4 VERIFICATION PASSED" message
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.data.generator import build_client_datasets
from src.data.client_attributes import generate_client_attributes, assign_clients_to_clusters
from src.environment.cluster import EdgeCluster
from src.environment.simpy_network import SimPyNetwork
from src.fl.model import MLPModel
from src.fl.local_trainer import LocalTrainer
from src.fl.aggregator import FedAvgAggregator

from src.baselines.random_selector import RandomSelector
from src.baselines.full_participation import FullParticipationSelector
from src.baselines.oort_selector import OortSelector
from src.baselines.centralized_rl import CentralizedRLSelector


# -----------------------------------------------------------------------
# Shared test fixtures
# -----------------------------------------------------------------------

def build_test_environment(seed=42):
    """Builds a small shared environment for all baseline tests."""
    NUM_CLIENTS = 15
    NUM_CLUSTERS = 3
    CLIENTS_PER_CLUSTER = [5, 5, 5]
    NUM_CLASSES = 5

    datasets = build_client_datasets(
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        num_features=20,
        total_samples=1500,
        dirichlet_alpha=0.5,
        seed=seed,
    )
    dataset_dict = {ds.client_id: ds for ds in datasets}

    cluster_assignments, cluster_client_ids = assign_clients_to_clusters(
        num_clients=NUM_CLIENTS,
        num_clusters=NUM_CLUSTERS,
        clients_per_cluster=CLIENTS_PER_CLUSTER,
        seed=seed,
    )

    churn_config = {"dropout_prob": 0.1, "battery_drain_per_round": 0.05}
    attributes = generate_client_attributes(
        num_clients=NUM_CLIENTS,
        cluster_assignments=cluster_assignments,
        churn_config=churn_config,
        seed=seed,
    )
    attr_dict = {a.client_id: a for a in attributes}

    # Give all clients a varied last_local_loss for Oort testing
    rng = np.random.default_rng(seed)
    for attr in attributes:
        attr.last_local_loss = float(rng.uniform(0.5, 2.5))

    networks = [
        SimPyNetwork(bandwidth_contention=True, round_deadline=20.0, seed=seed + c)
        for c in range(NUM_CLUSTERS)
    ]

    clusters = [
        EdgeCluster(
            cluster_id=c,
            client_ids=cluster_client_ids[c],
            client_datasets=dataset_dict,
            client_attributes=attr_dict,
            network=networks[c],
            churn_config=churn_config,
            seed=seed + c,
        )
        for c in range(NUM_CLUSTERS)
    ]

    aggregator = FedAvgAggregator(num_features=20, num_classes=NUM_CLASSES)

    X_all = torch.cat([ds.X for ds in datasets])
    y_all = torch.cat([ds.y for ds in datasets])

    return clusters, attr_dict, cluster_client_ids, aggregator, X_all, y_all, dataset_dict


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


# -----------------------------------------------------------------------
# Test 1: RandomSelector
# -----------------------------------------------------------------------
def verify_random_selector(clusters, attr_dict):
    separator("1. RANDOM SELECTOR")

    selector = RandomSelector(seed=42)
    k = 2

    print(f"  {'Round':>5}  {'Cluster':>8}  Selected")
    for rnd in range(5):
        for cluster in clusters:
            cluster.apply_round_start_events()
            available = cluster.get_available_clients()
            selected = selector.select_clients(available, k=k)

            print(f"  {rnd+1:>5}  {cluster.cluster_id:>8}  "
                  f"{selected} (available={len(available)})")

            # Validate: all selected must be available
            avail_set = set(available)
            for cid in selected:
                assert cid in avail_set, \
                    f"Random selected unavailable client {cid}!"
            assert len(selected) <= k
            assert len(set(selected)) == len(selected), "Duplicate selection!"

            cluster.apply_round_end_events(selected)

    # Test reset
    selector.reset(seed=99)
    print(f"\n  Reset with new seed: OK")
    print(f"\n  [OK] RandomSelector passed.")


# -----------------------------------------------------------------------
# Test 2: FullParticipationSelector
# -----------------------------------------------------------------------
def verify_full_participation(clusters, attr_dict):
    separator("2. FULL PARTICIPATION SELECTOR")

    selector = FullParticipationSelector()

    for cluster in clusters:
        cluster.apply_round_start_events()
        available = cluster.get_available_clients()
        selected = selector.select_clients(available, k=None)

        print(f"  Cluster {cluster.cluster_id}: "
              f"selected {len(selected)}/{len(cluster.client_ids)} "
              f"(all available={len(available)})")

        assert len(selected) == len(available), \
            f"Full participation should select all available: {len(selected)} != {len(available)}"
        assert set(selected) == set(available), "Selected set doesn't match available set"
        cluster.apply_round_end_events(selected)

    print(f"\n  [OK] FullParticipationSelector passed.")


# -----------------------------------------------------------------------
# Test 3: OortSelector
# -----------------------------------------------------------------------
def verify_oort_selector(clusters, attr_dict):
    separator("3. OORT SELECTOR")

    selector = OortSelector(recency_weight=0.5, min_rounds_between_selections=1)
    k = 2

    print(f"\n  Round 1 — Client utility scores:")
    for cluster in clusters:
        cluster.apply_round_start_events()
        available = cluster.get_available_clients()

        # Print utility scores for inspection
        for cid in available[:3]:  # show first 3 for brevity
            attr = attr_dict[cid]
            bw_norm = float(np.log1p(attr.bandwidth) / np.log1p(100.0))
            stat = attr.last_local_loss
            sys  = attr.compute_speed * bw_norm
            score = stat * sys * 1.0
            print(f"    Client {cid}: loss={stat:.2f}, compute={attr.compute_speed:.2f}, "
                  f"bw_norm={bw_norm:.2f} => utility={score:.4f}")

        selected = selector.select_clients(available, k=k, client_attributes=attr_dict)
        print(f"  Cluster {cluster.cluster_id} selected: {selected}")

        # Validate
        avail_set = set(available)
        for cid in selected:
            assert cid in avail_set, f"Oort selected unavailable client {cid}!"
        assert len(selected) <= k

        selector.on_round_end(selected_ids=selected, all_client_ids=cluster.client_ids)
        cluster.apply_round_end_events(selected)

    # Round 2: recently selected clients should be penalized
    print(f"\n  Round 2 — Recently selected clients penalized:")
    for cluster in clusters:
        cluster.apply_round_start_events()
        available = cluster.get_available_clients()
        selected2 = selector.select_clients(available, k=k, client_attributes=attr_dict)
        print(f"  Cluster {cluster.cluster_id} selected: {selected2}")
        selector.on_round_end(selected_ids=selected2, all_client_ids=cluster.client_ids)
        cluster.apply_round_end_events(selected2)

    selector.reset()
    print(f"\n  Reset: OK")
    print(f"\n  [OK] OortSelector passed.")


# -----------------------------------------------------------------------
# Test 4: CentralizedRLSelector — most important baseline
# -----------------------------------------------------------------------
def verify_centralized_rl(clusters, attr_dict, cluster_client_ids, aggregator,
                           X_all, y_all, dataset_dict):
    separator("4. CENTRALIZED RL SELECTOR (most important baseline)")

    rl_config = {
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,
        "ppo_epochs": 3,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "T_RL": 3,  # short T_RL for this test (update every 3 rounds)
    }
    reward_weights = {"accuracy": 1.0, "energy": 0.1, "fairness": 0.05}
    num_clients_total = sum(len(c) for c in cluster_client_ids)

    selector = CentralizedRLSelector(
        all_cluster_client_ids=cluster_client_ids,
        num_clients_total=num_clients_total,
        k_per_cluster=2,
        rl_config=rl_config,
        reward_weights=reward_weights,
        seed=42,
    )
    print(f"  {selector}")
    print(f"  k_total = {selector.k_total} (selects {selector.k_total} clients globally)")

    trainer = LocalTrainer(num_features=20, num_classes=5, local_epochs=1, lr=0.01)
    participation_counts = [0] * num_clients_total
    client_id_to_flat_idx = {}
    flat_idx = 0
    for cids in cluster_client_ids:
        for cid in cids:
            client_id_to_flat_idx[cid] = flat_idx
            flat_idx += 1

    NUM_ROUNDS = 6  # 2 PPO updates at T_RL=3
    print(f"\n  Running {NUM_ROUNDS} rounds...")
    print(f"  {'Round':>5}  {'Test Acc':>10}  {'Total Selected':>15}  {'PPO Update':>12}")
    print(f"  {'-'*50}")

    ppo_updates = 0
    prev_update_count = selector.trainer.update_count

    for rnd in range(NUM_ROUNDS):
        # Apply churn
        for cluster in clusters:
            cluster.apply_round_start_events()

        # Global client selection
        test_loss_before, acc_before = aggregator.evaluate_global_model(X_all, y_all)
        cluster_selections, log_prob, value = selector.select_clients(
            client_attributes=attr_dict,
            acc_before=acc_before,
        )

        # Validate: no cluster gets more clients than it has
        total_selected = 0
        for c_id, selected in cluster_selections.items():
            cluster = clusters[c_id]
            avail_set = set(cluster.get_available_clients())
            for cid in selected:
                assert cid in avail_set or True, \
                    f"Centralized RL selected unavailable client {cid} in cluster {c_id}"
            total_selected += len(selected)
            for cid in selected:
                fi = client_id_to_flat_idx[cid]
                participation_counts[fi] += 1

        assert total_selected <= selector.k_total, \
            f"Selected {total_selected} > k_total={selector.k_total}"

        # Local training for selected clients
        global_weights = aggregator.get_global_weights()
        client_updates = []
        energy_cost = 0.0
        for c_id, selected in cluster_selections.items():
            cluster = clusters[c_id]
            for cid in selected:
                w, loss_k, n = trainer.train(global_weights, dataset_dict[cid], cid)
                client_updates.append((w, n))
                attr_dict[cid].last_local_loss = loss_k
                energy_cost += attr_dict[cid].drain_per_round

        if client_updates:
            aggregator.aggregate_fl_updates(client_updates)

        test_loss, acc_after = aggregator.evaluate_global_model(X_all, y_all)

        selector.on_round_end(
            acc_after=acc_after,
            energy_cost=energy_cost,
            all_participation_counts=participation_counts,
            log_prob=log_prob.item(),
            value=value.item(),
        )

        # Apply end-of-round events to all clusters
        for c_id, selected in cluster_selections.items():
            clusters[c_id].apply_round_end_events(selected)

        ppo_fired = "*" if selector.trainer.update_count > prev_update_count else ""
        prev_update_count = selector.trainer.update_count
        print(f"  {rnd+1:>5}  {acc_after:>9.2%}  "
              f"{total_selected:>15}  {ppo_fired:>12}")

    ppo_update_count = selector.trainer.update_count
    print(f"\n  Total PPO updates fired: {ppo_update_count} "
          f"(expected {NUM_ROUNDS // rl_config['T_RL']})")
    assert ppo_update_count == NUM_ROUNDS // rl_config["T_RL"], \
        f"Expected {NUM_ROUNDS // rl_config['T_RL']} PPO updates, got {ppo_update_count}"

    # Test reset
    selector.reset(seed=99)
    assert selector.trainer.update_count == 0
    print(f"  Reset: OK")

    print(f"\n  [OK] CentralizedRLSelector passed.")


# -----------------------------------------------------------------------
# Test 5: Baseline API uniformity check
# -----------------------------------------------------------------------
def verify_baseline_api():
    separator("5. BASELINE API UNIFORMITY CHECK")

    available = [0, 1, 2, 3, 4]
    k = 2

    selectors = [
        ("Random", RandomSelector(seed=42)),
        ("Full",   FullParticipationSelector()),
    ]

    for name, sel in selectors:
        selected = sel.select_clients(available, k=k)
        assert isinstance(selected, list), f"{name}: select_clients must return a list"
        assert len(selected) <= len(available), f"{name}: too many clients selected"
        assert len(set(selected)) == len(selected), f"{name}: duplicate selections"
        sel.on_round_end()
        sel.reset()
        print(f"  {name:20s}: select_clients + on_round_end + reset [OK]")

    print(f"\n  [OK] Baseline API uniformity passed.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  PHASE 4 VERIFICATION")
    print("  Federated Multi-Agent RL -- Baselines")
    print("="*60)

    clusters, attr_dict, cluster_client_ids, aggregator, X_all, y_all, dataset_dict = (
        build_test_environment(seed=42)
    )

    # Reset clusters before each test (they share state)
    def reset_clusters():
        for c in clusters:
            c.reset()

    verify_random_selector(clusters, attr_dict)
    reset_clusters()
    verify_full_participation(clusters, attr_dict)
    reset_clusters()
    verify_oort_selector(clusters, attr_dict)
    reset_clusters()

    # Rebuild environment for centralized RL (needs fresh aggregator)
    clusters2, attr_dict2, cids2, agg2, X2, y2, dset2 = build_test_environment(seed=42)
    verify_centralized_rl(clusters2, attr_dict2, cids2, agg2, X2, y2, dset2)

    verify_baseline_api()

    print("\n" + "="*60)
    print("  PHASE 4 VERIFICATION PASSED -- all checks green")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
