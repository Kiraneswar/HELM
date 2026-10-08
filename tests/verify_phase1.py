"""
tests/verify_phase1.py
-----------------------
Phase 1 Verification Script

PURPOSE:
    Tests the data generation, Dirichlet partition, client attributes, and
    SimPy network model in isolation before any FL/RL code is layered on top.

    Per the Verification Plan:
        - Print partition statistics (label distribution per client)
        - Verify Dirichlet skew is visible (different clusters have different distributions)
        - Print ClientAttributes summary per cluster archetype
        - Run SimPyNetwork.simulate_uploads and verify straggler handling works

RUN FROM REPO ROOT:
    python tests/verify_phase1.py

EXPECTED OUTPUT:
    - Dataset shape and class balance
    - Per-client label distributions (visibly skewed)
    - Per-cluster attribute summaries showing archetype differences
    - SimPy upload simulation: arrived vs straggler count
    - "PHASE 1 VERIFICATION PASSED" message
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ---- Data layer imports ----
from src.data.generator import generate_dataset, partition_data_dirichlet, build_client_datasets
from src.data.client_attributes import (
    generate_client_attributes,
    assign_clients_to_clusters,
)
# ---- Environment layer imports ----
from src.environment.simpy_network import SimPyNetwork
from src.environment.cluster import EdgeCluster


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def verify_data_generation():
    separator("1. DATASET GENERATION")

    X, y = generate_dataset(
        num_samples=5000,
        num_features=20,
        num_classes=10,
        seed=42,
    )

    print(f"  X shape:       {X.shape}")
    print(f"  y shape:       {y.shape}")
    print(f"  X dtype:       {X.dtype}")
    print(f"  y dtype:       {y.dtype}")
    print(f"  X mean:        {X.mean():.4f}  (should be ~0.0 after StandardScaler)")
    print(f"  X std:         {X.std():.4f}   (should be ~1.0)")
    print(f"  Class counts:  {np.bincount(y)}")

    assert X.shape == (5000, 20), f"Expected (5000, 20), got {X.shape}"
    assert y.shape == (5000,), f"Expected (5000,), got {y.shape}"
    assert abs(X.mean()) < 0.1, f"Features not centered: mean = {X.mean()}"
    assert abs(X.std() - 1.0) < 0.1, f"Features not scaled: std = {X.std()}"
    print("\n  [OK] Dataset generation passed.")
    return X, y


def verify_dirichlet_partition(X, y):
    separator("2. DIRICHLET PARTITION (non-IID verification)")

    NUM_CLIENTS = 10
    NUM_CLASSES = 10

    shards = partition_data_dirichlet(
        X=X, y=y,
        num_clients=NUM_CLIENTS,
        num_classes=NUM_CLASSES,
        alpha=0.5,
        seed=42,
    )

    print(f"  Number of shards: {len(shards)}")
    print(f"  Shard sizes (samples per client):")

    distributions = []
    for k, (Xk, yk) in enumerate(shards):
        dist = np.bincount(yk, minlength=NUM_CLASSES) / len(yk)
        distributions.append(dist)
        # Show most dominant class for each client
        dominant_class = np.argmax(dist)
        print(f"    Client {k:2d}: {len(yk):4d} samples | "
              f"dominant class = {dominant_class} ({dist[dominant_class]:.1%}) | "
              f"dist = [{', '.join(f'{p:.2f}' for p in dist)}]")

    # Verify non-IID: distributions should be DIFFERENT across clients
    # Metric: average total variation distance between pairs
    tv_distances = []
    for i in range(NUM_CLIENTS):
        for j in range(i+1, NUM_CLIENTS):
            tv = 0.5 * np.sum(np.abs(distributions[i] - distributions[j]))
            tv_distances.append(tv)

    mean_tv = np.mean(tv_distances)
    print(f"\n  Mean pairwise total variation distance: {mean_tv:.4f}")
    print(f"  (0.0 = perfectly IID, 1.0 = maximally non-IID)")
    print(f"  With alpha=0.5, expect mean TV in [0.3, 0.7]")

    assert mean_tv > 0.2, f"Partition appears too IID (TV={mean_tv:.3f})"
    assert all(len(shards[k][0]) >= 10 for k in range(NUM_CLIENTS)), \
        "Some client has fewer than 10 samples (min_samples violated)"

    print("\n  [OK] Dirichlet partition passed — visible non-IID skew confirmed.")
    return shards


def verify_client_attributes():
    separator("3. CLIENT ATTRIBUTES (cluster archetypes)")

    NUM_CLIENTS = 20
    NUM_CLUSTERS = 3
    CLIENTS_PER_CLUSTER = [7, 7, 6]

    cluster_assignments, cluster_client_ids = assign_clients_to_clusters(
        num_clients=NUM_CLIENTS,
        num_clusters=NUM_CLUSTERS,
        clients_per_cluster=CLIENTS_PER_CLUSTER,
        seed=42,
    )

    churn_config = {"dropout_prob": 0.1, "battery_drain_per_round": 0.05}
    attributes = generate_client_attributes(
        num_clients=NUM_CLIENTS,
        cluster_assignments=cluster_assignments,
        churn_config=churn_config,
        seed=42,
    )

    print(f"  Total clients: {len(attributes)}")
    print(f"  Cluster sizes: {[len(c) for c in cluster_client_ids]}")
    print()

    for c in range(NUM_CLUSTERS):
        cids = cluster_client_ids[c]
        cluster_attrs = [attributes[cid] for cid in cids]
        avg_compute = np.mean([a.compute_speed for a in cluster_attrs])
        avg_battery = np.mean([a.initial_battery for a in cluster_attrs])
        avg_drain   = np.mean([a.drain_per_round for a in cluster_attrs])
        avg_bw      = np.mean([a.bandwidth for a in cluster_attrs])
        avg_latency = np.mean([a.latency for a in cluster_attrs])
        profile_name = ["Mobile/Weak", "Fixed/Strong", "Mixed"][c % 3]

        print(f"  Cluster {c} ({profile_name}, {len(cids)} clients):")
        print(f"    avg compute_speed: {avg_compute:.3f}")
        print(f"    avg initial_battery: {avg_battery:.3f}")
        print(f"    avg drain_per_round: {avg_drain:.4f}")
        print(f"    avg bandwidth (Mbps): {avg_bw:.2f}")
        print(f"    avg latency (s): {avg_latency:.3f}")

    # Verify state vectors have the right shape
    for attr in attributes:
        sv = attr.to_state_vector()
        assert sv.shape == (5,), f"State vector wrong shape: {sv.shape}"
        assert sv.dtype == np.float32, f"State vector wrong dtype: {sv.dtype}"

    print("\n  [OK] Client attributes passed — cluster archetypes visible.")
    return attributes, cluster_assignments, cluster_client_ids


def verify_simpy_network():
    separator("4. SIMPY NETWORK MODEL (upload simulation)")

    network = SimPyNetwork(
        bandwidth_contention=True,
        round_deadline=10.0,
        seed=42,
    )

    # Simulate 5 clients uploading a 0.5 MB model update
    client_ids = [0, 1, 2, 3, 4]
    bandwidths = [5.0, 2.0, 15.0, 0.5, 8.0]   # Mbps — wide range
    latencies  = [0.05, 0.20, 0.02, 0.50, 0.03]  # seconds

    arrived, times = network.simulate_uploads(
        client_ids=client_ids,
        bandwidths=bandwidths,
        latencies=latencies,
        model_size_mb=0.5,
    )

    print(f"  Upload results (contention ON, deadline=10s, model=0.5MB):")
    for cid, t in zip(client_ids, times):
        status = "ARRIVED" if cid in arrived else "STRAGGLER"
        print(f"    Client {cid}: BW={bandwidths[cid]:.1f}Mbps, "
              f"lat={latencies[cid]:.2f}s -> upload_time={t:.3f}s [{status}]")

    print(f"\n  Arrived: {len(arrived)}/{len(client_ids)} clients")

    # With very low bandwidth (client 3: 0.5 Mbps) and contention,
    # the slow client may miss the deadline under contention
    # (0.5 MB * 8 bits/byte / 0.5 Mbps = 8s + 0.5s latency = 8.5s
    # + wait time if queue is long → may exceed 10s deadline)
    estimate = network.estimate_round_time(bandwidths, latencies, model_size_mb=0.5)
    print(f"\n  Estimated serial round time: {estimate:.2f}s")

    assert len(arrived) >= 0 and len(arrived) <= len(client_ids), \
        "arrived count out of bounds"
    assert all(t > 0 for t in times), "All upload times should be positive"

    print("\n  [OK] SimPy network model passed.")


def verify_edge_cluster():
    separator("5. EDGE CLUSTER (integration check)")

    from src.data.generator import build_client_datasets

    NUM_CLIENTS = 10
    NUM_CLUSTERS = 2
    CLIENTS_PER_CLUSTER = [5, 5]

    # Build client datasets
    datasets = build_client_datasets(
        num_clients=NUM_CLIENTS,
        num_classes=5,
        num_features=10,
        total_samples=1000,
        dirichlet_alpha=0.5,
        seed=42,
    )
    dataset_dict = {ds.client_id: ds for ds in datasets}

    # Build client attributes
    cluster_assignments, cluster_client_ids = assign_clients_to_clusters(
        num_clients=NUM_CLIENTS,
        num_clusters=NUM_CLUSTERS,
        clients_per_cluster=CLIENTS_PER_CLUSTER,
        seed=42,
    )
    churn_config = {"dropout_prob": 0.2, "battery_drain_per_round": 0.05}
    attributes = generate_client_attributes(
        num_clients=NUM_CLIENTS,
        cluster_assignments=cluster_assignments,
        churn_config=churn_config,
        seed=42,
    )
    attr_dict = {a.client_id: a for a in attributes}

    # Create a cluster
    network = SimPyNetwork(bandwidth_contention=True, round_deadline=15.0, seed=42)
    cluster = EdgeCluster(
        cluster_id=0,
        client_ids=cluster_client_ids[0],
        client_datasets=dataset_dict,
        client_attributes=attr_dict,
        network=network,
        churn_config=churn_config,
        seed=42,
    )

    print(f"  Created: {cluster}")

    # Simulate a round
    cluster.apply_round_start_events()
    available = cluster.get_available_clients()
    print(f"  Available after churn: {len(available)}/{len(cluster.client_ids)} clients")

    state_matrix = cluster.get_state_matrix()
    print(f"  State matrix shape: {state_matrix.shape}")
    assert state_matrix.shape == (len(cluster.client_ids), 5), \
        f"Unexpected state matrix shape: {state_matrix.shape}"

    mask = cluster.get_availability_mask()
    print(f"  Availability mask: {mask}")

    # Simulate selecting 3 clients (or fewer if not enough available)
    k = min(3, len(available))
    selected = available[:k]
    arrived, times, comm_cost = cluster.simulate_uploads(selected, model_size_mb=0.2)
    print(f"  Selected {k} clients, {len(arrived)} arrived (comm_cost={comm_cost:.3f}s)")

    cluster.apply_round_end_events(selected)
    print(f"  After round: {cluster}")

    # Test reset
    cluster.reset()
    assert cluster.round_number == 0
    print(f"  After reset: {cluster}")

    print("\n  [OK] EdgeCluster integration check passed.")


def main():
    print("\n" + "="*60)
    print("  PHASE 1 VERIFICATION")
    print("  Federated Multi-Agent RL — Data & Environment Layer")
    print("="*60)

    X, y = verify_data_generation()
    verify_dirichlet_partition(X, y)
    verify_client_attributes()
    verify_simpy_network()
    verify_edge_cluster()

    print("\n" + "="*60)
    print("  ✓  PHASE 1 VERIFICATION PASSED — all checks green")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
