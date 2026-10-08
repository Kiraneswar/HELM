"""
src/environment/cluster.py
---------------------------
WHAT THIS FILE DOES:
    Defines the EdgeCluster class — the central object representing one edge
    datacenter (or edge server) in the federated learning system.  Each cluster:
        1. Holds a subset of clients (with their data + system attributes)
        2. Owns a local RL policy that selects which clients train each round
        3. Manages per-round churn, battery drain, and staleness bookkeeping
        4. Calls the SimPy network model to simulate upload delays
        5. Returns arrived model updates to the aggregator (aggregator.py)

WHERE IT SITS IN THE ARCHITECTURE:
    EdgeCluster is the primary "actor" in the simulation loop.  The runner
    (simulation/runner.py) creates one EdgeCluster per cluster at startup, then
    calls cluster.run_round(...) each FL round.  The cluster's RL policy is
    periodically federated by the cloud aggregator (fl/aggregator.py).

    Upstream (receives from):
        - data/generator.py          → ClientDataset objects
        - data/client_attributes.py  → ClientAttributes objects
        - environment/simpy_network.py → SimPyNetwork for upload simulation

    Downstream (sends to):
        - fl/aggregator.py           → (weights, num_samples) tuples
        - rl/policy.py               → state vectors for RL training
        - simulation/metrics.py      → energy, participation, timing data

DEPENDENCIES:
    - data.generator.ClientDataset
    - data.client_attributes.ClientAttributes
    - environment.simpy_network.SimPyNetwork
    - numpy: for state vector construction

KEY DESIGN DECISION:
    The EdgeCluster does NOT run FL training directly — it delegates training to
    fl/local_trainer.py (called from runner.py).  This keeps the cluster class
    focused on client management and coordination, while training logic lives in
    the FL layer.  The alternative (putting training inside cluster) was NOT chosen
    because it would create a circular dependency with fl/ modules and make the
    cluster class a "god object" that's harder to test and extend.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from src.data.generator import ClientDataset
from src.data.client_attributes import ClientAttributes
from src.environment.simpy_network import SimPyNetwork


class EdgeCluster:
    """
    Represents one edge cluster in the hierarchical FL system.

    An EdgeCluster groups clients into a managed unit, exposes their current
    state for RL policy queries, handles churn/battery updates, and interfaces
    with the SimPy network model for communication simulation.

    Parameters
    ----------
    cluster_id : int
        Unique identifier for this cluster (0 … num_clusters-1).
    client_ids : List[int]
        List of client IDs assigned to this cluster.
    client_datasets : Dict[int, ClientDataset]
        Maps client_id → ClientDataset (the client's local data).
    client_attributes : Dict[int, ClientAttributes]
        Maps client_id → ClientAttributes (compute, battery, bandwidth, etc.)
    network : SimPyNetwork
        The SimPy network model used to simulate upload delays for this cluster.
    churn_config : dict
        Churn configuration dict from YAML config['churn'].
    seed : int
        Per-cluster RNG seed for reproducible churn events.
    """

    def __init__(
        self,
        cluster_id: int,
        client_ids: List[int],
        client_datasets: Dict[int, ClientDataset],
        client_attributes: Dict[int, ClientAttributes],
        network: SimPyNetwork,
        churn_config: dict,
        seed: int = 42,
    ):
        self.cluster_id = cluster_id
        self.client_ids = list(client_ids)
        self.client_datasets = client_datasets
        self.client_attributes = client_attributes
        self.network = network
        self.churn_config = churn_config

        # Per-cluster RNG: offset by cluster_id so clusters have different churn patterns
        self.rng = np.random.default_rng(seed + cluster_id * 1000)

        # Round counter: tracks how many FL rounds this cluster has participated in
        self.round_number: int = 0

        # History for dashboard and RL reward computation
        self.round_history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # State access: available clients, state vectors, etc.
    # -----------------------------------------------------------------------

    def get_available_clients(self) -> List[int]:
        """
        Returns the list of client IDs that are currently reachable (not churned
        and with sufficient battery) and therefore eligible for selection this round.

        Called by:
            - RL policy (to know which clients it can select from)
            - Baselines (same)
            - runner.py (for full-participation baseline)

        Returns
        -------
        List[int]: Available client IDs (subset of self.client_ids).
        """
        return [
            cid for cid in self.client_ids
            if self.client_attributes[cid].is_available
        ]

    def get_state_matrix(self) -> np.ndarray:
        """
        Returns the current state matrix for all clients in this cluster.

        The RL policy (rl/policy.py) takes this matrix as its observation.
        Each row is one client's 5-dimensional system state vector.

        MATH — State vector for client i:
            s_i = [compute_speed_i, battery_i, log(bandwidth_i)/norm,
                   staleness_i / 20, last_local_loss_i]

        Unavailable clients are included with their state set to zeros, so the
        policy can always receive a fixed-size input regardless of churn.
        The policy learns to avoid selecting zero-state clients (they represent
        unavailable/churned clients).

        Returns
        -------
        np.ndarray, shape (num_clients_in_cluster, 5)
            Row i corresponds to client self.client_ids[i].
        """
        rows = []
        for cid in self.client_ids:
            attr = self.client_attributes[cid]
            if attr.is_available:
                rows.append(attr.to_state_vector())
            else:
                # Zero vector signals "unavailable" to the RL policy
                rows.append(np.zeros(5, dtype=np.float32))
        return np.stack(rows, axis=0)  # shape: (N_cluster, 5)

    def get_availability_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (num_clients_in_cluster,) where True
        means the client is currently available for selection.

        Used by the RL policy to mask out unavailable clients before sampling,
        ensuring the policy never selects a churned client.

        Returns
        -------
        np.ndarray[bool], shape (num_clients_in_cluster,)
        """
        return np.array(
            [self.client_attributes[cid].is_available for cid in self.client_ids],
            dtype=bool,
        )

    # -----------------------------------------------------------------------
    # Round management: churn, battery, staleness
    # -----------------------------------------------------------------------

    def apply_round_start_events(self) -> None:
        """
        Updates client availability and battery at the START of each FL round.

        Call this before asking the RL policy to select clients, so the policy
        sees the updated availability mask.

        What this does:
            1. Apply random churn (Bernoulli dropout with configured probability)
            2. Apply idle battery drain to all clients (selected clients will
               get additional active drain in apply_round_end_events)

        Note on ordering: churn is applied first (a churned client doesn't consume
        active battery), then idle drain applies to all remaining clients.
        """
        dropout_prob = self.churn_config.get("dropout_prob", 0.0)

        for cid in self.client_ids:
            attr = self.client_attributes[cid]
            # Step 1: Random churn check (also checks battery threshold internally)
            attr.apply_churn(dropout_prob=dropout_prob, rng=self.rng)

    def apply_round_end_events(self, selected_ids: List[int]) -> None:
        """
        Updates client state at the END of each FL round.

        Call this after the round's communication phase completes.

        What this does:
            1. Active battery drain for selected clients
            2. Idle battery drain for non-selected clients
            3. Staleness counter update for all clients
            4. Increment round counter

        Parameters
        ----------
        selected_ids : List[int]
            The client IDs that were selected (and trained) this round.
        """
        selected_set = set(selected_ids)
        for cid in self.client_ids:
            attr = self.client_attributes[cid]
            was_selected = (cid in selected_set)
            attr.apply_battery_drain(was_selected=was_selected)
            attr.update_staleness(was_selected=was_selected)

        self.round_number += 1

    # -----------------------------------------------------------------------
    # Communication simulation: model uploads via SimPy
    # -----------------------------------------------------------------------

    def simulate_uploads(
        self,
        selected_ids: List[int],
        model_size_mb: float = 1.0,
    ) -> Tuple[List[int], List[float], float]:
        """
        Simulates the upload phase for selected clients using the SimPy model.

        After local training, each selected client must upload its model update
        to the edge server.  This function delegates to SimPyNetwork.simulate_uploads
        to compute which clients' updates arrive in time (before the round deadline)
        and how long each upload took.

        Parameters
        ----------
        selected_ids : List[int]
            Client IDs that completed local training and are attempting upload.
        model_size_mb : float
            Size of the model update in megabytes.  For the MLP in this project,
            a small model with ~10K parameters = ~0.04 MB (float32), but we use
            a configurable value (default 1.0) to model realistic upload times.

        Returns
        -------
        arrived_ids : List[int]
            Subset of selected_ids whose updates arrived before the deadline.
        upload_times : List[float]
            Simulated upload completion times (seconds) for each selected_id.
        total_comm_cost : float
            Sum of upload times for all arriving clients (logged as communication cost).
        """
        if not selected_ids:
            return [], [], 0.0

        bandwidths = [self.client_attributes[cid].bandwidth for cid in selected_ids]
        latencies = [self.client_attributes[cid].latency for cid in selected_ids]

        arrived_ids, upload_times = self.network.simulate_uploads(
            client_ids=selected_ids,
            bandwidths=bandwidths,
            latencies=latencies,
            model_size_mb=model_size_mb,
        )

        total_comm_cost = float(np.sum([t for t in upload_times if t < float("inf")]))
        return arrived_ids, upload_times, total_comm_cost

    # -----------------------------------------------------------------------
    # Metrics snapshot: for logging and RL reward computation
    # -----------------------------------------------------------------------

    def get_round_metrics(
        self,
        selected_ids: List[int],
        arrived_ids: List[int],
        upload_times: List[float],
        local_losses: Dict[int, float],
    ) -> Dict[str, Any]:
        """
        Collects a snapshot of all per-round metrics for this cluster.

        Called by runner.py after the round completes, and by metrics.py to
        aggregate cross-cluster statistics for logging/plotting.

        Parameters
        ----------
        selected_ids : List[int]
            Clients that were selected for training this round.
        arrived_ids : List[int]
            Clients whose updates arrived in time (after SimPy simulation).
        upload_times : List[float]
            Upload completion times for each selected client.
        local_losses : Dict[int, float]
            Maps client_id → local training loss for this round.

        Returns
        -------
        Dict with keys: round, cluster_id, selected_count, arrived_count,
            straggler_count, energy_cost, avg_battery, avg_staleness,
            participation_counts, local_losses, upload_times.
        """
        energy_cost = sum(
            self.client_attributes[cid].energy_consumed
            for cid in selected_ids
        )
        batteries = [self.client_attributes[cid].battery for cid in self.client_ids]
        stalenesses = [self.client_attributes[cid].staleness for cid in self.client_ids]
        participation_counts = [
            self.client_attributes[cid].participation_count for cid in self.client_ids
        ]

        metrics = {
            "round": self.round_number,
            "cluster_id": self.cluster_id,
            "selected_count": len(selected_ids),
            "arrived_count": len(arrived_ids),
            "straggler_count": len(selected_ids) - len(arrived_ids),
            "energy_cost": energy_cost,
            "avg_battery": float(np.mean(batteries)),
            "avg_staleness": float(np.mean(stalenesses)),
            "participation_counts": participation_counts,
            "local_losses": {cid: local_losses.get(cid, float("nan")) for cid in selected_ids},
            "upload_times": dict(zip(selected_ids, upload_times)),
        }

        self.round_history.append(metrics)
        return metrics

    def reset(self) -> None:
        """
        Resets the cluster to its initial state for a fresh experiment run.

        Called between seeds in multi-seed experiments to ensure clean separation.
        Resets all client attributes (battery, staleness, participation counts)
        but does NOT regenerate data partitions or system attributes — those are
        fixed per-experiment-config and should not change between seeds.
        (Seeds control training randomness, not the data/hardware setup.)
        """
        self.round_number = 0
        self.round_history = []
        for cid in self.client_ids:
            self.client_attributes[cid].reset()
        # Re-seed the RNG for clean reproducibility
        self.rng = np.random.default_rng(
            int(self.rng.integers(0, 2**31)) + self.cluster_id * 1000
        )

    def __repr__(self) -> str:
        available = len(self.get_available_clients())
        return (
            f"EdgeCluster(id={self.cluster_id}, "
            f"clients={len(self.client_ids)}, "
            f"available={available}, "
            f"round={self.round_number})"
        )
