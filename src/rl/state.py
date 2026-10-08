"""
src/rl/state.py
---------------
WHAT THIS FILE DOES:
    Constructs the RL observation (state) tensor that each edge cluster's
    Actor-Critic policy receives before making a client-selection decision
    each round.

    The state is a matrix of shape (N_cluster, STATE_DIM) where each row
    is a 5-dimensional feature vector for one client in the cluster:
        [0] compute_speed     — device throughput, normalized to [0, 1]
        [1] battery           — current battery level in [0, 1]
        [2] bandwidth_norm    — log-scaled, normalized bandwidth
        [3] staleness_norm    — rounds since last selection, clipped and normalized
        [4] last_local_loss   — most recent local training loss (raw float)

    Unavailable/churned clients are represented as zero vectors so the policy
    network sees a consistent input size regardless of churn, and can learn
    to avoid selecting zero-state (unavailable) clients.

WHERE IT SITS IN THE ARCHITECTURE:
    Called by simulation/runner.py once per FL round, per cluster, BEFORE the
    RL policy makes its selection.  Also called by baselines/centralized_rl.py
    which concatenates state matrices from all clusters into one global observation.

    Upstream: environment/cluster.py (provides ClientAttributes per client)
    Downstream: rl/policy.py (consumes the state matrix as its observation)

DEPENDENCIES:
    - numpy: for the per-client feature construction
    - torch: converts the numpy state matrix to a tensor for the policy network
    - data/client_attributes.py: ClientAttributes.to_state_vector()

KEY DESIGN DECISION:
    We use a FLAT per-client feature vector (5 scalars) rather than a graph
    neural network (GNN) that would model client-to-client relationships.
    A GNN could theoretically capture cluster topology but adds significant
    complexity (message passing, edge features, variable-size graphs).  For
    this paper, the flat feature vector is sufficient because clients within
    a cluster are treated as independent (the key relationships are
    client-to-server, captured by bandwidth/latency in the feature vector).
    GNN was NOT chosen because it would shift the paper's focus from the
    federation mechanism to the encoder architecture, which is not the contribution.
"""

import numpy as np
import torch
from typing import List, Dict, Optional

from src.data.client_attributes import ClientAttributes


# Number of features per client in the RL state vector (keep in sync with policy.py)
STATE_DIM = 5


class StateBuilder:
    """
    Converts a cluster's list of ClientAttributes into an RL observation tensor.

    This class is intentionally lightweight — it does not hold state itself.
    It is a pure transformation: attributes → tensor.  This makes it easy to
    use in both the per-cluster RL policy and the centralized baseline (which
    concatenates observations from all clusters).

    Parameters
    ----------
    max_staleness : int
        Staleness values are clipped to this maximum before normalization.
        Default 20: after 20 missed rounds, additional staleness carries
        diminishing information (the model is very stale regardless).
    max_loss : float
        Loss values are clipped to this maximum for normalization.
        Default 5.0: losses above this are already "very bad", clipping
        prevents a single outlier client from dominating the state scale.
    """

    # State vector feature indices (for readable access in debugging/logging)
    IDX_COMPUTE   = 0
    IDX_BATTERY   = 1
    IDX_BANDWIDTH = 2
    IDX_STALENESS = 3
    IDX_LOSS      = 4

    def __init__(self, max_staleness: int = 20, max_loss: float = 5.0):
        self.max_staleness = max_staleness
        self.max_loss = max_loss

    def build_state_matrix(
        self,
        client_ids: List[int],
        client_attributes: Dict[int, ClientAttributes],
    ) -> torch.Tensor:
        """
        Builds the state observation matrix for one cluster.

        MATH — Per-client state vector s_i (5-dimensional):
        ---------------------------------------------------
        Let attr_i be the ClientAttributes object for client i:

            s_i[0] = attr_i.compute_speed                         ∈ [0.1, 1.0]
            s_i[1] = attr_i.battery                               ∈ [0.0, 1.0]
            s_i[2] = log(1 + attr_i.bandwidth) / log(101)        ∈ [0.0, 1.0]
                     (log-scale normalizes the long-tail bandwidth distribution)
            s_i[3] = min(attr_i.staleness, max_staleness)
                     / max_staleness                              ∈ [0.0, 1.0]
            s_i[4] = min(attr_i.last_local_loss, max_loss)
                     / max_loss                                   ∈ [0.0, 1.0]

        Unavailable clients: s_i = 0 vector (all zeros).
        The policy network learns to assign very low selection scores to zero
        vectors, effectively never selecting churned/dead-battery clients.

        Parameters
        ----------
        client_ids : List[int]
            Ordered list of client IDs in this cluster.  The ordering is fixed
            and consistent across rounds so the policy's output indices map
            correctly to client IDs.
        client_attributes : Dict[int, ClientAttributes]
            Maps client_id → ClientAttributes (current round state).

        Returns
        -------
        torch.Tensor, shape (len(client_ids), STATE_DIM), dtype float32.
            The RL observation matrix.  Row i corresponds to client_ids[i].
        """
        rows = []
        for cid in client_ids:
            attr = client_attributes[cid]
            if not attr.is_available:
                # Zero vector: signals "unavailable" to the policy
                rows.append(np.zeros(STATE_DIM, dtype=np.float32))
            else:
                # Normalize each feature to [0, 1] range:
                compute = float(attr.compute_speed)                                     # already in [0.1,1.0]
                battery = float(attr.battery)                                           # already in [0.0,1.0]
                bw_norm = float(np.log1p(attr.bandwidth) / np.log1p(100.0))            # log-normalize
                stale_norm = float(min(attr.staleness, self.max_staleness)
                                   / self.max_staleness)                                # clip and normalize
                loss_norm = float(min(attr.last_local_loss, self.max_loss)
                                  / self.max_loss)                                      # clip and normalize

                rows.append(np.array(
                    [compute, battery, bw_norm, stale_norm, loss_norm],
                    dtype=np.float32,
                ))

        state_matrix = np.stack(rows, axis=0)  # shape: (N_cluster, 5)
        return torch.from_numpy(state_matrix)

    def build_availability_mask(
        self,
        client_ids: List[int],
        client_attributes: Dict[int, ClientAttributes],
    ) -> torch.Tensor:
        """
        Returns a boolean mask where True = client is available for selection.

        Used by the policy's masked softmax to prevent selecting unavailable clients.

        Parameters
        ----------
        client_ids, client_attributes : same as build_state_matrix.

        Returns
        -------
        torch.Tensor[bool], shape (len(client_ids),)
        """
        mask = [client_attributes[cid].is_available for cid in client_ids]
        return torch.tensor(mask, dtype=torch.bool)

    def build_global_state_matrix(
        self,
        all_cluster_client_ids: List[List[int]],
        client_attributes: Dict[int, ClientAttributes],
    ) -> tuple:
        """
        Builds a GLOBAL state matrix by concatenating all clusters' client
        feature vectors into a single matrix.

        Used by baselines/centralized_rl.py: the single centralized RL agent
        sees ALL clients across ALL clusters as one flat observation.

        Parameters
        ----------
        all_cluster_client_ids : List[List[int]]
            One inner list per cluster; each contains the client_ids in that cluster.
        client_attributes : Dict[int, ClientAttributes]
            All clients' attributes.

        Returns
        -------
        state_matrix : torch.Tensor, shape (total_clients, STATE_DIM)
        availability_mask : torch.Tensor[bool], shape (total_clients,)
        flat_client_ids : List[int]
            The flattened ordering of client IDs (row i of state_matrix = flat_client_ids[i]).
        """
        flat_ids = []
        all_rows = []
        all_masks = []

        for cluster_ids in all_cluster_client_ids:
            for cid in cluster_ids:
                flat_ids.append(cid)
                attr = client_attributes[cid]
                if not attr.is_available:
                    all_rows.append(np.zeros(STATE_DIM, dtype=np.float32))
                    all_masks.append(False)
                else:
                    compute   = float(attr.compute_speed)
                    battery   = float(attr.battery)
                    bw_norm   = float(np.log1p(attr.bandwidth) / np.log1p(100.0))
                    stale_n   = float(min(attr.staleness, self.max_staleness)
                                      / self.max_staleness)
                    loss_n    = float(min(attr.last_local_loss, self.max_loss)
                                      / self.max_loss)
                    all_rows.append(np.array([compute, battery, bw_norm, stale_n, loss_n],
                                             dtype=np.float32))
                    all_masks.append(True)

        state_matrix = torch.from_numpy(np.stack(all_rows, axis=0))
        availability_mask = torch.tensor(all_masks, dtype=torch.bool)
        return state_matrix, availability_mask, flat_ids
