"""
src/baselines/centralized_rl.py
---------------------------------
WHAT THIS FILE DOES:
    Implements the Single-Agent Centralized RL baseline: ONE ActorCriticPolicy
    that observes the system state of ALL clients across ALL clusters and
    selects K_total = num_clusters × k_per_cluster clients globally, ignoring
    cluster boundaries.

    This is the MOST IMPORTANT baseline in the paper.  It answers the question:
        "Does federating the RL policy actually help, or could we get the
         same performance by just having one big centralized RL agent?"

    If the federated multi-agent RL (FMARL) method outperforms this centralized
    baseline, it demonstrates that policy federation adds value beyond simply
    having a larger (global) policy.  If FMARL merely matches centralized RL,
    the contribution is efficiency/privacy (decentralization) rather than
    accuracy improvement.

WHERE IT SITS IN THE ARCHITECTURE:
    The CentralizedRLSelector replaces ALL per-cluster RL policies with a
    single global policy.  In the runner, when method='centralized_rl':
        - Instead of one policy per cluster, there is ONE CentralizedRLSelector
        - It is called once per FL round (not once per cluster)
        - It returns a Dict[cluster_id → List[selected_ids]] mapping
        - It trains via PPO on the global reward (same reward signal)

    Upstream: simulation/runner.py provides global state from all clusters.
    Downstream: fl/local_trainer.py trains the selected clients.

DEPENDENCIES:
    - rl/state.py:     StateBuilder.build_global_state_matrix()
    - rl/policy.py:    ActorCriticPolicy (same architecture, larger input)
    - rl/ppo_trainer.py: PPOTrainer (same update logic)
    - data/client_attributes.py: ClientAttributes for all clients

KEY DESIGN DECISION:
    The centralized agent selects clients GLOBALLY (k_total = sum of k_per_cluster
    across all clusters), ignoring cluster membership when ranking clients.
    The selected clients are then assigned back to their respective clusters for
    local training.  This gives the centralized agent the maximum possible
    information and flexibility — it can select 0 clients from one cluster and
    all clients from another if it learns that's optimal.  A constrained version
    (exactly k clients per cluster, selected centrally) was NOT chosen because it
    would artificially limit the centralized agent's advantage and make the
    comparison unfair to the paper's hierarchical method.
"""

import numpy as np
import torch
import copy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.rl.state import StateBuilder
from src.rl.policy import ActorCriticPolicy
from src.rl.ppo_trainer import PPOTrainer, compute_reward
from src.data.client_attributes import ClientAttributes


class CentralizedRLSelector:
    """
    Single-agent RL selector that sees and selects across all clients globally.

    This is the ablation point for the paper: it tests whether the hierarchical
    structure (decentralized policies + policy federation) beats having one
    omniscient RL agent with full visibility.

    Parameters
    ----------
    all_cluster_client_ids : List[List[int]]
        one inner list per cluster; cluster_i's clients are inner list i.
        Defines the mapping from flat global indices back to cluster/client.
    num_clients_total : int
        Total number of clients across all clusters.
    k_per_cluster : int
        Nominal clients to select per cluster.  The global selection target is
        k_total = k_per_cluster * num_clusters, but the centralized agent can
        deviate from equal per-cluster allocation.
    rl_config : dict
        RL hyperparameters from YAML: lr_actor, lr_critic, ppo_epochs, etc.
    reward_weights : dict
        From YAML config['rl']['reward_weights'].
    hidden_dim : int
        Width of the policy network's hidden layers.
    seed : int
        RNG seed for reproducible training.
    device : torch.device
    """

    def __init__(
        self,
        all_cluster_client_ids: List[List[int]],
        num_clients_total: int,
        k_per_cluster: int,
        rl_config: dict,
        reward_weights: dict,
        hidden_dim: int = 64,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
    ):
        self.all_cluster_client_ids = all_cluster_client_ids
        self.num_clusters = len(all_cluster_client_ids)
        self.k_per_cluster = k_per_cluster
        self.k_total = k_per_cluster * self.num_clusters  # global selection budget
        self.reward_weights = reward_weights
        self.device = device

        # Build a flat index → (cluster_id, client_id) lookup table
        # so we can map global selections back to per-cluster assignments
        self._flat_to_cluster: Dict[int, int] = {}
        flat_idx = 0
        for c, cids in enumerate(all_cluster_client_ids):
            for cid in cids:
                self._flat_to_cluster[flat_idx] = (c, cid)
                flat_idx += 1

        # State builder (shared with main system)
        from src.rl.state import STATE_DIM
        self.state_builder = StateBuilder()

        # Single global policy: same architecture, larger input (all clients)
        torch.manual_seed(seed)
        self.policy = ActorCriticPolicy(
            state_dim=STATE_DIM,
            hidden_dim=hidden_dim,
        )

        # PPO trainer (same hyperparameters as per-cluster policies)
        self.trainer = PPOTrainer(
            policy=self.policy,
            lr_actor=rl_config.get("lr_actor", 3e-4),
            lr_critic=rl_config.get("lr_critic", 1e-3),
            ppo_epochs=rl_config.get("ppo_epochs", 3),
            clip_eps=rl_config.get("clip_eps", 0.2),
            entropy_coef=rl_config.get("entropy_coef", 0.01),
        )

        # Track participation counts across ALL clients for Gini computation
        self._participation_counts: List[int] = [0] * num_clients_total
        self._acc_before: float = 0.0  # stored between select and on_round_end calls

        # PPO buffer flush cadence — controlled by update_every_n_rounds.
        # Falls back to T_RL for backward compatibility with old configs that
        # don't specify update_every_n_rounds separately.
        self._update_every: int = rl_config.get(
            "update_every_n_rounds", rl_config.get("T_RL", 5)
        )
        self._round_number: int = 0
        # Set to True only in the round where PPO actually ran.
        # The runner reads this to avoid carry-forward logging of stale loss values.
        self.last_update_fired: bool = False

    def select_clients(
        self,
        client_attributes: Dict[int, ClientAttributes],
        acc_before: float,
        deterministic: bool = False,
    ) -> Tuple[Dict[int, List[int]], torch.Tensor, torch.Tensor]:
        """
        Selects k_total clients globally and assigns them back to their clusters.

        Parameters
        ----------
        client_attributes : Dict[int, ClientAttributes]
            All clients' current state (across all clusters).
        acc_before : float
            Current global model accuracy (stored for reward computation).
        deterministic : bool
            If True, use greedy (argmax) selection for evaluation.

        Returns
        -------
        cluster_selections : Dict[int, List[int]]
            Maps cluster_id → list of selected client_ids for that cluster.
        log_prob : torch.Tensor (scalar)
            Log-probability of the global selection (for PPO storage).
        value : torch.Tensor (scalar)
            Critic's value estimate V(s).
        """
        self._acc_before = acc_before

        # Build global state: all clients from all clusters concatenated
        global_state, global_mask, flat_ids = self.state_builder.build_global_state_matrix(
            self.all_cluster_client_ids, client_attributes
        )

        # Centralized policy selects k_total clients from the global pool
        selected_flat_indices, log_prob, value = self.policy.select_clients(
            state_matrix=global_state,
            availability_mask=global_mask,
            k=self.k_total,
            deterministic=deterministic,
        )

        # Map flat indices back to (cluster_id, client_id) pairs
        cluster_selections: Dict[int, List[int]] = defaultdict(list)
        for flat_idx in selected_flat_indices:
            cluster_id, client_id = self._flat_to_cluster[flat_idx]
            cluster_selections[cluster_id].append(client_id)

        # Store global state for PPO transition
        self._last_state = global_state.numpy()
        self._last_mask  = global_mask.numpy()
        self._last_action = selected_flat_indices

        return dict(cluster_selections), log_prob, value

    def on_round_end(
        self,
        acc_after: float,
        energy_cost: float,
        all_participation_counts: List[int],
        log_prob: float,
        value: float,
    ) -> None:
        """
        Stores the round's transition in the PPO buffer and (every
        update_every_n_rounds rounds) triggers a PPO update.

        Parameters
        ----------
        acc_after : float
            Global model accuracy AFTER this round's FL update.
        energy_cost : float
            Total energy consumed across ALL selected clients (all clusters).
        all_participation_counts : List[int]
            Per-client participation counts (all clients, for Gini fairness).
        log_prob : float
            Log-probability of the selection action (from select_clients).
        value : float
            Critic's value estimate at selection time (from select_clients).
        """
        reward = compute_reward(
            acc_before=self._acc_before,
            acc_after=acc_after,
            energy_cost=energy_cost,
            participation_counts=all_participation_counts,
            reward_weights=self.reward_weights,
        )
        # Expose the per-round reward for the learning-curve log (L7).
        self.last_reward = float(reward)

        self._round_number += 1
        done = (self._round_number % self._update_every == 0)
        # Reset the fired flag; only set True if PPO actually runs this round.
        self.last_update_fired = False

        self.trainer.store_transition(
            state_matrix=self._last_state,
            availability_mask=self._last_mask,
            action_indices=self._last_action,
            log_prob_old=log_prob,
            reward=reward,
            value_old=value,
            done=done,
        )

        # Run PPO update every update_every_n_rounds rounds
        if done and len(self.trainer.buffer) > 0:
            self.trainer.update()
            self.trainer.clear_buffer()
            self.last_update_fired = True

    def get_policy_weights(self) -> dict:
        """Returns the global policy weights (for logging / analysis)."""
        return self.policy.get_weights()

    def reset(self, seed: int = None) -> None:
        """Resets for a new experiment seed."""
        if seed is not None:
            torch.manual_seed(seed)
            self.policy = ActorCriticPolicy(
                state_dim=self.policy.state_dim,
                hidden_dim=self.policy.hidden_dim,
            )
            self.trainer.policy = self.policy
        self.trainer.reset()
        self._participation_counts = [0] * len(self._participation_counts)
        self._round_number = 0

    def __repr__(self) -> str:
        return (
            f"CentralizedRLSelector("
            f"k_total={self.k_total}, "
            f"clusters={self.num_clusters}, "
            f"policy={self.policy})"
        )
