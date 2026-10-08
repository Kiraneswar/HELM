"""
src/rl/policy.py
-----------------
WHAT THIS FILE DOES:
    Defines the Actor-Critic neural network that is the per-cluster RL
    client-selection policy.  Each edge cluster owns one instance of this
    network.  Given a state matrix describing all clients in the cluster,
    it produces:
        - Actor:  a probability distribution over clients → select top-k
        - Critic: a scalar value estimate V(s) for PPO advantage computation

    The policy's weights are what get federated every T_RL rounds — the
    core novel contribution of this project.

WHERE IT SITS IN THE ARCHITECTURE:
    This is the EDGE tier's local RL policy.  It runs once per edge cluster,
    decides which clients get selected each FL round, and periodically sends
    its weights up to the cloud aggregator defined in fl/aggregator.py.

    Each EdgeCluster in simulation/runner.py owns exactly one ActorCriticPolicy.
    The centralized RL baseline (baselines/centralized_rl.py) also uses this
    same class but with a larger input (all clients across all clusters).

    Upstream:  rl/state.py builds the state matrix this network consumes.
    Downstream: rl/ppo_trainer.py collects (state, action, log_prob, value)
                tuples and runs PPO updates on this network.
                fl/aggregator.py receives this network's state_dict for federation.

DEPENDENCIES:
    - torch, torch.nn: the policy network.
    - rl/state.py: STATE_DIM constant (must match feature vector width).

KEY DESIGN DECISION:
    We use PPO (Proximal Policy Optimization) instead of vanilla REINFORCE.
    REINFORCE's high-variance gradient estimates (no baseline subtraction, or
    only a simple scalar baseline) caused the client-selection policy to be
    very unstable during early development testing: the policy would oscillate
    between selecting the same 1-2 clients every round (degenerate) and
    randomly exploring (high entropy).  PPO's clipped objective keeps updates
    conservative (within a trust region), and the critic's advantage estimate
    (GAE) dramatically reduces gradient variance.  REINFORCE was NOT chosen
    for these reasons.

    For the ACTION SPACE, we use sequential multinomial sampling without
    replacement.  The alternative (Gumbel-softmax top-k) provides a
    differentiable approximation but produces non-integer "soft" selections
    that are harder to interpret and log.  Sequential multinomial gives exact
    discrete selections with well-defined log-probabilities for PPO's ratio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import copy

from src.rl.state import STATE_DIM


# ---------------------------------------------------------------------------
# Actor-Critic Policy Network
# ---------------------------------------------------------------------------

class ActorCriticPolicy(nn.Module):
    """
    Shared-encoder Actor-Critic network for top-k client selection.

    Architecture:
        Shared per-client encoder (processes each client's feature vector):
            Linear(STATE_DIM, hidden_dim) → LayerNorm → ReLU
            → Linear(hidden_dim, hidden_dim) → ReLU

        Actor head (outputs a score per client):
            Linear(hidden_dim, 1)  → squeeze → masked softmax → multinomial

        Critic head (outputs scalar V(s) from mean-pooled client encodings):
            Linear(hidden_dim, hidden_dim) → ReLU → Linear(hidden_dim, 1)

    Why shared encoder?
        The actor and critic share a common representation of client features.
        This reduces the total parameter count (important for federation — fewer
        parameters → faster aggregation, less communication overhead) and means
        the critic's value estimate is informed by the same client feature space
        as the actor's selection scores.

    Parameters
    ----------
    state_dim : int
        Width of each client's feature vector.  Must equal STATE_DIM from state.py.
    hidden_dim : int
        Width of the shared encoder's hidden layers.  Default 64 — small enough
        for fast federation (64²*2 ≈ 8K params per policy) while expressive enough
        to learn non-trivial selection strategies in 30-100 FL rounds.
    dropout_rate : float
        Dropout applied after the first encoder layer (mild regularization).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 64,
        dropout_rate: float = 0.05,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------------------
        # Shared per-client encoder
        # Processes each client's feature row independently (no cross-client
        # attention) — the simplest design that still lets the network learn
        # non-linear mappings from raw features to selection scores.
        # ----------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # LayerNorm (not BatchNorm) so it works
            nn.ReLU(),                  # with a single client (batch_size=1)
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: maps per-client encoding → scalar selection score
        # (no softmax here — the masking + softmax happens in get_action_scores)
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic head: takes mean-pooled encoding → scalar value estimate V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Orthogonal initialization for all Linear layers.
        Recommended for PPO actor-critic networks (Andrychowicz et al., 2021):
        orthogonal init preserves gradient magnitudes through the network and
        gives more stable early training than He/Xavier for RL policies.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # Core forward methods
    # -----------------------------------------------------------------------

    def encode(self, state_matrix: torch.Tensor) -> torch.Tensor:
        """
        Runs the shared encoder on all client feature vectors.

        Parameters
        ----------
        state_matrix : torch.Tensor, shape (N_clients, STATE_DIM)
            The RL observation for one cluster (from StateBuilder).

        Returns
        -------
        torch.Tensor, shape (N_clients, hidden_dim)
            Per-client encoded representations.
        """
        return self.encoder(state_matrix)

    def get_action_scores(
        self,
        state_matrix: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes per-client selection logits and encodes client states.

        Unavailable clients receive a very large negative logit so that
        softmax(logits) ≈ 0 for them — they are effectively excluded from
        selection without requiring the sampler to track availability separately.

        Parameters
        ----------
        state_matrix : torch.Tensor, shape (N, STATE_DIM)
        availability_mask : torch.Tensor[bool], shape (N,)

        Returns
        -------
        scores : torch.Tensor, shape (N,)
            Raw (unmasked softmax) logits for each client.
        encoded : torch.Tensor, shape (N, hidden_dim)
            Per-client encodings (reused by the critic to avoid double forward pass).
        """
        encoded = self.encode(state_matrix)            # (N, hidden)
        scores = self.actor_head(encoded).squeeze(-1)  # (N,)

        # Mask unavailable clients: set their logit to -1e9 so softmax gives ≈ 0
        # This is the standard "mask before softmax" approach for constrained action spaces
        scores = scores.masked_fill(~availability_mask, -1e9)

        return scores, encoded

    def get_value(
        self,
        encoded: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Critic: estimates V(s) from mean-pooled available-client encodings.

        MATH — Mean-pooled critic input:
            e_global = (Σ_{i: available} encoded_i) / |available clients|

        Mean-pooling over available clients gives a permutation-invariant
        global state summary.  The alternative (concatenating all client
        encodings) would give a fixed-size input only if N_clients is fixed,
        which breaks when clusters have different sizes.

        Parameters
        ----------
        encoded : torch.Tensor, shape (N, hidden_dim)
            Output of encode() — reused to avoid redundant computation.
        availability_mask : torch.Tensor[bool], shape (N,)

        Returns
        -------
        torch.Tensor, scalar — the state value estimate V(s).
        """
        # Mean-pool only available clients
        avail_float = availability_mask.float().unsqueeze(-1)  # (N, 1)
        n_available = avail_float.sum().clamp(min=1.0)
        pooled = (encoded * avail_float).sum(dim=0) / n_available  # (hidden_dim,)

        value = self.critic_head(pooled)  # (1,)
        return value.squeeze()            # scalar

    # -----------------------------------------------------------------------
    # Action sampling: used during rollout (data collection)
    # -----------------------------------------------------------------------

    def select_clients(
        self,
        state_matrix: torch.Tensor,
        availability_mask: torch.Tensor,
        k: int,
        deterministic: bool = False,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Samples k clients without replacement using score-based multinomial.

        ACTION SPACE DESIGN:
        ---------------------
        The top-k client selection problem is framed as SEQUENTIAL MULTINOMIAL
        SAMPLING WITHOUT REPLACEMENT.  At each step t ∈ {1,...,k}:

            MATH:
                remaining_mask = availability_mask AND (not yet selected)
                probs_t = softmax(scores * remaining_mask_float)   (masked softmax)
                a_t ~ Categorical(probs_t)
                log π(a_t | s, a_{<t}) = log probs_t[a_t]

            The full action log-probability is:
                log π(A | s) = Σ_{t=1}^{k} log π(a_t | s, a_{<t})

        This sequential decomposition makes the log-probability tractable
        (no need to enumerate all C(N,k) subsets) and naturally handles
        varying numbers of available clients.

        Parameters
        ----------
        state_matrix : torch.Tensor, shape (N, STATE_DIM)
        availability_mask : torch.Tensor[bool], shape (N,)
        k : int
            Number of clients to select.  Clamped to min(k, n_available).
        deterministic : bool
            If True, use argmax (greedy) instead of sampling.  Used during
            evaluation/baselines to get the policy's "best guess" selection.

        Returns
        -------
        selected_indices : List[int]
            Indices into client_ids list (row indices of state_matrix).
        log_prob : torch.Tensor (scalar, requires_grad)
            Log-probability of this action for PPO training.
        value : torch.Tensor (scalar)
            Critic's value estimate V(s).
        """
        with torch.set_grad_enabled(not deterministic):
            scores, encoded = self.get_action_scores(state_matrix, availability_mask)
            value = self.get_value(encoded, availability_mask)

            n_available = availability_mask.sum().item()
            k_actual = min(k, int(n_available))

            if k_actual == 0:
                # No available clients — return empty selection
                return [], torch.tensor(0.0), value

            selected_indices: List[int] = []
            log_prob = torch.tensor(0.0)
            # Track which positions are still available using a plain Python list
            # (avoids inplace tensor mutation inside the autograd graph)
            remaining: List[bool] = availability_mask.tolist()

            for step in range(k_actual):
                # Build a fresh boolean tensor from the Python list each step
                # so masked_fill never sees an inplace-modified tensor
                step_mask = torch.tensor(remaining, dtype=torch.bool)

                # Recompute masked scores over remaining (not yet selected) clients
                masked_scores = scores.masked_fill(~step_mask, -1e9)

                # MATH: p_t = softmax(masked_scores)
                probs = F.softmax(masked_scores, dim=0)

                if deterministic:
                    # Greedy: always pick the highest-probability client
                    idx = int(probs.argmax().item())
                else:
                    # Stochastic: sample one client from the distribution
                    idx = int(torch.multinomial(probs, num_samples=1).item())

                selected_indices.append(idx)

                # MATH: log π(a_t | s, a_{<t}) = log p_t[idx]
                log_prob = log_prob + torch.log(probs[idx] + 1e-8)

                # Remove the selected client from the remaining pool (Python list, not tensor)
                remaining[idx] = False

        return selected_indices, log_prob, value

    # -----------------------------------------------------------------------
    # Action evaluation: used during PPO update
    # -----------------------------------------------------------------------

    def evaluate_action(
        self,
        state_matrix: torch.Tensor,
        availability_mask: torch.Tensor,
        action_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluates a previously taken action under the CURRENT (updated) policy.

        This is the critical PPO method: during the update step, we need to
        compute log π_new(A|s) for an action A that was originally taken by
        π_old.  The ratio r_t = π_new/π_old is what PPO clips to stabilize learning.

        MATH — PPO Probability Ratio:
            r_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
                = exp(log_prob_new - log_prob_old)

        We also compute the per-step entropy to use as the PPO entropy bonus,
        which encourages the policy to maintain exploration rather than
        collapsing to a deterministic selection strategy prematurely.

        MATH — Per-step entropy (averaged over k selection steps):
            H_t = -Σ_{i: remaining} p_t[i] * log(p_t[i])
            H = (1/k) Σ_{t=1}^{k} H_t

        Parameters
        ----------
        state_matrix : torch.Tensor, shape (N, STATE_DIM)
        availability_mask : torch.Tensor[bool], shape (N,)
        action_indices : List[int]
            The exact client indices that were selected in the stored rollout.

        Returns
        -------
        log_prob_new : torch.Tensor (scalar, requires_grad)
            Log-probability of the action under the CURRENT policy weights.
        entropy : torch.Tensor (scalar)
            Average entropy across the k selection steps.
        value : torch.Tensor (scalar)
            Current critic's value estimate V(s) (used for value loss).
        """
        scores, encoded = self.get_action_scores(state_matrix, availability_mask)
        value = self.get_value(encoded, availability_mask)

        log_prob_new = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)
        # Use a plain Python list to track which indices are still available,
        # avoiding inplace tensor mutations that break autograd during backward().
        remaining: List[bool] = availability_mask.tolist()
        valid_steps = 0

        for idx in action_indices:
            # Safety: skip if the selected index is no longer available
            # (can happen if churn state changed between collection and update)
            if not remaining[idx]:
                continue

            # Build a fresh boolean tensor from the Python list each step
            step_mask = torch.tensor(remaining, dtype=torch.bool)

            masked_scores = scores.masked_fill(~step_mask, -1e9)
            probs = F.softmax(masked_scores, dim=0)

            # MATH: log π_new(a_t | s, a_{<t}) = log p_new[idx]
            log_prob_new = log_prob_new + torch.log(probs[idx] + 1e-8)

            # MATH: H_t = -Σ_i p_i log(p_i)  over remaining clients
            valid_probs = probs[step_mask]
            step_entropy = -(valid_probs * torch.log(valid_probs + 1e-8)).sum()
            total_entropy = total_entropy + step_entropy

            remaining[idx] = False
            valid_steps += 1

        entropy = total_entropy / max(valid_steps, 1)
        return log_prob_new, entropy, value

    # -----------------------------------------------------------------------
    # Weight management (for federation)
    # -----------------------------------------------------------------------

    def get_weights(self) -> dict:
        """
        Returns a deep copy of the policy's state_dict.

        Called by simulation/runner.py every T_RL rounds to collect policy
        weights from all clusters for federation in fl/aggregator.py.
        Deep copy is mandatory — federation modifies these dicts and must not
        affect the live policy network.
        """
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict: dict) -> None:
        """
        Loads a state_dict into this policy network.

        Called after fl/aggregator.federate_policies() returns updated weights.
        Used for BOTH soft-update (caller passes blended weights) and
        hard-replace (caller passes the meta-policy weights directly).

        Parameters
        ----------
        state_dict : dict
            Must have the same keys and tensor shapes as this policy's state_dict.
        """
        self.load_state_dict(state_dict)

    def count_parameters(self) -> int:
        """Total trainable parameters (used in logging and federation cost analysis)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ActorCriticPolicy("
            f"state_dim={self.state_dim}, "
            f"hidden={self.hidden_dim}, "
            f"params={self.count_parameters():,})"
        )
