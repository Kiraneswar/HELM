"""
src/rl/ppo_trainer.py
----------------------
WHAT THIS FILE DOES:
    Implements the PPO (Proximal Policy Optimization) training loop that improves
    each cluster's ActorCriticPolicy over the course of the FL experiment.

    Each FL round, the runner:
        1. Calls policy.select_clients() to get a client selection + log_prob
        2. Executes that selection (FL training + aggregation)
        3. Computes a reward from the FL outcome
        4. Calls ppo_trainer.store_transition() to record the experience

    Every T_RL rounds (the RL policy federation cadence), the runner calls
    ppo_trainer.update() to run PPO on the accumulated buffer of transitions,
    then clears the buffer for the next interval.

WHERE IT SITS IN THE ARCHITECTURE:
    One PPOTrainer instance is paired with each ActorCriticPolicy (one per
    edge cluster).  The centralized RL baseline has exactly one PPOTrainer
    paired with its single global policy.

    Upstream: simulation/runner.py calls store_transition() and update()
    Downstream: ActorCriticPolicy is updated in-place by update()

DEPENDENCIES:
    - torch, torch.optim: PyTorch optimizer and gradient machinery
    - rl/policy.py: ActorCriticPolicy is updated here

KEY DESIGN DECISION:
    We use PPO with Generalized Advantage Estimation (GAE, Schulman et al., 2016)
    for the advantage function rather than plain Monte Carlo returns.  GAE
    interpolates between the high-variance Monte Carlo estimate (lambda=1)
    and the high-bias one-step TD estimate (lambda=0) via a decay parameter
    lambda.  For this project (episodes are short, ~5 transitions per PPO
    update at T_RL=5), GAE with lambda=0.95 gives low variance with
    acceptable bias.  Plain Monte Carlo was NOT chosen because with only 5
    transitions per buffer, the variance is too high for stable policy improvement.

    The reward function is defined here (compute_reward) rather than in runner.py
    because the reward is part of the RL algorithm's specification.  The weights
    (accuracy, energy, fairness) are configurable per the ablation requirements.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from src.rl.policy import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Reward computation: the RL signal for client selection
# ---------------------------------------------------------------------------

def compute_reward(
    acc_before: float,
    acc_after: float,
    energy_cost: float,
    participation_counts: List[int],
    reward_weights: dict,
    energy_normalizer: float = 1.0,
) -> float:
    """
    Computes the scalar reward for one FL round's client selection.

    MATH — Reward Function:
    -----------------------
        r = w_acc * Δacc
          - w_energy * (energy_cost / energy_normalizer)
          - w_fair * gini(participation_counts)

    where:
        Δacc = acc_after - acc_before       (global model accuracy improvement this round)
        energy_cost                          (sum of selected clients' energy drain)
        gini(participation_counts)           (Gini coefficient of selection fairness)

    WHY THIS REWARD STRUCTURE?
        - Δacc (positive term): the RL policy should learn to select clients
          whose training improves the global model.  Using delta rather than
          absolute accuracy makes the reward stationary (the global model's
          absolute accuracy increases monotonically, which would cause reward
          non-stationarity if used directly).

        - energy_cost (negative term): penalizes selecting high-drain clients,
          especially important in the mobile/weak cluster where battery drain
          is severe.  The RL policy learns to trade off accuracy vs. energy.

        - gini coefficient (negative term): penalizes unfair selection patterns
          where a few clients are always chosen and most are never selected.
          Gini=0 means perfectly fair (all clients selected equally often);
          Gini=1 means maximally unfair (one client selected all the time).
          This term encourages the policy to periodically rotate its selections.

    The reward weight parameters are exposed in the YAML config under
    `rl.reward_weights` and are an ablation axis for the paper.

    Parameters
    ----------
    acc_before : float
        Global model accuracy BEFORE this round's FL update.
    acc_after : float
        Global model accuracy AFTER this round's FL update.
    energy_cost : float
        Total energy consumed by selected clients this round.
    participation_counts : List[int]
        Per-client count of how many rounds each client has been selected so far.
        Used to compute the Gini fairness coefficient.
    reward_weights : dict
        Contains keys 'accuracy', 'energy', 'fairness' with float weight values.
    energy_normalizer : float
        Scales energy_cost to keep it in a similar range as Δacc.
        Default 1.0; tune upward if energy values dwarf the accuracy delta.

    Returns
    -------
    float: scalar reward value (can be positive or negative).
    """
    w_acc    = reward_weights.get("accuracy", 1.0)
    w_energy = reward_weights.get("energy",   0.1)
    w_fair   = reward_weights.get("fairness", 0.05)

    # Term 1: accuracy improvement (main learning signal)
    delta_acc = acc_after - acc_before

    # Term 2: energy cost (negative penalty)
    energy_term = energy_cost / max(energy_normalizer, 1e-8)

    # Term 3: Gini coefficient of participation fairness
    gini = compute_gini(participation_counts)

    # MATH: r = w_acc * Δacc - w_energy * E_norm - w_fair * Gini
    reward = w_acc * delta_acc - w_energy * energy_term - w_fair * gini

    return float(reward)


def compute_gini(values: List[int]) -> float:
    """
    Computes the Gini coefficient of an integer distribution.

    MATH — Gini Coefficient:
    ------------------------
        G = (Σ_i Σ_j |x_i - x_j|) / (2 * n^2 * mean(x))

    G = 0: perfect equality (all values identical, i.e., every client was
           selected exactly the same number of times).
    G = 1: maximum inequality (one client was selected for all rounds, others
           for none).

    This is used as the fairness penalty in the reward function.

    Parameters
    ----------
    values : List[int]
        Participation counts per client.

    Returns
    -------
    float in [0, 1]: Gini coefficient.
    """
    if len(values) == 0:
        return 0.0

    x = np.array(values, dtype=np.float64)
    mean_x = x.mean()

    if mean_x < 1e-8:
        return 0.0  # No one has participated — by convention, Gini = 0

    # Efficient O(n log n) Gini implementation:
    # G = (2 * Σ_i (i+1)*x_i) / (n * Σ_i x_i) - (n+1)/n
    # (equivalent to the pairwise formula but O(n log n) instead of O(n²))
    n = len(x)
    x_sorted = np.sort(x)
    i = np.arange(1, n + 1, dtype=np.float64)
    gini = (2 * np.sum(i * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Transition: one experience tuple stored in the rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """
    Stores one FL round's experience tuple for PPO training.

    Fields (all stored as Python scalars/lists, converted to tensors in update()):
        state_matrix        : np.ndarray, shape (N, STATE_DIM) — raw state
        availability_mask   : np.ndarray[bool], shape (N,)
        action_indices      : List[int] — which clients were selected
        log_prob_old        : float — log π_old(a|s) at collection time
        reward              : float — scalar reward received after this step
        value_old           : float — critic's V(s) estimate at collection time
        done                : bool — True at end of a "training episode"
                              (in our setup, each T_RL-step interval is one episode)
    """
    state_matrix: np.ndarray
    availability_mask: np.ndarray
    action_indices: List[int]
    log_prob_old: float
    reward: float
    value_old: float
    done: bool = False


# ---------------------------------------------------------------------------
# PPOTrainer: updates the ActorCriticPolicy via PPO
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO trainer paired with one ActorCriticPolicy instance.

    One PPOTrainer exists per edge cluster (plus one for the centralized baseline).
    It accumulates transitions for T_RL rounds, then runs PPO_EPOCHS gradient
    updates over the buffer, then clears the buffer.

    Parameters
    ----------
    policy : ActorCriticPolicy
        The policy network this trainer will update in-place.
    lr_actor : float
        Learning rate for the actor parameters (scores and encoder).
    lr_critic : float
        Learning rate for the critic head (typically higher than actor LR
        for faster value function fitting).
    ppo_epochs : int
        Number of passes over the rollout buffer per PPO update call.
        More epochs → better sample efficiency but risk of overfitting the buffer.
    clip_eps : float
        PPO clipping epsilon ε.  The probability ratio r_t is clipped to
        [1-ε, 1+ε].  Default 0.2 (standard from the PPO paper).
    entropy_coef : float
        Weight of the entropy bonus in the PPO loss.  Higher values encourage
        more exploration (higher action entropy).
    value_coef : float
        Weight of the value loss term in the combined PPO loss.
    gamma : float
        Discount factor for future rewards.  γ=0.99 in standard RL; we use
        a slightly lower value (0.95) because FL "episodes" are short (T_RL
        rounds) and future rewards are less discounted in practice.
    gae_lambda : float
        GAE decay parameter λ.  Interpolates between Monte Carlo (λ=1) and
        TD(0) (λ=0) advantage estimates.  Default 0.95 (PPO paper default).
    max_grad_norm : float
        Gradient clipping norm for PPO updates.  Prevents occasional large
        gradient spikes when the policy ratio is far from 1.0.
    device : torch.device
        Device for tensor operations.
    """

    def __init__(
        self,
        policy: ActorCriticPolicy,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        ppo_epochs: int = 3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Single optimizer over ALL policy parameters.
        # Different LRs for actor vs. critic are approximated by using the
        # lower LR for all but scaling the value loss by value_coef.
        # A true two-optimizer setup complicates the federation (we'd need to
        # also share optimizer states, which is non-standard).
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=lr_actor,
            eps=1e-5,   # Adam epsilon: smaller value gives more stable updates
        )

        # Rollout buffer: accumulates transitions until update() is called
        self.buffer: List[Transition] = []

        # Training statistics (for logging)
        self.update_count: int = 0
        self.last_policy_loss: float = 0.0
        self.last_value_loss: float = 0.0
        self.last_entropy: float = 0.0

    # -----------------------------------------------------------------------
    # Buffer management
    # -----------------------------------------------------------------------

    def store_transition(
        self,
        state_matrix: np.ndarray,
        availability_mask: np.ndarray,
        action_indices: List[int],
        log_prob_old: float,
        reward: float,
        value_old: float,
        done: bool = False,
    ) -> None:
        """
        Appends one FL round's experience to the rollout buffer.

        Called by runner.py after each FL round completes and the reward is
        computed.  The buffer is filled for T_RL rounds, then update() is called.

        Parameters
        ----------
        state_matrix : np.ndarray, shape (N_clients, STATE_DIM)
            The state observed before selecting clients this round.
        availability_mask : np.ndarray[bool], shape (N_clients,)
            Which clients were available when the selection was made.
        action_indices : List[int]
            The client indices that were selected (returned by policy.select_clients).
        log_prob_old : float
            log π_old(a|s) — the log-probability of the action under the policy
            AT THE TIME of selection.  Used to compute the PPO ratio.
        reward : float
            The scalar reward received AFTER the FL round completed.
        value_old : float
            The critic's V(s) estimate AT THE TIME of selection (for GAE).
        done : bool
            True if this is the last transition in the current "episode".
            In this project, each T_RL-round interval is treated as one episode.
        """
        self.buffer.append(Transition(
            state_matrix=state_matrix.copy() if isinstance(state_matrix, np.ndarray)
                         else state_matrix.numpy().copy(),
            availability_mask=availability_mask.numpy().copy()
                              if isinstance(availability_mask, torch.Tensor)
                              else np.array(availability_mask),
            action_indices=list(action_indices),
            log_prob_old=float(log_prob_old),
            reward=float(reward),
            value_old=float(value_old),
            done=bool(done),
        ))

    def clear_buffer(self) -> None:
        """Empties the rollout buffer after a PPO update."""
        self.buffer = []

    # -----------------------------------------------------------------------
    # GAE: advantage estimation
    # -----------------------------------------------------------------------

    def _compute_gae_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """
        Computes Generalized Advantage Estimation (GAE) for the stored transitions.

        MATH — GAE (Schulman et al., 2016):
        ------------------------------------
        For a trajectory of T steps with rewards r_t and values V(s_t):

            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
                  (TD error at step t)

            A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
                  (discounted sum of future TD errors)

        where:
            γ (gamma)      = discount factor for future rewards
            λ (gae_lambda) = exponential decay on the advantage estimate

        The returns (value targets) are:
            R_t = A_t + V(s_t)

        Parameters
        ----------
        rewards : List[float], length T
        values  : List[float], length T (critic estimates at each step)
        dones   : List[bool],  length T

        Returns
        -------
        advantages : List[float], length T
        returns    : List[float], length T (value regression targets)
        """
        T = len(rewards)
        advantages = [0.0] * T
        returns = [0.0] * T

        # Bootstrap from 0 at episode end (no future reward after last step)
        gae = 0.0
        next_value = 0.0  # V(s_{T+1}) = 0 at episode boundary

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
                gae = 0.0

            # MATH: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value - values[t]

            # MATH: A_t = δ_t + γ * λ * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return advantages, returns

    # -----------------------------------------------------------------------
    # PPO update step
    # -----------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Runs PPO_EPOCHS gradient update passes over the accumulated rollout buffer.

        Called by runner.py every T_RL rounds (RL policy federation cadence).
        After calling update(), the caller should also call clear_buffer().

        MATH — PPO Clipped Objective (Schulman et al., 2017):
        ------------------------------------------------------
        For each transition t in the buffer, with current policy π_θ:

            r_t = exp(log π_θ(a_t|s_t) - log π_{θ_old}(a_t|s_t))
                  (probability ratio — how much the policy has changed)

            L_clip_t = min(r_t * A_t,
                           clip(r_t, 1-ε, 1+ε) * A_t)
                  (clipped surrogate objective — prevents too-large policy updates)

            L_value_t = (V_θ(s_t) - R_t)^2
                  (value function regression loss)

            L_entropy_t = H(π_θ(·|s_t))
                  (entropy bonus — encourages exploration)

            L_total = -E[L_clip] + c_value * E[L_value] - c_entropy * E[L_entropy]

        The negative signs: we MINIMIZE L_total via gradient descent, but we want
        to MAXIMIZE the clipped surrogate and entropy.

        Returns
        -------
        Dict with keys: 'policy_loss', 'value_loss', 'entropy', 'n_transitions'
            Logged by runner.py for per-round metrics.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_transitions": 0}

        # ----------------------------------------------------------------
        # Step 1: Extract buffer contents and compute GAE advantages
        # ----------------------------------------------------------------
        rewards   = [t.reward     for t in self.buffer]
        values    = [t.value_old  for t in self.buffer]
        dones     = [t.done       for t in self.buffer]

        advantages, returns = self._compute_gae_advantages(rewards, values, dones)

        # Normalize advantages: zero-mean, unit-variance.
        # This is a critical PPO implementation detail: without normalization,
        # advantage magnitudes vary wildly across different reward scales,
        # making the learning rate sensitive to reward engineering choices.
        # NOTE: skip normalization when buffer has only 1 sample — with a single
        # transition std=0, which collapses all advantages to ~0 and produces a
        # near-zero policy_loss even though the gradient update still runs.
        adv_array = np.array(advantages, dtype=np.float32)
        if len(adv_array) > 1:
            adv_mean = adv_array.mean()
            adv_std  = adv_array.std() + 1e-8
            adv_array = (adv_array - adv_mean) / adv_std

        # Convert log_probs_old and returns to tensors
        log_probs_old_t = torch.tensor(
            [t.log_prob_old for t in self.buffer], dtype=torch.float32
        )
        returns_t = torch.tensor(returns, dtype=torch.float32)
        adv_t     = torch.tensor(adv_array, dtype=torch.float32)

        # ----------------------------------------------------------------
        # Step 2: PPO update loop (multiple passes over the buffer)
        # ----------------------------------------------------------------
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        self.policy.train()

        for epoch in range(self.ppo_epochs):
            # Shuffle buffer indices each epoch to decorrelate updates
            indices = np.random.permutation(len(self.buffer))

            for idx in indices:
                transition = self.buffer[idx]
                adv = adv_t[idx]
                ret = returns_t[idx]
                log_prob_old = log_probs_old_t[idx]

                # Rebuild tensors for this transition
                state_matrix = torch.from_numpy(transition.state_matrix).float()
                avail_mask   = torch.from_numpy(transition.availability_mask).bool()

                # Re-evaluate the action under CURRENT policy weights
                log_prob_new, entropy, value_new = self.policy.evaluate_action(
                    state_matrix, avail_mask, transition.action_indices
                )

                # MATH: r_t = exp(log π_new - log π_old)
                ratio = torch.exp(log_prob_new - log_prob_old.detach())

                # MATH: L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
                L_clip_unclipped = ratio * adv
                L_clip_clipped   = torch.clamp(ratio, 1.0 - self.clip_eps,
                                                       1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(L_clip_unclipped, L_clip_clipped)

                # MATH: L_value = (V_θ(s) - R)^2
                value_loss = (value_new - ret) ** 2

                # MATH: L_entropy = -H(π) (negative because we MAXIMIZE entropy)
                entropy_loss = -entropy

                # MATH: L_total = L_clip + c_value * L_value + c_entropy * L_entropy
                total_loss = (
                    policy_loss
                    + self.value_coef  * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping: keeps updates within a safe norm ball
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()

        n_steps = len(self.buffer) * self.ppo_epochs
        stats = {
            "policy_loss":   total_policy_loss / max(n_steps, 1),
            "value_loss":    total_value_loss  / max(n_steps, 1),
            "entropy":       total_entropy     / max(n_steps, 1),
            "n_transitions": len(self.buffer),
        }

        self.last_policy_loss = stats["policy_loss"]
        self.last_value_loss  = stats["value_loss"]
        self.last_entropy     = stats["entropy"]
        self.update_count += 1

        return stats

    def reset(self) -> None:
        """
        Resets trainer state for a fresh experiment seed.
        Reinitializes the optimizer (clears Adam momentum buffers) and
        clears the rollout buffer.
        """
        self.clear_buffer()
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.optimizer.param_groups[0]["lr"], eps=1e-5
        )
        self.update_count = 0
        self.last_policy_loss = 0.0
        self.last_value_loss  = 0.0
        self.last_entropy     = 0.0

    def __repr__(self) -> str:
        return (
            f"PPOTrainer(epochs={self.ppo_epochs}, "
            f"clip={self.clip_eps}, "
            f"buffer={len(self.buffer)}, "
            f"updates={self.update_count})"
        )
