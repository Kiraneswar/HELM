"""
rl/__init__.py
--------------
Package initializer for the `rl` (Reinforcement Learning) sub-package.

Architecture position:
    The RL layer sits between the FL layer (src/fl/) and the experiment runner
    (simulation/runner.py).  It is responsible for the per-cluster CLIENT
    SELECTION decision each FL round: observing cluster state, choosing which
    k clients to train, and learning from the resulting reward signal.

    This package contains three modules:
        state.py        — builds the RL observation tensor from raw cluster data
        policy.py       — Actor-Critic network that maps state → client selection
        ppo_trainer.py  — PPO update engine that improves the policy over rounds

Exports:
    StateBuilder, ActorCriticPolicy, PPOTrainer, compute_reward
"""
from .state import StateBuilder
from .policy import ActorCriticPolicy
from .ppo_trainer import PPOTrainer, compute_reward

__all__ = ["StateBuilder", "ActorCriticPolicy", "PPOTrainer", "compute_reward"]
