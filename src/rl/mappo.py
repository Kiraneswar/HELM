"""
src/rl/mappo.py  (Phase 6 / L4)
-------------------------------
MAPPO-style backbone: decentralized per-cluster ACTORS with a single CENTRALIZED
CRITIC, giving proper multi-agent credit assignment — instead of the independent
per-cluster PPO critics used by the `federated_marl` method.

Design (centralized training, decentralized execution):
  - Each edge cluster keeps its own ActorCriticPolicy; only its ACTOR is used to
    select clients (the policy's own critic head is ignored here).
  - One CentralCritic V(s_global) estimates the value of the GLOBAL state
    (mean-pooled per-client features across all clusters).
  - Advantages are computed once per round from the central critic and the global
    reward, then SHARED across all cluster actors — this is the credit-assignment
    signal a set of independent PPO critics cannot provide.
  - The policy-federation novelty is preserved: the actors are still federated
    across clusters every T_RL rounds (handled by the runner via the aggregator).

This is a faithful, self-contained MAPPO variant; it deliberately reuses the
existing ActorCriticPolicy actor and GAE math so the comparison to `federated_marl`
isolates exactly one change: independent critics -> one centralized critic.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

from src.rl.policy import ActorCriticPolicy


class CentralCritic(nn.Module):
    """V(s_global): maps a pooled global state summary to a scalar value."""
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        return self.net(global_feat).squeeze(-1)


def global_summary(cluster_states: Dict[int, np.ndarray]) -> np.ndarray:
    """Mean-pool per-client state vectors across all clusters -> (STATE_DIM,)."""
    rows = [s for s in cluster_states.values() if s is not None and len(s) > 0]
    if not rows:
        # fall back to a zero vector of the right width
        any_s = next(iter(cluster_states.values()))
        return np.zeros(any_s.shape[1], dtype=np.float32)
    allrows = np.concatenate(rows, axis=0)
    return allrows.mean(axis=0).astype(np.float32)


class MAPPOTrainer:
    """
    One trainer for the whole system: holds the per-cluster actors and the shared
    central critic, and runs a centralized-critic PPO update over a rollout buffer.
    """
    def __init__(self, actors: Dict[int, ActorCriticPolicy], state_dim: int,
                 lr_actor=3e-4, lr_critic=1e-3, ppo_epochs=3, clip_eps=0.2,
                 entropy_coef=0.01, gamma=0.95, gae_lambda=0.95, max_grad_norm=0.5,
                 hidden=64, seed=42):
        torch.manual_seed(seed)
        self.actors = actors
        self.critic = CentralCritic(state_dim, hidden=hidden)
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        params = [p for a in actors.values() for p in a.parameters()]
        self.actor_opt = optim.Adam(params, lr=lr_actor, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        # buffer of per-round records
        self.buffer: List[dict] = []
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0

    def central_value(self, global_feat: np.ndarray) -> float:
        with torch.no_grad():
            v = self.critic(torch.from_numpy(global_feat).float())
        return float(v.item())

    def store(self, global_feat, per_cluster, global_reward, value_old, done):
        """per_cluster: list of dicts {c_id,state,mask,action,log_prob_old}."""
        self.buffer.append(dict(global_feat=global_feat, per_cluster=per_cluster,
                                reward=float(global_reward), value_old=float(value_old),
                                done=bool(done)))

    def _gae(self, rewards, values, dones):
        T = len(rewards); adv = [0.0]*T; ret = [0.0]*T; gae = 0.0; next_v = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                next_v = 0.0; gae = 0.0
            delta = rewards[t] + self.gamma*next_v - values[t]
            gae = delta + self.gamma*self.gae_lambda*gae
            adv[t] = gae; ret[t] = gae + values[t]; next_v = values[t]
        return adv, ret

    def update(self) -> dict:
        if not self.buffer:
            return {}
        rewards = [b["reward"] for b in self.buffer]
        values = [b["value_old"] for b in self.buffer]
        dones = [b["done"] for b in self.buffer]
        adv, ret = self._gae(rewards, values, dones)
        adv = np.array(adv, dtype=np.float32)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_t = torch.tensor(ret, dtype=torch.float32)
        gfeat = torch.tensor(np.stack([b["global_feat"] for b in self.buffer]),
                             dtype=torch.float32)

        tot_pl = tot_vl = tot_ent = 0.0; nstep = 0
        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(len(self.buffer))
            for t in idx:
                b = self.buffer[t]
                a_t = torch.tensor(float(adv[t]))
                # actor loss: shared centralized advantage across all clusters
                pol_loss = 0.0; ent_sum = 0.0; ncl = 0
                for tr in b["per_cluster"]:
                    a = self.actors[tr["c_id"]]
                    lp_new, ent, _ = a.evaluate_action(
                        torch.from_numpy(tr["state"]).float(),
                        torch.from_numpy(tr["mask"]).bool(),
                        tr["action"])
                    ratio = torch.exp(lp_new - torch.tensor(tr["log_prob_old"]))
                    unclipped = ratio * a_t
                    clipped = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * a_t
                    pol_loss = pol_loss - torch.min(unclipped, clipped) - self.entropy_coef*ent
                    ent_sum = ent_sum + ent.item(); ncl += 1
                pol_loss = pol_loss / max(ncl, 1)
                # critic loss: central critic regresses the global return
                v_pred = self.critic(gfeat[t])
                val_loss = (v_pred - ret_t[t])**2

                self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
                (pol_loss + 0.5*val_loss).backward()
                for a in self.actors.values():
                    torch.nn.utils.clip_grad_norm_(a.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step(); self.critic_opt.step()
                tot_pl += float(pol_loss.item()); tot_vl += float(val_loss.item())
                tot_ent += ent_sum/max(ncl,1); nstep += 1

        self.last_policy_loss = tot_pl/max(nstep,1)
        self.last_value_loss = tot_vl/max(nstep,1)
        self.last_entropy = tot_ent/max(nstep,1)
        return {"policy_loss": self.last_policy_loss, "value_loss": self.last_value_loss,
                "entropy": self.last_entropy}

    def clear(self):
        self.buffer = []
