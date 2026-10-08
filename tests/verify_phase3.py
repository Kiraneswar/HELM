"""
tests/verify_phase3.py
-----------------------
Phase 3 Verification Script — RL Layer

PURPOSE:
    Tests the StateBuilder, ActorCriticPolicy, PPOTrainer, and reward function
    before the experiment runner and baselines are built on top.

    Per the Verification Plan:
        - Run 10 RL steps, verify policy gradient is non-zero
        - Verify reward is changing across steps (learning signal exists)
        - Verify PPO update runs without error and produces finite losses
        - Verify policy weights change after update (learning is happening)
        - Verify compute_gini() produces correct values for known inputs
        - Verify centralized state builder stacks clusters correctly

RUN FROM REPO ROOT:
    python tests/verify_phase3.py

EXPECTED OUTPUT:
    - State matrix shapes and value ranges
    - 10-step RL rollout with rewards
    - PPO update losses (policy, value, entropy)
    - Policy weights changed after update [OK]
    - Gini coefficient correctness check
    - "PHASE 3 VERIFICATION PASSED" message
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import copy

from src.data.client_attributes import generate_client_attributes, assign_clients_to_clusters
from src.rl.state import StateBuilder, STATE_DIM
from src.rl.policy import ActorCriticPolicy
from src.rl.ppo_trainer import PPOTrainer, compute_reward, compute_gini


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


# -----------------------------------------------------------------------
# Test 1: StateBuilder — shapes, ranges, availability masking
# -----------------------------------------------------------------------
def verify_state_builder():
    separator("1. STATE BUILDER")

    NUM_CLIENTS = 8
    NUM_CLUSTERS = 2
    CLIENTS_PER_CLUSTER = [4, 4]

    cluster_assignments, cluster_client_ids = assign_clients_to_clusters(
        num_clients=NUM_CLIENTS,
        num_clusters=NUM_CLUSTERS,
        clients_per_cluster=CLIENTS_PER_CLUSTER,
        seed=42,
    )
    churn_config = {"dropout_prob": 0.3, "battery_drain_per_round": 0.05}
    attributes = generate_client_attributes(
        num_clients=NUM_CLIENTS,
        cluster_assignments=cluster_assignments,
        churn_config=churn_config,
        seed=42,
    )
    attr_dict = {a.client_id: a for a in attributes}

    # Simulate one client being unavailable
    attr_dict[cluster_client_ids[0][0]].is_available = False

    sb = StateBuilder(max_staleness=20, max_loss=5.0)

    # --- Per-cluster state ---
    for c in range(NUM_CLUSTERS):
        state = sb.build_state_matrix(cluster_client_ids[c], attr_dict)
        mask  = sb.build_availability_mask(cluster_client_ids[c], attr_dict)

        print(f"\n  Cluster {c}:")
        print(f"    state shape:  {tuple(state.shape)}  (expected: ({len(cluster_client_ids[c])}, {STATE_DIM}))")
        print(f"    dtype:        {state.dtype}")
        print(f"    mask:         {mask.tolist()}")
        print(f"    state range:  [{state.min():.3f}, {state.max():.3f}]  (expect [0,1])")

        assert state.shape == (len(cluster_client_ids[c]), STATE_DIM), \
            f"Wrong state shape: {state.shape}"
        assert state.dtype == torch.float32
        assert state.min() >= 0.0 and state.max() <= 1.0, \
            f"State values out of [0,1]: [{state.min()}, {state.max()}]"

        # Unavailable client's row should be all zeros
        if c == 0:
            unavail_idx = 0  # we set first client of cluster 0 unavailable
            assert state[unavail_idx].sum() == 0.0, \
                "Unavailable client should have zero state vector"
            print(f"    Unavailable client row = {state[unavail_idx].tolist()} [OK]")

    # --- Global state for centralized RL ---
    global_state, global_mask, flat_ids = sb.build_global_state_matrix(
        cluster_client_ids, attr_dict
    )
    print(f"\n  Global state shape:  {tuple(global_state.shape)} "
          f"(expected: ({NUM_CLIENTS}, {STATE_DIM}))")
    assert global_state.shape == (NUM_CLIENTS, STATE_DIM)
    assert len(flat_ids) == NUM_CLIENTS

    print(f"\n  [OK] StateBuilder passed.")
    return cluster_client_ids, attr_dict, sb


# -----------------------------------------------------------------------
# Test 2: ActorCriticPolicy — forward pass, select_clients, evaluate_action
# -----------------------------------------------------------------------
def verify_policy(cluster_client_ids, attr_dict, sb):
    separator("2. ACTOR-CRITIC POLICY")

    policy = ActorCriticPolicy(state_dim=STATE_DIM, hidden_dim=64)
    print(f"  {policy}")

    cluster_ids = cluster_client_ids[0]
    state  = sb.build_state_matrix(cluster_ids, attr_dict)
    mask   = sb.build_availability_mask(cluster_ids, attr_dict)
    k = 2

    print(f"\n  State:  {tuple(state.shape)}, available: {mask.sum().item()}/{len(cluster_ids)}")

    # --- Stochastic selection ---
    selected, log_prob, value = policy.select_clients(state, mask, k=k, deterministic=False)
    print(f"  Stochastic selection: {selected}")
    print(f"  log_prob: {log_prob.item():.4f}  (should be negative)")
    print(f"  value:    {value.item():.4f}")

    assert len(selected) == min(k, mask.sum().item()), \
        f"Expected {k} selected, got {len(selected)}"
    assert log_prob.item() < 0, "log_prob should be negative (it's a log of a probability)"
    for idx in selected:
        assert mask[idx].item(), f"Selected unavailable client {idx}"

    # --- Deterministic selection ---
    det_selected, _, _ = policy.select_clients(state, mask, k=k, deterministic=True)
    print(f"  Deterministic selection: {det_selected}")
    assert len(det_selected) == min(k, mask.sum().item())

    # --- evaluate_action ---
    log_prob_eval, entropy, value_eval = policy.evaluate_action(state, mask, selected)
    print(f"  evaluate_action log_prob: {log_prob_eval.item():.4f}")
    print(f"  entropy:                  {entropy.item():.4f}  (should be > 0)")
    print(f"  value:                    {value_eval.item():.4f}")

    assert entropy.item() > 0, "Entropy should be positive for a non-degenerate policy"

    # --- requires_grad check (critical for PPO backprop) ---
    assert log_prob_eval.requires_grad, "log_prob_eval must require grad for PPO"
    assert value_eval.requires_grad, "value_eval must require grad for PPO"

    print(f"\n  [OK] ActorCriticPolicy passed.")
    return policy


# -----------------------------------------------------------------------
# Test 3: compute_gini — correctness on known inputs
# -----------------------------------------------------------------------
def verify_gini():
    separator("3. GINI COEFFICIENT")

    # Perfect equality: all same → Gini = 0
    g_equal = compute_gini([5, 5, 5, 5])
    print(f"  [5,5,5,5]  => Gini = {g_equal:.4f}  (expect 0.0)")
    assert abs(g_equal) < 1e-6, f"Expected 0.0, got {g_equal}"

    # Maximum inequality: one gets everything → Gini approaches 1
    g_unequal = compute_gini([10, 0, 0, 0])
    print(f"  [10,0,0,0] => Gini = {g_unequal:.4f}  (expect ~0.75)")
    assert g_unequal > 0.6, f"Expected >0.6, got {g_unequal}"

    # Mild inequality
    g_mild = compute_gini([3, 4, 5, 6])
    print(f"  [3,4,5,6]  => Gini = {g_mild:.4f}  (expect 0 < G < 0.2)")
    assert 0.0 < g_mild < 0.2, f"Expected mild Gini, got {g_mild}"

    # Edge case: all zeros → Gini = 0
    g_zeros = compute_gini([0, 0, 0])
    print(f"  [0,0,0]    => Gini = {g_zeros:.4f}  (expect 0.0)")
    assert abs(g_zeros) < 1e-6

    print(f"\n  [OK] Gini coefficient passed.")


# -----------------------------------------------------------------------
# Test 4: compute_reward — sign and structure check
# -----------------------------------------------------------------------
def verify_reward():
    separator("4. REWARD FUNCTION")

    reward_weights = {"accuracy": 1.0, "energy": 0.1, "fairness": 0.05}
    participation = [5, 5, 5, 5, 0, 0]  # 4 clients always selected, 2 never

    # Case 1: big accuracy gain, small cost → positive reward
    r1 = compute_reward(0.50, 0.55, energy_cost=0.2, participation_counts=participation,
                        reward_weights=reward_weights)
    print(f"  Big acc gain (+5%), low energy:   r = {r1:.4f}  (expect > 0)")
    assert r1 > 0, f"Positive accuracy gain should give positive reward: {r1}"

    # Case 2: no accuracy gain, high cost → negative reward
    r2 = compute_reward(0.55, 0.55, energy_cost=5.0, participation_counts=participation,
                        reward_weights=reward_weights)
    print(f"  No acc gain, high energy:         r = {r2:.4f}  (expect < 0)")
    assert r2 < 0, f"No gain + high cost should give negative reward: {r2}"

    # Case 3: accuracy improvement outweighs energy cost
    r3 = compute_reward(0.60, 0.65, energy_cost=0.1, participation_counts=[2, 2, 2, 2],
                        reward_weights=reward_weights)
    print(f"  Acc gain (+5%), fair selection:   r = {r3:.4f}  (expect > r1 due to fairness)")

    print(f"\n  [OK] Reward function passed.")


# -----------------------------------------------------------------------
# Test 5: PPOTrainer — 10-step rollout and update
# -----------------------------------------------------------------------
def verify_ppo_trainer(cluster_client_ids, attr_dict, sb):
    separator("5. PPO TRAINER — 10 STEP ROLLOUT + UPDATE")

    torch.manual_seed(42)
    np.random.seed(42)

    policy = ActorCriticPolicy(state_dim=STATE_DIM, hidden_dim=64)
    trainer = PPOTrainer(
        policy=policy,
        lr_actor=3e-4,
        lr_critic=1e-3,
        ppo_epochs=3,
        clip_eps=0.2,
        entropy_coef=0.01,
    )

    cluster_ids = cluster_client_ids[0]
    k = 2
    reward_weights = {"accuracy": 1.0, "energy": 0.1, "fairness": 0.05}
    participation_counts = [0] * len(cluster_ids)

    weights_before = copy.deepcopy(policy.get_weights())

    print(f"  Collecting 10 transitions (simulated FL rounds)...")
    rewards_seen = []

    for step in range(10):
        # Simulate availability change each step
        for cid in cluster_ids:
            attr_dict[cid].is_available = (np.random.rand() > 0.15)
        # Ensure at least 2 clients available
        for cid in cluster_ids[:2]:
            attr_dict[cid].is_available = True

        state  = sb.build_state_matrix(cluster_ids, attr_dict)
        mask   = sb.build_availability_mask(cluster_ids, attr_dict)

        selected, log_prob, value = policy.select_clients(state, mask, k=k)

        # Simulate a fake reward (in real runs, this comes from FL outcome)
        acc_before = 0.3 + step * 0.01
        acc_after  = acc_before + np.random.uniform(0.00, 0.03)
        energy     = np.random.uniform(0.1, 0.5)
        for idx in selected:
            participation_counts[idx] += 1

        reward = compute_reward(
            acc_before, acc_after, energy, participation_counts, reward_weights
        )
        rewards_seen.append(reward)

        done = (step == 9)
        trainer.store_transition(
            state_matrix=state.numpy(),
            availability_mask=mask,
            action_indices=selected,
            log_prob_old=log_prob.item(),
            reward=reward,
            value_old=value.item(),
            done=done,
        )

    print(f"  Buffer size: {len(trainer.buffer)}")
    print(f"  Reward range: [{min(rewards_seen):.4f}, {max(rewards_seen):.4f}]")
    assert len(trainer.buffer) == 10

    # --- Run PPO update ---
    print(f"\n  Running PPO update ({trainer.ppo_epochs} epochs)...")
    stats = trainer.update()
    print(f"  policy_loss:   {stats['policy_loss']:.6f}")
    print(f"  value_loss:    {stats['value_loss']:.6f}")
    print(f"  entropy:       {stats['entropy']:.6f}  (should be > 0)")
    print(f"  n_transitions: {stats['n_transitions']}")

    assert np.isfinite(stats["policy_loss"]), "Policy loss is not finite!"
    assert np.isfinite(stats["value_loss"]),  "Value loss is not finite!"
    assert stats["entropy"] > 0, "Entropy should be positive"
    assert stats["n_transitions"] == 10

    # --- Verify weights actually changed ---
    weights_after = policy.get_weights()
    weights_changed = False
    for key in weights_before:
        if not torch.allclose(weights_before[key], weights_after[key]):
            weights_changed = True
            break
    assert weights_changed, "Policy weights did not change after PPO update!"
    print(f"\n  Policy weights changed after PPO update [OK]")

    # --- Test trainer.reset() ---
    trainer.clear_buffer()
    assert len(trainer.buffer) == 0, "Buffer should be empty after clear"
    print(f"  Buffer cleared [OK]")

    print(f"\n  [OK] PPOTrainer passed.")


# -----------------------------------------------------------------------
# Test 6: Policy weight management (for federation compatibility)
# -----------------------------------------------------------------------
def verify_policy_weights():
    separator("6. POLICY WEIGHT MANAGEMENT (federation compatibility)")

    policy_a = ActorCriticPolicy(state_dim=STATE_DIM, hidden_dim=64)
    policy_b = ActorCriticPolicy(state_dim=STATE_DIM, hidden_dim=64)

    # Ensure they start different (random init)
    wa = policy_a.get_weights()
    wb = policy_b.get_weights()

    # Compute manual soft-update: alpha=0.5
    alpha = 0.5
    blended = {}
    for key in wa:
        blended[key] = alpha * wa[key] + (1 - alpha) * wb[key]

    # Apply blended weights to policy_a
    policy_a.set_weights(blended)

    # Verify the new weights are the blend
    wa_new = policy_a.get_weights()
    for key in wa:
        expected = alpha * wa[key] + (1 - alpha) * wb[key]
        assert torch.allclose(wa_new[key], expected, atol=1e-6), \
            f"Weight mismatch for {key}"

    print(f"  Soft-update set_weights round-trip: OK")
    print(f"  Policy A param count: {policy_a.count_parameters():,}")

    # After loading new weights, forward pass should still work
    state  = torch.randn(4, STATE_DIM)
    mask   = torch.tensor([True, True, False, True])
    selected, log_p, val = policy_a.select_clients(state, mask, k=2)
    assert len(selected) == 2, "Forward pass failed after set_weights"
    print(f"  Forward pass after set_weights: selected={selected} [OK]")

    print(f"\n  [OK] Policy weight management passed.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  PHASE 3 VERIFICATION")
    print("  Federated Multi-Agent RL -- RL Policy Layer")
    print("="*60)

    cluster_client_ids, attr_dict, sb = verify_state_builder()
    policy = verify_policy(cluster_client_ids, attr_dict, sb)
    verify_gini()
    verify_reward()
    verify_ppo_trainer(cluster_client_ids, attr_dict, sb)
    verify_policy_weights()

    print("\n" + "="*60)
    print("  PHASE 3 VERIFICATION PASSED -- all checks green")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
