"""
src/simulation/runner.py
-------------------------
WHAT THIS FILE DOES:
    The central experiment runner — the top-level orchestrator that connects
    every module built in Phases 1-4 into a complete, runnable FL simulation.

    Given a YAML config (loaded by load_config) and a seed, it:
        1. Builds the synthetic dataset and partitions it across clients
        2. Assigns clients to clusters and generates their system attributes
        3. Creates SimPy network models, EdgeClusters, and the FedAvg aggregator
        4. Instantiates the client-selection method (random / full / oort /
           centralized_rl / federated_marl)
        5. Runs num_rounds FL rounds with the full loop:
               churn → select → train → upload → aggregate → evaluate → log
        6. Handles the DUAL CADENCE scheduling:
               - FL model aggregation every T_FL rounds
               - RL policy updates + federation every T_RL rounds
        7. Logs per-round metrics to a MetricsLogger and saves CSV
        8. Generates matplotlib comparison plots

WHERE IT SITS IN THE ARCHITECTURE:
    This file is the entrypoint for all experiments.  run_experiment.py and
    demo.py both call ExperimentRunner.run().

    Upstream: loads config, instantiates everything from all other modules.
    Downstream: writes results/ CSVs and PNG plots.

DEPENDENCIES:
    Imports from all src/ subpackages: data, environment, fl, rl, baselines, simulation.

KEY DESIGN DECISION:
    We use pure PyTorch (no Flower/flwr) for the simulation loop.  This was a
    deliberate choice, not an oversight.  Flower's client-server round structure
    does not cleanly support independent T_FL/T_RL cadences plus in-round SimPy
    churn/network events.  With pure PyTorch, the runner has full control over
    the round timing, cadence scheduling, and SimPy event injection.  The
    alternative (Flower) was NOT chosen because its abstraction layer would
    require reimplementing the SimPy integration outside Flower's round lifecycle,
    defeating the purpose of using Flower at all.
"""

import os
import copy
import time
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

from src.data.generator import build_client_datasets
from src.data.client_attributes import generate_client_attributes, assign_clients_to_clusters
from src.environment.simpy_network import SimPyNetwork
from src.environment.cluster import EdgeCluster
from src.fl.model import MLPModel
from src.fl.local_trainer import LocalTrainer
from src.fl.aggregator import FedAvgAggregator
from src.rl.state import StateBuilder
from src.rl.policy import ActorCriticPolicy
from src.rl.ppo_trainer import PPOTrainer, compute_reward
from src.baselines.random_selector import RandomSelector
from src.baselines.full_participation import FullParticipationSelector
from src.baselines.oort_selector import OortSelector
from src.baselines.centralized_rl import CentralizedRLSelector
from src.simulation.metrics import MetricsLogger, load_all_results
from src.simulation.energy import client_round_energy, mlp_param_count


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Loads a YAML experiment config file and returns it as a Python dict.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict: parsed config with all nested keys.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_global_seed(seed: int) -> None:
    """
    Sets random seeds for all libraries to ensure reproducibility.

    A single seed controls: numpy, Python random, PyTorch CPU ops,
    and the dataset partition + client attribute generation.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)


# ---------------------------------------------------------------------------
# Environment builder: shared across all methods
# ---------------------------------------------------------------------------

def build_environment(config: dict, seed: int):
    """
    Builds all shared environment objects from config + seed.

    Called once per (method, seed) pair.  Returns everything needed to run
    any of the five selection methods.

    Parameters
    ----------
    config : dict
        Parsed YAML config.
    seed : int
        Seed for this run.

    Returns
    -------
    env : dict containing all environment objects.
    """
    set_global_seed(seed)

    num_clients  = config["num_clients"]
    num_clusters = config["num_clusters"]
    num_classes  = config.get("num_classes", 10)
    num_features = config.get("num_features", 20)
    clients_per_cluster = config.get("clients_per_cluster", None)

    # ── Unified data pipeline ──────────────────────────────────────────────
    # Generate ONE large dataset (train_samples + test_samples) with one seed.
    # The last n_test rows become the fixed held-out test set; the first
    # total_samples rows are partitioned across clients via Dirichlet.
    #
    # WHY ONE CALL: sklearn.make_classification does NOT produce prefix-consistent
    # datasets across different n_samples values — so generate_dataset(N, seed=s)
    # and the first N rows of generate_dataset(N+M, seed=s) are not the same.
    # We must generate all samples in one call, then slice.
    #
    # BUG HISTORY: the original code called generate_dataset(seed=seed+9999) for
    # the test set, which creates a completely independent synthetic problem.
    # A model trained on seed=42 data scores at chance (11%) on seed=9041 data.
    # Fixed here: train and test both come from generate_dataset(seed=seed).
    from src.data.generator import generate_dataset, partition_data_dirichlet, ClientDataset

    total_samples   = (config.get("data", {}).get("num_samples_per_client", 200)
                       * num_clients)
    n_test          = max(500, total_samples // 5)
    n_total_gen     = total_samples + n_test
    dirichlet_alpha = config.get("data", {}).get("dirichlet_alpha", 0.5)

    X_all, y_all = generate_dataset(
        num_samples=n_total_gen,
        num_features=num_features,
        num_classes=num_classes,
        seed=seed,
    )

    # Deterministic train/test split — last n_test rows held out
    X_train_global = X_all[:total_samples]
    y_train_global = y_all[:total_samples]
    X_test_np      = X_all[total_samples:]
    y_test_np      = y_all[total_samples:]

    X_test = torch.from_numpy(X_test_np.copy())
    y_test = torch.from_numpy(y_test_np.copy())

    # Dirichlet partition of training pool across clients
    shards = partition_data_dirichlet(
        X=X_train_global,
        y=y_train_global,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=dirichlet_alpha,
        seed=seed,
    )
    datasets    = [ClientDataset(X=s[0], y=s[1], client_id=k) for k, s in enumerate(shards)]
    dataset_dict = {ds.client_id: ds for ds in datasets}

    # --- Client system attributes + cluster assignment ---
    cluster_assignments, cluster_client_ids = assign_clients_to_clusters(
        num_clients=num_clients,
        num_clusters=num_clusters,
        clients_per_cluster=clients_per_cluster,
        seed=seed,
    )

    churn_config = config.get("churn", {
        "mode": "mild", "dropout_prob": 0.1, "battery_drain_per_round": 0.05
    })
    attributes = generate_client_attributes(
        num_clients=num_clients,
        cluster_assignments=cluster_assignments,
        churn_config=churn_config,
        seed=seed,
    )
    attr_dict = {a.client_id: a for a in attributes}

    # --- SimPy network models (one per cluster) ---
    net_cfg = config.get("network", {})
    networks = [
        SimPyNetwork(
            bandwidth_contention=True,
            round_deadline=net_cfg.get("round_deadline", 30.0),
            seed=seed + c,
        )
        for c in range(num_clusters)
    ]

    # --- Edge clusters ---
    clusters = [
        EdgeCluster(
            cluster_id=c,
            client_ids=cluster_client_ids[c],
            client_datasets=dataset_dict,
            client_attributes=attr_dict,
            network=networks[c],
            churn_config=churn_config,
            seed=seed + c * 1000,
        )
        for c in range(num_clusters)
    ]

    # --- FL model + aggregator ---
    fl_cfg   = config.get("fl", {})
    fed_cfg  = config.get("federation", {})
    agg = FedAvgAggregator(
        num_features=num_features,
        num_classes=num_classes,
        hidden1=fl_cfg.get("hidden1", 128),
        hidden2=fl_cfg.get("hidden2", 64),
        federation_alpha=fed_cfg.get("alpha", 0.5),
        federation_mode=fed_cfg.get("mode", "soft"),
        federation_dp_noise_std=fed_cfg.get("dp_noise_std", 0.0),     # L9
        federation_dp_clip=fed_cfg.get("dp_clip", 0.0),               # L9
        federation_personalize=fed_cfg.get("personalize", False),     # L10
        federation_local_keys=tuple(fed_cfg.get("local_keys", ("actor_head", "critic_head"))),
    )

    # --- Local trainer (shared across all clients, stateless) ---
    trainer = LocalTrainer(
        num_features=num_features,
        num_classes=num_classes,
        local_epochs=fl_cfg.get("local_epochs", 2),
        batch_size=fl_cfg.get("batch_size", 32),
        lr=fl_cfg.get("lr", 0.01),
        hidden1=fl_cfg.get("hidden1", 128),
        hidden2=fl_cfg.get("hidden2", 64),
    )

    # --- Energy model constants (L2 fix): grounded compute+comm energy ---
    model_params = mlp_param_count(
        num_features=num_features,
        hidden1=fl_cfg.get("hidden1", 128),
        hidden2=fl_cfg.get("hidden2", 64),
        num_classes=num_classes,
    )
    model_size_bytes = model_params * 4  # float32 weights uploaded per round

    return {
        "datasets": dataset_dict,
        "X_test": X_test,
        "y_test": y_test,
        "attr_dict": attr_dict,
        "cluster_client_ids": cluster_client_ids,
        "clusters": clusters,
        "aggregator": agg,
        "trainer": trainer,
        "num_clients": num_clients,
        "num_clusters": num_clusters,
        "num_classes": num_classes,
        "num_features": num_features,
        "churn_config": churn_config,
        "model_params": model_params,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_bytes / 1e6,
        "energy_cfg": config.get("energy", {}),
        "local_epochs": fl_cfg.get("local_epochs", 2),
        "_round_client_energy": {},  # cid -> Joules for the current round (L2)
    }


# ---------------------------------------------------------------------------
# Main experiment runner class
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Runs a complete FL experiment for one method and one seed.

    Handles all five methods:
        'random'          — RandomSelector
        'full'            — FullParticipationSelector
        'oort'            — OortSelector
        'centralized_rl'  — CentralizedRLSelector (single global policy)
        'federated_marl'  — Per-cluster ActorCriticPolicy + RL policy federation

    Parameters
    ----------
    config : dict
        Parsed YAML experiment configuration.
    method : str
        Client-selection method to run.
    results_dir : str
        Directory to save CSV logs and plots.
    verbose : bool
        If True, print per-round progress to stdout.
    """

    def __init__(
        self,
        config: dict,
        method: str,
        results_dir: str = "results",
        verbose: bool = True,
    ):
        self.config = config
        self.method = method
        self.results_dir = results_dir
        self.verbose = verbose

    # -----------------------------------------------------------------------
    # Public API: run one seed
    # -----------------------------------------------------------------------

    def run(self, seed: int) -> MetricsLogger:
        """
        Runs the full experiment for one seed.  Returns a populated MetricsLogger.

        Parameters
        ----------
        seed : int
            Random seed for this run.

        Returns
        -------
        MetricsLogger with all rounds logged.
        """
        t_start = time.time()

        # Build environment
        env = build_environment(self.config, seed)

        # Build method-specific selector(s)
        selector_ctx = self._build_selector(env, seed)

        # Create metrics logger
        logger = MetricsLogger(method=self.method, seed=seed)

        # Scheduling cadences
        T_FL  = self.config.get("T_FL", 1)
        T_RL  = self.config.get("T_RL", 5)
        # update_every: PPO flush cadence — held fixed across CRL and FMARL
        # so that both methods see the same number of gradient updates.
        # T_RL on federated_marl now controls ONLY policy federation frequency.
        update_every = self.config.get("rl", {}).get("update_every_n_rounds", T_RL)
        num_rounds = self.config.get("num_rounds", 30)
        k_select = self.config.get("rl", {}).get("k_select", 5)
        rl_round_counter = 0

        # Participation tracking across all clients (for Gini)
        all_participation = [0] * env["num_clients"]
        # Flat index for participation tracking
        flat_client_order = [
            cid
            for cluster_ids in env["cluster_client_ids"]
            for cid in cluster_ids
        ]
        cid_to_flat = {cid: i for i, cid in enumerate(flat_client_order)}

        acc_before = 0.0
        _, acc_before = env["aggregator"].evaluate_global_model(env["X_test"], env["y_test"])

        if self.verbose:
            print(f"\n  Method={self.method}  seed={seed}  rounds={num_rounds}")
            print(f"  {'Round':>5}  {'Acc':>7}  {'Loss':>7}  {'Energy':>8}  "
                  f"{'Gini':>6}  {'Selected':>10}  {'Arrived':>8}")
            print(f"  {'-'*60}")

        for fl_round in range(1, num_rounds + 1):

            # ── 1. Apply churn: update client availability ──────────────────
            for cluster in env["clusters"]:
                cluster.apply_round_start_events()

            # ── 2. Select clients ─────────────────────────────────────────
            (cluster_selections, log_probs, values,
             sel_log_meta) = self._select_clients(
                selector_ctx=selector_ctx,
                env=env,
                k_select=k_select,
                acc_before=acc_before,
                fl_round=fl_round,
            )

            # ── 3. Local training ─────────────────────────────────────────
            # client_updates maps cid -> (weights, n_samples) so we can later
            # filter which updates actually ARRIVED before aggregating (L5 fix).
            global_weights = env["aggregator"].get_global_weights()
            client_updates = {}          # cid -> (w, n)
            total_energy = 0.0
            round_client_energy = {}     # cid -> Joules this round (L2 grounded model)

            for c_id, selected_cids in cluster_selections.items():
                for cid in selected_cids:
                    w, loss_k, n = env["trainer"].train(
                        global_weights, env["datasets"][cid], cid
                    )
                    env["attr_dict"][cid].last_local_loss = loss_k
                    client_updates[cid] = (w, n)
                    # L2: grounded compute+comm energy (Joules), not a fixed proxy.
                    # Energy is spent whether or not the upload later beats the
                    # deadline (the device already trained + attempted to send).
                    e_cid = client_round_energy(
                        n_samples=n,
                        local_epochs=env["local_epochs"],
                        model_params=env["model_params"],
                        model_size_bytes=env["model_size_bytes"],
                        compute_speed=env["attr_dict"][cid].compute_speed,
                        cfg=env["energy_cfg"],
                    )
                    round_client_energy[cid] = e_cid
                    total_energy += e_cid
                    all_participation[cid_to_flat[cid]] += 1

            env["_round_client_energy"] = round_client_energy

            # ── 4. SimPy upload simulation → determine ARRIVALS ───────────
            # L5 fix: the network model now actually decides which updates are
            # aggregated. Stragglers (uploads that miss the round deadline) are
            # dropped from aggregation, not silently kept.
            total_arrived = 0
            total_comm_cost = 0.0
            arrived_global = set()

            for c_id, selected_cids in cluster_selections.items():
                if not selected_cids:
                    continue
                cluster = env["clusters"][c_id]
                arrived_ids, upload_times, comm_cost = cluster.simulate_uploads(
                    selected_cids,
                    model_size_mb=env["model_size_mb"],
                )
                total_arrived += len(arrived_ids)
                total_comm_cost += comm_cost
                arrived_global.update(arrived_ids)

            total_selected_this_round = sum(len(v) for v in cluster_selections.values())
            num_stragglers = total_selected_this_round - total_arrived

            # Only updates that ARRIVED before the deadline are aggregated (L5).
            arrived_updates = [client_updates[cid] for cid in arrived_global
                               if cid in client_updates]

            # ── 5. FL model aggregation (every T_FL rounds) ───────────────
            if fl_round % T_FL == 0 and arrived_updates:
                env["aggregator"].aggregate_fl_updates(arrived_updates)

            # ── 6. Evaluate global model ──────────────────────────────────
            test_loss, acc_after = env["aggregator"].evaluate_global_model(
                env["X_test"], env["y_test"]
            )

            # ── 7. RL update + policy federation (every T_RL rounds) ──────
            # NOVELTY: dual cadence scheduling — RL policies update on their
            # own T_RL schedule, independent of the T_FL FL aggregation cadence.
            policy_loss_val = 0.0
            value_loss_val  = 0.0
            entropy_val     = 0.0
            policy_sim      = 0.0
            mean_reward_val = float("nan")

            if self.method in ("federated_marl", "centralized_rl", "mappo_marl"):
                (policy_loss_val, value_loss_val, entropy_val, policy_sim,
                 mean_reward_val) = (
                    self._update_rl(
                        selector_ctx=selector_ctx,
                        env=env,
                        cluster_selections=cluster_selections,
                        log_probs=log_probs,
                        values=values,
                        acc_before=acc_before,
                        acc_after=acc_after,
                        total_energy=total_energy,
                        all_participation=all_participation,
                        fl_round=fl_round,
                        T_RL=T_RL,
                        update_every=update_every,
                    )
                )
                rl_round_counter = fl_round // T_RL

            # ── 8. Apply round-end events (battery drain, staleness) ───────
            total_selected = 0
            for c_id, selected_cids in cluster_selections.items():
                env["clusters"][c_id].apply_round_end_events(selected_cids)
                total_selected += len(selected_cids)

            # ── 9. Log metrics ────────────────────────────────────────────
            record = logger.log_round(
                fl_round=fl_round,
                accuracy=acc_after,
                loss=test_loss,
                energy_cost=total_energy,
                participation_counts=list(all_participation),
                num_selected=total_selected,
                num_arrived=total_arrived if total_arrived > 0 else total_selected,
                comm_cost=total_comm_cost,
                policy_loss=policy_loss_val,
                value_loss=value_loss_val,
                rl_entropy=entropy_val,
                policy_similarity=policy_sim,
                mean_reward=mean_reward_val,
                num_stragglers=num_stragglers,
            )

            if self.verbose:
                print(
                    f"  {fl_round:>5}  {acc_after:>6.2%}  "
                    f"{test_loss:>7.4f}  {total_energy:>8.3f}  "
                    f"{record['gini']:>6.3f}  {total_selected:>10}  "
                    f"{record['num_arrived']:>8}"
                )

            acc_before = acc_after

        # Save CSV
        csv_path = logger.save_csv(self.results_dir)
        elapsed = time.time() - t_start

        if self.verbose:
            print(f"\n  Done in {elapsed:.1f}s — CSV saved: {csv_path}")
            print(f"  Final accuracy: {logger.get_final_accuracy():.2%}  "
                  f"Total energy: {logger.get_final_energy():.3f}  "
                  f"Final Gini: {logger.get_final_gini():.4f}")

        return logger

    # -----------------------------------------------------------------------
    # Multi-seed runner
    # -----------------------------------------------------------------------

    def run_multi_seed(self, num_seeds: int = None) -> List[MetricsLogger]:
        """
        Runs the experiment for multiple seeds and returns all loggers.

        Parameters
        ----------
        num_seeds : int, optional
            Overrides config['num_seeds'] if provided.

        Returns
        -------
        List[MetricsLogger]: one per seed.
        """
        n = num_seeds or self.config.get("num_seeds", 1)
        base_seed = self.config.get("seed", 42)
        loggers = []

        for i in range(n):
            seed = base_seed + i
            print(f"\n[Seed {i+1}/{n}  (seed={seed})]")
            loggers.append(self.run(seed=seed))

        return loggers

    # -----------------------------------------------------------------------
    # Selector builders (method-specific)
    # -----------------------------------------------------------------------

    def _build_selector(self, env: dict, seed: int) -> dict:
        """
        Instantiates the client-selection mechanism for the configured method.

        Returns a context dict with everything the selector needs at runtime.
        """
        method = self.method
        rl_cfg = self.config.get("rl", {})
        rw = rl_cfg.get("reward_weights", {"accuracy": 1.0, "energy": 0.1, "fairness": 0.05})
        hidden_dim = 64

        if method == "random":
            return {"selector": RandomSelector(seed=seed)}

        elif method == "full":
            return {"selector": FullParticipationSelector()}

        elif method == "oort":
            return {"selector": OortSelector(
                recency_weight=0.5,
                min_rounds_between_selections=1,
                seed=seed,
            )}

        elif method == "centralized_rl":
            sel = CentralizedRLSelector(
                all_cluster_client_ids=env["cluster_client_ids"],
                num_clients_total=env["num_clients"],
                k_per_cluster=rl_cfg.get("k_select", 5),
                rl_config={
                    **rl_cfg,
                    "T_RL": self.config.get("T_RL", 5),
                    # Pass update_every_n_rounds explicitly so CRL uses the
                    # same PPO cadence as federated_marl, not T_RL.
                    "update_every_n_rounds": rl_cfg.get(
                        "update_every_n_rounds", self.config.get("T_RL", 5)
                    ),
                },
                reward_weights=rw,
                hidden_dim=hidden_dim,
                seed=seed,
            )
            return {"selector": sel}

        elif method == "federated_marl":
            # NOVELTY: one ActorCriticPolicy + PPOTrainer per edge cluster
            from src.rl.state import STATE_DIM
            torch.manual_seed(seed)

            policies = {}
            trainers = {}
            for c in range(env["num_clusters"]):
                pol = ActorCriticPolicy(state_dim=STATE_DIM, hidden_dim=hidden_dim)
                trn = PPOTrainer(
                    policy=pol,
                    lr_actor=rl_cfg.get("lr_actor", 3e-4),
                    lr_critic=rl_cfg.get("lr_critic", 1e-3),
                    ppo_epochs=rl_cfg.get("ppo_epochs", 3),
                    clip_eps=rl_cfg.get("clip_eps", 0.2),
                    entropy_coef=rl_cfg.get("entropy_coef", 0.01),
                )
                policies[c] = pol
                trainers[c] = trn

            state_builder = StateBuilder()
            return {
                "policies": policies,
                "trainers": trainers,
                "state_builder": state_builder,
                "reward_weights": rw,
                # Store acc_before per cluster for reward computation
                "_acc_before": 0.0,
                "_last_states": {},
                "_last_masks": {},
                "_last_actions": {},
                "_last_log_probs": {},
                "_last_values": {},
            }

        elif method == "mappo_marl":
            # L4: per-cluster actors + ONE centralized critic (credit assignment).
            from src.rl.state import STATE_DIM
            from src.rl.mappo import MAPPOTrainer
            torch.manual_seed(seed)
            actors = {c: ActorCriticPolicy(state_dim=STATE_DIM, hidden_dim=hidden_dim)
                      for c in range(env["num_clusters"])}
            trainer = MAPPOTrainer(
                actors=actors, state_dim=STATE_DIM,
                lr_actor=rl_cfg.get("lr_actor", 3e-4),
                lr_critic=rl_cfg.get("lr_critic", 1e-3),
                ppo_epochs=rl_cfg.get("ppo_epochs", 3),
                clip_eps=rl_cfg.get("clip_eps", 0.2),
                entropy_coef=rl_cfg.get("entropy_coef", 0.01),
                hidden=hidden_dim, seed=seed,
            )
            return {
                "actors": actors,
                "mappo": trainer,
                "state_builder": StateBuilder(),
                "reward_weights": rw,
                "_last_states": {},
                "_last_masks": {},
                "_last_actions": {},
                "_last_log_probs": {},
                "_last_global_feat": None,
                "_last_central_value": 0.0,
            }

        else:
            raise ValueError(f"Unknown method: {method!r}. Choose from: random, full, "
                             f"oort, centralized_rl, federated_marl, mappo_marl")

    # -----------------------------------------------------------------------
    # Per-round selection dispatch
    # -----------------------------------------------------------------------

    def _select_clients(
        self,
        selector_ctx: dict,
        env: dict,
        k_select: int,
        acc_before: float,
        fl_round: int,
    ) -> Tuple[Dict[int, List[int]], Dict, Dict, Dict]:
        """
        Dispatches client selection to the appropriate method.

        Returns:
            cluster_selections : Dict[cluster_id → List[client_ids]]
            log_probs          : Dict[cluster_id → log_prob tensor] (RL methods)
            values             : Dict[cluster_id → value tensor] (RL methods)
            meta               : Dict for any extra logging data
        """
        method = self.method
        cluster_selections = {}
        log_probs = {}
        values = {}
        meta = {}

        if method == "random":
            sel = selector_ctx["selector"]
            for cluster in env["clusters"]:
                avail = cluster.get_available_clients()
                selected = sel.select_clients(avail, k=k_select)
                cluster_selections[cluster.cluster_id] = selected

        elif method == "full":
            sel = selector_ctx["selector"]
            for cluster in env["clusters"]:
                avail = cluster.get_available_clients()
                selected = sel.select_clients(avail)
                cluster_selections[cluster.cluster_id] = selected

        elif method == "oort":
            sel = selector_ctx["selector"]
            for cluster in env["clusters"]:
                avail = cluster.get_available_clients()
                selected = sel.select_clients(
                    avail, k=k_select, client_attributes=env["attr_dict"]
                )
                cluster_selections[cluster.cluster_id] = selected
                sel.on_round_end(
                    selected_ids=selected,
                    all_client_ids=cluster.client_ids,
                )

        elif method == "centralized_rl":
            sel = selector_ctx["selector"]
            cs, lp, val = sel.select_clients(
                client_attributes=env["attr_dict"],
                acc_before=acc_before,
            )
            cluster_selections = cs
            log_probs = {0: lp}   # centralized: single log_prob
            values    = {0: val}
            # Ensure all clusters have an entry
            for c in range(env["num_clusters"]):
                if c not in cluster_selections:
                    cluster_selections[c] = []

        elif method == "federated_marl":
            # Per-cluster RL selection
            sb = selector_ctx["state_builder"]
            policies = selector_ctx["policies"]

            for cluster in env["clusters"]:
                c_id = cluster.cluster_id
                state  = sb.build_state_matrix(cluster.client_ids, env["attr_dict"])
                mask   = sb.build_availability_mask(cluster.client_ids, env["attr_dict"])

                policy_indices, lp, val = policies[c_id].select_clients(
                    state, mask, k=k_select
                )
                # Map policy output indices (positions in cluster.client_ids)
                # back to actual client IDs
                actual_cids = [cluster.client_ids[i] for i in policy_indices]
                cluster_selections[c_id] = actual_cids
                log_probs[c_id] = lp
                values[c_id]    = val

                # Store for PPO update later
                selector_ctx["_last_states"][c_id]    = state.numpy()
                selector_ctx["_last_masks"][c_id]     = mask.numpy()
                selector_ctx["_last_actions"][c_id]   = policy_indices
                selector_ctx["_last_log_probs"][c_id] = lp.item()
                selector_ctx["_last_values"][c_id]    = val.item()

        elif method == "mappo_marl":
            # Per-cluster ACTOR selection; value comes from the central critic.
            from src.rl.mappo import global_summary
            sb = selector_ctx["state_builder"]
            actors = selector_ctx["actors"]
            states_np = {}
            for cluster in env["clusters"]:
                c_id = cluster.cluster_id
                state = sb.build_state_matrix(cluster.client_ids, env["attr_dict"])
                mask  = sb.build_availability_mask(cluster.client_ids, env["attr_dict"])
                policy_indices, lp, _val = actors[c_id].select_clients(state, mask, k=k_select)
                actual_cids = [cluster.client_ids[i] for i in policy_indices]
                cluster_selections[c_id] = actual_cids
                log_probs[c_id] = lp
                selector_ctx["_last_states"][c_id]    = state.numpy()
                selector_ctx["_last_masks"][c_id]     = mask.numpy()
                selector_ctx["_last_actions"][c_id]   = policy_indices
                selector_ctx["_last_log_probs"][c_id] = lp.item()
                states_np[c_id] = state.numpy()
            gfeat = global_summary(states_np)
            selector_ctx["_last_global_feat"] = gfeat
            selector_ctx["_last_central_value"] = selector_ctx["mappo"].central_value(gfeat)

        return cluster_selections, log_probs, values, meta

    # -----------------------------------------------------------------------
    # RL update dispatcher (handles both centralized and federated_marl)
    # -----------------------------------------------------------------------

    def _update_rl(
        self,
        selector_ctx: dict,
        env: dict,
        cluster_selections: Dict[int, List],
        log_probs: Dict,
        values: Dict,
        acc_before: float,
        acc_after: float,
        total_energy: float,
        all_participation: List[int],
        fl_round: int,
        T_RL: int,
        update_every: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Stores RL transitions and runs PPO updates / policy federation on
        their independent schedules.

        PPO updates fire every `update_every` rounds (same for CRL and FMARL).
        Policy federation fires every `T_RL` rounds (FMARL only).
        These two events are now fully decoupled.

        Returns (policy_loss, value_loss, entropy, policy_similarity, mean_reward).
        policy_similarity is np.nan on rounds where federation did not fire;
        mean_reward is the RL reward this round (for the learning-curve, L7).
        """
        rw = (selector_ctx.get("reward_weights")
              or self.config.get("rl", {}).get("reward_weights",
                                               {"accuracy": 1.0, "energy": 0.1, "fairness": 0.05}))
        method = self.method

        policy_loss = 0.0
        value_loss  = 0.0
        entropy     = 0.0
        policy_sim  = 0.0
        mean_reward = float("nan")
        round_energy = env.get("_round_client_energy", {})

        if method == "centralized_rl":
            sel = selector_ctx["selector"]
            lp  = log_probs.get(0)
            val = values.get(0)
            sel.on_round_end(
                acc_after=acc_after,
                energy_cost=total_energy,
                all_participation_counts=all_participation,
                log_prob=lp.item() if lp is not None else 0.0,
                value=val.item() if val is not None else 0.0,
            )
            mean_reward = getattr(sel, "last_reward", float("nan"))
            # Only read the trainer's loss stats in the round PPO actually ran.
            # Without this guard the stale last_policy_loss would carry forward
            # into every round after the first PPO update, making 96/100 rows
            # look like PPO updates when only ~(100/update_every) actually fired.
            if sel.last_update_fired:
                policy_loss = sel.trainer.last_policy_loss
                value_loss  = sel.trainer.last_value_loss
                entropy     = sel.trainer.last_entropy

        elif method == "federated_marl":
            trainers = selector_ctx["trainers"]
            policies = selector_ctx["policies"]

            # ── Decoupled cadences ──────────────────────────────────────────
            # ppo_done:  flush PPO buffer + run gradient updates
            # fed_done:  federate policy weights across clusters
            # These are now independent: PPO can fire without federation
            # and federation can happen on a round PPO did not fire on.
            ppo_done = (fl_round % update_every == 0)
            fed_done = (fl_round % T_RL == 0)

            # Store one transition per cluster
            _cluster_rewards = []
            for c_id, cluster in enumerate(env["clusters"]):
                selected_cids = cluster_selections.get(c_id, [])
                # L2: per-cluster energy uses the grounded per-client energy (J)
                # computed this round, not the old drain proxy.
                energy_c = sum(round_energy.get(cid, 0.0) for cid in selected_cids)
                participation_c = [
                    env["attr_dict"][cid].participation_count
                    for cid in cluster.client_ids
                ]

                reward = compute_reward(
                    acc_before=acc_before,
                    acc_after=acc_after,
                    energy_cost=energy_c,
                    participation_counts=participation_c,
                    reward_weights=rw,
                )
                _cluster_rewards.append(reward)

                if c_id in selector_ctx["_last_states"]:
                    trainers[c_id].store_transition(
                        state_matrix=selector_ctx["_last_states"][c_id],
                        availability_mask=selector_ctx["_last_masks"][c_id],
                        action_indices=selector_ctx["_last_actions"][c_id],
                        log_prob_old=selector_ctx["_last_log_probs"][c_id],
                        reward=reward,
                        value_old=selector_ctx["_last_values"][c_id],
                        done=ppo_done,  # signals end-of-episode for GAE
                    )

            # ── PPO update (every update_every rounds) ──────────────────────
            if ppo_done:
                all_stats = []
                for c_id in range(env["num_clusters"]):
                    if len(trainers[c_id].buffer) > 0:
                        stats = trainers[c_id].update()
                        trainers[c_id].clear_buffer()
                        all_stats.append(stats)

                if all_stats:
                    policy_loss = np.mean([s["policy_loss"] for s in all_stats])
                    value_loss  = np.mean([s["value_loss"]  for s in all_stats])
                    entropy     = np.mean([s["entropy"]     for s in all_stats])

            # ── Policy federation (every T_RL rounds) ── NOVELTY ───────────
            # policy_sim is NaN on non-federation rounds so the plot does not
            # zero-fill the metric between actual federation events.
            if fed_done:
                policy_weights = {c_id: policies[c_id].get_weights()
                                  for c_id in range(env["num_clusters"])}

                updated_weights = env["aggregator"].federate_policies(
                    cluster_policy_weights=policy_weights,
                    rl_round=fl_round // T_RL,
                )

                # Cosine similarity between first two cluster policies (post-federation)
                if len(updated_weights) >= 2:
                    ids = list(updated_weights.keys())
                    policy_sim = env["aggregator"].compute_policy_similarity(
                        updated_weights[ids[0]], updated_weights[ids[1]]
                    )
                else:
                    policy_sim = np.nan

                # Push federated weights back to each cluster's local policy
                for c_id, new_weights in updated_weights.items():
                    policies[c_id].set_weights(new_weights)
            else:
                # Not a federation round — log NaN so the plot is not zero-filled
                policy_sim = np.nan

            if _cluster_rewards:
                mean_reward = float(np.mean(_cluster_rewards))

        elif method == "mappo_marl":
            trainer = selector_ctx["mappo"]
            actors = selector_ctx["actors"]
            ppo_done = (fl_round % update_every == 0)
            fed_done = (fl_round % T_RL == 0)

            # per-cluster rewards (grounded energy), global reward = their mean
            cluster_rewards = []
            per_cluster = []
            for c_id, cluster in enumerate(env["clusters"]):
                selected_cids = cluster_selections.get(c_id, [])
                energy_c = sum(round_energy.get(cid, 0.0) for cid in selected_cids)
                participation_c = [env["attr_dict"][cid].participation_count
                                   for cid in cluster.client_ids]
                r = compute_reward(acc_before=acc_before, acc_after=acc_after,
                                   energy_cost=energy_c, participation_counts=participation_c,
                                   reward_weights=rw)
                cluster_rewards.append(r)
                if c_id in selector_ctx["_last_states"]:
                    per_cluster.append(dict(
                        c_id=c_id,
                        state=selector_ctx["_last_states"][c_id],
                        mask=selector_ctx["_last_masks"][c_id],
                        action=selector_ctx["_last_actions"][c_id],
                        log_prob_old=selector_ctx["_last_log_probs"][c_id]))
            global_reward = float(np.mean(cluster_rewards)) if cluster_rewards else 0.0
            mean_reward = global_reward
            if selector_ctx["_last_global_feat"] is not None and per_cluster:
                trainer.store(global_feat=selector_ctx["_last_global_feat"],
                              per_cluster=per_cluster, global_reward=global_reward,
                              value_old=selector_ctx["_last_central_value"], done=ppo_done)

            if ppo_done and trainer.buffer:
                stats = trainer.update(); trainer.clear()
                if stats:
                    policy_loss = stats["policy_loss"]; value_loss = stats["value_loss"]
                    entropy = stats["entropy"]

            # policy federation on the ACTORS (novelty preserved)
            if fed_done:
                aw = {c_id: actors[c_id].get_weights() for c_id in range(env["num_clusters"])}
                updated = env["aggregator"].federate_policies(aw, rl_round=fl_round // T_RL)
                if len(updated) >= 2:
                    ids = list(updated.keys())
                    policy_sim = env["aggregator"].compute_policy_similarity(
                        updated[ids[0]], updated[ids[1]])
                else:
                    policy_sim = np.nan
                for c_id, nw in updated.items():
                    actors[c_id].set_weights(nw)
            else:
                policy_sim = np.nan

        return policy_loss, value_loss, entropy, policy_sim, mean_reward


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

# Colour palette: consistent across all plots and the dashboard
METHOD_COLORS = {
    "random":           "#9E9E9E",  # grey
    "full":             "#4CAF50",  # green
    "oort":             "#FF9800",  # orange
    "centralized_rl":   "#2196F3",  # blue
    "federated_marl":   "#E91E63",  # pink/magenta (hero method)
}

METHOD_LABELS = {
    "random":           "Random",
    "full":             "Full Participation",
    "oort":             "Oort (Heuristic)",
    "centralized_rl":   "Centralized RL",
    "federated_marl":   "Federated MARL (Ours)",
}


def plot_accuracy_curves(
    results_dir: str,
    methods: List[str],
    num_seeds: int,
    output_path: str,
    title: str = "Global Model Accuracy vs. FL Round",
) -> None:
    """
    Plots per-round accuracy (mean ± std) for all methods on one figure.

    Parameters
    ----------
    results_dir : str
        Directory containing per-seed CSV files.
    methods : List[str]
        Methods to include in the plot (must have corresponding CSVs).
    num_seeds : int
        Number of seeds to aggregate.
    output_path : str
        Full path for the output PNG.
    title : str
        Plot title.
    """
    df_all = load_all_results(results_dir, methods, num_seeds)
    if df_all.empty:
        print(f"  [WARNING] No results found in {results_dir} for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        df_m = df_all[df_all["method"] == method]
        if df_m.empty:
            continue
        color  = METHOD_COLORS.get(method, "#333333")
        label  = METHOD_LABELS.get(method, method)
        rounds = df_m["fl_round"].values
        mean   = df_m["accuracy_mean"].values * 100  # to percentage
        std    = df_m["accuracy_std"].fillna(0).values * 100

        ax.plot(rounds, mean, label=label, color=color, linewidth=2)
        if num_seeds > 1:
            ax.fill_between(rounds, mean - std, mean + std,
                            alpha=0.15, color=color)

    ax.set_xlabel("FL Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


def plot_cost_accuracy_tradeoff(
    results_dir: str,
    methods: List[str],
    num_seeds: int,
    output_path: str,
) -> None:
    """
    Plots total energy cost vs. final accuracy for all methods.

    Each method is a single point (mean across seeds).  Ideal position:
    top-left (high accuracy, low energy cost).

    Parameters
    ----------
    results_dir, methods, num_seeds, output_path : same as plot_accuracy_curves.
    """
    df_all = load_all_results(results_dir, methods, num_seeds)
    if df_all.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in methods:
        df_m = df_all[df_all["method"] == method]
        if df_m.empty:
            continue
        # Final accuracy = last round's mean accuracy
        final_acc  = df_m.iloc[-1]["accuracy_mean"] * 100
        total_energy = df_m["energy_mean"].sum()
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        ax.scatter(total_energy, final_acc, color=color, s=120, zorder=5)
        ax.annotate(label, (total_energy, final_acc),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Total Cumulative Energy Cost", fontsize=13)
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs. Energy Cost Tradeoff", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


def plot_fairness_curves(
    results_dir: str,
    methods: List[str],
    num_seeds: int,
    output_path: str,
) -> None:
    """
    Plots the Gini coefficient (fairness) over FL rounds for all methods.

    Lower Gini = more fair client participation.
    """
    df_all = load_all_results(results_dir, methods, num_seeds)
    if df_all.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for method in methods:
        df_m = df_all[df_all["method"] == method]
        if df_m.empty:
            continue
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        ax.plot(df_m["fl_round"], df_m["gini_mean"],
                label=label, color=color, linewidth=2)

    ax.set_xlabel("FL Round", fontsize=13)
    ax.set_ylabel("Gini Coefficient (lower = fairer)", fontsize=13)
    ax.set_title("Client Participation Fairness over Rounds", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


def generate_all_plots(
    results_dir: str,
    methods: List[str],
    num_seeds: int,
    plots_dir: str = None,
    run_name: str = "demo",
) -> None:
    """
    Convenience function: generates all three standard plots for a completed run.

    Parameters
    ----------
    results_dir : str
        Where CSVs are stored.
    methods : List[str]
        Methods to include.
    num_seeds : int
    plots_dir : str, optional
        Where to save plots.  Defaults to results_dir.
    run_name : str
        Prefix for plot filenames.
    """
    if plots_dir is None:
        plots_dir = results_dir

    plot_accuracy_curves(
        results_dir=results_dir,
        methods=methods,
        num_seeds=num_seeds,
        output_path=os.path.join(plots_dir, f"{run_name}_accuracy.png"),
    )
    plot_cost_accuracy_tradeoff(
        results_dir=results_dir,
        methods=methods,
        num_seeds=num_seeds,
        output_path=os.path.join(plots_dir, f"{run_name}_cost_accuracy.png"),
    )
    plot_fairness_curves(
        results_dir=results_dir,
        methods=methods,
        num_seeds=num_seeds,
        output_path=os.path.join(plots_dir, f"{run_name}_fairness.png"),
    )
