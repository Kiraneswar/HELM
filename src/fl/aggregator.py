"""
src/fl/aggregator.py
---------------------
WHAT THIS FILE DOES:
    Implements the cloud-level aggregator — the component that:

    1. STANDARD: Runs FedAvg to merge client FL model updates into a single
       global model (the standard part of any federated learning system).

    2. NOVEL: Periodically runs *policy federation* — taking the RL client-selection
       policy networks from each edge cluster, averaging their weights, and sending
       the averaged policy back down to regularize each cluster's local policy.

    These two aggregation operations run on INDEPENDENT cadences:
        - FL model aggregation:    every T_FL rounds  (typically every round, T_FL=1)
        - RL policy federation:    every T_RL rounds  (e.g., every 5 rounds, T_RL=5)

    The separation of these cadences is the paper's key structural contribution.

WHERE IT SITS IN THE ARCHITECTURE:
    The aggregator is the "cloud" in the hierarchical cloud-edge system.  It sits
    above all edge clusters and sees updates from all of them.  In the simulation,
    it runs in the same process as everything else (no network required at the
    cloud level, since cloud-to-cloud communication is fast and not the bottleneck
    being studied).

    Upstream (receives from):
        - fl/local_trainer.py  → (weights, num_samples) per client, per round
        - rl/policy.py         → policy network state_dicts, every T_RL rounds

    Downstream (sends to):
        - environment/cluster.py → updated global weights for next-round distribution
        - rl/policy.py           → federated (averaged) policy weights for soft/hard update

DEPENDENCIES:
    - torch: weight averaging (no torch.nn needed here — we work with state_dicts
      directly, not with model instances, to keep the aggregator model-agnostic)
    - fl/model.py: MLPModel holds the global model state

KEY DESIGN DECISION:
    The aggregator works with raw state_dicts (OrderedDicts of tensors) rather than
    model instances.  This makes it model-agnostic: the same FedAvg code works for
    both MLPModel weights and RL policy weights without knowing the model architecture.
    The alternative (passing model instances) was NOT chosen because it would require
    the aggregator to import both fl/model.py AND rl/policy.py, creating a circular
    dependency chain.
"""

import copy
import torch
from typing import Dict, List, OrderedDict, Optional, Tuple
from collections import OrderedDict as ODict

from src.fl.model import MLPModel


# ---------------------------------------------------------------------------
# FedAvgAggregator: the cloud-level aggregation engine
# ---------------------------------------------------------------------------

class FedAvgAggregator:
    """
    Manages both FL model aggregation and RL policy federation at the cloud level.

    This class is the central coordination point of the hierarchical system.
    It holds the single global FL model and performs two kinds of aggregation:
        1. FedAvg on FL model weights   (every T_FL rounds)
        2. FedAvg on RL policy weights  (every T_RL rounds) ← NOVEL

    Parameters
    ----------
    num_features : int
        Input dimensionality of the MLP (must match all clients).
    num_classes : int
        Output classes of the MLP.
    hidden1 : int
        MLP first hidden layer width.
    hidden2 : int
        MLP second hidden layer width.
    federation_alpha : float
        Soft-update coefficient for RL policy federation (see federate_policies).
        alpha=1.0 means no federation (local policy unchanged).
        alpha=0.0 means full replacement by the meta-policy.
        Exposed as `federation.alpha` in the YAML config — this is an important
        ablation axis in the paper.
    federation_mode : str
        'soft'  → θ_local = α*θ_local + (1-α)*θ_meta  (soft update)
        'hard'  → θ_local = θ_meta                     (full replacement, α=0)
    device : torch.device
        Device for global model tensors.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden1: int = 128,
        hidden2: int = 64,
        federation_alpha: float = 0.5,
        federation_mode: str = "soft",
        federation_dp_noise_std: float = 0.0,   # L9: Gaussian DP noise on meta-policy (0=off)
        federation_dp_clip: float = 0.0,          # L9: per-tensor L2 clip before averaging (0=off)
        federation_personalize: bool = False,     # L10: keep decision head local, federate backbone
        federation_local_keys: tuple = ("actor_head", "critic_head"),
        device: torch.device = torch.device("cpu"),
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.federation_alpha = federation_alpha
        self.federation_mode = federation_mode
        self.federation_dp_noise_std = federation_dp_noise_std
        self.federation_dp_clip = federation_dp_clip
        self.federation_personalize = federation_personalize
        self.federation_local_keys = tuple(federation_local_keys)
        self.device = device

        # The single global FL model — kept as an MLPModel instance so we can
        # call .evaluate() on it directly from the runner.
        self.global_model = MLPModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden1=hidden1,
            hidden2=hidden2,
        ).to(device)

        # Round counters for scheduling T_FL and T_RL events
        self.fl_round: int = 0

        # History for logging: track how many clients contributed each round
        self.aggregation_history: List[Dict] = []

    # -----------------------------------------------------------------------
    # Global model access
    # -----------------------------------------------------------------------

    def get_global_weights(self) -> OrderedDict:
        """
        Returns a deep copy of the current global model's state_dict.

        Called by runner.py before each round to distribute starting weights
        to all selected clients.  Deep copy is mandatory — clients must not
        share tensor storage with the global model.

        Returns
        -------
        OrderedDict: {layer_name: weight_tensor}, deep-copied.
        """
        return self.global_model.get_weights()

    def evaluate_global_model(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Evaluates the current global model on a shared test set.

        Called by runner.py after each FL aggregation step to compute the
        per-round accuracy and loss that go into the logging CSV.

        Parameters
        ----------
        X_test : torch.Tensor, shape (N_test, num_features)
        y_test : torch.Tensor, shape (N_test,)

        Returns
        -------
        loss : float, accuracy : float
        """
        return self.global_model.evaluate(X_test, y_test, device=self.device)

    # -----------------------------------------------------------------------
    # FL Model Aggregation: standard FedAvg
    # -----------------------------------------------------------------------

    def aggregate_fl_updates(
        self,
        client_updates: List[Tuple[OrderedDict, int]],
    ) -> None:
        """
        Runs one round of FedAvg to merge client model updates into the global model.

        Called by runner.py every T_FL rounds after collecting local training results.

        MATH — FedAvg Weighted Average (McMahan et al., 2017):
        -------------------------------------------------------
        Given K clients, each with n_k local samples and updated weights θ_k:

            N = Σ_{k=1}^{K} n_k          (total samples across participating clients)

            θ_global ← Σ_{k=1}^{K} (n_k / N) * θ_k

        Each client's contribution is weighted by its data volume.  A client with
        more data has proportionally more influence on the global model.  This is
        the standard FedAvg weighting scheme.  Equal weighting (1/K) was NOT used
        because it ignores data heterogeneity and produces a worse-calibrated model.

        Parameters
        ----------
        client_updates : List of (weights, num_samples) tuples
            - weights: OrderedDict — the client's updated model state_dict
            - num_samples: int — number of samples this client trained on
            Both must be in the same order (same layer keys) as the global model.

        Side effects:
            Updates self.global_model in-place with the FedAvg result.
            Logs aggregation metadata to self.aggregation_history.
        """
        if not client_updates:
            # No updates this round (e.g., all selected clients dropped out)
            # Keep global model unchanged.
            return

        # Compute total sample count for normalization
        total_samples = sum(n for _, n in client_updates)

        if total_samples == 0:
            return

        # Initialize the aggregated weights to zero
        # We'll accumulate the weighted sum into this dict
        global_weights = self.global_model.get_weights()
        aggregated: ODict = ODict()
        for key in global_weights:
            # Use float64 for the accumulation to avoid precision loss
            # when summing many small weight contributions
            aggregated[key] = torch.zeros_like(global_weights[key], dtype=torch.float64)

        # MATH: θ_global = Σ_k (n_k / N) * θ_k
        for weights, num_samples in client_updates:
            # Compute this client's normalized weight
            client_weight = num_samples / total_samples
            for key in aggregated:
                # Accumulate: aggregated[key] += (n_k / N) * θ_k
                aggregated[key] += client_weight * weights[key].double()

        # Cast back to float32 for the model
        new_global_weights: ODict = ODict()
        for key in aggregated:
            new_global_weights[key] = aggregated[key].float()

        # Push the FedAvg result into the global model
        self.global_model.set_weights(new_global_weights)
        self.fl_round += 1

        # Log aggregation event
        self.aggregation_history.append({
            "fl_round": self.fl_round,
            "num_clients": len(client_updates),
            "total_samples": total_samples,
        })

    # -----------------------------------------------------------------------
    # RL Policy Federation: the paper's core novel contribution
    # -----------------------------------------------------------------------

    def federate_policies(
        self,
        cluster_policy_weights: Dict[int, OrderedDict],
        rl_round: int,
    ) -> Dict[int, OrderedDict]:
        """
        Averages the RL client-selection policies from all edge clusters and
        returns per-cluster updated weights via soft or hard update.

        # NOVELTY: this is the policy-federation step — the core contribution of this
        # project. Standard FL only federates the ML model; here we ALSO federate the
        # RL policy deciding client selection, which is what most prior work does not do.
        # The intuition: if one cluster's policy learns "prefer high-bandwidth clients",
        # and another cluster's policy learns "prefer low-staleness clients", the
        # meta-policy carries both insights and injects them into all local policies.

        MATH — RL Policy Federation (equal weight across clusters):
        -----------------------------------------------------------
        Given C clusters with local policy weights {φ_1, ..., φ_C}:

            φ_meta = (1/C) Σ_{c=1}^{C} φ_c      ← meta-policy (simple average)

        Then, for each cluster c, update its local policy via soft update:

            φ_c ← α * φ_c + (1 - α) * φ_meta

        where α = self.federation_alpha (config: federation.alpha):
            α = 1.0  → no federation (local policy unchanged)
            α = 0.0  → hard replacement by meta-policy
            α = 0.5  → balanced blend (default for this paper)

        Hard-replace mode (federation.mode = 'hard'):
            φ_c ← φ_meta  (equivalent to α=0.0 always)
            Exposed as a distinct mode because it's a common ablation point —
            "is soft blending actually better than just replacing?"

        WHY EQUAL WEIGHTING?
            We weight all cluster policies equally (1/C) regardless of cluster size
            because we're federating policy PARAMETERS, not data samples.  A cluster
            with more clients doesn't necessarily have a better policy — it may just
            have learned a very cluster-specific strategy.  Equal weighting encourages
            broader generalization.  Data-proportional weighting (like FL model FedAvg)
            was NOT used because it would bias the meta-policy toward the largest
            cluster's local strategy, defeating the purpose of sharing across clusters.

        # NOVELTY: the T_RL cadence — this function is called every T_RL rounds (not
        # every round), independently of T_FL. This dual-cadence scheduling means the
        # RL policies can be federated less frequently than the FL model, which is
        # intentional: RL policies need more local experience between federation events
        # to develop meaningful cluster-specific specializations worth sharing.

        Parameters
        ----------
        cluster_policy_weights : Dict[int, OrderedDict]
            Maps cluster_id → that cluster's current policy network state_dict.
            All state_dicts must have the same keys and tensor shapes.
        rl_round : int
            Current RL federation round number (for logging only).

        Returns
        -------
        Dict[int, OrderedDict]
            Maps cluster_id → the updated policy state_dict for that cluster.
            Each cluster receives its own copy (not shared tensor storage).
        """
        if not cluster_policy_weights:
            return {}

        cluster_ids = list(cluster_policy_weights.keys())
        num_clusters = len(cluster_ids)

        # ----------------------------------------------------------------
        # Step 1: Compute the meta-policy (simple average across all clusters)
        # MATH: φ_meta = (1/C) Σ_{c=1}^{C} φ_c
        # ----------------------------------------------------------------
        reference_weights = cluster_policy_weights[cluster_ids[0]]
        meta_policy: ODict = ODict()

        for key in reference_weights:
            # Initialize accumulator at float64 for precision
            meta_policy[key] = torch.zeros_like(reference_weights[key], dtype=torch.float64)

        for cid in cluster_ids:
            for key in reference_weights:
                w = cluster_policy_weights[cid][key].double()
                # L9 (DP): clip each cluster's per-tensor contribution to bound
                # its sensitivity before averaging (a simplified DP-SGD clip).
                if self.federation_dp_clip > 0:
                    nrm = w.norm()
                    if nrm > self.federation_dp_clip:
                        w = w * (self.federation_dp_clip / (nrm + 1e-12))
                meta_policy[key] += w / num_clusters

        # Cast back to float32
        for key in meta_policy:
            meta_policy[key] = meta_policy[key].float()

        # L9 (DP): add Gaussian noise to the shared meta-policy so the federated
        # signal is differentially private w.r.t. any single cluster's policy.
        if self.federation_dp_noise_std > 0:
            for key in meta_policy:
                meta_policy[key] = meta_policy[key] + torch.randn_like(
                    meta_policy[key]) * self.federation_dp_noise_std

        # ----------------------------------------------------------------
        # Step 2: Compute per-cluster updated policy via soft/hard update
        # NOVELTY: the soft-update alpha mechanism — alpha is the paper's
        # primary ablation axis for policy federation strength.
        # ----------------------------------------------------------------
        updated_cluster_weights: Dict[int, OrderedDict] = {}

        for cid in cluster_ids:
            local_weights = cluster_policy_weights[cid]
            new_weights: ODict = ODict()

            def _is_local(k):
                # L10 (personalization): keep the decision head local; only the
                # shared backbone (encoder) is federated across clusters.
                return self.federation_personalize and any(
                    lk in k for lk in self.federation_local_keys)

            if self.federation_mode == "hard":
                # Hard replace: local policy is fully overwritten by meta-policy
                # MATH: φ_c ← φ_meta  (α=0.0 always in hard mode)
                for key in local_weights:
                    if _is_local(key):
                        new_weights[key] = local_weights[key].clone()
                    else:
                        new_weights[key] = copy.deepcopy(meta_policy[key])
            else:
                # Soft update: blend local and meta policies
                # MATH: φ_c ← α * φ_c + (1 - α) * φ_meta
                alpha = self.federation_alpha
                for key in local_weights:
                    if _is_local(key):
                        # Personalized head: never federated, stays fully local.
                        new_weights[key] = local_weights[key].clone()
                        continue
                    # alpha: float in [0,1], controls how much of the local
                    # policy is kept vs. replaced with the federated average.
                    new_weights[key] = (
                        alpha * local_weights[key].float()
                        + (1.0 - alpha) * meta_policy[key].float()
                    )

            updated_cluster_weights[cid] = new_weights

        return updated_cluster_weights

    def compute_policy_similarity(
        self,
        weights_a: OrderedDict,
        weights_b: OrderedDict,
    ) -> float:
        """
        Computes a cosine-similarity-based measure of how similar two policy
        networks are after federation.

        Used by:
            - simulation/metrics.py for logging the federation event impact
            - The dashboard panel (c) to show before/after policy similarity

        MATH — Cosine Similarity of flattened weight vectors:
            sim(φ_a, φ_b) = (φ_a · φ_b) / (||φ_a|| * ||φ_b||)

        A value near 1.0 means the policies are nearly identical (strong
        federation effect).  A value near 0.0 means very different policies.

        Parameters
        ----------
        weights_a, weights_b : OrderedDict
            Two policy state_dicts with the same keys.

        Returns
        -------
        float: cosine similarity in [-1, 1].
        """
        flat_a = torch.cat([v.float().flatten() for v in weights_a.values()])
        flat_b = torch.cat([v.float().flatten() for v in weights_b.values()])

        cos_sim = torch.nn.functional.cosine_similarity(
            flat_a.unsqueeze(0), flat_b.unsqueeze(0)
        ).item()
        return float(cos_sim)

    def reset(self) -> None:
        """
        Resets the global model to random initialization.
        Called between seeds in multi-seed experiments.

        Note: We re-initialize the model rather than resetting to the first
        seed's weights, because each seed should start from a fresh random state.
        The seed is set in the runner before calling reset().
        """
        self.global_model = MLPModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
        ).to(self.device)
        self.fl_round = 0
        self.aggregation_history = []

    def __repr__(self) -> str:
        return (
            f"FedAvgAggregator("
            f"model={self.global_model}, "
            f"alpha={self.federation_alpha}, "
            f"mode={self.federation_mode}, "
            f"fl_round={self.fl_round})"
        )
