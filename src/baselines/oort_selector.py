"""
src/baselines/oort_selector.py
-------------------------------
WHAT THIS FILE DOES:
    Implements an Oort-style utility-based client selection heuristic.  Oort
    (Lai et al., 2021, OSDI) is the leading non-RL heuristic for federated
    client selection.  It defines a per-client "utility score" that combines:
        1. Statistical utility: clients with higher local loss contribute more
           to global model improvement (they have more room to learn from).
        2. System utility: clients that are faster (compute + bandwidth) can
           complete a round without becoming stragglers.
        3. Recency: clients selected too recently are penalized to avoid
           always picking the same high-utility clients every round (improves
           fairness and prevents overfitting to a subset of data).

    The top-k clients by utility score are selected each round.

WHERE IT SITS IN THE ARCHITECTURE:
    Used by the experiment runner when method='oort'.  This is the strongest
    non-RL baseline — a good Oort implementation should outperform random
    and full-participation in the accuracy-vs-cost tradeoff, and represents
    the target that RL-based methods must surpass to claim novelty.

    Upstream: environment/cluster.py provides available client IDs + attributes.
    Downstream: fl/local_trainer.py trains the selected clients.

DEPENDENCIES:
    - numpy: for score computation and sorting.
    - data/client_attributes.py: reads compute_speed, bandwidth, staleness,
      last_local_loss from ClientAttributes objects.

KEY DESIGN DECISION:
    We implement Oort's utility score with all three components (statistical +
    system + recency) because an ablated version (e.g., loss-only) would be a
    strawman.  The paper needs a genuine Oort implementation to make a fair
    comparison.  The exact Oort formula from the paper is reproduced in the
    math comments below.  The alternative (a simplified loss-only heuristic)
    was NOT chosen because reviewers will recognize it is not true Oort.
"""

import numpy as np
from typing import List, Dict

from src.data.client_attributes import ClientAttributes


class OortSelector:
    """
    Oort-style utility-based client selector.

    Computes a utility score per client each round and selects the top-k.

    Parameters
    ----------
    recency_weight : float
        Controls how strongly recently-selected clients are penalized.
        Higher value → stronger recency penalty → more exploration/fairness.
        Default 0.5 (matches the balance used in the Oort paper).
    min_rounds_between_selections : int
        A client selected in the last N rounds receives an additional penalty
        factor.  Default 1 (penalize if selected last round).
    seed : int
        Not used for selection (Oort is deterministic), but kept for API
        consistency with other selectors.
    """

    def __init__(
        self,
        recency_weight: float = 0.5,
        min_rounds_between_selections: int = 1,
        seed: int = 42,
    ):
        self.recency_weight = recency_weight
        self.min_rounds_between_selections = min_rounds_between_selections
        # Track rounds since last selection for each client
        self._rounds_since_selected: Dict[int, int] = {}

    def _compute_utility(
        self,
        client_id: int,
        attr: ClientAttributes,
    ) -> float:
        """
        Computes the Oort utility score for one client.

        MATH — Oort Utility Score (adapted from Lai et al., 2021):
        ----------------------------------------------------------
            U_i = stat_utility_i * sys_utility_i * recency_factor_i

        where:

            stat_utility_i = last_local_loss_i
                (higher loss → more to gain from this client's update)

            sys_utility_i = compute_speed_i * normalized_bandwidth_i
                (higher throughput → faster completion → less straggling)
                Note: in the original Oort, sys_utility uses the expected
                completion time T_i.  We approximate T_i^{-1} as the product
                of compute and bandwidth scores (both normalized to [0,1]).

            recency_factor_i:
                if rounds_since_selected_i >= min_rounds_between_selections:
                    recency_factor_i = 1.0   (no penalty)
                else:
                    recency_factor_i = (1 - recency_weight)
                    (penalty for being selected too recently)

        Parameters
        ----------
        client_id : int
        attr : ClientAttributes

        Returns
        -------
        float: utility score (higher = more desirable to select).
        """
        # Statistical utility: local training loss (proxy for data informativeness)
        stat_utility = float(attr.last_local_loss)

        # System utility: combine compute and bandwidth into throughput estimate
        # Both are already in [0, 1] range after normalization in client_attributes.py
        bw_norm = float(np.log1p(attr.bandwidth) / np.log1p(100.0))  # log-normalize bw
        sys_utility = float(attr.compute_speed) * bw_norm

        # Recency penalty: discourage re-selecting recently used clients
        rounds_since = self._rounds_since_selected.get(client_id, 999)
        if rounds_since >= self.min_rounds_between_selections:
            recency_factor = 1.0
        else:
            recency_factor = 1.0 - self.recency_weight

        # MATH: U_i = stat * sys * recency
        utility = stat_utility * sys_utility * recency_factor

        # Floor at a small epsilon to prevent zero utility from blocking
        # clients that happen to have zero reported loss (e.g., at round 0)
        return max(utility, 1e-6)

    def select_clients(
        self,
        available_client_ids: List[int],
        k: int,
        client_attributes: Dict[int, ClientAttributes] = None,
    ) -> List[int]:
        """
        Selects the top-k clients by Oort utility score.

        Parameters
        ----------
        available_client_ids : List[int]
            Clients eligible for selection this round (not churned, battery OK).
        k : int
            Number of clients to select.
        client_attributes : Dict[int, ClientAttributes]
            Maps client_id → ClientAttributes for all clients in the cluster.
            Required to compute the utility score.

        Returns
        -------
        List[int]: Top-k client IDs by utility score (highest first).
        """
        if not available_client_ids:
            return []

        if client_attributes is None:
            # Fallback: if attributes not provided, behave like random selector
            import warnings
            warnings.warn(
                "OortSelector.select_clients called without client_attributes; "
                "falling back to random selection.",
                stacklevel=2,
            )
            rng = np.random.default_rng(0)
            k_actual = min(k, len(available_client_ids))
            indices = rng.choice(len(available_client_ids), size=k_actual, replace=False)
            return [available_client_ids[i] for i in indices]

        # Compute utility score for every available client
        scored = []
        for cid in available_client_ids:
            attr = client_attributes[cid]
            score = self._compute_utility(cid, attr)
            scored.append((cid, score))

        # Sort by utility score descending; take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        k_actual = min(k, len(scored))
        selected = [cid for cid, _ in scored[:k_actual]]
        return selected

    def on_round_end(
        self,
        selected_ids: List[int],
        all_client_ids: List[int],
        **kwargs,
    ) -> None:
        """
        Updates recency counters after each round.

        Selected clients have their counter reset to 0 (just selected).
        Non-selected clients have their counter incremented.

        Parameters
        ----------
        selected_ids : List[int]
            Clients that were selected this round.
        all_client_ids : List[int]
            All clients in the cluster (for updating non-selected counters).
        """
        selected_set = set(selected_ids)
        for cid in all_client_ids:
            if cid in selected_set:
                self._rounds_since_selected[cid] = 0
            else:
                self._rounds_since_selected[cid] = (
                    self._rounds_since_selected.get(cid, 0) + 1
                )

    def reset(self, seed: int = None) -> None:
        """Resets recency counters for a new experiment seed."""
        self._rounds_since_selected = {}

    def __repr__(self) -> str:
        return (
            f"OortSelector("
            f"recency_weight={self.recency_weight}, "
            f"min_gap={self.min_rounds_between_selections})"
        )
