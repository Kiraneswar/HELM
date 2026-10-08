"""
src/baselines/random_selector.py
---------------------------------
WHAT THIS FILE DOES:
    Implements the Random client selection baseline: each FL round, uniformly
    randomly samples k clients from the pool of currently available (not churned,
    battery above threshold) clients in a cluster.

    This is the simplest possible baseline and serves as the LOWER BOUND in the
    paper's comparison plots.  If the RL methods cannot outperform random
    selection, something is fundamentally wrong with the RL reward design or
    training stability.

WHERE IT SITS IN THE ARCHITECTURE:
    A RandomSelector instance is created per cluster (or globally) by the
    experiment runner (simulation/runner.py) when method='random' is configured.
    The runner calls select_clients() once per FL round.

    Upstream: environment/cluster.py provides the list of available client IDs.
    Downstream: fl/local_trainer.py trains the selected clients.

DEPENDENCIES:
    - numpy: for reproducible random sampling (seeded RNG).

KEY DESIGN DECISION:
    Selection is performed WITHOUT replacement (torch.multinomial-style), so
    the same client is never selected twice in one round.  Selection WITH
    replacement was NOT chosen because it would allow a lucky client to provide
    duplicate model updates to FedAvg, which distorts the weighted average and
    is not how real FL systems operate.
"""

import numpy as np
from typing import List


class RandomSelector:
    """
    Selects k clients uniformly at random from the available pool each round.

    Parameters
    ----------
    seed : int
        Base seed for the random number generator.  Ensures reproducibility
        across runs with the same config + seed pair.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def select_clients(
        self,
        available_client_ids: List[int],
        k: int,
    ) -> List[int]:
        """
        Randomly selects up to k clients from the available pool.

        If fewer than k clients are available (due to churn or battery),
        all available clients are selected (no padding with unavailable ones).

        Parameters
        ----------
        available_client_ids : List[int]
            Client IDs currently reachable and eligible for selection.
            Computed by EdgeCluster.get_available_clients() before calling here.
        k : int
            Target number of clients to select.

        Returns
        -------
        List[int]: Selected client IDs (length = min(k, len(available_client_ids))).
        """
        if not available_client_ids:
            return []

        k_actual = min(k, len(available_client_ids))

        # Sampling WITHOUT replacement: each client can appear at most once
        selected_indices = self.rng.choice(
            len(available_client_ids),
            size=k_actual,
            replace=False,
        )
        return [available_client_ids[i] for i in selected_indices]

    def on_round_end(self, **kwargs) -> None:
        """No-op for the random baseline (no state to update between rounds)."""
        pass

    def reset(self, seed: int = None) -> None:
        """
        Resets the RNG for a new experiment seed.

        Parameters
        ----------
        seed : int, optional
            New seed; if None, uses the original seed.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return "RandomSelector()"
