"""
src/baselines/full_participation.py
-------------------------------------
WHAT THIS FILE DOES:
    Implements the Full Participation baseline (vanilla FedAvg with all
    available clients selected every round).  There is no client selection
    decision — all non-churned, battery-alive clients participate every round.

    In the original FedAvg paper (McMahan et al., 2017), full participation
    is the standard setup.  In practice, full participation is often
    infeasible (too many clients, high communication cost, stragglers) but
    it represents the ACCURACY UPPER BOUND: if a method cannot approach
    full-participation accuracy, the selection strategy is lossy.

WHERE IT SITS IN THE ARCHITECTURE:
    Used by the experiment runner when method='full'.  Like RandomSelector,
    it is called once per FL round per cluster.

    In this simulation, "full participation" means ALL currently available
    clients — churned clients and dead-battery clients are still excluded
    (they are physically unable to participate, regardless of selection policy).

KEY DESIGN DECISION:
    We do NOT shuffle the returned list because the order does not matter for
    FedAvg (the aggregation is order-invariant).  The alternative (shuffling
    for consistency with other baselines) was NOT chosen because it introduces
    unnecessary randomness that could make debugging harder.
"""

from typing import List


class FullParticipationSelector:
    """
    Selects ALL currently available clients in a cluster every round.

    No hyperparameters — the selection is fully deterministic given the
    availability state of clients.

    This baseline achieves the highest possible per-round data utilization
    but has the highest communication and energy cost per round.
    """

    def select_clients(
        self,
        available_client_ids: List[int],
        k: int = None,  # ignored — all available clients are selected
    ) -> List[int]:
        """
        Returns all currently available client IDs.

        Parameters
        ----------
        available_client_ids : List[int]
            All clients that passed the availability check this round.
        k : int, optional
            Ignored for this baseline — all available clients are selected.
            Kept in the signature for API compatibility with other selectors.

        Returns
        -------
        List[int]: All available client IDs.
        """
        return list(available_client_ids)

    def on_round_end(self, **kwargs) -> None:
        """No-op — full participation has no adaptive state."""
        pass

    def reset(self, seed: int = None) -> None:
        """No-op — no state to reset."""
        pass

    def __repr__(self) -> str:
        return "FullParticipationSelector()"
