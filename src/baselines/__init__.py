"""
baselines/__init__.py
----------------------
Package initializer for the `baselines` sub-package.

Architecture position:
    Baselines are alternative client-selection strategies used as comparison
    points for the paper's main contribution (Federated Multi-Agent RL).
    All baselines plug into the same simulation/runner.py loop via the
    BaseSelector interface defined in this package.

    The four baselines in order of sophistication:
        1. random         — no intelligence, pure random selection
        2. full           — no selection at all (train everyone)
        3. oort           — utility-based heuristic (state-of-the-art non-RL)
        4. centralized_rl — single RL agent over all clients (strongest baseline)

Exports:
    BaseSelector, RandomSelector, FullParticipationSelector,
    OortSelector, CentralizedRLSelector
"""
from .random_selector import RandomSelector
from .full_participation import FullParticipationSelector
from .oort_selector import OortSelector
from .centralized_rl import CentralizedRLSelector

__all__ = [
    "RandomSelector",
    "FullParticipationSelector",
    "OortSelector",
    "CentralizedRLSelector",
]
