"""
src/simulation/metrics.py
--------------------------
WHAT THIS FILE DOES:
    Provides the MetricsLogger class, which accumulates per-round experiment
    measurements and provides utilities to:
        1. Log one round's worth of data (accuracy, loss, energy, fairness, etc.)
        2. Save the full run's log to a CSV file
        3. Compute summary statistics (mean ± std across seeds)
        4. Compute the Gini coefficient for fairness measurement

    Also re-exports `compute_gini` so callers don't need to import it separately
    from rl/ppo_trainer.py.

WHERE IT SITS IN THE ARCHITECTURE:
    MetricsLogger is instantiated by simulation/runner.py (one per method per seed)
    and records data at the end of each FL round.  After the run completes, runner.py
    calls save_csv() and the plotting functions use the saved CSVs.

DEPENDENCIES:
    - pandas: for the per-round data table and CSV output
    - numpy: for statistical summaries
    - rl/ppo_trainer.py: re-exports compute_gini (single source of truth)

KEY DESIGN DECISION:
    We store metrics as a flat list of dicts (one per round) rather than a
    multi-dimensional array because the dict structure maps naturally to a
    pandas DataFrame and makes adding new metrics trivial — just add a new key
    to the dict in log_round().  A numpy array would require pre-defining all
    columns upfront, making extension harder during development.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

# Re-export compute_gini as canonical source of truth
from src.rl.ppo_trainer import compute_gini


class MetricsLogger:
    """
    Accumulates per-round metrics for one experiment run (one method, one seed).

    Usage in runner.py:
        logger = MetricsLogger(method='federated_marl', seed=42)
        for round in range(num_rounds):
            ...training...
            logger.log_round(round=round, accuracy=acc, loss=loss, ...)
        logger.save_csv(results_dir='results/')

    Parameters
    ----------
    method : str
        Name of the client-selection method (e.g., 'random', 'oort',
        'centralized_rl', 'federated_marl').  Used in the CSV filename and plots.
    seed : int
        Experiment seed.  Multiple seeds produce multiple CSVs that are
        aggregated by compute_aggregated_stats().
    """

    def __init__(self, method: str, seed: int):
        self.method = method
        self.seed = seed
        self._records: List[Dict[str, Any]] = []

    def log_round(
        self,
        fl_round: int,
        accuracy: float,
        loss: float,
        energy_cost: float,
        participation_counts: List[int],
        num_selected: int,
        num_arrived: int,
        comm_cost: float = 0.0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        rl_entropy: float = 0.0,
        policy_similarity: float = 0.0,
        mean_reward: float = float("nan"),
        num_stragglers: int = 0,
        extra: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Records all metrics for one FL round.

        METRICS LOGGED:
        ---------------
        - accuracy        : global model test accuracy after this round
        - loss            : global model test NLL loss after this round
        - energy_cost     : total energy consumed by selected clients
        - gini            : Gini coefficient of client participation counts
                            (0 = perfect fairness, 1 = one client always picked)
        - num_selected    : clients selected this round across all clusters
        - num_arrived     : clients whose update arrived before deadline
        - comm_cost       : total simulated communication time (seconds)
        - policy_loss     : PPO clipped policy loss (RL methods only)
        - value_loss      : PPO value function loss (RL methods only)
        - rl_entropy      : action distribution entropy (RL methods only)
        - policy_similarity: cosine similarity between local and meta policy
                             after a federation event (federated_marl only)

        Parameters
        ----------
        fl_round : int
            Current FL round number (1-indexed).
        ... (all others described above)
        extra : dict, optional
            Additional method-specific metrics to log.

        Returns
        -------
        Dict[str, Any]: the record that was stored (for immediate logging).
        """
        # MATH — Gini fairness coefficient:
        # G = 0 means every client has been selected exactly equally often.
        # G → 1 means one client monopolizes all selections.
        gini = compute_gini(participation_counts)

        record = {
            "method":             self.method,
            "seed":               self.seed,
            "fl_round":           fl_round,
            "accuracy":           float(accuracy),
            "loss":               float(loss),
            "energy_cost":        float(energy_cost),
            "gini":               float(gini),
            "num_selected":       int(num_selected),
            "num_arrived":        int(num_arrived),
            "comm_cost":          float(comm_cost),
            "policy_loss":        float(policy_loss),
            "value_loss":         float(value_loss),
            "rl_entropy":         float(rl_entropy),
            "policy_similarity":  float(policy_similarity),
            "mean_reward":        float(mean_reward),
            "num_stragglers":     int(num_stragglers),
        }
        if extra:
            record.update(extra)

        self._records.append(record)
        return record

    def get_dataframe(self) -> pd.DataFrame:
        """Returns all logged rounds as a pandas DataFrame."""
        return pd.DataFrame(self._records)

    def save_csv(self, results_dir: str) -> str:
        """
        Saves the logged metrics to a CSV file.

        Filename format: {results_dir}/{method}_seed{seed}.csv
        Creates results_dir if it does not exist.

        Parameters
        ----------
        results_dir : str
            Directory path to write CSVs into.

        Returns
        -------
        str: Full path of the saved CSV file.
        """
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, f"{self.method}_seed{self.seed}.csv")
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        return filename

    def get_final_accuracy(self) -> float:
        """Returns the accuracy from the last logged round."""
        if not self._records:
            return 0.0
        return self._records[-1]["accuracy"]

    def get_final_energy(self) -> float:
        """Returns cumulative energy cost across all rounds."""
        return sum(r["energy_cost"] for r in self._records)

    def get_final_gini(self) -> float:
        """Returns the Gini coefficient from the last logged round."""
        if not self._records:
            return 0.0
        return self._records[-1]["gini"]


# ---------------------------------------------------------------------------
# Multi-seed aggregation utilities
# ---------------------------------------------------------------------------

def compute_aggregated_stats(
    results_dir: str,
    method: str,
    num_seeds: int,
) -> Optional[pd.DataFrame]:
    """
    Loads all seed CSVs for a given method and computes per-round mean +/- std.

    Scans the results_dir for any file matching '{method}_seed*.csv' rather than
    assuming 0-indexed seeds, so it works regardless of the base seed in config.

    Parameters
    ----------
    results_dir : str
    method : str
    num_seeds : int
        Maximum number of seed files to load.

    Returns
    -------
    pd.DataFrame with per-round aggregated stats, or None if no files found.
    """
    import glob as _glob
    pattern = os.path.join(results_dir, f"{method}_seed*.csv")
    found_files = sorted(_glob.glob(pattern))[:num_seeds]

    if not found_files:
        return None

    dfs = []
    for fpath in found_files:
        try:
            dfs.append(pd.read_csv(fpath))
        except Exception:
            pass

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    agg = combined.groupby("fl_round").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std"),
        energy_mean=("energy_cost", "mean"),
        energy_std=("energy_cost", "std"),
        gini_mean=("gini", "mean"),
        gini_std=("gini", "std"),
        comm_cost_mean=("comm_cost", "mean"),
        policy_loss_mean=("policy_loss", "mean"),
    ).reset_index()

    agg["method"] = method
    return agg


def load_all_results(results_dir: str, methods: List[str], num_seeds: int) -> pd.DataFrame:
    """
    Loads and aggregates results for all methods into one DataFrame for plotting.

    Parameters
    ----------
    results_dir : str
    methods : List[str]
        E.g. ['random', 'full', 'oort', 'centralized_rl', 'federated_marl']
    num_seeds : int

    Returns
    -------
    pd.DataFrame: combined aggregated stats with 'method' column for grouping.
    """
    all_dfs = []
    for method in methods:
        agg = compute_aggregated_stats(results_dir, method, num_seeds)
        if agg is not None:
            all_dfs.append(agg)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
