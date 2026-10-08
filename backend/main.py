"""
backend/main.py
---------------
FastAPI backend for the FMARL dashboard.

Reads the per-seed CSVs written by simulation/runner.py and serves them
as JSON over a small REST API consumed by the React frontend.

ENDPOINTS:
    GET /api/methods              — list methods with available data
    GET /api/results/{method}     — aggregated per-round stats for one method
    GET /api/summary              — final-round summary card values (all methods)
    GET /api/raw/{method}/{seed}  — raw per-round rows for one seed (for debugging)
    GET /api/config               — experiment config (num_rounds, methods, etc.)

LAUNCH:
    uvicorn backend.main:app --reload --port 8000

CORS:
    Allows all origins in development (React dev server runs on :5173).
    Restrict to specific origins for any deployment.
"""

import os
import glob
from typing import List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FMARL Dashboard API", version="1.0.0")

# Allow the Vite dev server to call this backend without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config: results directory (can be overridden via env var)
# ---------------------------------------------------------------------------

RESULTS_DIR = os.environ.get("FMARL_RESULTS_DIR", "results/demo/")

ALL_METHODS = ["random", "full", "oort", "centralized_rl", "federated_marl"]

METHOD_LABELS = {
    "random":           "Random",
    "full":             "Full Participation",
    "oort":             "Oort (Heuristic)",
    "centralized_rl":   "Centralized RL",
    "federated_marl":   "Federated MARL (Ours)",
}

METHOD_COLORS = {
    "random":           "#9E9E9E",
    "full":             "#4CAF50",
    "oort":             "#FF9800",
    "centralized_rl":   "#2196F3",
    "federated_marl":   "#E91E63",
}

NOVEL_METHOD = "federated_marl"


# ---------------------------------------------------------------------------
# Helper: load all seeds for a method
# ---------------------------------------------------------------------------

def _load_method(method: str, results_dir: str) -> Optional[pd.DataFrame]:
    pattern = os.path.join(results_dir, f"{method}_seed*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-round mean ± std across seeds."""
    agg = df.groupby("fl_round").agg(
        accuracy_mean   =("accuracy",     "mean"),
        accuracy_std    =("accuracy",     lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        loss_mean       =("loss",         "mean"),
        loss_std        =("loss",         lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        energy_mean     =("energy_cost",  "mean"),
        energy_std      =("energy_cost",  lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        gini_mean       =("gini",         "mean"),
        gini_std        =("gini",         lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        comm_cost_mean  =("comm_cost",    "mean"),
        policy_loss_mean=("policy_loss",  "mean"),
        entropy_mean    =("rl_entropy",   "mean"),
        policy_sim_mean =("policy_similarity", "mean"),
        num_selected_mean=("num_selected","mean"),
        n_seeds         =("seed",         "count"),
    ).reset_index()
    # Replace NaN std with 0
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/methods")
def list_methods(results_dir: str = Query(default=None)):
    """Lists which methods have result CSVs in the results directory."""
    rdir = results_dir or RESULTS_DIR
    available = []
    for method in ALL_METHODS:
        pattern = os.path.join(rdir, f"{method}_seed*.csv")
        files   = glob.glob(pattern)
        if files:
            seeds = len(files)
            df    = _load_method(method, rdir)
            n_rounds = int(df["fl_round"].max()) if df is not None else 0
            available.append({
                "method":    method,
                "label":     METHOD_LABELS[method],
                "color":     METHOD_COLORS[method],
                "is_novel":  method == NOVEL_METHOD,
                "num_seeds": seeds,
                "num_rounds": n_rounds,
            })
    return {"methods": available, "results_dir": rdir}


@app.get("/api/results/{method}")
def get_results(method: str, results_dir: str = Query(default=None)):
    """Returns aggregated per-round stats (mean ± std) for one method."""
    rdir = results_dir or RESULTS_DIR
    if method not in ALL_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    df = _load_method(method, rdir)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No data found for method: {method}")
    agg = _aggregate(df)
    return {
        "method": method,
        "label":  METHOD_LABELS[method],
        "color":  METHOD_COLORS[method],
        "is_novel": method == NOVEL_METHOD,
        "rounds": agg.to_dict(orient="records"),
    }


@app.get("/api/summary")
def get_summary(results_dir: str = Query(default=None)):
    """
    Returns final-round summary cards for all available methods.
    Used by the Overview page.
    """
    rdir = results_dir or RESULTS_DIR
    cards = []
    for method in ALL_METHODS:
        df = _load_method(method, rdir)
        if df is None:
            continue
        agg = _aggregate(df)
        final = agg.iloc[-1]
        n_seeds = int(df["seed"].nunique())
        cards.append({
            "method":       method,
            "label":        METHOD_LABELS[method],
            "color":        METHOD_COLORS[method],
            "is_novel":     method == NOVEL_METHOD,
            "final_accuracy_mean": float(final["accuracy_mean"]),
            "final_accuracy_std":  float(final["accuracy_std"]),
            "final_loss_mean":     float(final["loss_mean"]),
            "total_energy":        float(agg["energy_mean"].sum()),
            "final_gini":          float(final["gini_mean"]),
            "final_gini_std":      float(final["gini_std"]),
            "num_seeds":           n_seeds,
            "num_rounds":          int(final["fl_round"]),
        })
    return {"summary": cards, "results_dir": rdir}


@app.get("/api/raw/{method}/{seed}")
def get_raw(method: str, seed: int, results_dir: str = Query(default=None)):
    """Returns the raw per-round rows for a specific method + seed. For debugging."""
    rdir = results_dir or RESULTS_DIR
    fpath = os.path.join(rdir, f"{method}_seed{seed}.csv")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"File not found: {fpath}")
    df = pd.read_csv(fpath)
    return {"method": method, "seed": seed, "rows": df.to_dict(orient="records")}


@app.get("/api/config")
def get_config(results_dir: str = Query(default=None)):
    """Returns experiment metadata inferred from available CSVs."""
    rdir = results_dir or RESULTS_DIR
    info = {"results_dir": rdir, "methods": []}
    for method in ALL_METHODS:
        df = _load_method(method, rdir)
        if df is not None:
            info["methods"].append({
                "method":     method,
                "num_seeds":  int(df["seed"].nunique()),
                "num_rounds": int(df["fl_round"].max()),
            })
    return info


@app.get("/health")
def health():
    return {"status": "ok"}
