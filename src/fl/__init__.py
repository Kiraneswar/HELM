"""
fl/__init__.py
--------------
Package initializer for the `fl` (Federated Learning) sub-package.

Architecture position:
    The FL layer is the second tier in the system, sitting above the data/environment
    layer and below the RL layer.  It provides three things:
        1. The shared global model (MLP)
        2. Local client training (SGD on each client's private dataset)
        3. Aggregation — both FedAvg for the FL model AND the novel RL policy federation

    "FL" here means the standard federated learning machinery — the RL layer
    (src/rl/) plugs into this layer by deciding WHICH clients get trained.

Exports:
    MLPModel, LocalTrainer, FedAvgAggregator
"""
from .model import MLPModel
from .local_trainer import LocalTrainer
from .aggregator import FedAvgAggregator

__all__ = ["MLPModel", "LocalTrainer", "FedAvgAggregator"]
