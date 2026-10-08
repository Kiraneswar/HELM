"""
src/fl/local_trainer.py
------------------------
WHAT THIS FILE DOES:
    Performs local federated learning training on a single client.  Given:
        - A copy of the current global model weights
        - The client's private local dataset (ClientDataset)
        - Training hyperparameters (epochs, batch size, learning rate)
    ...it runs SGD for the specified number of local epochs and returns:
        - The updated model weight dictionary (to be sent to the edge server)
        - The average training loss over all local epochs
        - The number of local samples (used as the FedAvg weight)

    This is the "compute" step that happens on each selected client every FL round.

WHERE IT SITS IN THE ARCHITECTURE:
    LocalTrainer is called by simulation/runner.py in the inner loop of each FL
    round, once per selected client.  In real FL, this would run on-device;
    here we simulate it sequentially in-process, which is fine for a simulation.

    Upstream: fl/aggregator.py sends global weights → runner.py → LocalTrainer
    Downstream: LocalTrainer returns updated weights → runner.py → aggregator.py

DEPENDENCIES:
    - torch, torch.optim, torch.utils.data: PyTorch training machinery
    - fl/model.py: MLPModel is instantiated here as a fresh local copy
    - data/generator.py: ClientDataset provides the DataLoader-compatible dataset

KEY DESIGN DECISION:
    We use a FRESH copy of the global model for each client (set_weights at the
    start of each round) rather than persisting a local model between rounds.
    This is standard FedAvg — clients start from the global model each round,
    not from where they left off last round.  The alternative (persistent local
    models, i.e., FedProx-style) was NOT chosen because it requires a proximal
    term and extra hyperparameters that would complicate the RL reward signal.
    FedAvg is the standard against which the paper's contribution is measured.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, OrderedDict

from src.fl.model import MLPModel
from src.data.generator import ClientDataset


# ---------------------------------------------------------------------------
# LocalTrainer: runs one client's local training for one FL round
# ---------------------------------------------------------------------------

class LocalTrainer:
    """
    Encapsulates local SGD training for a single FL client.

    Designed to be stateless between rounds (no persistent optimizer state
    is kept — each round starts fresh from the global model weights).  This
    matches the standard FedAvg protocol.

    Parameters
    ----------
    num_features : int
        MLP input dimension (must match the global model).
    num_classes : int
        MLP output classes (must match the global model).
    local_epochs : int
        Number of full passes over the local dataset per FL round.
        More epochs → bigger local drift from global model → can hurt convergence
        but helps clients with slow connections (fewer rounds needed).
        The paper uses local_epochs=2 as a balanced default.
    batch_size : int
        Mini-batch size for local SGD.
    lr : float
        SGD learning rate for local training.
    device : torch.device
        Device to run training on (CPU for this project).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        local_epochs: int = 2,
        batch_size: int = 32,
        lr: float = 0.01,
        hidden1: int = 128,
        hidden2: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.device = device

        # Loss function: Negative Log-Likelihood, paired with LogSoftmax output
        # from the MLP.  Mathematically equivalent to CrossEntropyLoss but
        # keeps the NLLLoss formulation consistent with the model's log_softmax.
        self.criterion = nn.NLLLoss()

    def train(
        self,
        global_weights: OrderedDict,
        dataset: ClientDataset,
        client_id: int,
    ) -> Tuple[OrderedDict, float, int]:
        """
        Runs local SGD on one client for `local_epochs` epochs.

        PROTOCOL (standard FedAvg local training):
        ------------------------------------------
        1. Initialize local model with current global weights (fresh copy each round)
        2. Create DataLoader over the client's private dataset
        3. For each local epoch:
               For each mini-batch:
                   forward pass → compute NLL loss → backward → SGD step
        4. Return updated weights, average loss, and dataset size

        The returned weights are the "model update" this client contributes to
        the federated aggregation.  These are FULL weights (not gradients) because
        FedAvg averages weights directly, not gradients.

        MATH — FedAvg local update step (standard SGD):
            θ_local ← θ_global   (initialize from global each round)
            for each mini-batch (x_b, y_b):
                ℓ = NLLLoss(f(x_b; θ_local), y_b)
                θ_local ← θ_local - lr * ∇_θ ℓ

        Parameters
        ----------
        global_weights : OrderedDict
            The current global model weights (from FedAvgAggregator.get_global_weights()).
            This is a deep copy — we do NOT modify the original.
        dataset : ClientDataset
            The client's private local dataset (never leaves the client).
        client_id : int
            Used for logging/debugging only (not involved in training logic).

        Returns
        -------
        updated_weights : OrderedDict
            The trained local model weights after `local_epochs` of local SGD.
            These are sent to the edge server for aggregation.
        avg_loss : float
            Mean training loss across all batches and epochs this round.
            Used by the RL state vector (last_local_loss) and the Oort selector.
        num_samples : int
            Number of samples in this client's dataset.
            Used as the weight in the FedAvg weighted average.
        """
        # Step 1: Build a fresh local model initialized from global weights
        local_model = MLPModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
        )
        # Load a deep copy of global weights so we can modify without affecting
        # the global model held by the aggregator
        local_model.set_weights(copy.deepcopy(global_weights))
        local_model.to(self.device)
        local_model.train()

        # Step 2: Create DataLoader
        # drop_last=True: avoids batch_size=1 which breaks BatchNorm in the MLP
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=(len(dataset) > self.batch_size),
        )

        # Step 3: SGD optimizer — fresh each round, no momentum carry-over
        # Why no momentum?  Momentum accumulates velocity across steps; starting
        # fresh from global weights each round makes momentum misleading (the
        # previous round's direction is now irrelevant).  Plain SGD is simpler
        # and more faithful to the FedAvg paper (McMahan et al., 2017).
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr, momentum=0.0)

        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.local_epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass: model outputs log-probabilities
                log_probs = local_model(X_batch)

                # MATH — NLL Loss:
                #   ℓ = -(1/B) Σ_{i=1}^{B} log p(y_i | x_i; θ)
                #   where p(y_i|x_i;θ) = softmax output for the true class y_i
                loss = self.criterion(log_probs, y_batch)

                # Backward pass: compute gradients w.r.t. all parameters
                loss.backward()

                # Gradient clipping: prevents exploding gradients on clients with
                # extreme data distributions (very non-IID shards can produce
                # large gradient norms)
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=10.0)

                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        # Average loss across all batches and epochs
        avg_loss = total_loss / max(total_batches, 1)

        # Return updated weights (deep copy so the local model can be GC'd)
        updated_weights = local_model.get_weights()

        return updated_weights, avg_loss, len(dataset)

    def train_batch(
        self,
        global_weights: OrderedDict,
        dataset: ClientDataset,
    ) -> Tuple[OrderedDict, float, int]:
        """
        Convenience alias: runs exactly 1 epoch (used by some baselines for
        faster round simulation in the full_participation baseline where ALL
        clients train and we want to minimize simulation time).

        Parameters
        ----------
        global_weights, dataset: same as train().

        Returns
        -------
        Same as train() but with local_epochs=1 override.
        """
        original_epochs = self.local_epochs
        self.local_epochs = 1
        result = self.train(global_weights, dataset, client_id=-1)
        self.local_epochs = original_epochs
        return result
