"""
src/fl/model.py
---------------
WHAT THIS FILE DOES:
    Defines the MLP (Multi-Layer Perceptron) that serves as the shared global
    model trained via federated learning.  This model is trained collaboratively
    across all edge clients — no client ever shares its raw data, only its
    locally-computed weight updates (gradients already applied via local SGD).

WHERE IT SITS IN THE ARCHITECTURE:
    This is the "payload" of the entire federated learning system — the thing
    all the client selection, communication, and aggregation machinery exists
    to train.  One instance of MLPModel is held by the cloud aggregator
    (fl/aggregator.py) as the global model.  After each FL round, each selected
    client receives a copy of the global weights, trains locally, and returns
    updated weights.  The aggregator then FedAvg-merges those updates back into
    the global model.

    Upstream: fl/aggregator.py creates the global model and distributes copies.
    Downstream: fl/local_trainer.py receives a model copy and runs local SGD on it.
                simulation/runner.py calls model.evaluate() to log round accuracy.

DEPENDENCIES:
    - torch, torch.nn: PyTorch for model definition and forward pass.

KEY DESIGN DECISION:
    We use a plain MLP (3 fully-connected layers with ReLU) rather than a CNN.
    This is deliberate — the paper's contribution is the CLIENT SELECTION
    mechanism, not the model architecture.  The MLP trains orders of magnitude
    faster on CPU, which is critical for running 5-seed × 100-round experiments
    in reasonable wall-clock time.  A CNN was NOT chosen because it requires
    image-shaped input (mandating a different data generator), adds ~10× training
    time, and contributes nothing to the research question about RL-based selection.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict


# ---------------------------------------------------------------------------
# MLPModel: the shared federated model
# ---------------------------------------------------------------------------

class MLPModel(nn.Module):
    """
    A three-layer Multi-Layer Perceptron for multi-class classification.

    Architecture:
        Input (num_features)
            → Linear(num_features, hidden1) → BatchNorm → ReLU → Dropout
            → Linear(hidden1, hidden2)       → BatchNorm → ReLU → Dropout
            → Linear(hidden2, num_classes)
            → LogSoftmax (for NLLLoss training)

    Design notes:
        - BatchNorm: stabilizes training when client data is non-IID, because
          local batch statistics differ from the global distribution.  This is
          actually a known tension in FL (local BN statistics hurt global model
          quality) — we use it anyway because it improves local convergence speed,
          and the paper's focus is client selection, not BN-in-FL.
        - Dropout: mild regularization to prevent local overfitting on small
          client datasets.
        - LogSoftmax + NLLLoss: numerically equivalent to CrossEntropyLoss but
          keeps the output interpretable (log-probabilities, easy to check sanity).

    Parameters
    ----------
    num_features : int
        Input dimensionality.  Must match the feature count in ClientDataset.
    num_classes : int
        Number of output classes.  Must match the label space in ClientDataset.
    hidden1 : int
        Width of the first hidden layer.  Default 128 is sufficient for the
        make_classification dataset with 20 features.
    hidden2 : int
        Width of the second hidden layer.
    dropout_rate : float
        Dropout probability applied after each hidden layer activation.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        # Layer 1: input → hidden1
        self.fc1 = nn.Linear(num_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)

        # Layer 2: hidden1 → hidden2
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)

        # Output layer: hidden2 → num_classes
        self.fc3 = nn.Linear(hidden2, num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize weights with He initialization (good default for ReLU networks)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        He (Kaiming) uniform initialization for all linear layers.

        Why He init?
            ReLU networks with default (Xavier) init can have poor gradient flow
            in early training.  He init sets variance proportional to 2/fan_in,
            correcting for the fact that ReLU kills half of its inputs on average.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, num_features)
            Input feature batch.

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Log-probabilities for each class.  Use with nn.NLLLoss.
        """
        # Layer 1
        x = self.fc1(x)
        # BatchNorm requires batch_size > 1 in training mode;
        # handled by local_trainer.py (drop_last=True in DataLoader)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output: log-softmax for NLLLoss
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    # -----------------------------------------------------------------------
    # Weight management: used heavily by the aggregator and local trainer
    # -----------------------------------------------------------------------

    def get_weights(self) -> OrderedDict:
        """
        Returns a deep copy of the model's state_dict.

        Using a deep copy is CRITICAL in federated learning: if we returned a
        reference to the state_dict, local training would modify the global model
        in-place, corrupting the aggregation.  Every client gets its own copy.

        Returns
        -------
        OrderedDict: key → tensor, fully detached and copied.
        """
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict: OrderedDict) -> None:
        """
        Loads a state_dict into this model instance.

        Used by:
            - local_trainer.py to initialize a client's local copy from global weights
            - aggregator.py to push the FedAvg result back into the global model

        Parameters
        ----------
        state_dict : OrderedDict
            Weight dictionary (typically from get_weights() or FedAvg result).
        """
        self.load_state_dict(state_dict)

    # -----------------------------------------------------------------------
    # Evaluation: called by runner.py after each aggregation round
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> tuple:
        """
        Evaluates the model on a held-out test set and returns loss + accuracy.

        This is called by simulation/runner.py after each FedAvg round to log
        the global model's progress.  The test set is a separate partition of
        the global dataset not seen by any client during training.

        Parameters
        ----------
        X : torch.Tensor, shape (N, num_features)
            Test feature matrix.
        y : torch.Tensor, shape (N,)
            Test labels.
        device : torch.device
            Device to run evaluation on (CPU for this project).

        Returns
        -------
        loss : float
            Average NLL loss over the test set.
        accuracy : float
            Fraction of correctly classified samples in [0, 1].
        """
        self.eval()
        self.to(device)
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            log_probs = self(X)
            loss = F.nll_loss(log_probs, y).item()
            predictions = log_probs.argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()

        return loss, accuracy

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters (for logging)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MLPModel(features={self.num_features}, "
            f"classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
