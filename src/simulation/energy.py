"""
src/simulation/energy.py
------------------------
Grounded device-energy model for one client's participation in one FL round.

REPLACES the previous energy proxy (a fixed per-client `drain_per_round` drawn
from a Gamma distribution), which had no physical units. This model separates the
two real costs a device pays to participate:

  1. On-device COMPUTE energy for local training, and
  2. COMMUNICATION (radio) energy to upload the model update.

MATH
----
  E_client = E_compute + E_comm

  E_compute = e_mac * (P * n * epochs * fb) / eff
      P      : number of model parameters (≈ MACs per sample-example, one layer pass)
      n      : local samples trained on
      epochs : local epochs
      fb     : forward+backward multiplier (backprop ≈ 2× forward, so ≈ 3)
      eff    : device compute efficiency in (0,1]; weaker/older devices ( -> eff small)
               spend MORE energy per operation, so we divide by eff.
      e_mac  : energy per multiply-accumulate (J/MAC)

  E_comm = e_bit * (model_size_bytes * 8)
      e_bit  : radio energy per uploaded bit (J/bit)

UNITS / CALIBRATION
-------------------
The constants below are ORDER-OF-MAGNITUDE values consistent with published
mobile figures (per-MAC compute energy ~ pic-to-nano-joules; per-bit radio energy
~ tens-to-hundreds of nano-joules). They are deliberately exposed in the YAML
`energy:` block so they are tunable and auditable, and are intended to be replaced
by MEASURED per-device power profiles in the real-data phase. Reported energy is
therefore in physically-meaningful joules (order of magnitude), not a unitless proxy.
"""

DEFAULTS = {
    "e_compute_per_mac_j": 1.0e-9,   # J per multiply-accumulate
    "e_comm_per_bit_j":    1.0e-7,   # J per uploaded bit
    "fwd_bwd_factor":      3.0,      # forward + backward passes
    "min_efficiency":      0.1,      # floor so weak devices are at most 10x costlier
}


def client_round_energy(
    n_samples: int,
    local_epochs: int,
    model_params: int,
    model_size_bytes: int,
    compute_speed: float,
    cfg: dict = None,
) -> float:
    """Energy (Joules) for one client to train locally and upload once."""
    c = dict(DEFAULTS)
    if cfg:
        c.update({k: cfg[k] for k in DEFAULTS if k in cfg})

    eff = max(float(compute_speed), c["min_efficiency"])
    macs = model_params * n_samples * local_epochs * c["fwd_bwd_factor"]
    e_compute = c["e_compute_per_mac_j"] * macs / eff
    e_comm = c["e_comm_per_bit_j"] * model_size_bytes * 8.0
    return float(e_compute + e_comm)


def mlp_param_count(num_features: int, hidden1: int, hidden2: int, num_classes: int) -> int:
    """Trainable parameter count of the MLP (weights + biases)."""
    return (num_features * hidden1 + hidden1
            + hidden1 * hidden2 + hidden2
            + hidden2 * num_classes + num_classes)
