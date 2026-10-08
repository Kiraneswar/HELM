"""
environment/__init__.py
-----------------------
Package initializer for the environment sub-package.

Architecture position:
    The environment layer sits between the raw data layer (src/data/) and the
    FL/RL layers above.  It wraps clients into clusters and models the network
    physics (delay, bandwidth contention) using SimPy.

Exports:
    EdgeCluster, SimPyNetwork — the two core environment objects.
"""
from .cluster import EdgeCluster
from .simpy_network import SimPyNetwork

__all__ = ["EdgeCluster", "SimPyNetwork"]
