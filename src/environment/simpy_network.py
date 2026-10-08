"""
src/environment/simpy_network.py
---------------------------------
WHAT THIS FILE DOES:
    Models the network communication layer between clients and their edge server
    using SimPy discrete-event simulation.  It computes how long it takes each
    selected client to upload its model update to the edge server, accounting for:
        1. Transfer time = model_size_MB / bandwidth_Mbps  (bandwidth-limited)
        2. Base latency (round-trip delay independent of bandwidth)
        3. Bandwidth CONTENTION: if multiple clients share the same uplink resource
           (edge server's ingress), they queue up — early clients block later ones.

    In real federated learning, all clients in a cluster upload concurrently but
    share the server's receive bandwidth.  SimPy's Resource(capacity=1) models a
    shared uplink bottleneck cleanly.

WHERE IT SITS IN THE ARCHITECTURE:
    Called by environment/cluster.py's `simulate_round_communication` method after
    clients finish local training.  The simulated communication times are used to
    determine which clients' updates actually arrive before the round deadline,
    and to compute per-round communication cost for logging.

    Upstream: cluster.py provides client attributes (bandwidth, latency, update size)
    Downstream: cluster.py returns arrived updates + timing data to runner.py

DEPENDENCIES:
    - simpy: the discrete-event simulation engine.
    - numpy: for small random jitter on network delays.

KEY DESIGN DECISION:
    We use SimPy's synchronous discrete-event simulation rather than real asyncio
    because the simulation runs in a single Python process and does not need actual
    wall-clock time.  SimPy lets us fast-forward through thousands of simulated
    seconds in milliseconds of real time, making it ideal for the paper's
    experiments where we simulate 100+ FL rounds.  The alternative (asyncio) was
    NOT chosen because it adds concurrency complexity without any benefit in a
    single-process simulation.
"""

import simpy
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# SimPyNetwork: discrete-event network model for one edge cluster's uplink
# ---------------------------------------------------------------------------

class SimPyNetwork:
    """
    Simulates the communication phase of a federated learning round for one
    edge cluster using SimPy discrete-event simulation.

    The model:
        - Each selected client is a SimPy process that requests the server uplink
          resource, waits for its transfer to complete, then releases the resource.
        - Transfer time is computed from the client's bandwidth and the model size.
        - A configurable round deadline causes stragglers to be dropped (their
          updates never arrive at the server this round).

    Parameters
    ----------
    bandwidth_contention : bool
        If True, the server uplink is modeled as a shared resource (capacity=1)
        so only one client uploads at a time.  If False, all uploads happen in
        parallel (no contention).  The paper uses True to model realistic edge
        servers with limited ingress bandwidth.
    round_deadline : float
        Maximum simulated seconds allowed for the communication phase.
        Clients whose upload finishes after this deadline are treated as
        stragglers and their updates are dropped.
    seed : int
        RNG seed for jitter in transmission delays.
    """

    def __init__(
        self,
        bandwidth_contention: bool = True,
        round_deadline: float = 30.0,
        seed: int = 42,
    ):
        self.bandwidth_contention = bandwidth_contention
        self.round_deadline = round_deadline
        self.rng = np.random.default_rng(seed)

    def simulate_uploads(
        self,
        client_ids: List[int],
        bandwidths: List[float],   # Mbps per client
        latencies: List[float],    # base latency in seconds per client
        model_size_mb: float = 1.0,
    ) -> Tuple[List[int], List[float]]:
        """
        Simulates all selected clients uploading their model updates.

        Returns which clients' updates arrived within the deadline and the
        simulated upload time for each.

        MATH — Transfer Time:
            transfer_time_i = (model_size_MB * 8) / bandwidth_i_Mbps
                              + latency_i
                              + jitter_i
            where jitter_i ~ Exponential(0.01)  (small random delay perturbation)

        If bandwidth_contention is True, clients queue on a SimPy Resource(1)
        and their effective transfer starts when the server is free.

        Parameters
        ----------
        client_ids : List[int]
            IDs of the clients uploading this round.
        bandwidths : List[float]
            Each client's upload bandwidth in Mbps.
        latencies : List[float]
            Each client's base round-trip latency in seconds.
        model_size_mb : float
            Size of the model update being uploaded in megabytes.
            Determines upload time together with bandwidth.

        Returns
        -------
        arrived_ids : List[int]
            Subset of client_ids whose updates arrived before round_deadline.
        upload_times : List[float]
            Simulated upload completion time (seconds) for each client_id,
            in the same order as client_ids.  Clients that missed the deadline
            still have their time recorded (>= round_deadline).
        """
        env = simpy.Environment()
        results: dict = {}  # client_id → completion_time

        # The server uplink is a shared resource.
        # capacity=1 means only one client uploads at a time (fully serialized).
        # A real edge server would have capacity > 1, but capacity=1 gives the
        # strongest contention effect and makes the timing model cleanest.
        if self.bandwidth_contention:
            uplink = simpy.Resource(env, capacity=1)
        else:
            uplink = None

        def upload_process(cid: int, bw: float, lat: float):
            """
            SimPy generator process for a single client uploading its update.

            This function is a SimPy process: each `yield env.timeout(t)` means
            'wait t simulated seconds before proceeding'.  The process structure
            mirrors real network behavior: request uplink → wait for slot →
            transmit → release slot.

            MATH:
                transmission_time = (model_size_MB * 8.0) / bandwidth_Mbps
                jitter            ~ Exponential(0.01 seconds)
                effective_delay   = latency + transmission_time + jitter
            """
            # Jitter: small random perturbation (network variability)
            jitter = float(self.rng.exponential(0.01))
            # Convert MB to Megabits, divide by bandwidth in Mbps
            transmission_time = (model_size_mb * 8.0) / max(bw, 0.1)

            if uplink is not None:
                # Request the shared server uplink resource
                with uplink.request() as req:
                    yield req  # wait until the uplink is free
                    # Simulate the actual upload duration
                    yield env.timeout(lat + transmission_time + jitter)
            else:
                # No contention: upload takes exactly transfer + latency + jitter
                yield env.timeout(lat + transmission_time + jitter)

            results[cid] = env.now

        # Register one process per selected client
        for cid, bw, lat in zip(client_ids, bandwidths, latencies):
            env.process(upload_process(cid, bw, lat))

        # Run the SimPy simulation until all processes complete or deadline is hit
        env.run(until=self.round_deadline * 2)  # 2× deadline gives stragglers time to finish

        # Determine which clients arrived within the deadline
        upload_times: List[float] = []
        arrived_ids: List[int] = []

        for cid in client_ids:
            t = results.get(cid, float("inf"))
            upload_times.append(t)
            if t <= self.round_deadline:
                arrived_ids.append(cid)

        return arrived_ids, upload_times

    def estimate_round_time(
        self,
        bandwidths: List[float],
        latencies: List[float],
        model_size_mb: float = 1.0,
    ) -> float:
        """
        Quickly estimates the expected round communication time without running
        the full SimPy simulation.  Used by the RL reward computation to estimate
        time cost before knowing which clients will be selected.

        MATH:
            If contention:  round_time ≈ sum of serial transfer times
            If no contention: round_time ≈ max of parallel transfer times

        Parameters
        ----------
        bandwidths, latencies : List[float]
            System attributes of the candidate clients.
        model_size_mb : float
            Model update size in megabytes.

        Returns
        -------
        float: Estimated total communication time in simulated seconds.
        """
        times = [
            (model_size_mb * 8.0) / max(bw, 0.1) + lat
            for bw, lat in zip(bandwidths, latencies)
        ]
        if self.bandwidth_contention:
            return float(np.sum(times))  # serial: all must complete sequentially
        else:
            return float(np.max(times))  # parallel: bottleneck is the slowest
