from __future__ import annotations

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan


def test_two_rank_ghost_to_owner_sum():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        HaloPlan.from_edges(comm, [], entity_count=0).exchange(np.empty(0), op="sum")
        return

    rank = comm.Get_rank()
    peer = 1 - rank
    plan = HaloPlan.from_edges(comm, [HaloEdge(peer, [0], [0])], entity_count=1)
    values = np.array([rank + 1.0])
    result = plan.exchange(values, op="sum")
    assert np.array_equal(result, [3.0])


def test_two_rank_nonblocking_sum_matches_blocking():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        return

    rank = comm.Get_rank()
    peer = 1 - rank
    plan = HaloPlan.from_edges(comm, [HaloEdge(peer, [0], [0])], entity_count=1)
    values = np.array([rank + 1.0])
    result = plan.begin_exchange(values, op="sum").wait()
    assert np.array_equal(result, [3.0])
