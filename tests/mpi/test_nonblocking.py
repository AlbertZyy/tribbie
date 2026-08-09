from __future__ import annotations

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan


def test_two_rank_nonblocking_replace_and_local_work():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        HaloPlan.from_edges(comm, [], entity_count=0).begin_exchange(np.empty(0))
        return

    rank = comm.Get_rank()
    peer = 1 - rank
    plan = HaloPlan.from_edges(
        comm,
        [HaloEdge(peer, [0, 1], [1, 2])],
        entity_count=3,
    )
    values = np.array([rank + 1, rank + 2, rank + 3], dtype=np.float64)
    request = plan.begin_exchange(values)
    local_work = sum(range(1000))
    assert local_work == 499500
    result = request.wait()
    assert request.completed is True
    assert result is values
    assert np.array_equal(values, [rank + 1, 2 - rank, 3 - rank])
    plan.close()


def test_two_rank_nonblocking_vector_non_in_place():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        return

    rank = comm.Get_rank()
    peer = 1 - rank
    plan = HaloPlan.from_edges(comm, [HaloEdge(peer, [0], [1])], entity_count=2)
    source = np.array([[rank + 1, 10], [rank + 2, 20]], dtype=np.float32)
    target = np.full_like(source, -1)
    request = plan.begin_exchange(source, target)
    complete, result = request.test()
    if not complete:
        result = request.wait()
    assert result is target
    assert np.array_equal(target, [[-1, -1], [2 - rank, 10]])
    plan.close()
