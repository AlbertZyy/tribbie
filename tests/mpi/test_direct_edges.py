from __future__ import annotations

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan


def _plan_for_rank(comm):
    rank = comm.Get_rank()
    if comm.Get_size() != 2:
        return HaloPlan.from_edges(comm, [], entity_count=0)
    peer = 1 - rank
    if rank == 0:
        edge = HaloEdge(peer, [0, 1], [1, 2])
    else:
        edge = HaloEdge(peer, [0, 1], [1, 2])
    return HaloPlan.from_edges(comm, [edge], entity_count=3)


def test_two_rank_blocking_replace_scalar_and_vector():
    comm = MPI.COMM_WORLD
    plan = _plan_for_rank(comm)
    if comm.Get_size() != 2:
        assert plan.exchange(np.empty((0, 2), dtype=np.float64)).shape == (0, 2)
        return

    rank = comm.Get_rank()
    src = np.array([[rank + 1, 10], [rank + 2, 20], [rank + 3, 30]], dtype=np.float64)
    dst = plan.exchange(src, dst=np.full_like(src, -1), op="replace")
    expected = np.array([[-1, -1], [2 - rank, 10], [3 - rank, 20]], dtype=np.float64)
    assert np.array_equal(dst, expected)


def test_two_rank_blocking_replace_in_place():
    comm = MPI.COMM_WORLD
    plan = _plan_for_rank(comm)
    if comm.Get_size() != 2:
        return

    rank = comm.Get_rank()
    values = np.array([rank + 1, rank + 2, rank + 3], dtype=np.float64)
    result = plan.exchange(values, op="replace")
    assert result is values
    assert np.array_equal(values, [rank + 1, 2 - rank, 3 - rank])


def test_two_rank_asymmetric_counts():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        HaloPlan.from_edges(comm, [], entity_count=0)
        return

    rank = comm.Get_rank()
    peer = 1 - rank
    edge = HaloEdge(peer, [0, 1], [2]) if rank == 0 else HaloEdge(peer, [0], [1, 2])
    plan = HaloPlan.from_edges(comm, [edge], entity_count=3)
    values = np.array([10 * rank + 1, 10 * rank + 2, 10 * rank + 3], dtype=np.int64)
    result = plan.exchange(values.copy())
    expected = np.array([11, 1, 2]) if rank == 1 else np.array([1, 2, 11])
    assert np.array_equal(result, expected)
