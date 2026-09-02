from __future__ import annotations

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloPlan


def test_two_rank_global_id_owner_to_ghost_noncontiguous_and_shuffled():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        HaloPlan.from_global_ids(comm, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.array([9000000000007, 41], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    else:
        ids = np.array([41, 9000000000007], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    plan = HaloPlan.from_global_ids(comm, ids, owners, direction="owner_to_ghost", validation="full")

    assert plan.neighbors == ((1,) if rank == 0 else (0,))
    values = np.array([100.0 + rank, 200.0 + rank])
    result = plan.exchange(values.copy())
    if rank == 1:
        assert np.array_equal(result, [200.0, 100.0])
    else:
        assert np.array_equal(result, values)


def test_three_rank_shared_entity_owner_to_all_ghosts():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 3:
        HaloPlan.from_global_ids(comm, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
        return

    rank = comm.Get_rank()
    ids = np.array([1234567890123], dtype=np.int64)
    owners = np.array([0], dtype=np.int32)
    plan = HaloPlan.from_global_ids(comm, ids, owners, direction="owner_to_ghost", validation="full")
    values = np.array([float(rank + 1)])
    result = plan.exchange(values.copy())
    assert np.array_equal(result, [1.0])


def test_two_rank_global_id_ghost_to_owner_sum():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        HaloPlan.from_global_ids(comm, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.array([9000000000007, 41], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    else:
        ids = np.array([41, 9000000000007], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    plan = HaloPlan.from_global_ids(
        comm, ids, owners, direction="ghost_to_owner", validation="full"
    )

    values = np.array([100.0 + rank, 200.0 + rank])
    result = plan.exchange(values.copy(), op="sum")
    if rank == 0:
        # gid 9000000000007: 100 + 201 = 301; gid 41: 200 + 101 = 301.
        assert np.array_equal(result, [301.0, 301.0])
    else:
        assert np.array_equal(result, values)


def test_three_rank_shared_entity_ghost_to_owner_sum():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 3:
        HaloPlan.from_global_ids(comm, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
        return

    rank = comm.Get_rank()
    ids = np.array([1234567890123], dtype=np.int64)
    owners = np.array([0], dtype=np.int32)
    plan = HaloPlan.from_global_ids(
        comm, ids, owners, direction="ghost_to_owner", validation="full"
    )
    values = np.array([float(rank + 1)])
    result = plan.exchange(values.copy(), op="sum")
    if rank == 0:
        assert np.array_equal(result, [6.0])
    else:
        assert np.array_equal(result, values)


def test_reduce_and_broadcast_from_two_way():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        HaloPlan.from_global_ids(comm, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.array([9000000000007, 41], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    else:
        ids = np.array([41, 9000000000007], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    reduce_plan, broadcast_plan = HaloPlan.from_global_ids(
        comm, ids, owners, validation="full"
    )
    values = np.array([100.0 + rank, 200.0 + rank])
    result = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)
    assert np.array_equal(result, [301.0, 301.0])
