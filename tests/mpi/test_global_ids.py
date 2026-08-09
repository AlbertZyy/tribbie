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
    plan = HaloPlan.from_global_ids(comm, ids, owners, validation="full")

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
    plan = HaloPlan.from_global_ids(comm, ids, owners, validation="full")
    values = np.array([float(rank + 1)])
    result = plan.exchange(values.copy())
    assert np.array_equal(result, [1.0])
