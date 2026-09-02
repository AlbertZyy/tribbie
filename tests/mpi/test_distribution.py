from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.distribution import (
    IndexDistribution,
    RankMismatchError,
    make_halo_plan,
)
from tribbie.halo import HaloPlan
from tribbie.halo.errors import InvalidOwnerError as HaloInvalidOwnerError


def _empty(comm):
    return IndexDistribution(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int32),
        global_size=0,
        self_rank=comm.Get_rank(),
    )


def test_two_rank_two_way_reduce_and_broadcast_matches_expected():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        make_halo_plan(_empty(comm), comm)
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.array([9000000000007, 41], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    else:
        ids = np.array([41, 9000000000007], dtype=np.int64)
        owners = np.array([0, 0], dtype=np.int32)
    dist = IndexDistribution(ids, owners, global_size=10**13, self_rank=rank)

    reduce_plan, broadcast_plan = make_halo_plan(dist, comm, direction="two_way")
    values = np.array([100.0 + rank, 200.0 + rank])
    result = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)
    assert np.array_equal(result, [301.0, 301.0])


def test_two_rank_mixed_ownership_noncontiguous_ids():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        make_halo_plan(_empty(comm), comm)
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.array([9000000000007, 41, 5000000000003], dtype=np.int64)
        owners = np.array([0, 0, 1], dtype=np.int32)
        values = np.array([100.0, 200.0, 300.0])
        expected = np.array([100.0, 401.0, 401.0])
    else:
        ids = np.array([5000000000003, 41], dtype=np.int64)
        owners = np.array([1, 0], dtype=np.int32)
        values = np.array([101.0, 201.0])
        expected = np.array([401.0, 401.0])
    dist = IndexDistribution(ids, owners, global_size=10**13, self_rank=rank)

    reduce_plan, broadcast_plan = make_halo_plan(dist, comm, direction="two_way")
    result = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)
    assert np.array_equal(result, expected)


def test_three_rank_shared_entity_owned_and_ghosts():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 3:
        make_halo_plan(_empty(comm), comm)
        return

    rank = comm.Get_rank()
    ids = np.array([1234567890123], dtype=np.int64)
    owners = np.array([0], dtype=np.int32)
    dist = IndexDistribution(ids, owners, global_size=10**13, self_rank=rank)

    reduce_plan, broadcast_plan = make_halo_plan(dist, comm, direction="two_way")
    values = np.array([float(rank + 1)])
    result = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)
    assert np.array_equal(result, [6.0])


def test_empty_distribution_on_ghost_process():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        make_halo_plan(_empty(comm), comm)
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.array([5], dtype=np.int64)
        owners = np.array([0], dtype=np.int32)
        values = np.array([7.0])
    else:
        ids = np.empty(0, dtype=np.int64)
        owners = np.empty(0, dtype=np.int32)
        values = np.empty(0, dtype=np.float64)
    dist = IndexDistribution(ids, owners, global_size=10, self_rank=rank)

    reduce_plan, broadcast_plan = make_halo_plan(dist, comm, direction="two_way")
    result = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)
    if rank == 0:
        assert np.array_equal(result, [7.0])
    else:
        assert result.shape == (0,)


def test_full_validation_detects_cross_rank_owner_inconsistency():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        make_halo_plan(_empty(comm), comm, validation="full")
        return

    rank = comm.Get_rank()
    ids = np.array([7], dtype=np.int64)
    owners = np.array([rank], dtype=np.int32)
    dist = IndexDistribution(ids, owners, global_size=10, self_rank=rank)

    with pytest.raises(HaloInvalidOwnerError):
        make_halo_plan(dist, comm, direction="two_way", validation="full")


def test_self_rank_mismatch_raises_collectively():
    comm = MPI.COMM_WORLD
    dist = IndexDistribution(
        np.array([7], dtype=np.int64),
        np.array([comm.Get_rank()], dtype=np.int32),
        global_size=10,
        self_rank=comm.Get_size() + 10,
    )
    with pytest.raises(RankMismatchError):
        make_halo_plan(dist, comm, direction="two_way")
