from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan
from tribbie.halo.errors import ReplaceConflictError


def test_sum_keeps_destination_and_accumulates_duplicate_indices():
    plan = HaloPlan.from_edges(
        MPI.COMM_SELF,
        [HaloEdge(0, [0, 1], [1, 1])],
        entity_count=2,
    )
    values = np.array([10.0, 100.0])
    result = plan.exchange(values, op="sum")
    assert result is values
    assert np.array_equal(values, [10.0, 210.0])


def test_replace_rejects_duplicate_receive_indices_at_exchange_time():
    plan = HaloPlan.from_edges(
        MPI.COMM_SELF,
        [HaloEdge(0, [0, 1], [1, 1])],
        entity_count=2,
    )
    with pytest.raises(ReplaceConflictError):
        plan.exchange(np.array([1.0, 2.0]), op="replace")


def test_reduce_and_broadcast_uses_explicit_plans():
    reduce_plan = HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(0, [1], [0])], entity_count=2)
    broadcast_plan = HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(0, [0], [1])], entity_count=2)
    values = np.array([5.0, 7.0])
    result = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)
    assert np.array_equal(result, [12.0, 12.0])
