from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan
from tribbie.halo.errors import (
    InvalidIndexError,
    InvalidPeerError,
    InconsistentCommunicationError,
    PayloadMismatchError,
    ReplaceConflictError,
    UnsupportedArrayError,
)


def test_from_edges_exposes_immutable_edge_metadata():
    plan = HaloPlan.from_edges(
        MPI.COMM_SELF,
        [HaloEdge(peer=0, send_indices=[0, 2], recv_indices=[1, 3])],
    )

    assert plan.neighbors == (0,)
    assert plan.send_counts == {0: 2}
    assert plan.recv_counts == {0: 2}
    assert np.array_equal(plan.edges[0].send_indices, [0, 2])
    with pytest.raises(ValueError):
        plan.edges[0].send_indices[0] = 9


def test_from_edges_rejects_invalid_peer_and_indices():
    with pytest.raises(InvalidPeerError):
        HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(1, [0], [0])])
    with pytest.raises(InvalidIndexError):
        HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(0, [-1], [0])], entity_count=1)
    with pytest.raises(InconsistentCommunicationError):
        HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(0, [0], [0, 1])])


def test_from_edges_rejects_ambiguous_replace_targets():
    plan = HaloPlan.from_edges(
        MPI.COMM_SELF,
        [HaloEdge(0, [0], [1]), HaloEdge(0, [2], [1])],
        entity_count=3,
    )
    with pytest.raises(ReplaceConflictError):
        plan.exchange(np.zeros(3))


def test_exchange_rejects_invalid_payload():
    plan = HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(0, [0], [0])], entity_count=1)
    with pytest.raises(UnsupportedArrayError):
        plan.exchange([1.0])
    with pytest.raises(PayloadMismatchError):
        plan.exchange(np.zeros((2, 1), dtype=np.float64))


def test_self_exchange_replace_and_sum_support_payloads_and_in_place():
    plan = HaloPlan.from_edges(
        MPI.COMM_SELF,
        [HaloEdge(0, [0, 1], [1, 2])],
        entity_count=3,
    )
    values = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)

    replaced = plan.exchange(values, op="replace")
    assert np.array_equal(replaced, [[1, 10], [1, 10], [2, 20]])

    summed = plan.exchange(np.array([1.0, 2.0, 3.0]), op="sum")
    assert np.array_equal(summed, [1.0, 3.0, 5.0])


def test_empty_plan_accepts_empty_and_nonempty_payload_without_mpi_traffic():
    plan = HaloPlan.from_edges(MPI.COMM_SELF, [], entity_count=0)
    values = np.empty((0, 2), dtype=np.float32)
    result = plan.exchange(values)
    assert result is values
    assert result.shape == (0, 2)


def test_fixed_payload_reuses_neighbor_buffers():
    plan = HaloPlan.from_edges(MPI.COMM_SELF, [HaloEdge(0, [0], [1])], entity_count=2)
    plan.exchange(np.array([1.0, 2.0]))
    first_buffers = plan._buffers
    plan.exchange(np.array([3.0, 4.0]))
    assert plan._buffers is first_buffers
