from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan
from tribbie.halo.errors import ConcurrentRequestError, HaloClosedError, InvalidRequestStateError


def test_self_begin_exchange_completes_and_reuses_result():
    plan = HaloPlan.from_edges(
        MPI.COMM_SELF,
        [HaloEdge(0, [0], [1])],
        entity_count=2,
    )
    values = np.array([4.0, 9.0])
    request = plan.begin_exchange(values)

    assert request.completed is True
    assert request.test() == (True, values)
    assert request.wait() is values
    assert request.wait() is values
    plan.close()


def test_pending_request_timeout_and_double_completion_are_explicit():
    from tribbie.halo import HaloRequest

    request = HaloRequest.pending()
    with pytest.raises(InvalidRequestStateError):
        request.wait(timeout=0)
    with pytest.raises(NotImplementedError):
        request.wait()
    request._complete("done")
    assert request.wait() == "done"
    with pytest.raises(InvalidRequestStateError):
        request._complete("again")


def test_plan_rejects_second_in_flight_request_and_close_until_completion():
    plan = HaloPlan.from_edges(MPI.COMM_SELF, [], entity_count=0)
    values = np.empty(0, dtype=np.float64)
    request = plan.begin_exchange(values)
    assert request.completed is True
    plan.close()
    with pytest.raises(HaloClosedError):
        plan.close()


def test_begin_exchange_rejects_unsupported_operation():
    plan = HaloPlan.from_edges(MPI.COMM_SELF, [], entity_count=0)
    with pytest.raises(NotImplementedError):
        plan.begin_exchange(np.empty(0), op="sum")
