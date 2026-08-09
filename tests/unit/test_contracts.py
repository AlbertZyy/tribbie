from __future__ import annotations

import numpy as np
import pytest

from tribbie.halo import HaloPlan, HaloRequest
from tribbie.halo.errors import (
    HaloClosedError,
    HaloError,
    InvalidIndexError,
    InvalidOwnerError,
    InvalidPeerError,
    InconsistentCommunicationError,
    InvalidRequestStateError,
    UnsupportedArrayError,
)


def test_public_error_types_share_halo_error():
    assert issubclass(InvalidIndexError, HaloError)
    assert issubclass(InvalidOwnerError, HaloError)
    assert issubclass(InvalidPeerError, HaloError)
    assert issubclass(InconsistentCommunicationError, HaloError)
    assert issubclass(HaloClosedError, HaloError)
    assert issubclass(InvalidRequestStateError, HaloError)
    assert issubclass(UnsupportedArrayError, HaloError)


def test_request_starts_pending_and_can_be_completed_once():
    request = HaloRequest.completed_request(result="result")

    assert request.completed is True
    assert request.test() == (True, "result")
    assert request.wait() == "result"
    assert request.wait() == "result"


def test_pending_request_has_explicit_lifecycle():
    request = HaloRequest.pending()

    assert request.completed is False
    assert request.test() == (False, None)
    with pytest.raises(InvalidRequestStateError):
        request.wait(timeout=0)


def test_plan_metadata_is_read_only_and_close_is_terminal():
    plan = HaloPlan(
        neighbors=(1, 3),
        send_counts={1: 2, 3: 1},
        recv_counts={1: 1, 3: 2},
        supported_dtypes=(np.dtype("float64"),),
    )

    assert plan.neighbors == (1, 3)
    assert plan.send_counts == {1: 2, 3: 1}
    assert plan.recv_counts == {1: 1, 3: 2}
    assert plan.total_send_count == 3
    assert plan.total_recv_count == 3
    assert plan.supported_dtypes == (np.dtype("float64"),)

    with pytest.raises((AttributeError, TypeError)):
        plan.neighbors += (4,)
    with pytest.raises(TypeError):
        plan.send_counts[1] = 99

    plan.close()
    assert plan.closed is True
    with pytest.raises(HaloClosedError):
        plan.close()
    with pytest.raises(HaloClosedError):
        plan.exchange(np.zeros(2))


def test_stage_a_operations_are_explicitly_deferred():
    plan = HaloPlan()
    assert plan.exchange(np.zeros(0)).shape == (0,)
    with pytest.raises(NotImplementedError):
        plan.begin_exchange(np.zeros(0))
