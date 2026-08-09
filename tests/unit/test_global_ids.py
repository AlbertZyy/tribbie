from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.halo import HaloPlan
from tribbie.halo.errors import (
    DuplicateGlobalIdError,
    InvalidGlobalIdError,
    InvalidOwnerError,
)


def test_single_rank_global_id_plan_is_empty_for_local_owned_entities():
    plan = HaloPlan.from_global_ids(
        MPI.COMM_SELF,
        np.array([9007199254740991, 77], dtype=np.int64),
        np.array([0, 0], dtype=np.int32),
    )
    assert plan.neighbors == ()
    assert plan.total_send_count == 0
    assert plan.total_recv_count == 0


def test_global_id_input_validation():
    with pytest.raises(InvalidGlobalIdError):
        HaloPlan.from_global_ids(MPI.COMM_SELF, np.array([-1], dtype=np.int64), np.array([0]))
    with pytest.raises(DuplicateGlobalIdError):
        HaloPlan.from_global_ids(MPI.COMM_SELF, np.array([4, 4], dtype=np.int64), np.array([0, 0]))
    with pytest.raises(InvalidOwnerError):
        HaloPlan.from_global_ids(MPI.COMM_SELF, np.array([4], dtype=np.int64), np.array([1]))


def test_global_id_plan_requires_int64_ids():
    with pytest.raises(InvalidGlobalIdError):
        HaloPlan.from_global_ids(MPI.COMM_SELF, np.array([4], dtype=np.int32), np.array([0]))
