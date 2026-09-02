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
        direction="owner_to_ghost",
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


def test_direction_rejects_unknown_value():
    with pytest.raises(ValueError):
        HaloPlan.from_global_ids(
            MPI.COMM_SELF,
            np.array([4], dtype=np.int64),
            np.array([0]),
            direction="sideways",
        )


def test_ghost_to_owner_direction_empty_on_single_rank():
    plan = HaloPlan.from_global_ids(
        MPI.COMM_SELF,
        np.array([9007199254740991, 77], dtype=np.int64),
        np.array([0, 0], dtype=np.int32),
        direction="ghost_to_owner",
    )
    assert plan.neighbors == ()
    assert plan.total_send_count == 0
    assert plan.total_recv_count == 0


def test_two_way_direction_returns_pair_on_single_rank():
    reduce_plan, broadcast_plan = HaloPlan.from_global_ids(
        MPI.COMM_SELF,
        np.array([9007199254740991, 77], dtype=np.int64),
        np.array([0, 0], dtype=np.int32),
    )
    assert reduce_plan.neighbors == ()
    assert reduce_plan.total_send_count == 0
    assert broadcast_plan.neighbors == ()
    assert broadcast_plan.total_recv_count == 0
