from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from tribbie.halo import HaloPlan
from tribbie.halo.errors import InvalidGlobalIdError


def test_missing_owner_global_id_fails_collectively():
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        return

    rank = comm.Get_rank()
    if rank == 0:
        ids = np.empty(0, dtype=np.int64)
        owners = np.empty(0, dtype=np.int32)
    else:
        ids = np.array([7777777777777], dtype=np.int64)
        owners = np.array([0], dtype=np.int32)
    with pytest.raises(InvalidGlobalIdError):
        HaloPlan.from_global_ids(comm, ids, owners, validation="full")
