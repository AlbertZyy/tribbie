from __future__ import annotations

from mpi4py import MPI


def test_mpi_entry_is_executable():
    comm = MPI.COMM_WORLD
    assert comm.Get_size() >= 1
    assert 0 <= comm.Get_rank() < comm.Get_size()
