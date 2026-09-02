"""Reduce-and-broadcast from a single persistent distribution.

Run with e.g. ``mpiexec -n 3 python examples/distribution/roundtrip.py``.

The example builds one ``IndexDistribution`` describing a shared global entity
owned by rank 0, derives the bidirectional ``(ghost_to_owner, owner_to_ghost)``
plans with ``make_halo_plan``, then reduces every rank's local contribution to
the owner and broadcasts the total back to all replicas.
"""

from __future__ import annotations

from mpi4py import MPI
import numpy as np

from tribbie.distribution import IndexDistribution, make_halo_plan
from tribbie.halo import HaloPlan


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    global_id = 42
    ids = np.array([global_id], dtype=np.int64)
    owners = np.array([0], dtype=np.int32)  # rank 0 owns the shared entity
    dist = IndexDistribution(ids, owners, global_size=100, self_rank=rank)

    reduce_plan, broadcast_plan = make_halo_plan(dist, comm, direction="two_way")

    local = np.array([float(rank + 1)])  # each rank contributes rank + 1
    total = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, local)

    expected = float(sum(range(1, comm.Get_size() + 1)))
    assert np.isclose(total[0], expected), (rank, total, expected)
    if rank == 0:
        print(f"owner total for global entity {global_id}: {total[0]}")


if __name__ == "__main__":
    main()
