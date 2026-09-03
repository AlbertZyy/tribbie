from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from mpi4py import MPI

    from tribbie.halo import HaloPlan

from .errors import RankMismatchError
from .index_map import IndexDistribution


@overload
def make_halo_plan(
    distribution: IndexDistribution,
    comm: "MPI.Comm",
    *,
    direction: Literal["two_way"],
    validation: str = "basic",
) -> "tuple[HaloPlan, HaloPlan]": ...


@overload
def make_halo_plan(
    distribution: IndexDistribution,
    comm: "MPI.Comm",
    *,
    direction: Literal["owner_to_ghost", "ghost_to_owner"],
    validation: str = "basic",
) -> "HaloPlan": ...


def make_halo_plan(
    distribution: IndexDistribution,
    comm: "MPI.Comm",
    *,
    direction: Literal["ghost_to_owner", "owner_to_ghost", "two_way"] = "two_way",
    validation: str = "basic",
) -> "HaloPlan | tuple[HaloPlan, HaloPlan]":
    """Build a halo plan from persistent distribution metadata.

    This is the bridge between structural metadata and an executable plan: it
    verifies that ``distribution.self_rank`` matches ``comm``'s rank (so the
    owned/ghost classification is interpreted against the right rank space)
    and then delegates to :meth:`HaloPlan.from_global_ids`, whose discovery
    produces directed edges from the distribution's ``global_ids`` and
    ``owners``.

    ``direction`` selects the edge orientation and matches
    :meth:`HaloPlan.from_global_ids`:

    * ``"owner_to_ghost"``: a single plan whose ``send_indices`` are owned
      positions and ``recv_indices`` are ghost positions (owner scatters,
      usually paired with ``replace``).
    * ``"ghost_to_owner"``: a single plan whose ``send_indices`` are ghost
      positions and ``recv_indices`` are owned positions (ghosts contribute,
      usually paired with ``sum``).
    * ``"two_way"`` (default): returns ``(ghost_to_owner, owner_to_ghost)``
      from one discovery, in the order expected by
      :meth:`HaloPlan.reduce_and_broadcast`.

    This function is collective over ``comm``: the self-rank check and plan
    discovery are collective, and a mismatch or validation failure is raised
    on every rank so no rank hangs.
    """
    from tribbie.halo import HaloPlan

    ok = distribution.self_rank == comm.Get_rank()
    if not all(comm.allgather(ok)):
        raise RankMismatchError(
            f"distribution self rank {distribution.self_rank} does not match "
            f"communicator rank {comm.Get_rank()}"
        )
    return HaloPlan.from_global_ids(
        comm,
        distribution.global_ids,
        distribution.owners,
        direction=direction,
        validation=validation,
    )
