from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .errors import (
    DuplicateGlobalIdError,
    InvalidGlobalIdError,
    InvalidIndexError,
    InvalidOwnerError,
    UnsupportedLayoutError,
)
from .version import DistributionLayout, DistributionVersion

_INT32_MAX = np.iinfo(np.int32).max


class IndexDistribution:
    """Persistent structural metadata for one abstract index space.

    An ``IndexDistribution`` describes how a single numbering space (vertices,
    cells, degrees of freedom, particles, constraints, ...) is distributed,
    independent of any concrete distributed object and independent of any
    communication plan.  In the general layout the caller's local ordering is
    preserved: an entity at local index ``i`` is *owned* when
    ``owners[i] == self_rank`` and a *ghost* otherwise.  No ``MPI.Comm`` is
    held; the owning rank is recorded as a plain integer so the metadata can
    be serialized and queried without communication.
    """

    def __init__(
        self,
        global_ids: Any,
        owners: Any,
        *,
        global_size: int,
        self_rank: int,
        version: DistributionVersion | None = None,
        layout: DistributionLayout = DistributionLayout.GENERAL,
    ) -> None:
        if not isinstance(layout, DistributionLayout):
            raise TypeError("layout must be a DistributionLayout")
        if layout is DistributionLayout.OWNED_FIRST_GHOST_SORTED:
            raise UnsupportedLayoutError(
                "owned-first + ghost-sorted layout is reserved and not implemented yet"
            )
        ids = np.asarray(global_ids)
        own = np.asarray(owners)
        size = int(global_size)
        rank = int(self_rank)
        if size < 0:
            raise InvalidGlobalIdError("global_size must be non-negative")
        if rank < 0:
            raise InvalidOwnerError("self_rank must be non-negative")
        _validate_global_inputs(ids, own, size)

        ids = np.array(ids, dtype=np.int64, copy=True)
        own = np.array(own, dtype=np.int32, copy=True)
        ids.setflags(write=False)
        own.setflags(write=False)

        self._global_ids = ids
        self._owners = own
        self._global_size = size
        self._self_rank = rank
        self._version = version if version is not None else DistributionVersion.zero()
        if not isinstance(self._version, DistributionVersion):
            raise TypeError("version must be a DistributionVersion")
        self._layout = layout

        owned_mask = own == rank
        self._owned_mask = owned_mask
        self._owned_size = int(np.count_nonzero(owned_mask))
        owned_ids = ids[owned_mask]
        ghost_ids = ids[~owned_mask]
        ghost_owners = own[~owned_mask]
        owned_ids.setflags(write=False)
        ghost_ids.setflags(write=False)
        ghost_owners.setflags(write=False)
        self._owned_global_ids = owned_ids
        self._ghost_global_ids = ghost_ids
        self._ghost_owners = ghost_owners
        self._inverse: dict[int, int] | None = None

    @classmethod
    def from_owned_first_ghost_sorted(cls, *args: Any, **kwargs: Any) -> "IndexDistribution":
        """Reserved constructor for the owned-first + ghost-sorted layout.

        In that layout owned entities occupy ``[0, owned_size)`` and ghosts
        follow ordered by owner rank then ascending local index, which enables
        asymmetric send-only / recv-only halo plans.  The ``HaloPlan`` side
        does not support that layout yet, so construction is deferred.
        """
        raise UnsupportedLayoutError(
            "owned-first + ghost-sorted layout is reserved and not implemented yet"
        )

    @property
    def self_rank(self) -> int:
        """Return the rank that owns this distribution's owned entities."""
        return self._self_rank

    @property
    def global_size(self) -> int:
        """Return the number of entities in the global numbering space."""
        return self._global_size

    @property
    def local_size(self) -> int:
        """Return the number of local entities (owned plus ghost)."""
        return int(self._global_ids.size)

    @property
    def owned_size(self) -> int:
        """Return the number of locally owned entities."""
        return self._owned_size

    @property
    def ghost_size(self) -> int:
        """Return the number of local ghost entities."""
        return self.local_size - self._owned_size

    @property
    def version(self) -> DistributionVersion:
        """Return the persistent distribution version."""
        return self._version

    @property
    def layout(self) -> DistributionLayout:
        """Return the structural layout of the local index space."""
        return self._layout

    @property
    def is_owned_first(self) -> bool:
        """Return whether owned entities precede ghosts in local order.

        Always ``False`` for the general layout; reserved for the future
        owned-first + ghost-sorted layout.
        """
        return self._layout is DistributionLayout.OWNED_FIRST_GHOST_SORTED

    @property
    def global_ids(self) -> np.ndarray:
        """Return the read-only int64 array of local global identifiers."""
        return self._global_ids

    @property
    def owners(self) -> np.ndarray:
        """Return the read-only int32 array of owner ranks."""
        return self._owners

    @property
    def owned_global_ids(self) -> np.ndarray:
        """Return the read-only global identifiers of owned entities."""
        return self._owned_global_ids

    @property
    def ghost_global_ids(self) -> np.ndarray:
        """Return the read-only global identifiers of ghost entities."""
        return self._ghost_global_ids

    @property
    def ghost_owners(self) -> np.ndarray:
        """Return the read-only owner ranks of ghost entities."""
        return self._ghost_owners

    def is_owned(self, i: int) -> bool:
        """Return whether local index ``i`` refers to an owned entity."""
        self._check_index(i)
        return bool(self._owned_mask[i])

    def is_ghost(self, i: int) -> bool:
        """Return whether local index ``i`` refers to a ghost entity."""
        self._check_index(i)
        return not bool(self._owned_mask[i])

    def local_to_global(self, i: int) -> int:
        """Return the global identifier of local index ``i``."""
        self._check_index(i)
        return int(self._global_ids[i])

    def global_to_local(self, g: int) -> int | None:
        """Return the local index of global identifier ``g``, or ``None``."""
        if self._inverse is None:
            self._inverse = {
                int(gid): idx for idx, gid in enumerate(self._global_ids)
            }
        return self._inverse.get(int(g))

    def owner(self, i: int) -> int:
        """Return the owner rank of local index ``i``."""
        self._check_index(i)
        return int(self._owners[i])

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict snapshot suitable for persistence.

        The returned arrays are independent copies, so later mutation of the
        snapshot cannot affect this distribution.
        """
        return {
            "global_ids": self._global_ids.copy(),
            "owners": self._owners.copy(),
            "global_size": self._global_size,
            "self_rank": self._self_rank,
            "version": {
                "topology": self._version.topology,
                "numbering": self._version.numbering,
                "ghost_layout": self._version.ghost_layout,
            },
            "layout": self._layout.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndexDistribution":
        """Reconstruct a distribution from a :meth:`to_dict` snapshot."""
        version_data = data["version"]
        return cls(
            data["global_ids"],
            data["owners"],
            global_size=data["global_size"],
            self_rank=data["self_rank"],
            version=DistributionVersion(
                topology=version_data["topology"],
                numbering=version_data["numbering"],
                ghost_layout=version_data["ghost_layout"],
            ),
            layout=DistributionLayout(data["layout"]),
        )

    def make_halo_plan(
        self,
        comm: Any,
        *,
        direction: str = "two_way",
        validation: str = "basic",
    ) -> Any:
        """Build a halo plan (or a ``(ghost_to_owner, owner_to_ghost)`` pair).

        Delegates to :func:`tribbie.distribution.make_halo_plan`; the
        communicator is passed explicitly so this metadata object stays free of
        any ``MPI.Comm``.
        """
        from .plan import make_halo_plan

        return make_halo_plan(self, comm, direction=direction, validation=validation)

    def _check_index(self, i: int) -> None:
        if not 0 <= int(i) < self.local_size:
            raise InvalidIndexError(f"local index {i} is outside [0, {self.local_size})")


def _validate_global_inputs(
    global_ids: np.ndarray,
    owners: np.ndarray,
    global_size: int,
) -> None:
    if global_ids.ndim != 1 or global_ids.dtype != np.dtype(np.int64):
        raise InvalidGlobalIdError("global_ids must be a one-dimensional int64 array")
    if owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer):
        raise InvalidOwnerError("owners must be a one-dimensional integer array")
    if global_ids.size != owners.size:
        raise InvalidOwnerError("owners must have the same length as global_ids")
    if np.any(global_ids < 0):
        raise InvalidGlobalIdError("global_ids must be non-negative")
    if np.any(global_ids >= global_size):
        raise InvalidGlobalIdError("global_ids must be smaller than global_size")
    if np.unique(global_ids).size != global_ids.size:
        raise DuplicateGlobalIdError("global_ids must be unique within each rank")
    if np.any(owners < 0):
        raise InvalidOwnerError("owners must be non-negative")
    if np.any(owners > _INT32_MAX):
        raise InvalidOwnerError("owners must fit in int32")
