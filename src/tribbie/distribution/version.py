from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class DistributionVersion:
    """Persistent version of a distribution's structural semantics.

    A distribution changes identity when any of its structural aspects change:
    re-partitioning or topology edits bump ``topology``, global renumbering
    bumps ``numbering``, and ghost-layer rebuilds bump ``ghost_layout``.
    Comparison is by value, so a plan built from one version can be checked
    against a distribution's current version.
    """

    topology: int = 0
    numbering: int = 0
    ghost_layout: int = 0

    @classmethod
    def zero(cls) -> "DistributionVersion":
        """Return the all-zero initial version."""
        return cls()


class DistributionLayout(Enum):
    """Structural layout of the local index space.

    ``GENERAL`` keeps the caller's local ordering and lets ``HaloPlan`` carry
    both ``send_indices`` and ``recv_indices`` for bidirectional exchange.

    ``OWNED_FIRST_GHOST_SORTED`` is reserved: owned entities come first,
    ghosts follow ordered by owner rank then local order, enabling asymmetric
    send-only / recv-only plans.  It is not implemented yet.
    """

    GENERAL = "general"
    OWNED_FIRST_GHOST_SORTED = "owned_first_ghost_sorted"
