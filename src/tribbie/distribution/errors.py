from __future__ import annotations


class DistributionError(Exception):
    """Base class for Distribution contract and validation errors."""


class InvalidGlobalIdError(DistributionError, ValueError):
    """A global identifier is invalid."""


class DuplicateGlobalIdError(DistributionError, ValueError):
    """A local global identifier is duplicated."""


class InvalidOwnerError(DistributionError, ValueError):
    """An owner rank array is invalid."""


class InvalidIndexError(DistributionError, IndexError):
    """A local entity index is invalid."""


class RankMismatchError(DistributionError, ValueError):
    """A distribution's self rank does not match the communicator rank."""


class UnsupportedLayoutError(DistributionError, NotImplementedError):
    """A distribution layout is reserved and not yet implemented."""
