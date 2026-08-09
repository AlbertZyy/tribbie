from __future__ import annotations


class HaloError(Exception):
    """Base class for Halo contract and runtime errors."""


class InvalidGlobalIdError(HaloError, ValueError):
    """A global identifier is invalid."""


class DuplicateGlobalIdError(HaloError, ValueError):
    """A local global identifier is duplicated."""


class InvalidOwnerError(HaloError, ValueError):
    """An owner rank is invalid."""


class InvalidPeerError(HaloError, ValueError):
    """A peer rank is invalid."""


class InvalidIndexError(HaloError, IndexError):
    """A local entity index is invalid."""


class InconsistentCommunicationError(HaloError, ValueError):
    """Communication counts or ordering are inconsistent."""


class PayloadMismatchError(HaloError, ValueError):
    """Payload shape or dtype is incompatible."""


class ReplaceConflictError(HaloError, ValueError):
    """Replace would perform an ambiguous multi-source write."""


class HaloClosedError(HaloError, RuntimeError):
    """An operation was attempted on a closed plan."""


class InvalidRequestStateError(HaloError, RuntimeError):
    """A request lifecycle operation is invalid."""


class UnsupportedArrayError(HaloError, TypeError):
    """An array type or dtype is unsupported."""


class ConcurrentRequestError(HaloError, RuntimeError):
    """A plan has an illegal concurrent request."""
