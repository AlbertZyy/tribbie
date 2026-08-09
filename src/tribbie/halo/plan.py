from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .errors import HaloClosedError
from .request import HaloRequest


class HaloPlan:
    """Static Halo metadata and lifecycle contract for stage A."""

    def __init__(
        self,
        *,
        neighbors: tuple[int, ...] = (),
        send_counts: Mapping[int, int] | None = None,
        recv_counts: Mapping[int, int] | None = None,
        supported_dtypes: tuple[np.dtype[Any], ...] = (),
    ):
        self._neighbors = tuple(neighbors)
        self._send_counts = MappingProxyType(dict(send_counts or {}))
        self._recv_counts = MappingProxyType(dict(recv_counts or {}))
        self._supported_dtypes = tuple(supported_dtypes)
        self._closed = False

    @classmethod
    def from_global_ids(cls, comm, global_ids, owners, *, validation="basic"):
        raise NotImplementedError("global-id plan construction starts in stage E")

    @classmethod
    def from_edges(cls, comm, edges, *, validation="basic"):
        raise NotImplementedError("direct-edge plan construction starts in stage B")

    @property
    def neighbors(self) -> tuple[int, ...]:
        return self._neighbors

    @property
    def send_counts(self) -> Mapping[int, int]:
        return self._send_counts

    @property
    def recv_counts(self) -> Mapping[int, int]:
        return self._recv_counts

    @property
    def total_send_count(self) -> int:
        return sum(self._send_counts.values())

    @property
    def total_recv_count(self) -> int:
        return sum(self._recv_counts.values())

    @property
    def supported_dtypes(self) -> tuple[np.dtype[Any], ...]:
        return self._supported_dtypes

    @property
    def closed(self) -> bool:
        return self._closed

    def exchange(self, src, dst=None, *, op="replace"):
        self._ensure_open()
        raise NotImplementedError("blocking communication starts in stage B")

    def begin_exchange(self, src, dst=None, *, op="replace") -> HaloRequest[Any]:
        self._ensure_open()
        raise NotImplementedError("non-blocking communication starts in stage C")

    def close(self) -> None:
        if self._closed:
            raise HaloClosedError("Halo plan is already closed")
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise HaloClosedError("Halo plan is closed")
