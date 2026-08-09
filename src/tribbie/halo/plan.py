from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mpi4py import MPI

from .errors import (
    ConcurrentRequestError,
    HaloClosedError,
    InconsistentCommunicationError,
    InvalidIndexError,
    InvalidPeerError,
    PayloadMismatchError,
    ReplaceConflictError,
    UnsupportedArrayError,
)
from .request import HaloRequest

_Array = NDArray[Any]
_BufferPair = tuple[list[_Array], list[_Array]]


@dataclass(frozen=True, slots=True)
class HaloEdge:
    peer: int
    send_indices: np.ndarray
    recv_indices: np.ndarray

    def __init__(self, peer: int, send_indices: Any, recv_indices: Any) -> None:
        send = _index_array(send_indices)
        recv = _index_array(recv_indices)
        send.setflags(write=False)
        recv.setflags(write=False)
        object.__setattr__(self, "peer", int(peer))
        object.__setattr__(self, "send_indices", send)
        object.__setattr__(self, "recv_indices", recv)


_EdgeInput = HaloEdge | tuple[int, Any, Any]


class HaloPlan:
    """Static direct-edge Halo plan for typed-buffer exchanges.

    A plan stores the fixed mapping ``source[send_indices]`` ->
    ``target[recv_indices]`` for each peer.  Exchange therefore computes a
    sparse gather, transfers only the gathered payload, and applies either
    replacement or addition at the receive positions.  The implementation
    duplicates the communicator, reuses buffers for a stable payload shape,
    and uses mpi4py buffer requests for the runtime data path.
    """

    def __init__(
        self,
        *,
        comm: MPI.Comm | None = None,
        edges: tuple[HaloEdge, ...] = (),
        entity_count: int | None = None,
        neighbors: tuple[int, ...] = (),
        send_counts: Mapping[int, int] | None = None,
        recv_counts: Mapping[int, int] | None = None,
        supported_dtypes: tuple[np.dtype[Any], ...] = (),
    ) -> None:
        self._comm = comm
        self._edges = tuple(edges)
        self._entity_count = entity_count
        self._neighbors = tuple(neighbors)
        self._send_counts = dict(send_counts or {})
        self._recv_counts = dict(recv_counts or {})
        self._supported_dtypes = tuple(supported_dtypes)
        self._buffers: _BufferPair | None = None
        self._owns_comm = False
        self._active_request: HaloRequest[Any] | None = None
        self._closed = False

    @classmethod
    def from_global_ids(
        cls,
        comm: MPI.Comm,
        global_ids: Any,
        owners: Any,
        *,
        validation: str = "basic",
    ) -> HaloPlan:
        raise NotImplementedError("global-id plan construction starts in stage E")

    @classmethod
    def from_edges(
        cls,
        comm: MPI.Comm,
        edges: Iterable[_EdgeInput],
        *,
        validation: str = "basic",
        entity_count: int | None = None,
    ) -> HaloPlan:
        """Build a plan from explicit directed peer edges.

        Each edge declares the local send and receive positions.  The
        resulting plan aggregates edges by peer, sorts peers deterministically,
        duplicates ``comm`` for tag isolation, and collectively verifies that
        each peer's send count matches the other side's receive count.  No
        payload data is transferred during this construction step.
        """
        size = comm.Get_size()
        normalized = tuple(edge if isinstance(edge, HaloEdge) else HaloEdge(*edge) for edge in edges)
        for edge in normalized:
            if not 0 <= edge.peer < size:
                raise InvalidPeerError(f"peer {edge.peer} is outside communicator size {size}")
            if entity_count is not None:
                if np.any(edge.send_indices >= entity_count) or np.any(edge.recv_indices >= entity_count):
                    raise InvalidIndexError("edge index is outside local entity range")
        grouped: dict[int, list[HaloEdge]] = {}
        for edge in normalized:
            grouped.setdefault(edge.peer, []).append(edge)
        aggregated = []
        for peer, peer_edges in grouped.items():
            aggregated.append(
                HaloEdge(
                    peer,
                    np.concatenate([edge.send_indices for edge in peer_edges]),
                    np.concatenate([edge.recv_indices for edge in peer_edges]),
                )
            )
        normalized = tuple(aggregated)
        ordered = tuple(sorted(normalized, key=lambda edge: edge.peer))
        plan_comm = comm.Dup()
        plan = cls(
            comm=plan_comm,
            edges=ordered,
            entity_count=entity_count,
            neighbors=tuple(edge.peer for edge in ordered),
            send_counts={edge.peer: int(edge.send_indices.size) for edge in ordered},
            recv_counts={edge.peer: int(edge.recv_indices.size) for edge in ordered},
            supported_dtypes=tuple(np.dtype(code) for code in ("int32", "int64", "float32", "float64", "complex64", "complex128")),
        )
        plan._owns_comm = True
        plan._validate_peer_counts()
        return plan

    @property
    def edges(self) -> tuple[HaloEdge, ...]:
        """Return the immutable tuple of planned peer edges."""
        return self._edges

    @property
    def neighbors(self) -> tuple[int, ...]:
        """Return peer ranks in the deterministic plan order."""
        return self._neighbors

    @property
    def send_counts(self) -> Mapping[int, int]:
        """Return the number of entities gathered for each peer."""
        from types import MappingProxyType
        return MappingProxyType(self._send_counts)

    @property
    def recv_counts(self) -> Mapping[int, int]:
        """Return the number of entities received from each peer."""
        from types import MappingProxyType
        return MappingProxyType(self._recv_counts)

    @property
    def total_send_count(self) -> int:
        """Return the total number of entities sent by this rank."""
        return sum(self._send_counts.values())

    @property
    def total_recv_count(self) -> int:
        """Return the total number of entities received by this rank."""
        return sum(self._recv_counts.values())

    @property
    def supported_dtypes(self) -> tuple[np.dtype[Any], ...]:
        """Return the NumPy dtypes accepted by this plan's typed buffers."""
        return self._supported_dtypes

    @property
    def closed(self) -> bool:
        """Report whether this plan has released its duplicated communicator."""
        return self._closed

    def exchange(self, src: _Array, dst: _Array | None = None, *, op: str = "replace") -> _Array:
        """Perform a blocking sparse exchange and return the destination.

        For ``replace``, each received payload ``v`` assigns
        ``dst[index] = v``.  For ``sum``, it computes
        ``dst[index] = dst[index] + v`` and applies repeated indices one at a
        time, preserving every contribution.  ``dst=None`` aliases ``src``;
        all send buffers are filled before any receive data can overwrite it.
        The method delegates setup and completion to the non-blocking path and
        immediately waits on its request, so the MPI transport remains the
        same for both APIs.
        """
        return self.begin_exchange(src, dst, op=op).wait()

    def begin_exchange(
        self,
        src: _Array,
        dst: _Array | None = None,
        *,
        op: str = "replace",
    ) -> HaloRequest[_Array]:
        """Start a non-blocking sparse exchange and return its request.

        The operation posts one typed ``Irecv`` and one typed ``Isend`` per
        real neighbor after copying ``src[send_indices]`` into reusable
        buffers.  ``test()`` polls MPI without waiting and ``wait()`` completes
        all requests before applying the mathematical replacement or sum to
        ``dst[recv_indices]``.  The request retains arrays and buffers until
        completion; callers must not mutate them meanwhile.
        """
        self._ensure_open()
        if self._active_request is not None and not self._active_request.completed:
            raise ConcurrentRequestError("plan already has an in-flight request")
        source, target, send_buffers, recv_buffers = self._prepare_exchange(src, dst, op)
        request: HaloRequest[_Array]
        if self._comm is None or self._comm.Get_size() == 1:
            for send_buffer, recv_buffer in zip(send_buffers, recv_buffers):
                recv_buffer[...] = send_buffer
            self._apply_received(target, recv_buffers, op)
            request = HaloRequest.completed_request(target)
        else:
            from mpi4py import MPI

            mpi_type = _mpi_dtype(source.dtype)
            mpi_requests = [
                self._comm.Irecv([buffer, mpi_type], source=edge.peer, tag=0)
                for edge, buffer in zip(self._edges, recv_buffers)
            ]
            mpi_requests.extend(
                self._comm.Isend([buffer, mpi_type], dest=edge.peer, tag=0)
                for edge, buffer in zip(self._edges, send_buffers)
            )

            def poll() -> tuple[bool, Any | None]:
                complete = MPI.Request.Testall(mpi_requests)
                if complete:
                    self._apply_received(target, recv_buffers, op)
                    return True, target
                return False, None

            def wait() -> Any:
                MPI.Request.Waitall(mpi_requests)
                self._apply_received(target, recv_buffers, op)
                return target

            request = HaloRequest(
                poll=poll,
                wait_fn=wait,
                on_complete=lambda: self._release_request(request),
            )
        self._active_request = request
        return request

    def _prepare_exchange(
        self,
        src: _Array,
        dst: _Array | None,
        op: str,
    ) -> tuple[_Array, _Array, list[_Array], list[_Array]]:
        if op not in {"replace", "sum"}:
            raise ValueError(f"unsupported operation: {op}")
        if op == "replace" and any(
            len(np.unique(edge.recv_indices)) != edge.recv_indices.size for edge in self._edges
        ):
            raise ReplaceConflictError("duplicate receive indices are ambiguous for replace")
        source = _validate_array(src)
        if self._supported_dtypes and source.dtype not in self._supported_dtypes:
            raise UnsupportedArrayError(f"dtype {source.dtype} is not supported by this plan")
        if self._entity_count is not None and source.shape[0] != self._entity_count:
            raise PayloadMismatchError("payload entity count does not match plan")
        target = source if dst is None else _validate_array(dst)
        if self._supported_dtypes and target.dtype not in self._supported_dtypes:
            raise UnsupportedArrayError(f"dtype {target.dtype} is not supported by this plan")
        if target.shape != source.shape or target.dtype != source.dtype:
            raise PayloadMismatchError("src and dst must have matching shape and dtype")
        if self._entity_count is not None and target.shape[0] != self._entity_count:
            raise PayloadMismatchError("destination entity count does not match plan")
        payload_shape = source.shape[1:]
        buffers_match = self._buffers is not None and all(
            buffer.dtype == source.dtype and buffer.shape[1:] == payload_shape
            for buffers in self._buffers
            for buffer in buffers
        )
        if not buffers_match:
            self._buffers = (
                [np.empty((edge.send_indices.size, *payload_shape), dtype=source.dtype) for edge in self._edges],
                [np.empty((edge.recv_indices.size, *payload_shape), dtype=source.dtype) for edge in self._edges],
            )
        assert self._buffers is not None
        send_buffers, recv_buffers = self._buffers
        for edge, buffer in zip(self._edges, send_buffers):
            buffer[...] = source[edge.send_indices]
        return source, target, send_buffers, recv_buffers

    def _apply_received(self, target: _Array, recv_buffers: list[_Array], op: str) -> None:
        for edge, buffer in zip(self._edges, recv_buffers):
            if op == "replace":
                target[edge.recv_indices] = buffer
            else:
                for position, index in enumerate(edge.recv_indices):
                    target[index] += buffer[position]

    @staticmethod
    def reduce_and_broadcast(
        reduce_plan: HaloPlan,
        broadcast_plan: HaloPlan,
        values: _Array,
        *,
        dst: _Array | None = None,
    ) -> _Array:
        """Reduce contributions to an owner plan, then broadcast its result.

        Mathematically this performs ``owner = owner + sum(contributions)``
        followed by ``ghost = owner``.  Both communication directions are
        explicit plans supplied by the caller; this helper does not infer
        ownership or construct a global index map.
        """
        reduced = reduce_plan.exchange(values, dst=dst, op="sum")
        return broadcast_plan.exchange(reduced, op="replace")

    def _release_request(self, request: HaloRequest[Any]) -> None:
        if self._active_request is request:
            self._active_request = None

    def close(self) -> None:
        """Close the plan and free its duplicated communicator.

        Closing is permitted only after any non-blocking request completes.
        Runtime payload buffers become unreachable with the plan; the caller's
        NumPy arrays remain the caller's responsibility.
        """
        if self._closed:
            raise HaloClosedError("Halo plan is already closed")
        if self._active_request is not None and not self._active_request.completed:
            raise HaloClosedError("cannot close plan with an in-flight request")
        if self._owns_comm:
            assert self._comm is not None
            self._comm.Free()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise HaloClosedError("Halo plan is closed")

    def _validate_peer_counts(self) -> None:
        assert self._comm is not None
        local_counts = {
            edge.peer: (int(edge.send_indices.size), int(edge.recv_indices.size))
            for edge in self._edges
        }
        rank_counts = self._comm.allgather(local_counts)
        rank = self._comm.Get_rank()
        for edge in self._edges:
            remote = rank_counts[edge.peer].get(rank)
            if remote is None or edge.send_indices.size != remote[1] or edge.recv_indices.size != remote[0]:
                raise InconsistentCommunicationError(f"communication counts disagree with peer {edge.peer}")


def _index_array(values: Any) -> NDArray[np.intp]:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise InvalidIndexError("indices must be a one-dimensional integer array")
    if np.any(array < 0):
        raise InvalidIndexError("indices must be non-negative")
    return np.array(array, dtype=np.intp, copy=True)


def _validate_array(values: Any) -> _Array:
    if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.number):
        raise UnsupportedArrayError("exchange requires a numeric NumPy ndarray")
    if values.ndim < 1 or not values.flags.c_contiguous:
        raise UnsupportedArrayError("exchange requires a C-contiguous ndarray")
    return values


def _mpi_dtype(dtype: np.dtype[Any]) -> MPI.Datatype:
    from mpi4py import MPI

    try:
        return MPI._typedict[dtype.char]
    except (KeyError, AttributeError):
        raise UnsupportedArrayError(f"no MPI datatype mapping for {dtype}")
