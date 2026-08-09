from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from .errors import InvalidRequestStateError

_ResultT = TypeVar("_ResultT")


class HaloRequest(Generic[_ResultT]):
    """Stage A request lifecycle placeholder."""

    def __init__(
        self,
        *,
        result: _ResultT | None = None,
        completed: bool = False,
        poll: Callable[[], tuple[bool, _ResultT | None]] | None = None,
        wait_fn: Callable[[], _ResultT] | None = None,
        on_complete: Callable[[], None] | None = None,
    ):
        self._result = result
        self._completed = completed
        self._poll_fn = poll
        self._wait_fn = wait_fn
        self._on_complete = on_complete

    @classmethod
    def pending(cls) -> HaloRequest[None]:
        return cls()

    @classmethod
    def completed_request(cls, result: _ResultT) -> HaloRequest[_ResultT]:
        return cls(result=result, completed=True)

    @property
    def completed(self) -> bool:
        return self._completed

    def test(self) -> tuple[bool, _ResultT | None]:
        if not self._completed and self._poll_fn is not None:
            complete, result = self._poll_fn()
            if complete:
                self._result = result
                self._completed = True
                self._notify_complete()
        return self._completed, self._result if self._completed else None

    def wait(self, timeout: float | None = None) -> _ResultT:
        if not self._completed:
            if timeout == 0:
                raise InvalidRequestStateError("request is not complete")
            if self._wait_fn is None:
                raise NotImplementedError("request has no completion operation")
            self._result = self._wait_fn()
            self._completed = True
            self._notify_complete()
        return self._result  # type: ignore[return-value]

    def _complete(self, result: _ResultT) -> None:
        if self._completed:
            raise InvalidRequestStateError("request is already complete")
        self._result = result
        self._completed = True

    def _notify_complete(self) -> None:
        if self._on_complete is not None:
            callback, self._on_complete = self._on_complete, None
            callback()
