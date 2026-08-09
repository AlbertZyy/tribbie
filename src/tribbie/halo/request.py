from __future__ import annotations

from typing import Generic, TypeVar

from .errors import InvalidRequestStateError

_ResultT = TypeVar("_ResultT")


class HaloRequest(Generic[_ResultT]):
    """Stage A request lifecycle placeholder."""

    def __init__(self, *, result: _ResultT | None = None, completed: bool = False):
        self._result = result
        self._completed = completed

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
        return self._completed, self._result if self._completed else None

    def wait(self, timeout: float | None = None) -> _ResultT:
        if not self._completed:
            if timeout == 0:
                raise InvalidRequestStateError("request is not complete")
            raise NotImplementedError("non-blocking communication starts in stage C")
        return self._result  # type: ignore[return-value]
