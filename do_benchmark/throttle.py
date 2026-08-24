"""Compatibility wrapper around the public per-endpoint AIMD throttle."""

from __future__ import annotations

from .aimd_throttle import AIMDThrottle


class AdaptiveProviderThrottle:
    """Expose the small interface used by the benchmark runners."""

    def __init__(
        self,
        max_concurrency: int,
        *,
        min_concurrency: int = 1,
        initial_concurrency: int | None = None,
        grow_after: int = 4,
    ) -> None:
        self._inner = AIMDThrottle(
            max_concurrency,
            min_concurrency=min_concurrency,
            initial_concurrency=initial_concurrency,
            grow_after=grow_after,
            reduction_factor=0.75,
        )

    def acquire(self, endpoint_id: str) -> None:
        self._inner.acquire(endpoint_id)

    def release(self, endpoint_id: str, *, error: BaseException | None = None) -> None:
        self._inner.release(endpoint_id, error=error)

    async def aacquire(self, endpoint_id: str) -> None:
        await self._inner.aacquire(endpoint_id)

    async def arelease(
        self, endpoint_id: str, *, error: BaseException | None = None
    ) -> None:
        await self._inner.arelease(endpoint_id, error=error)

    def snapshot(self) -> dict[str, int]:
        return self._inner.limits()
