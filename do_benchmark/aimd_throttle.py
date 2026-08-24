"""General-purpose AIMD (Additive Increase / Multiplicative Decrease) concurrency throttle.

Thread-safe per-key concurrency limiter that reacts to provider back-pressure
(429s, 5xx, timeouts) by halving the concurrency limit for that key, and
additively grows it back on sustained success.

Callers choose the key granularity:

- ``route_id`` so one flaky model on a shared provider cannot drag down
  every other model on that provider (scoring pipeline default).
- ``provider_id`` when you genuinely want per-provider isolation.
- ``deployment_id`` or any other string for finer-grained control.

Usage::

    throttle = AIMDThrottle(max_concurrency=512)

    # Context manager (recommended)
    with throttle.limit("digitalocean_model_id"):
        result = call_provider(...)

    # Manual acquire/release
    throttle.acquire("digitalocean_model_id")
    try:
        result = call_provider(...)
    except RateLimitError as exc:
        throttle.release("digitalocean_model_id", error=exc)
        raise
    throttle.release("digitalocean_model_id")

    # Async context manager
    async with throttle.alimit("digitalocean_model_id"):
        result = await async_call_provider(...)
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator


# Markers that indicate provider back-pressure (rate limiting, overload,
# transient server errors). Checked case-insensitively against str(error).
_PRESSURE_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "throttl",
    "503",
    "502",
    "500",
    "timed out",
    "timeout",
    "overloaded",
    "capacity",
    "too many requests",
    "resource exhausted",
    "quota exceeded",
)


def is_pressure_error(error: BaseException) -> bool:
    """Return True if *error* or its exception chain indicates back-pressure.

    Provider gateways deliberately wrap the final transport exception with a
    route-level error.  Inspecting only the wrapper text hides the underlying
    429/5xx/timeout signal and prevents AIMD from reducing concurrency.
    """

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        details = [str(current)]
        # ``subprocess.CalledProcessError.__str__`` omits captured provider
        # stderr/stdout. Inference-skill runners deliberately capture those
        # streams, so inspect them without logging or persisting their content.
        for attribute in ("stderr", "stdout", "output"):
            value = getattr(current, attribute, None)
            if value:
                details.append(str(value))
        if any(
            marker in detail.lower()
            for detail in details
            for marker in _PRESSURE_MARKERS
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class AIMDThrottle:
    """Per-key AIMD concurrency limiter.

    Parameters
    ----------
    max_concurrency:
        Ceiling per key.  No key will ever have more than this many
        in-flight requests.
    min_concurrency:
        Floor per key.  Even under sustained errors the limit never
        drops below this.  Set higher (e.g. 16) for scoring pipelines
        where you want to keep making progress even when a provider
        is partially failing.
    initial_concurrency:
        Starting limit for each new key. Defaults to ``max_concurrency``
        for backwards compatibility. Set it near the floor when the caller
        needs to measure an additive ramp from cold start.
    grow_after:
        Number of consecutive successes before additively increasing
        the concurrency limit by 1.
    reduction_factor:
        Multiplicative decrease factor on back-pressure.  0.75 means
        "reduce to 75% of current limit" (gentler than halving).
    """

    def __init__(
        self,
        max_concurrency: int,
        *,
        min_concurrency: int = 16,
        initial_concurrency: int | None = None,
        grow_after: int = 4,
        reduction_factor: float = 0.75,
    ) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.min_concurrency = min(self.max_concurrency, max(1, min_concurrency))
        if initial_concurrency is None:
            initial_concurrency = self.max_concurrency
        self.initial_concurrency = min(
            self.max_concurrency,
            max(self.min_concurrency, int(initial_concurrency)),
        )
        self.grow_after = max(1, grow_after)
        self.reduction_factor = max(0.1, min(0.99, reduction_factor))

        self._limits: dict[str, int] = {}
        self._in_flight: dict[str, int] = {}
        self._successes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)

        # Async support: per-key asyncio.Semaphore would fight with the
        # adaptive limits, so we use a single asyncio.Event that is set
        # whenever a slot might be available.
        self._async_event: asyncio.Event | None = None

    # ── Synchronous API ─────────────────────────────────────────────────

    def acquire(self, key: str) -> None:
        """Block until a slot is available for *key*."""
        with self._available:
            while self._in_flight.get(key, 0) >= self._limits.setdefault(
                key, self.initial_concurrency
            ):
                self._available.wait(timeout=5.0)
            self._in_flight[key] = self._in_flight.get(key, 0) + 1

    def release(self, key: str, *, error: BaseException | None = None) -> None:
        """Release a slot for *key*, adjusting limits based on outcome."""
        with self._available:
            self._in_flight[key] = max(0, self._in_flight.get(key, 0) - 1)
            limit = self._limits.setdefault(key, self.initial_concurrency)

            if error is not None and is_pressure_error(error):
                # Multiplicative decrease
                new_limit = max(
                    self.min_concurrency,
                    int(limit * self.reduction_factor),
                )
                self._limits[key] = new_limit
                self._successes[key] = 0
            elif error is None:
                # Additive increase
                streak = self._successes.get(key, 0) + 1
                if streak >= self.grow_after and limit < self.max_concurrency:
                    self._limits[key] = limit + 1
                    streak = 0
                self._successes[key] = streak

            self._available.notify_all()

        # Wake any async waiters
        if self._async_event is not None:
            self._async_event.set()

    @contextmanager
    def limit(self, key: str) -> Iterator[None]:
        """Context manager: acquire before yield, release on exit."""
        self.acquire(key)
        try:
            yield
        except BaseException as exc:
            self.release(key, error=exc)
            raise
        self.release(key)

    # ── Async API ───────────────────────────────────────────────────────

    async def aacquire(self, key: str) -> None:
        """Async version of acquire: yields to the event loop while waiting."""
        if self._async_event is None:
            self._async_event = asyncio.Event()

        while True:
            with self._lock:
                if self._in_flight.get(key, 0) < self._limits.setdefault(
                    key, self.initial_concurrency
                ):
                    self._in_flight[key] = self._in_flight.get(key, 0) + 1
                    return
            self._async_event.clear()
            try:
                await asyncio.wait_for(self._async_event.wait(), timeout=5.0)
            except TimeoutError:
                # Periodically re-check the adaptive limit even when no release
                # happened in this event loop tick. Long provider calls can
                # legitimately hold every slot for more than five seconds.
                continue

    async def arelease(self, key: str, *, error: BaseException | None = None) -> None:
        """Async version of release."""
        self.release(key, error=error)

    @asynccontextmanager
    async def alimit(self, key: str) -> AsyncIterator[None]:
        """Async context manager: acquire before yield, release on exit."""
        await self.aacquire(key)
        try:
            yield
        except BaseException as exc:
            await self.arelease(key, error=exc)
            raise
        await self.arelease(key)

    # ── Introspection ───────────────────────────────────────────────────

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return current limits and in-flight counts per key."""
        with self._lock:
            return {
                key: {
                    "limit": self._limits.get(key, self.initial_concurrency),
                    "in_flight": self._in_flight.get(key, 0),
                    "successes": self._successes.get(key, 0),
                }
                for key in sorted(set(self._limits) | set(self._in_flight))
            }

    def limits(self) -> dict[str, int]:
        """Return current concurrency limit per key."""
        with self._lock:
            return dict(self._limits)

    def total_in_flight(self) -> int:
        """Return total in-flight requests across all keys."""
        with self._lock:
            return sum(self._in_flight.values())
