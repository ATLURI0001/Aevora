"""Async provider circuit breaker and token/cost execution budget guard for browser agents."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a provider circuit breaker is open and rejecting requests."""

    def __init__(self, provider: str, retry_after_seconds: float) -> None:
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit breaker OPEN for provider '{provider}'; retry after {retry_after_seconds:.2f}s"
        )


class BudgetExceededError(RuntimeError):
    """Raised when an agent execution exceeds configured token, cost, or wall-time budgets."""

    def __init__(self, reason: str, snapshot: dict[str, Any]) -> None:
        self.reason = reason
        self.snapshot = snapshot
        super().__init__(f"Execution budget exceeded: {reason}")


@dataclass(frozen=True)
class BudgetStatus:
    consumed_tokens: int
    max_tokens: int | None
    consumed_usd: float
    max_usd: float | None
    elapsed_seconds: float
    max_wall_seconds: float | None
    warning_triggered: bool
    warning_reasons: tuple[str, ...]


class ProviderCircuitBreaker:
    """Three-state (CLOSED / OPEN / HALF_OPEN) async circuit breaker for LLM and CDP calls."""

    def __init__(
        self,
        provider: str = "default",
        *,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        success_threshold: int = 2,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if not math.isfinite(recovery_timeout_seconds) or recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be positive and finite")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")

        self.provider = provider
        self.failure_threshold = int(failure_threshold)
        self.recovery_timeout_seconds = float(recovery_timeout_seconds)
        self.success_threshold = int(success_threshold)
        self._clock = clock if callable(clock) else time.monotonic

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_successes = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False
        self._lock = asyncio.Lock()

        self.total_calls = 0
        self.total_failures = 0
        self.total_rejected = 0

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (self._clock() - self._opened_at) >= self.recovery_timeout_seconds
        ):
            return CircuitState.HALF_OPEN
        return self._state

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute an async callable protected by circuit-breaker state transitions."""
        async with self._lock:
            now = self._clock()
            if self._state == CircuitState.OPEN:
                elapsed = now - (self._opened_at or now)
                if elapsed < self.recovery_timeout_seconds:
                    self.total_rejected += 1
                    raise CircuitOpenError(self.provider, self.recovery_timeout_seconds - elapsed)
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                self._half_open_probe_in_flight = False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_probe_in_flight:
                    self.total_rejected += 1
                    raise CircuitOpenError(self.provider, 0.5)
                self._half_open_probe_in_flight = True

            self.total_calls += 1

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                self.total_failures += 1
                self._half_open_probe_in_flight = False
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.OPEN
                    self._opened_at = self._clock()
                    self._half_open_successes = 0
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self.failure_threshold:
                        self._state = CircuitState.OPEN
                        self._opened_at = self._clock()
            raise

        async with self._lock:
            self._half_open_probe_in_flight = False
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._half_open_successes = 0
                    self._opened_at = None
            else:
                self._consecutive_failures = 0

        return result

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
        }


class ExecutionBudgetGuard:
    """Hard token, USD cost, and wall-clock execution budget guard with early-warning telemetry."""

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        max_usd: float | None = None,
        max_wall_seconds: float | None = None,
        warning_ratio: float = 0.8,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_usd is not None and (not math.isfinite(max_usd) or max_usd <= 0):
            raise ValueError("max_usd must be positive and finite")
        if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
            raise ValueError("max_wall_seconds must be positive and finite")

        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.max_wall_seconds = max_wall_seconds
        self.warning_ratio = max(0.1, min(0.99, float(warning_ratio)))
        self._clock = clock if callable(clock) else time.monotonic
        self._started_at = self._clock()
        self._consumed_tokens = 0
        self._consumed_usd = 0.0

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0) -> BudgetStatus:
        """Record LLM step consumption and enforce hard ceilings immediately."""
        tokens = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
        cost = max(0.0, float(cost_usd)) if math.isfinite(cost_usd) else 0.0

        self._consumed_tokens += tokens
        self._consumed_usd += cost
        return self.check()

    def check(self) -> BudgetStatus:
        """Evaluate current utilization against hard limits and soft warning thresholds."""
        elapsed = max(0.0, self._clock() - self._started_at)
        snapshot = {
            "consumed_tokens": self._consumed_tokens,
            "max_tokens": self.max_tokens,
            "consumed_usd": round(self._consumed_usd, 6),
            "max_usd": self.max_usd,
            "elapsed_seconds": round(elapsed, 3),
            "max_wall_seconds": self.max_wall_seconds,
        }

        if self.max_tokens is not None and self._consumed_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"token limit ({self._consumed_tokens} > {self.max_tokens})", snapshot
            )
        if self.max_usd is not None and self._consumed_usd > self.max_usd:
            raise BudgetExceededError(
                f"cost limit (${self._consumed_usd:.4f} > ${self.max_usd:.4f})", snapshot
            )
        if self.max_wall_seconds is not None and elapsed > self.max_wall_seconds:
            raise BudgetExceededError(
                f"wall time limit ({elapsed:.2f}s > {self.max_wall_seconds:.2f}s)", snapshot
            )

        warnings: list[str] = []
        if self.max_tokens is not None and self._consumed_tokens >= self.max_tokens * self.warning_ratio:
            warnings.append("tokens_near_limit")
        if self.max_usd is not None and self._consumed_usd >= self.max_usd * self.warning_ratio:
            warnings.append("cost_near_limit")
        if self.max_wall_seconds is not None and elapsed >= self.max_wall_seconds * self.warning_ratio:
            warnings.append("wall_time_near_limit")

        return BudgetStatus(
            consumed_tokens=self._consumed_tokens,
            max_tokens=self.max_tokens,
            consumed_usd=round(self._consumed_usd, 6),
            max_usd=self.max_usd,
            elapsed_seconds=round(elapsed, 3),
            max_wall_seconds=self.max_wall_seconds,
            warning_triggered=bool(warnings),
            warning_reasons=tuple(warnings),
        )
