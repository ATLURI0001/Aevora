import pytest

from browser_use.agent.circuit_breaker import (
    BudgetExceededError,
    CircuitOpenError,
    CircuitState,
    ExecutionBudgetGuard,
    ProviderCircuitBreaker,
)


@pytest.mark.asyncio
async def test_circuit_breaker_full_lifecycle_and_probe_recovery():
    now = [100.0]
    cb = ProviderCircuitBreaker(
        provider="openai",
        failure_threshold=2,
        recovery_timeout_seconds=10.0,
        success_threshold=2,
        clock=lambda: now[0],
    )

    async def fail_fn():
        raise ConnectionError("503 service unavailable")

    async def ok_fn():
        return "ok"

    # Two consecutive failures trip CLOSED -> OPEN
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await cb.call(fail_fn)
    assert cb.state == CircuitState.OPEN

    # Calls during cooldown reject fast without invoking fn
    with pytest.raises(CircuitOpenError) as exc_info:
        await cb.call(ok_fn)
    assert exc_info.value.provider == "openai"
    assert exc_info.value.retry_after_seconds == pytest.approx(10.0, abs=0.1)

    # Advance past cooldown -> HALF_OPEN
    now[0] = 111.0
    assert cb.state == CircuitState.HALF_OPEN

    # Probe fails -> immediately re-opens for another 10s
    with pytest.raises(ConnectionError):
        await cb.call(fail_fn)
    assert cb.state == CircuitState.OPEN

    # Advance past second cooldown -> two probe successes close circuit
    now[0] = 122.0
    assert await cb.call(ok_fn) == "ok"
    assert cb.state == CircuitState.HALF_OPEN
    assert await cb.call(ok_fn) == "ok"
    assert cb.state == CircuitState.CLOSED

    diag = cb.get_diagnostics()
    assert diag["total_rejected"] == 1
    assert diag["total_failures"] == 3


def test_execution_budget_guard_warnings_and_hard_cutoffs():
    now = [0.0]
    guard = ExecutionBudgetGuard(
        max_tokens=1000,
        max_usd=1.0,
        max_wall_seconds=60.0,
        warning_ratio=0.8,
        clock=lambda: now[0],
    )

    status = guard.record_usage(prompt_tokens=300, completion_tokens=200, cost_usd=0.40)
    assert status.warning_triggered is False
    assert status.consumed_tokens == 500

    # Cross 80% warning threshold on tokens
    status_warn = guard.record_usage(prompt_tokens=250, completion_tokens=100, cost_usd=0.20)
    assert status_warn.warning_triggered is True
    assert "tokens_near_limit" in status_warn.warning_reasons

    # Exceed hard token cap
    with pytest.raises(BudgetExceededError) as exc_info:
        guard.record_usage(prompt_tokens=200, completion_tokens=0, cost_usd=0.10)
    assert "token limit" in exc_info.value.reason

    # Wall-time budget test
    wall_guard = ExecutionBudgetGuard(max_wall_seconds=10.0, clock=lambda: now[0])
    now[0] = 12.5
    with pytest.raises(BudgetExceededError) as wall_exc:
        wall_guard.check()
    assert "wall time limit" in wall_exc.value.reason
