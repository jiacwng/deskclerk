# Prometheus counters and histograms for /metrics. Labels hold only bounded values: an outcome, a
# decision or the name of an MCP server an admin set, never an employee or any text.

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from deskclerk.agent import AgentResult, ExecutedCall, ToolSpan


class Metrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self._registry = registry
        self._asks = Counter(
            "deskclerk_asks",
            "Asks and confirmations finished, by outcome",
            ["outcome"],
            registry=registry,
        )
        self._tool_calls = Counter(
            "deskclerk_tool_calls",
            "Tool calls run, by MCP server and outcome",
            ["server", "outcome"],
            registry=registry,
        )
        self._confirmations = Counter(
            "deskclerk_confirmations",
            "Decisions on calls that waited for the employee, by decision",
            ["decision"],
            registry=registry,
        )
        self._screen_flags = Counter(
            "deskclerk_screen_flags",
            "Tool results flagged by the injection screen",
            registry=registry,
        )
        self._cost = Counter(
            "deskclerk_model_cost_usd", "Model and screen cost in US dollars", registry=registry
        )
        self._ask_seconds = Histogram(
            "deskclerk_ask_duration_seconds",
            "Seconds from receiving an ask to finishing it",
            registry=registry,
        )
        self._tool_seconds = Histogram(
            "deskclerk_tool_call_duration_seconds",
            "Seconds a tool call took, by MCP server",
            ["server"],
            registry=registry,
        )

    def ask_finished(
        self,
        outcome: str,
        started: float,
        cost_usd: float,
        executed: Sequence[ExecutedCall],
        result: AgentResult | None,
    ) -> None:
        self._asks.labels(outcome).inc()
        self._ask_seconds.observe(time.monotonic() - started)
        self._cost.inc(cost_usd)
        for call in executed:
            self._tool_calls.labels(call.server, "error" if call.is_error else "ok").inc()
        if result is not None:
            self._screen_flags.inc(result.flagged)
            for decided in result.decided:
                self._confirmations.labels(decided.decision).inc()

    def timed(self, inner: ToolSpan) -> ToolSpan:
        @contextmanager
        def span(server: str, tool: str, arguments: dict[str, Any]) -> Iterator[Any]:
            started = time.monotonic()
            try:
                with inner(server, tool, arguments) as record_output:
                    yield record_output
            finally:
                self._tool_seconds.labels(server).observe(time.monotonic() - started)

        return span

    def render(self) -> bytes:
        return generate_latest(self._registry)
