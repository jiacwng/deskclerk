# The bench's loop that approves every pending write until the agent stops asking.

from types import SimpleNamespace
from typing import Any, cast

import pytest

from deskclerk.agent import AgentResult, Decision
from run_deskclerk import approve_until_done


def pausing(*ids: str) -> AgentResult:
    return cast(AgentResult, SimpleNamespace(pending=[SimpleNamespace(id=id_) for id_ in ids]))


def finished() -> AgentResult:
    return cast(AgentResult, SimpleNamespace(pending=[]))


class Script:
    def __init__(self, *results: AgentResult) -> None:
        self.results = list(results)
        self.decisions: list[dict[str, Decision] | None] = []

    async def start(self, decisions: dict[str, Decision] | None) -> AgentResult:
        self.decisions.append(decisions)
        return self.results.pop(0)


@pytest.mark.anyio
async def test_every_pending_call_is_approved_until_nothing_is_pending() -> None:
    script = Script(pausing("a", "b"), pausing("c"), finished())
    results = await approve_until_done(script.start, 8)
    assert len(results) == 3
    assert script.decisions == [None, {"a": "approve", "b": "approve"}, {"c": "approve"}]


@pytest.mark.anyio
async def test_a_run_that_never_stops_asking_ends_at_the_limit() -> None:
    results: list[Any] = [pausing("a")] * 10
    script = Script(*results)
    done = await approve_until_done(script.start, 3)
    assert len(done) == 4
    assert done[-1].pending
