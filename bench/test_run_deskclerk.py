# Runs one WorkBench task through deskclerk's agent with a scripted model against the real
# WorkBench MCP server: allowed and refused writes, the step limit, and the reset between tasks.
import ast
import asyncio
import csv
import itertools
import json
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.fernet import Fernet
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from deskclerk.agent import Gate, ProposedCall, Verdict
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.jev import Jev
from deskclerk.tokens import TokenStore
from run_deskclerk import (
    CUT_OFF_MESSAGE,
    WorkbenchText,
    Worker,
    allow_everything,
    load_workbench_text,
    mixed_domains,
    run_task,
    start_server,
)
from workbench import Task

WORKBENCH = Path(os.environ["WORKBENCH_PATH"]).resolve()
TASKS = WORKBENCH / "data/processed/tasks_and_outcomes/calendar_tasks_and_outcomes.csv"
MODEL = "anthropic:claude-haiku-5-5"


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call-0") -> AIMessage:
    call = {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def scripted(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies))


async def refuse_everything(call: ProposedCall) -> Verdict:
    return Verdict(False, "not asked for")


@pytest.fixture(scope="module")
def url() -> Iterator[str]:
    with ExitStack() as stack:
        yield start_server(stack, os.environ["SERVER_PYTHON"], WORKBENCH)


@pytest.fixture(scope="module")
def text() -> WorkbenchText:
    return load_workbench_text(WORKBENCH)


@pytest.fixture
def connections(tmp_path: Path) -> OAuthConnections:
    tokens = TokenStore(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())
    return OAuthConnections(tokens, "http://127.0.0.1/callback", "bench", 60)


def delete_task() -> tuple[Task, str, str]:
    with TASKS.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            outcome = ast.literal_eval(row["outcome"])
            if len(outcome) == 1 and ".delete_event.func(" in outcome[0]:
                event_id = ast.literal_eval(outcome[0].split("event_id=")[1].rstrip(")"))
                return Task("calendar", index, row["task"]), event_id, outcome[0]
    raise AssertionError("no single delete task")


def run(worker: Worker, task: Task, text: WorkbenchText) -> dict[str, Any]:
    return asyncio.run(run_task(worker, task, text))


def worker_for(url: str, connections: OAuthConnections, model: Any, gate: Gate) -> Worker:
    return Worker(url, connections, model, gate, MODEL)


def test_an_allowed_write_is_logged_in_workbench_format(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, expected = delete_task()
    model = scripted(
        tool_call("calendar__delete_event", {"event_id": event_id}),
        AIMessage(content="Deleted."),
    )
    row = run(worker_for(url, connections, model, allow_everything), task, text)
    assert ast.literal_eval(row["function_calls"]) == [expected]
    assert row["full_response"] == "Deleted."
    assert row["error"] == ""
    assert json.loads(row["blocked"]) == []


def test_a_refused_write_is_blocked_and_not_logged(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, _ = delete_task()
    model = scripted(
        tool_call("calendar__delete_event", {"event_id": event_id}),
        AIMessage(content="I could not delete it."),
    )
    row = run(worker_for(url, connections, model, refuse_everything), task, text)
    assert row["function_calls"] == "[]"
    assert json.loads(row["blocked"]) == [
        {
            "tool": "delete_event",
            "arguments": {"event_id": event_id},
            "reason": "not asked for",
        }
    ]


def test_the_step_limit_sets_the_stopped_error(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, _ = delete_task()
    looks = (
        tool_call(
            "calendar__get_event_information_by_id",
            {"event_id": event_id, "field": "event_name"},
            f"call-{number}",
        )
        for number in itertools.count()
    )
    model = ScriptedModel(messages=looks)
    row = run(worker_for(url, connections, model, allow_everything), task, text)
    assert row["error"] == text.stopped_message
    assert (
        len(ast.literal_eval(row["function_calls"])) == Settings.model_fields["step_limit"].default
    )


def test_the_server_is_reset_between_tasks(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, expected = delete_task()
    first = scripted(
        tool_call("calendar__delete_event", {"event_id": event_id}),
        AIMessage(content="Deleted."),
    )
    run(worker_for(url, connections, first, allow_everything), task, text)
    second = scripted(
        tool_call(
            "calendar__get_event_information_by_id",
            {"event_id": event_id, "field": "event_name"},
        ),
        AIMessage(content="Found."),
    )
    row = run(worker_for(url, connections, second, allow_everything), task, text)
    calls = ast.literal_eval(row["function_calls"])
    assert len(calls) == 1
    assert expected not in calls


def test_a_reply_cut_off_at_the_output_limit_sets_the_cut_off_error(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, _, _ = delete_task()
    cut = AIMessage(content="The event is", response_metadata={"stop_reason": "max_tokens"})
    row = run(worker_for(url, connections, scripted(cut), allow_everything), task, text)
    assert row["error"] == CUT_OFF_MESSAGE


class FakeJev:
    def __init__(self, wanted: set[str]) -> None:
        self.wanted = wanted
        self.calls = 0

    def noul_many(
        self, state: Any, instructions: dict[str, str], on_tokens: Any = None
    ) -> dict[str, float]:
        self.calls += 1
        on_tokens(100)
        return {name: 1.0 if name in self.wanted else 0.0 for name in instructions}


class RecordingModel(ScriptedModel):
    binds: list[list[str]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.binds.append([tool.name for tool in tools])
        return super().bind_tools(tools, **kwargs)


def selecting_worker(url: str, connections: OAuthConnections, selection: str, jev: Any) -> Worker:
    model = RecordingModel(messages=iter([AIMessage(content="Nothing to do.")]))
    model.binds = []
    return Worker(url, connections, model, allow_everything, MODEL, selection, cast(Jev, jev))


def test_the_server_arm_offers_only_the_servers_jev_picked_and_bills_jev(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, _, _ = delete_task()
    jev = FakeJev({"calendar"})
    worker = selecting_worker(url, connections, "server", jev)
    row = run(worker, task, text)
    [offered] = cast(RecordingModel, worker.model).binds
    assert offered and all(name.startswith("calendar__") for name in offered)
    assert jev.calls == 1
    assert row["jev_input_tokens"] == 100
    assert row["selection_usd"] == pytest.approx(row["jev_input_tokens"] * 0.042 / 1_000_000)


def test_the_all_arm_offers_every_tool_and_asks_nothing(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, _, _ = delete_task()
    worker = selecting_worker(url, connections, "all", None)
    row = run(worker, task, text)
    [offered] = cast(RecordingModel, worker.model).binds
    assert len(offered) == 27
    assert row["selection_usd"] == 0


def test_a_limited_run_takes_tasks_round_robin_over_the_domains() -> None:
    tasks = [Task(domain, index, "") for domain in ("a", "b", "c") for index in range(3)]
    mixed = mixed_domains(tasks, 5)
    assert [(task.domain, task.index) for task in mixed] == [
        ("a", 0),
        ("b", 0),
        ("c", 0),
        ("a", 1),
        ("b", 1),
    ]
