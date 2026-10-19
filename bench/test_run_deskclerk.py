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
from typing import Any

import pytest
from cryptography.fernet import Fernet
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from deskclerk.agent import Gate, ProposedCall, Verdict
from deskclerk.connections import OAuthConnections
from deskclerk.tokens import TokenStore
from run_deskclerk import (
    WorkbenchText,
    Worker,
    allow_everything,
    load_workbench_text,
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


def tool_call(name: str, arguments: dict[str, Any]) -> AIMessage:
    call = {"name": name, "args": arguments, "id": "call-0", "type": "tool_call"}
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
    return Worker(url, connections, model, gate, MODEL, None)


def test_an_allowed_write_is_logged_in_workbench_format(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, expected = delete_task()
    model = scripted(
        tool_call("workbench__calendar__delete_event", {"event_id": event_id}),
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
        tool_call("workbench__calendar__delete_event", {"event_id": event_id}),
        AIMessage(content="I could not delete it."),
    )
    row = run(worker_for(url, connections, model, refuse_everything), task, text)
    assert row["function_calls"] == "[]"
    assert json.loads(row["blocked"]) == [
        {
            "tool": "calendar__delete_event",
            "arguments": {"event_id": event_id},
            "reason": "not asked for",
        }
    ]


def test_the_step_limit_sets_the_stopped_error(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, _ = delete_task()
    look = tool_call(
        "workbench__calendar__get_event_information_by_id",
        {"event_id": event_id, "field": "event_name"},
    )
    model = ScriptedModel(messages=itertools.repeat(look))
    row = run(worker_for(url, connections, model, allow_everything), task, text)
    assert row["error"] == text.stopped_message
    assert len(ast.literal_eval(row["function_calls"])) == 20


def test_the_server_is_reset_between_tasks(
    url: str, connections: OAuthConnections, text: WorkbenchText
) -> None:
    task, event_id, expected = delete_task()
    first = scripted(
        tool_call("workbench__calendar__delete_event", {"event_id": event_id}),
        AIMessage(content="Deleted."),
    )
    run(worker_for(url, connections, first, allow_everything), task, text)
    second = scripted(
        tool_call(
            "workbench__calendar__get_event_information_by_id",
            {"event_id": event_id, "field": "event_name"},
        ),
        AIMessage(content="Found."),
    )
    row = run(worker_for(url, connections, second, allow_everything), task, text)
    calls = ast.literal_eval(row["function_calls"])
    assert len(calls) == 1
    assert expected not in calls
