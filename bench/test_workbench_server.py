# Runs the WorkBench MCP servers on a free port and checks tools, annotations, call log and grading.
import ast
import asyncio
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from workbench import SERVERS, server_of

SERVER = Path(__file__).parent / "workbench_server.py"
WORKBENCH = Path(os.environ["WORKBENCH_PATH"]).resolve()
TASKS = WORKBENCH / "data/processed/tasks_and_outcomes/calendar_tasks_and_outcomes.csv"


@pytest.fixture(scope="module")
def url() -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, str(SERVER), "--workbench", str(WORKBENCH), "--port", str(port)]
    )
    for _ in range(1200):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        process.terminate()
        pytest.fail("server did not start")
    yield f"http://127.0.0.1:{port}"
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def workbench() -> Iterator[Any]:
    sys.path.insert(0, str(WORKBENCH))
    previous = os.getcwd()
    os.chdir(WORKBENCH)
    import src.evals.evaluation as evaluation
    import src.tools.toolkits as toolkits

    yield type("WorkBench", (), {"evaluation": evaluation, "toolkits": toolkits})
    os.chdir(previous)
    sys.path.remove(str(WORKBENCH))


def single_call_task() -> tuple[str, list[str]]:
    with TASKS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            outcome = ast.literal_eval(row["outcome"])
            if len(outcome) == 1:
                return row["task"], outcome
    raise AssertionError("no single call task")


def call(url: str, name: str, arguments: dict[str, str] | None = None) -> tuple[bool, str]:
    server, tool = name.split("__", 1)

    async def run() -> tuple[bool, str]:
        async with Client(f"{url}/{server}/mcp") as client:
            result = await client.call_tool(tool, arguments or {})
            return result.is_error, result.content[0].text  # type: ignore[union-attr]

    return asyncio.run(run())


def reset(url: str) -> None:
    request = urllib.request.Request(url + "/bench/reset", method="POST")  # noqa: S310
    urllib.request.urlopen(request).close()  # noqa: S310


def bench_calls(url: str) -> list[str]:
    with urllib.request.urlopen(url + "/bench/calls") as response:  # noqa: S310
        return json.load(response)  # type: ignore[no-any-return]


def list_tools(url: str) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        tools: dict[str, Any] = {}
        for server in SERVERS:
            async with Client(f"{url}/{server}/mcp") as client:
                for tool in (await client.list_tools()).tools:
                    tools[f"{server}__{tool.name}"] = tool
        return tools

    return asyncio.run(run())


def test_each_domain_is_a_server_and_together_they_list_every_workbench_tool(
    url: str, workbench: Any
) -> None:
    names = set(list_tools(url))
    expected = {
        f"{server_of(tool.name)}__{tool.name.split('.', 1)[1]}"
        for tool in workbench.toolkits.all_tools
    }
    assert len(expected) == 27
    assert names == expected
    assert "email__find_email_address" in names
    assert {name.split("__")[0] for name in names} == set(SERVERS)


def test_writers_are_workbench_side_effect_tools(url: str, workbench: Any) -> None:
    tools = list_tools(url)
    writers = {n for n, tool in tools.items() if tool.annotations.read_only_hint is False}
    expected = {
        f"{server_of(t.name)}__{t.name.split('.', 1)[1]}"
        for t in workbench.toolkits.tools_with_side_effects
    }
    assert len(writers) == 14
    assert writers == expected


def test_annotations(url: str) -> None:
    tools = list_tools(url)
    read = tools["calendar__search_events"].annotations
    delete = tools["calendar__delete_event"].annotations
    create = tools["calendar__create_event"].annotations
    assert read.read_only_hint is True
    assert delete.read_only_hint is False
    assert delete.destructive_hint is True
    assert create.destructive_hint is False
    assert "Returns the events for a given query" in tools["calendar__search_events"].description


def test_logged_calls_are_replayed_by_workbench_grader(url: str, workbench: Any) -> None:
    _, truth = single_call_task()
    event_id = ast.literal_eval(truth[0].split("event_id=")[1].rstrip(")"))
    reset(url)
    call(
        url, "calendar__get_event_information_by_id", {"event_id": event_id, "field": "event_name"}
    )
    call(url, "calendar__delete_event", {"event_id": event_id})

    logged = bench_calls(url)
    assert logged[1:] == truth
    assert logged[0] == (
        f'calendar.get_event_information_by_id.func(event_id="{event_id}", field="event_name")'
    )
    assert workbench.evaluation.is_exact_match(logged, truth)
    assert workbench.evaluation.is_correct(logged, truth, "")
    wrong = [truth[0].replace(event_id, "00000000")]
    assert not workbench.evaluation.is_correct(wrong, truth, "")

    assert (
        "not found"
        in call(
            url,
            "calendar__get_event_information_by_id",
            {"event_id": event_id, "field": "event_name"},
        )[1]
    )
    reset(url)
    assert bench_calls(url) == []
    assert (
        "event_name"
        in call(
            url,
            "calendar__get_event_information_by_id",
            {"event_id": event_id, "field": "event_name"},
        )[1]
    )


def test_failing_tool_returns_error_and_server_survives(url: str) -> None:
    reset(url)
    is_error, text = call(url, "calendar__search_events", {"time_min": "not a date"})
    assert is_error
    assert "Unknown datetime string format" in text
    assert bench_calls(url) == []


def test_calls_on_different_servers_land_in_one_log_and_one_reset_clears_it(url: str) -> None:
    reset(url)
    call(url, "email__find_email_address", {"name": "Sam"})
    call(url, "calendar__search_events", {"query": "x"})
    logged = bench_calls(url)
    assert [entry.split(".func")[0] for entry in logged] == [
        "company_directory.find_email_address",
        "calendar.search_events",
    ]
    reset(url)
    assert bench_calls(url) == []
