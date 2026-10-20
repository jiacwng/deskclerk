# Checks stage 1 of the Jev measurement against the real WorkBench tools: call strings parse, the
# replay returns what the MCP server returns, and the labels follow the ground truth.
import ast
import csv
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jev_calls import Sandbox, collect, load_outcomes, load_sandbox, parse_call, replay, split_of
from test_workbench_server import call, reset, url  # noqa: F401
from workbench import RESULT_COLUMNS, split_tasks, write_results

WORKBENCH = Path(os.environ["WORKBENCH_PATH"]).resolve()
READ = 'calendar.get_event_information_by_id.func(event_id="00000000", field="event_name")'


@pytest.fixture(scope="module")
def sandbox() -> Iterator[Sandbox]:
    previous = os.getcwd()
    yield load_sandbox(WORKBENCH)
    os.chdir(previous)
    sys.path.remove(str(WORKBENCH))


def two_calendar_tasks() -> tuple[tuple[str, list[str]], tuple[str, list[str]]]:
    outcomes = load_outcomes(WORKBENCH, "calendar")
    dev = {task.text for task in split_tasks(WORKBENCH, "dev") if task.domain == "calendar"}
    csv.field_size_limit(sys.maxsize)
    path = WORKBENCH / "data/processed/tasks_and_outcomes/calendar_tasks_and_outcomes.csv"
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["task"] in dev]
    chosen = [
        (row["task"], ast.literal_eval(row["outcome"])) for row in rows if outcomes[row["task"]]
    ]
    first, second = chosen[0], next(item for item in chosen if set(item[1]) != set(chosen[0][1]))
    return first, second


def test_parse_call_reads_the_tool_and_unescapes_the_values() -> None:
    name, arguments = parse_call(
        'email.send_email.func(recipient="a@b.c", body="line\\none \\"quoted\\" \\\\ end")'
    )
    assert name == "email.send_email"
    assert arguments == {"recipient": "a@b.c", "body": 'line\none "quoted" \\ end'}


def test_parse_call_never_evaluates_code() -> None:
    with pytest.raises(ValueError):
        parse_call('email.delete_email.func(email_id=__import__("os").getcwd())')


def test_split_comes_from_the_file_name() -> None:
    assert split_of(Path("haiku55_dev_none.csv")) == "dev"
    assert split_of(Path("haiku55_test_rules.csv")) == "test"
    with pytest.raises(ValueError):
        split_of(Path("haiku55_none.csv"))


def test_replay_returns_what_the_server_returns(sandbox: Sandbox, url: str) -> None:  # noqa: F811
    (_, wanted), _ = two_calendar_tasks()
    calls = [READ, *wanted]
    replayed = replay(sandbox, calls)
    reset(url)
    for step in replayed:
        is_error, text = call(url, step.tool.name.replace(".", "__"), step.arguments)
        assert not is_error
        assert text == step.result


def test_replay_starts_from_a_fresh_sandbox_every_time(sandbox: Sandbox) -> None:
    (_, wanted), _ = two_calendar_tasks()
    first = [step.result for step in replay(sandbox, wanted)]
    second = [step.result for step in replay(sandbox, wanted)]
    assert first == second


def test_write_calls_carry_their_context_and_the_ground_truth_label(
    sandbox: Sandbox, tmp_path: Path
) -> None:
    (task, wanted), (_, other) = two_calendar_tasks()
    extra = next(action for action in other if action not in wanted)
    row: dict[str, Any] = dict.fromkeys(RESULT_COLUMNS, "")
    row |= {"domain": "calendar", "task": task, "function_calls": str([READ, *wanted, extra])}
    results = tmp_path / "haiku55_dev_made_up.csv"
    write_results(results, [row])

    lines = collect(sandbox, WORKBENCH, results)

    assert [line["label"] for line in lines] == ["wanted"] * len(wanted) + ["unwanted"]
    assert [line["call"] for line in lines] == [*wanted, extra]
    first = lines[0]
    assert first["run"] == "haiku55_dev_made_up"
    assert first["split"] == "dev"
    assert first["request"].endswith(f"\n\n{task}")
    assert "Do not ask for confirmation" in first["request"]
    assert first["request"].startswith("Today's date is")
    assert "." not in first["tool"]
    assert first["tool"].startswith("calendar__")
    assert first["description"]
    assert len(first["earlier_results"]) == 1
    assert len(lines[-1]["earlier_results"]) == 1 + len(wanted)
    json.dumps(lines)
