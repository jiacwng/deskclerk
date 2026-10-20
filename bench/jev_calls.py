# Stage 1 of the Jev measurement: replays the calls logged in WorkBench results files in a fresh
# sandbox and writes every write call as one JSON line with its request, tool, arguments, the
# results before it, and whether the task's ground truth contains it.
import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workbench import date_line, tasks_file

TOOL_ERRORS = (KeyError, ValueError, TypeError, IndexError, AttributeError)
SPLIT_IN_NAME = re.compile(r"_(dev|test)_")


@dataclass(frozen=True)
class Sandbox:
    request_prefix: str
    reset: Any
    tools: dict[str, Any]
    write_names: frozenset[str]


@dataclass(frozen=True)
class ReplayedCall:
    call: str
    tool: Any
    arguments: dict[str, str]
    result: str


def load_sandbox(workbench: Path) -> Sandbox:
    sys.path.insert(0, str(workbench))
    os.chdir(workbench)
    from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
    from src.evals.agent import ACT_WITHOUT_CONFIRMATION_SUFFIX
    from src.tools.state import reset_state
    from src.tools.toolkits import all_tools, tools_with_side_effects

    prefix = f"{date_line(HARDCODED_CURRENT_TIME)} {ACT_WITHOUT_CONFIRMATION_SUFFIX.strip()}"
    return Sandbox(
        prefix,
        reset_state,
        {tool.name: tool for tool in all_tools},
        frozenset(tool.name for tool in tools_with_side_effects),
    )


def parse_call(call: str) -> tuple[str, dict[str, str]]:
    expression = ast.parse(call, mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"not a call: {call}")
    parts: list[str] = []
    node = expression.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError(f"not a tool call: {call}")
    parts.append(node.id)
    parts.reverse()
    if parts[-1] == "func":
        parts.pop()
    arguments = {}
    for keyword in expression.keywords:
        if keyword.arg is not None:
            arguments[keyword.arg] = str(ast.literal_eval(keyword.value))
    return ".".join(parts), arguments


def replay(sandbox: Sandbox, calls: list[str]) -> list[ReplayedCall]:
    sandbox.reset()
    replayed = []
    for call in calls:
        name, arguments = parse_call(call)
        tool = sandbox.tools[name]
        try:
            result = str(tool(**arguments))
        except TOOL_ERRORS as error:
            result = f"{type(error).__name__}: {error}"
        replayed.append(ReplayedCall(call, tool, arguments, result))
    sandbox.reset()
    return replayed


def load_outcomes(workbench: Path, domain: str) -> dict[str, set[str]]:
    csv.field_size_limit(sys.maxsize)
    with tasks_file(workbench, domain).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["task"]: {action.replace("\n", "\n") for action in ast.literal_eval(row["outcome"])}
        for row in rows
    }


def split_of(path: Path) -> str:
    found = SPLIT_IN_NAME.search(path.name)
    if found is None:
        raise ValueError(f"cannot tell dev from test in the file name {path.name}")
    return found.group(1)


def write_calls_of(
    sandbox: Sandbox, run: str, split: str, row: dict[str, str], wanted: set[str]
) -> list[dict[str, Any]]:
    calls: list[str] = ast.literal_eval(row["function_calls"])
    replayed = replay(sandbox, calls)
    request = f"{sandbox.request_prefix}\n\n{row['task']}"
    lines = []
    for position, step in enumerate(replayed):
        if step.tool.name not in sandbox.write_names:
            continue
        lines.append(
            {
                "run": run,
                "split": split,
                "domain": row["domain"],
                "task": row["task"],
                "position": position,
                "call": step.call,
                "request": request,
                "tool": step.tool.name.replace(".", "__"),
                "description": step.tool.description,
                "arguments": step.arguments,
                "earlier_results": [earlier.result for earlier in replayed[:position]],
                "label": "wanted" if step.call.replace("\n", "\n") in wanted else "unwanted",
            }
        )
    return lines


def collect(sandbox: Sandbox, workbench: Path, results: Path) -> list[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    split = split_of(results)
    with results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    outcomes: dict[str, dict[str, set[str]]] = {}
    lines = []
    for row in rows:
        domain = row["domain"]
        if domain not in outcomes:
            outcomes[domain] = load_outcomes(workbench, domain)
        lines += write_calls_of(sandbox, results.stem, split, row, outcomes[domain][row["task"]])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="List the write calls of WorkBench runs.")
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    workbench = arguments.workbench.resolve()
    results = [path.resolve() for path in arguments.results]
    out = arguments.out.resolve()
    sandbox = load_sandbox(workbench)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for path in results:
            lines = collect(sandbox, workbench, path)
            for line in lines:
                handle.write(json.dumps(line) + "\n")
            counts = Counter(line["label"] for line in lines)
            print(f"{path.stem}: {len(lines)} write calls, {dict(counts)}")


if __name__ == "__main__":
    main()
