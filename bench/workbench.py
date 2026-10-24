# What both WorkBench runners share: the task files, the seeded dev/test split, the line WorkBench
# puts before every task, the server of each tool, and the columns of one run's results file.
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DOMAINS = (
    "analytics",
    "calendar",
    "customer_relationship_manager",
    "email",
    "project_management",
    "multi_domain",
)
SERVERS = ("email", "calendar", "analytics", "customer_relationship_manager", "project_management")
SERVER_OF_PREFIX = {"company_directory": "email"}
RESULT_COLUMNS = [
    "domain",
    "task",
    "function_calls",
    "full_response",
    "error",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "blocked",
    "selection_usd",
    "selector_input_tokens",
    "jev_input_tokens",
]


@dataclass(frozen=True)
class Task:
    domain: str
    index: int
    text: str


def server_of(tool_name: str) -> str:
    prefix = tool_name.split(".")[0]
    return SERVER_OF_PREFIX.get(prefix, prefix)


def agent_name(tool_name: str) -> str:
    return f"{server_of(tool_name)}__{tool_name.split('.', 1)[1]}"


def tasks_file(workbench: Path, domain: str) -> Path:
    return workbench / "data/processed/tasks_and_outcomes" / f"{domain}_tasks_and_outcomes.csv"


def load_tasks(workbench: Path, domain: str) -> list[Task]:
    with tasks_file(workbench, domain).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [Task(domain, index, row["task"]) for index, row in enumerate(rows)]


# Per domain, tasks are ordered by the sha256 of "<domain>-<index>" and the first third is dev.
def split_tasks(workbench: Path, part: str) -> list[Task]:
    chosen: list[Task] = []
    for domain in DOMAINS:
        tasks = load_tasks(workbench, domain)
        ordered = sorted(tasks, key=lambda task: _digest(task))
        dev_size = round(len(ordered) / 3)
        chosen += ordered[:dev_size] if part == "dev" else ordered[dev_size:]
    return chosen


def _digest(task: Task) -> str:
    return hashlib.sha256(f"{task.domain}-{task.index:03d}".encode()).hexdigest()


def date_line(now: datetime) -> str:
    return (
        f"Today's date is {now.strftime('%A')}, {now.date()} "
        f"and the current time is {now.time()}. "
        f"Remember the current date and time when completing tasks. "
        f"Meetings must not start before 9am or end after 6pm."
    )


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
