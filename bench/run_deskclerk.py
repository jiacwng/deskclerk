# Runs deskclerk's agent loop on one WorkBench split against the five WorkBench MCP servers, one
# per domain, offering the model every tool, and writes a results file in the format of the
# reference runner.
import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import anthropic
import httpx
import httpx2
from cryptography.fernet import Fernet
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.exceptions import MCPError

from deskclerk.agent import (
    AgentResult,
    ConversationThread,
    Decision,
    Gate,
    Limits,
    ProposedCall,
    ToolAccess,
    Verdict,
    run_agent,
)
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.database import Database
from deskclerk.instructions import INSTRUCTIONS
from deskclerk.providers import Usage, cost_of, model_options
from deskclerk.screen import pass_through
from deskclerk.servers import McpServer
from deskclerk.tokens import TokenStore
from deskclerk.writecheck import write_check
from workbench import SERVERS, Task, date_line, split_tasks, write_results

SERVER = Path(__file__).parent / "workbench_server.py"
# WorkBench's reference loop retries a busy or failed model call up to 10 times.
MODEL_RETRIES = 10
USER = "bench"
CUT_OFF_MESSAGE = "reply cut off at the output limit"
PROVIDER_FAILURES = (anthropic.APIError, MCPError, httpx2.HTTPError)


@dataclass(frozen=True)
class WorkbenchText:
    system: str
    stopped_message: str


@dataclass(frozen=True)
class Worker:
    url: str
    connections: OAuthConnections
    model: BaseChatModel
    gate: Gate
    model_name: str


def load_workbench_text(workbench: Path) -> WorkbenchText:
    sys.path.insert(0, str(workbench))
    from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
    from src.evals.agent import ACT_WITHOUT_CONFIRMATION_SUFFIX, AGENT_STOPPED_MESSAGE

    suffix = ACT_WITHOUT_CONFIRMATION_SUFFIX.strip()
    system = f"{INSTRUCTIONS}\n{date_line(HARDCODED_CURRENT_TIME)} {suffix}"
    return WorkbenchText(system, AGENT_STOPPED_MESSAGE)


async def approve_until_done(
    start: Callable[[dict[str, Decision] | None], Awaitable[AgentResult]], limit: int
) -> list[AgentResult]:
    results = [await start(None)]
    for _ in range(limit):
        if not results[-1].pending:
            break
        results.append(await start({call.id: "approve" for call in results[-1].pending}))
    return results


async def run_task(worker: Worker, task: Task, text: WorkbenchText) -> dict[str, Any]:
    base = worker.url
    servers = [McpServer(name, f"{base}/{name}/mcp", None, None, "none") for name in SERVERS]
    usage: list[Usage] = []
    answer, error = "", ""
    blocked: list[dict[str, Any]] = []
    step_limit: int = Settings.model_fields["step_limit"].default
    conversation = ConversationThread(InMemorySaver(), f"{USER}/{task.domain}-{task.index}", [])

    async def start(decisions: dict[str, Decision] | None) -> AgentResult:
        return await run_agent(
            USER,
            task.text,
            text.system,
            ToolAccess(servers, worker.connections, worker.connections.tokens),
            conversation,
            worker.model,
            worker.gate,
            pass_through,
            Limits(step_limit),
            usage.append,
            decisions=decisions,
        )

    async with httpx.AsyncClient() as http:
        (await http.post(f"{base}/bench/reset")).raise_for_status()
        try:
            results = await approve_until_done(start, step_limit)
            last = results[-1]
            answer = last.answer
            for result in results:
                for call in result.blocked:
                    blocked.append(
                        {"tool": call.tool, "arguments": call.arguments, "reason": call.reason}
                    )
            if last.cut_off:
                error = CUT_OFF_MESSAGE
            elif last.step_limit_hit or last.pending:
                error = text.stopped_message
        except PROVIDER_FAILURES as failure:
            error = str(failure)
        logged = (await http.get(f"{base}/bench/calls")).raise_for_status().json()

    total = Usage(
        sum(step.input_tokens for step in usage),
        sum(step.output_tokens for step in usage),
        sum(step.cache_read_tokens for step in usage),
        sum(step.cache_write_tokens for step in usage),
    )
    return {
        "domain": task.domain,
        "task": task.text,
        "function_calls": str(logged),
        "full_response": answer,
        "error": error,
        "input_tokens": total.input_tokens,
        "output_tokens": total.output_tokens,
        "cache_read_tokens": total.cache_read_tokens,
        "cache_write_tokens": total.cache_write_tokens,
        "cost_usd": cost_of(worker.model_name, total),
        "blocked": json.dumps(blocked),
    }


def start_server(stack: ExitStack, server_python: str, workbench: Path) -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(  # noqa: S603
        [server_python, str(SERVER), "--workbench", str(workbench), "--port", str(port)]
    )
    stack.callback(_stop, process)
    for _ in range(1200):
        if process.poll() is not None:
            raise RuntimeError(f"the workbench server exited with code {process.returncode}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("the workbench server did not start")


def _stop(process: "subprocess.Popen[bytes]") -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


async def run_worker(
    worker: Worker,
    tasks: list[tuple[int, Task]],
    text: WorkbenchText,
    rows: dict[int, dict[str, Any]],
    total: int,
) -> None:
    for position, task in tasks:
        row = await run_task(worker, task, text)
        rows[position] = row
        print(
            f"[{len(rows)}/{total}] {task.domain} #{task.index} "
            f"cost {row['cost_usd']:.4f} USD {row['error']}",
            flush=True,
        )


async def run_split(
    workers: list[Worker], tasks: list[Task], text: WorkbenchText
) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    numbered = list(enumerate(tasks))
    shares = [numbered[index :: len(workers)] for index in range(len(workers))]
    await asyncio.gather(
        *(
            run_worker(worker, share, text, rows, len(tasks))
            for worker, share in zip(workers, shares, strict=True)
        )
    )
    return [rows[position] for position in range(len(tasks))]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deskclerk's agent on a WorkBench split.")
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--gate", choices=["none", "rules"], required=True)
    parser.add_argument("--model", default="anthropic:claude-haiku-5-5")
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


async def allow_everything(call: ProposedCall) -> Verdict:
    return Verdict(True, "no check")


def make_gate(kind: str) -> Gate:
    if kind == "none":
        return allow_everything
    return write_check()


def mixed_domains(tasks: list[Task], limit: int) -> list[Task]:
    by_domain: dict[str, list[Task]] = {}
    for task in tasks:
        by_domain.setdefault(task.domain, []).append(task)
    interleaved = [task for group in zip_longest(*by_domain.values()) for task in group if task]
    return interleaved[:limit]


def make_model(name: str, provider: str) -> Any:
    return init_chat_model(
        name,
        model_provider=provider,
        max_retries=MODEL_RETRIES,
        **model_options(provider),
    )


def make_worker(url: str, connections: OAuthConnections, gate: Gate, model_id: str) -> Worker:
    provider, name = model_id.split(":", 1)
    return Worker(url, connections, make_model(name, provider), gate, model_id)


def main() -> None:
    arguments = parse_arguments()
    workbench: Path = arguments.workbench.resolve()
    cost_of(arguments.model, Usage(0, 0, 0, 0))
    gate = make_gate(arguments.gate)
    text = load_workbench_text(workbench)
    tasks = split_tasks(workbench, arguments.split)
    if arguments.limit is not None:
        tasks = mixed_domains(tasks, arguments.limit)
    database = Database(os.environ["DESKCLERK_DATABASE_URL"])
    with ExitStack() as stack:
        tokens = TokenStore(database, Fernet.generate_key().decode())
        connections = OAuthConnections(tokens, "http://127.0.0.1/callback", "bench", 60)
        workers = [
            make_worker(
                start_server(stack, arguments.server_python, workbench),
                connections,
                gate,
                arguments.model,
            )
            for _ in range(arguments.workers)
        ]
        rows = asyncio.run(run_split(workers, tasks, text))
    write_results(arguments.out, rows)
    print(f"{len(rows)} tasks, total cost {sum(row['cost_usd'] for row in rows):.4f} USD")


if __name__ == "__main__":
    main()
