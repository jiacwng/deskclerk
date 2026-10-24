# Runs deskclerk's agent loop on one WorkBench split against the five WorkBench MCP servers, one
# per domain, with one of four ways of choosing the tools the model sees, and writes a results
# file in the format of the reference runner.
import argparse
import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

import anthropic
import httpx
import httpx2
from cryptography.fernet import Fernet
from langchain.agents.middleware import AgentMiddleware, LLMToolSelectorMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.exceptions import MCPError

from deskclerk.agent import AgentResult, Gate, ProposedCall, Verdict, run_agent
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.instructions import INSTRUCTIONS
from deskclerk.jev import PRICE_PER_INPUT_TOKEN, Jev, jev_from_environment
from deskclerk.providers import Usage, cost_of, model_options, usage_from
from deskclerk.scope import jev_tool_selection
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
SELECTOR_CRASH = "tool selector crashed"
SELECTIONS = ("all", "server", "llm")


@dataclass(frozen=True)
class WorkbenchText:
    system: str
    stopped_message: str


class SelectorUsage(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[Usage] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration) and isinstance(
                    generation.message, AIMessage
                ):
                    self.calls.append(usage_from(generation.message))


@dataclass(frozen=True)
class Worker:
    url: str
    connections: OAuthConnections
    model: BaseChatModel
    gate: Gate
    model_name: str
    selection: str = "all"
    jev: Jev | None = None
    selector: BaseChatModel | None = None
    selector_usage: SelectorUsage = field(default_factory=SelectorUsage)


def load_workbench_text(workbench: Path) -> WorkbenchText:
    sys.path.insert(0, str(workbench))
    from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
    from src.evals.agent import ACT_WITHOUT_CONFIRMATION_SUFFIX, AGENT_STOPPED_MESSAGE

    suffix = ACT_WITHOUT_CONFIRMATION_SUFFIX.strip()
    system = f"{INSTRUCTIONS}\n{date_line(HARDCODED_CURRENT_TIME)} {suffix}"
    return WorkbenchText(system, AGENT_STOPPED_MESSAGE)


def make_selection(
    worker: Worker, on_jev_tokens: Callable[[int], None]
) -> AgentMiddleware[Any, Any, Any] | None:
    if worker.selection == "llm":
        return LLMToolSelectorMiddleware(model=worker.selector, on_parsing_failure="all")
    if worker.selection == "server":
        return jev_tool_selection(worker.jev, True, on_jev_tokens)
    return None


async def run_task(worker: Worker, task: Task, text: WorkbenchText) -> dict[str, Any]:
    base = worker.url
    servers = [McpServer(name, f"{base}/{name}/mcp", None, None, "none") for name in SERVERS]
    jev_tokens: list[int] = []
    selector_start = len(worker.selector_usage.calls)
    usage: list[Usage] = []
    answer, error = "", ""
    blocked: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as http:
        (await http.post(f"{base}/bench/reset")).raise_for_status()
        try:
            result: AgentResult = await run_agent(
                USER,
                task.text,
                text.system,
                servers,
                worker.connections,
                worker.connections.tokens,
                InMemorySaver(),
                f"{USER}/{task.domain}-{task.index}",
                [],
                worker.model,
                worker.gate,
                pass_through,
                Settings.model_fields["step_limit"].default,
                usage.append,
                selection=make_selection(worker, jev_tokens.append),
            )
            answer = result.answer
            for call in result.blocked:
                blocked.append(
                    {"tool": call.tool, "arguments": call.arguments, "reason": call.reason}
                )
            if result.cut_off:
                error = CUT_OFF_MESSAGE
            elif result.step_limit_hit:
                error = text.stopped_message
        except PROVIDER_FAILURES as failure:
            error = str(failure)
        except ValueError as failure:
            if worker.selection != "llm":
                raise
            error = f"{SELECTOR_CRASH}: {type(failure).__name__}: {failure}"
        logged = (await http.get(f"{base}/bench/calls")).raise_for_status().json()

    total = Usage(
        sum(step.input_tokens for step in usage),
        sum(step.output_tokens for step in usage),
        sum(step.cache_read_tokens for step in usage),
        sum(step.cache_write_tokens for step in usage),
    )
    selector = Usage(0, 0, 0, 0)
    for call in worker.selector_usage.calls[selector_start:]:
        selector = Usage(
            selector.input_tokens + call.input_tokens,
            selector.output_tokens + call.output_tokens,
            selector.cache_read_tokens + call.cache_read_tokens,
            selector.cache_write_tokens + call.cache_write_tokens,
        )
    selection_usd = sum(jev_tokens) * PRICE_PER_INPUT_TOKEN + cost_of(worker.model_name, selector)
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
        "cost_usd": cost_of(worker.model_name, total) + selection_usd,
        "blocked": json.dumps(blocked),
        "selection_usd": selection_usd,
        "selector_input_tokens": selector.input_tokens,
        "jev_input_tokens": sum(jev_tokens),
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
    parser.add_argument("--selection", choices=SELECTIONS, default="all")
    parser.add_argument("--jev-budget-usd", type=float, default=1.0)
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


def make_model(name: str, provider: str, callbacks: list[BaseCallbackHandler] | None) -> Any:
    return init_chat_model(
        name,
        model_provider=provider,
        max_retries=MODEL_RETRIES,
        callbacks=callbacks,
        **model_options(provider),
    )


def make_worker(
    url: str, connections: OAuthConnections, gate: Gate, jev: Jev | None, arguments: Any
) -> Worker:
    provider, name = arguments.model.split(":", 1)
    usage = SelectorUsage()
    return Worker(
        url,
        connections,
        make_model(name, provider, None),
        gate,
        arguments.model,
        arguments.selection,
        jev,
        make_model(name, provider, [usage]) if arguments.selection == "llm" else None,
        usage,
    )


def main() -> None:
    arguments = parse_arguments()
    workbench: Path = arguments.workbench.resolve()
    cost_of(arguments.model, Usage(0, 0, 0, 0))
    gate = make_gate(arguments.gate)
    text = load_workbench_text(workbench)
    tasks = split_tasks(workbench, arguments.split)
    if arguments.limit is not None:
        tasks = mixed_domains(tasks, arguments.limit)
    jev = jev_from_environment(arguments.jev_budget_usd)
    if arguments.selection == "server" and jev is None:
        raise SystemExit("TYPESAFE_API_KEY is not set")
    with ExitStack() as stack, tempfile.TemporaryDirectory() as scratch:
        tokens = TokenStore(Path(scratch) / "tokens.sqlite", Fernet.generate_key().decode())
        connections = OAuthConnections(tokens, "http://127.0.0.1/callback", "bench", 60)
        workers = [
            make_worker(
                start_server(stack, arguments.server_python, workbench),
                connections,
                gate,
                jev,
                arguments,
            )
            for _ in range(arguments.workers)
        ]
        rows = asyncio.run(run_split(workers, tasks, text))
    write_results(arguments.out, rows)
    print(f"{len(rows)} tasks, total cost {sum(row['cost_usd'] for row in rows):.4f} USD")


if __name__ == "__main__":
    main()
