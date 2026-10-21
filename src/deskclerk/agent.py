# One employee's request answered by a chat model that calls the tools of the MCP servers the
# employee has connected. Tools that are not read-only pass a gate first; a refused call never runs.
# Every result a tool returns passes the screen before the model reads it.

from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deskclerk.connections import (
    UPSTREAM_ERRORS,
    NotConnected,
    OAuthConnections,
    ToolInfo,
    first_leaf,
)
from deskclerk.providers import Usage, usage_from
from deskclerk.screen import UNCHECKED, WITHHELD, Screen
from deskclerk.servers import McpServer


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ProposedCall:
    request: str
    server: str
    tool: ToolInfo
    arguments: dict[str, Any]
    observed: tuple[str, ...]


Gate = Callable[[ProposedCall], Awaitable[Verdict]]


@dataclass(frozen=True)
class ExecutedCall:
    server: str
    tool: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


@dataclass(frozen=True)
class BlockedCall:
    server: str
    tool: str
    arguments: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class AgentResult:
    answer: str
    executed: tuple[ExecutedCall, ...]
    blocked: tuple[BlockedCall, ...]
    not_connected: tuple[str, ...]
    usage: tuple[Usage, ...]
    step_limit_hit: bool
    cut_off: bool
    withheld: int

    @property
    def total_usage(self) -> Usage:
        return Usage(
            sum(call.input_tokens for call in self.usage),
            sum(call.output_tokens for call in self.usage),
            sum(call.cache_read_tokens for call in self.usage),
            sum(call.cache_write_tokens for call in self.usage),
        )


ToolSpan = Callable[[str, str, dict[str, Any]], AbstractContextManager[Callable[[str], None]]]


@contextmanager
def no_tool_span(
    server: str, tool: str, arguments: dict[str, Any]
) -> Iterator[Callable[[str], None]]:
    yield lambda output: None


class McpServerFailed(Exception):
    def __init__(self, server_name: str, cause: str) -> None:
        super().__init__(f"{server_name} failed ({cause})")
        self.server_name = server_name
        self.cause = cause


def _server_failure(server_name: str, error: BaseException) -> McpServerFailed | None:
    if isinstance(error, BaseExceptionGroup):
        matching = error.subgroup(UPSTREAM_ERRORS)
        if matching is None:
            return None
        error = first_leaf(matching)
    return McpServerFailed(server_name, type(error).__name__)


@dataclass(frozen=True)
class _Tool:
    server: McpServer
    info: ToolInfo


async def run_agent(
    user_id: str,
    request: str,
    system: str,
    servers: list[McpServer],
    connections: OAuthConnections,
    model: BaseChatModel,
    gate: Gate,
    screen: Screen,
    step_limit: int,
    record_usage: Callable[[Usage], None],
    trace_config: RunnableConfig | None = None,
    tool_span: ToolSpan = no_tool_span,
) -> AgentResult:
    tools: dict[str, _Tool] = {}
    not_connected: list[str] = []
    for server in servers:
        try:
            listed = await connections.tools(user_id, server)
        except NotConnected:
            not_connected.append(server.name)
            continue
        except (*UPSTREAM_ERRORS, BaseExceptionGroup) as error:
            failure = _server_failure(server.name, error)
            if failure is None:
                raise
            raise failure from None
        for info in listed:
            tools[f"{server.name}__{info.name}"] = _Tool(server, info)

    chat_model: Any = model
    if tools:
        chat_model = model.bind_tools([_tool_schema(name, tool) for name, tool in tools.items()])

    messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(request)]
    executed: list[ExecutedCall] = []
    blocked: list[BlockedCall] = []
    usage: list[Usage] = []
    answer = ""
    step_limit_hit = True
    cut_off = False
    withheld = 0
    for _ in range(step_limit):
        response: AIMessage = await chat_model.ainvoke(messages, config=trace_config)
        messages.append(response)
        step_usage = usage_from(response)
        usage.append(step_usage)
        record_usage(step_usage)
        answer = response.text
        if not response.tool_calls:
            step_limit_hit = False
            cut_off = response.response_metadata.get("stop_reason") == "max_tokens"
            break
        for call in response.tool_calls:
            arguments: dict[str, Any] = call["args"]
            tool = tools.get(call["name"])
            if tool is None:
                text = f"There is no tool named {call['name']}."
                messages.append(ToolMessage(text, tool_call_id=call["id"] or "", status="error"))
                continue
            server_name, tool_name = tool.server.name, tool.info.name
            if not tool.info.read_only:
                proposed = ProposedCall(
                    request,
                    server_name,
                    tool.info,
                    arguments,
                    tuple(done.result for done in executed),
                )
                verdict = await gate(proposed)
                if not verdict.allowed:
                    blocked.append(BlockedCall(server_name, tool_name, arguments, verdict.reason))
                    text = f"This call was blocked and not run: {verdict.reason}"
                    messages.append(
                        ToolMessage(text, tool_call_id=call["id"] or "", status="error")
                    )
                    continue
            try:
                with tool_span(server_name, tool_name, arguments) as record_output:
                    try:
                        result = await connections.call(user_id, tool.server, tool_name, arguments)
                    except NotConnected:
                        raise
                    except (*UPSTREAM_ERRORS, BaseExceptionGroup) as error:
                        failure = _server_failure(server_name, error)
                        if failure is None:
                            raise
                        raise failure from None
                    result_text = await screen(request, result.text)
                    record_output(result_text)
                is_error = result.is_error
                if result_text in (WITHHELD, UNCHECKED):
                    withheld += 1
            except NotConnected as error:
                result_text, is_error = str(error), True
            executed.append(ExecutedCall(server_name, tool_name, arguments, result_text, is_error))
            messages.append(
                ToolMessage(
                    result_text,
                    tool_call_id=call["id"] or "",
                    status="error" if is_error else "success",
                )
            )

    return AgentResult(
        answer,
        tuple(executed),
        tuple(blocked),
        tuple(not_connected),
        tuple(usage),
        step_limit_hit,
        cut_off,
        withheld,
    )


def _tool_schema(name: str, tool: _Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.info.description,
            "parameters": tool.info.input_schema,
        },
    }
