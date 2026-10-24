# One employee's request answered by a LangChain agent whose model calls the tools of the MCP
# servers the employee has connected. Tools that are not read-only pass a gate first; a refused call
# never runs. Every result a tool returns passes the screen before the model reads it, a tool whose
# definition changed since the employee first used its server is not offered at all, and only
# read-only tools are retried after a transport error.

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from deskclerk.connections import (
    UPSTREAM_ERRORS,
    NotConnected,
    OAuthConnections,
    ToolInfo,
    first_leaf,
)
from deskclerk.pinning import pin_tools
from deskclerk.providers import Usage, usage_from
from deskclerk.screen import UNCHECKED, WITHHELD, Screen
from deskclerk.servers import McpServer
from deskclerk.tokens import TokenStore


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
    changed_tools: tuple[str, ...]

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


def _is_upstream_error(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return error.subgroup(UPSTREAM_ERRORS) is not None
    return isinstance(error, UPSTREAM_ERRORS) and not isinstance(error, NotConnected)


@dataclass(frozen=True)
class _Tool:
    server: McpServer
    info: ToolInfo


class _Guard(AgentMiddleware[AgentState[Any], None, Any]):
    def __init__(
        self,
        request: str,
        tools: dict[str, _Tool],
        gate: Gate,
        screen: Screen,
        tool_span: ToolSpan,
        record_usage: Callable[[Usage], None],
    ) -> None:
        super().__init__()
        self.request = request
        self.offered = tools
        self.gate = gate
        self.screen = screen
        self.tool_span = tool_span
        self.record_usage = record_usage
        self.one_call_at_a_time = asyncio.Lock()
        self.executed: list[ExecutedCall] = []
        self.blocked: list[BlockedCall] = []
        self.usage: list[Usage] = []
        self.last_response: AIMessage | None = None
        self.withheld = 0

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        response = await handler(request)
        message = response.result[-1]
        if not isinstance(message, AIMessage):
            raise TypeError(f"the model returned a {type(message).__name__}")
        step_usage = usage_from(message)
        self.usage.append(step_usage)
        self.record_usage(step_usage)
        self.last_response = message
        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        call_id = call["id"] or ""
        tool = self.offered.get(call["name"])
        if tool is None:
            text = f"There is no tool named {call['name']}."
            return ToolMessage(text, tool_call_id=call_id, status="error")
        arguments: dict[str, Any] = call["args"]
        server_name, tool_name = tool.server.name, tool.info.name
        async with self.one_call_at_a_time:
            if not tool.info.read_only:
                proposed = ProposedCall(
                    self.request,
                    server_name,
                    tool.info,
                    arguments,
                    tuple(done.result for done in self.executed),
                )
                verdict = await self.gate(proposed)
                if not verdict.allowed:
                    self.blocked.append(
                        BlockedCall(server_name, tool_name, arguments, verdict.reason)
                    )
                    text = f"This call was blocked and not run: {verdict.reason}"
                    return ToolMessage(text, tool_call_id=call_id, status="error")
            try:
                with self.tool_span(server_name, tool_name, arguments) as record_output:
                    try:
                        reply = await handler(request)
                    except Exception as error:
                        failure = _server_failure(server_name, error)
                        if failure is None or not _is_upstream_error(error):
                            raise
                        raise failure from None
                    if not isinstance(reply, ToolMessage):
                        raise TypeError(f"the tool returned a {type(reply).__name__}")
                    result_text = await self.screen(self.request, str(reply.content))
                    record_output(result_text)
                is_error = bool(reply.artifact)
                if result_text in (WITHHELD, UNCHECKED):
                    self.withheld += 1
            except NotConnected as error:
                result_text, is_error = str(error), True
            self.executed.append(
                ExecutedCall(server_name, tool_name, arguments, result_text, is_error)
            )
            return ToolMessage(
                result_text, tool_call_id=call_id, status="error" if is_error else "success"
            )


async def run_agent(
    user_id: str,
    request: str,
    system: str,
    servers: list[McpServer],
    connections: OAuthConnections,
    tokens: TokenStore,
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
    changed_tools: list[str] = []
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
        if server.tools is not None:
            listed = [info for info in listed if info.name in server.tools]
        pinned = pin_tools(tokens, user_id, server.name, listed)
        changed_tools += [f"{server.name}__{name}" for name in pinned.changed]
        for info in pinned.kept:
            tools[f"{server.name}__{info.name}"] = _Tool(server, info)

    guard = _Guard(request, tools, gate, screen, tool_span, record_usage)
    read_only_names: list[BaseTool | str] = [
        name for name, tool in tools.items() if tool.info.read_only
    ]
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=step_limit),
        guard,
        ToolRetryMiddleware(tools=read_only_names, retry_on=_is_upstream_error, on_failure="error"),
    ]
    agent = create_agent(
        model,
        [_langchain_tool(name, tool, user_id, connections) for name, tool in tools.items()],
        system_prompt=system,
        middleware=middleware,
    )
    await agent.ainvoke({"messages": [HumanMessage(request)]}, config=trace_config)

    last = guard.last_response
    return AgentResult(
        "" if last is None else last.text,
        tuple(guard.executed),
        tuple(guard.blocked),
        tuple(not_connected),
        tuple(guard.usage),
        last is None or bool(last.tool_calls),
        last is not None
        and not last.tool_calls
        and last.response_metadata.get("stop_reason") == "max_tokens",
        guard.withheld,
        tuple(changed_tools),
    )


def _langchain_tool(
    name: str, tool: _Tool, user_id: str, connections: OAuthConnections
) -> BaseTool:
    async def call_server(**arguments: Any) -> tuple[str, bool]:
        result = await connections.call(user_id, tool.server, tool.info.name, arguments)
        return result.text, result.is_error

    return StructuredTool(
        name=name,
        description=tool.info.description,
        args_schema=tool.info.input_schema,
        coroutine=call_server,
        response_format="content_and_artifact",
    )
