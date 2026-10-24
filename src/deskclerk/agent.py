# One employee's request, in a conversation that remembers the earlier ones, answered by a
# LangChain agent whose model calls the tools of the MCP servers the employee has connected. Tools
# that are not read-only pass a gate first; a refused call never runs. Every result a tool returns
# passes the screen before the model reads it, a tool whose definition changed since the employee
# first used its server is not offered at all, and only read-only tools are retried after a
# transport error.

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, cast

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
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime
from langgraph.types import Command

from deskclerk.config import DEFAULT_HISTORY_TOKENS, DEFAULT_TOOL_OUTPUT_CHARS
from deskclerk.connections import (
    UPSTREAM_ERRORS,
    NotConnected,
    OAuthConnections,
    ToolInfo,
    first_leaf,
)
from deskclerk.conversations import close_unanswered_calls, earlier_texts
from deskclerk.instructions import unavailable_note
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
    unavailable: tuple[str, ...]
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


@dataclass(frozen=True)
class _Listing:
    server: McpServer
    tools: list[ToolInfo]
    connected: bool
    failure: McpServerFailed | None


async def _list_tools(user_id: str, server: McpServer, connections: OAuthConnections) -> _Listing:
    try:
        return _Listing(server, await connections.tools(user_id, server), True, None)
    except NotConnected:
        return _Listing(server, [], False, None)
    except (*UPSTREAM_ERRORS, BaseExceptionGroup) as error:
        failure = _server_failure(server.name, error)
        if failure is None:
            raise
        return _Listing(server, [], True, failure)


class _Guard(AgentMiddleware[AgentState[Any], None, Any]):
    def __init__(
        self,
        request: str,
        earlier: tuple[str, ...],
        tools: dict[str, _Tool],
        gate: Gate,
        screen: Screen,
        tool_span: ToolSpan,
        record_usage: Callable[[Usage], None],
        tool_output_chars: int,
        history_tokens: int,
    ) -> None:
        super().__init__()
        self.request = request
        self.earlier = earlier
        self.offered = tools
        self.gate = gate
        self.screen = screen
        self.tool_span = tool_span
        self.record_usage = record_usage
        self.tool_output_chars = tool_output_chars
        self.history_tokens = history_tokens
        self.one_call_at_a_time = asyncio.Lock()
        self.executed: list[ExecutedCall] = []
        self.blocked: list[BlockedCall] = []
        self.usage: list[Usage] = []
        self.last_response: AIMessage | None = None
        self.withheld = 0
        self.shown: set[str] = set()

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        self.shown = {tool.name for tool in request.tools if isinstance(tool, BaseTool)}
        response = await handler(request.override(messages=self._within_budget(request.messages)))
        message = response.result[-1]
        if not isinstance(message, AIMessage):
            raise TypeError(f"the model returned a {type(message).__name__}")
        step_usage = usage_from(message)
        self.usage.append(step_usage)
        self.record_usage(step_usage)
        self.last_response = message
        return response

    def _within_budget(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        current = max(
            index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
        )
        earlier, this_ask = messages[:current], messages[current:]
        room = self.history_tokens - count_tokens_approximately(this_ask)
        if room <= 0:
            return this_ask
        kept = trim_messages(
            earlier,
            max_tokens=room,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
        )
        return [*cast(list[AnyMessage], kept), *this_ask]

    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        last = state["messages"][-1]
        spoken = self.last_response
        if isinstance(last, AIMessage) and (spoken is None or last.id != spoken.id):
            return {"messages": [RemoveMessage(id=last.id or "")]}
        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        call_id = call["id"] or ""
        tool = self.offered.get(call["name"])
        if tool is None or call["name"] not in self.shown:
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
                    self.earlier + tuple(done.result for done in self.executed),
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
                    result_text = await self.screen(
                        self.request, _cut(str(reply.content), self.tool_output_chars)
                    )
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


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[deskclerk: {limit:,} of {len(text):,} characters shown]"


async def run_agent(
    user_id: str,
    request: str,
    system: str,
    servers: list[McpServer],
    connections: OAuthConnections,
    tokens: TokenStore,
    checkpointer: BaseCheckpointSaver[Any],
    thread: str,
    history: list[BaseMessage],
    model: BaseChatModel,
    gate: Gate,
    screen: Screen,
    step_limit: int,
    record_usage: Callable[[Usage], None],
    trace_config: RunnableConfig | None = None,
    tool_span: ToolSpan = no_tool_span,
    selection: AgentMiddleware[Any, Any, Any] | None = None,
    tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
    history_tokens: int = DEFAULT_HISTORY_TOKENS,
) -> AgentResult:
    tools: dict[str, _Tool] = {}
    not_connected: list[str] = []
    unavailable: list[McpServerFailed] = []
    changed_tools: list[str] = []
    listings = await asyncio.gather(
        *(_list_tools(user_id, server, connections) for server in servers)
    )
    for listing in listings:
        server = listing.server
        if not listing.connected:
            not_connected.append(server.name)
            continue
        if listing.failure is not None:
            unavailable.append(listing.failure)
            continue
        listed = listing.tools
        if server.tools is not None:
            listed = [info for info in listed if info.name in server.tools]
        pinned = pin_tools(tokens, user_id, server.name, listed)
        changed_tools += [f"{server.name}__{name}" for name in pinned.changed]
        for info in pinned.kept:
            tools[f"{server.name}__{info.name}"] = _Tool(server, info)
    listed_any = len(listings) > len(not_connected) + len(unavailable)
    if unavailable and not listed_any:
        names = ", ".join(failure.server_name for failure in unavailable)
        raise McpServerFailed(names, unavailable[0].cause)
    unavailable_names = tuple(failure.server_name for failure in unavailable)

    earlier = earlier_texts(history)
    guard = _Guard(
        request,
        earlier,
        tools,
        gate,
        screen,
        tool_span,
        record_usage,
        tool_output_chars,
        history_tokens,
    )
    read_only_names: list[BaseTool | str] = [
        name for name, tool in tools.items() if tool.info.read_only
    ]
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=step_limit),
        *([] if selection is None else [selection]),
        guard,
        ToolRetryMiddleware(tools=read_only_names, retry_on=_is_upstream_error, on_failure="error"),
    ]
    agent = create_agent(
        model,
        [_langchain_tool(name, tool, user_id, connections) for name, tool in tools.items()],
        system_prompt=system + unavailable_note(unavailable_names),
        middleware=middleware,
        checkpointer=checkpointer,
    )
    config: RunnableConfig = {
        **(trace_config or {}),
        "configurable": {"thread_id": thread},
    }
    await agent.ainvoke(
        {"messages": [*close_unanswered_calls(history), HumanMessage(request)]}, config=config
    )

    last = guard.last_response
    return AgentResult(
        "" if last is None else last.text,
        tuple(guard.executed),
        tuple(guard.blocked),
        tuple(not_connected),
        unavailable_names,
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
