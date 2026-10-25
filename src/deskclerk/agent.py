# One employee's request, answered by a LangChain agent whose model calls the tools of the MCP
# servers the employee has connected. Writes pass the write check and then wait for the employee to
# approve or reject them. Every tool result is screened before the model reads it.

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import InputAgentState
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
from langgraph.types import Command, Interrupt

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
from deskclerk.screen import Screen, is_flagged
from deskclerk.servers import McpServer
from deskclerk.tokens import TokenStore

TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")

_log = logging.getLogger(__name__)


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


Decision = Literal["approve", "reject"]


@dataclass(frozen=True)
class PendingCall:
    id: str
    server: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DecidedCall:
    server: str
    tool: str
    decision: Decision


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
    flagged: int
    changed_tools: tuple[str, ...]
    pending: tuple[PendingCall, ...]
    decided: tuple[DecidedCall, ...]

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


class ConversationWaiting(Exception):
    pass


class NothingToConfirm(Exception):
    pass


class WrongDecisions(Exception):
    pass


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


def _names_the_model_accepts(server: str, listed: list[ToolInfo]) -> list[ToolInfo]:
    accepted = []
    for info in listed:
        full_name = f"{server}__{info.name}"
        if TOOL_NAME.fullmatch(full_name) is None:
            _log.warning(
                "tool %s is not offered: its name must be 1 to 64 letters, digits, "
                "underscores or hyphens",
                full_name,
            )
            continue
        accepted.append(info)
    return accepted


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


@dataclass
class _Shared:
    request: str
    earlier: tuple[str, ...]
    offered: dict[str, _Tool]
    gate: Gate
    screen: Screen
    tool_span: ToolSpan
    record_usage: Callable[[Usage], None]
    tool_output_chars: int
    history_tokens: int
    one_call_at_a_time: asyncio.Lock = field(default_factory=asyncio.Lock)
    executed: list[ExecutedCall] = field(default_factory=list)
    blocked: list[BlockedCall] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    last_response: AIMessage | None = None
    flagged: int = 0
    flag_raised: bool = False
    shown: set[str] = field(default_factory=set)


class _Guard(AgentMiddleware[AgentState[Any], None, Any]):
    def __init__(self, shared: _Shared) -> None:
        super().__init__()
        self.shared = shared

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        shared = self.shared
        shared.shown = {tool.name for tool in request.tools if isinstance(tool, BaseTool)}
        response = await handler(request.override(messages=self._within_budget(request.messages)))
        message = response.result[-1]
        if not isinstance(message, AIMessage):
            raise TypeError(f"the model returned a {type(message).__name__}")
        step_usage = usage_from(message)
        shared.usage.append(step_usage)
        shared.record_usage(step_usage)
        shared.last_response = message
        return response

    def _within_budget(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        current = max(
            index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
        )
        earlier, this_ask = messages[:current], messages[current:]
        room = self.shared.history_tokens - count_tokens_approximately(this_ask)
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
        spoken = self.shared.last_response
        if isinstance(last, AIMessage) and (spoken is None or last.id != spoken.id):
            return {"messages": [RemoveMessage(id=last.id or "")]}
        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        shared = self.shared
        call = request.tool_call
        call_id = call["id"] or ""
        tool = shared.offered.get(call["name"])
        if tool is None or call["name"] not in shared.shown:
            text = f"There is no tool named {call['name']}."
            return ToolMessage(text, tool_call_id=call_id, status="error")
        async with shared.one_call_at_a_time:
            if not tool.info.read_only:
                proposed = ProposedCall(
                    shared.request,
                    tool.server.name,
                    tool.info,
                    call["args"],
                    shared.earlier + tuple(done.result for done in shared.executed),
                )
                verdict = await shared.gate(proposed)
                if not verdict.allowed:
                    shared.blocked.append(
                        BlockedCall(tool.server.name, tool.info.name, call["args"], verdict.reason)
                    )
                    text = f"This call was blocked and not run: {verdict.reason}"
                    return ToolMessage(text, tool_call_id=call_id, status="error")
            return await handler(request)


class _Run(AgentMiddleware[AgentState[Any], None, Any]):
    def __init__(self, shared: _Shared) -> None:
        super().__init__()
        self.shared = shared

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        shared = self.shared
        call = request.tool_call
        tool = shared.offered[call["name"]]
        arguments: dict[str, Any] = call["args"]
        server_name, tool_name = tool.server.name, tool.info.name
        try:
            with shared.tool_span(server_name, tool_name, arguments) as record_output:
                try:
                    reply = await handler(request)
                except Exception as error:
                    failure = _server_failure(server_name, error)
                    if failure is None or not _is_upstream_error(error):
                        raise
                    raise failure from None
                if not isinstance(reply, ToolMessage):
                    raise TypeError(f"the tool returned a {type(reply).__name__}")
                result_text = await shared.screen(
                    shared.request, _cut(str(reply.content), shared.tool_output_chars)
                )
                record_output(result_text)
            is_error = bool(reply.artifact)
            if is_flagged(result_text):
                shared.flagged += 1
                shared.flag_raised = True
        except NotConnected as error:
            result_text, is_error = str(error), True
        shared.executed.append(
            ExecutedCall(server_name, tool_name, arguments, result_text, is_error)
        )
        return ToolMessage(
            result_text, tool_call_id=call["id"] or "", status="error" if is_error else "success"
        )


def _hit_output_limit(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("stop_reason") == "max_tokens"
        or metadata.get("finish_reason") == "length"
        or metadata.get("stopReason") == "max_tokens"
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
    *,
    confirm_writes: bool,
    decisions: dict[str, Decision] | None = None,
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
        listed = _names_the_model_accepts(server.name, listed)
        pinned = pin_tools(tokens, user_id, server.name, listed)
        changed_tools += [f"{server.name}__{name}" for name in pinned.changed]
        for info in pinned.kept:
            tools[f"{server.name}__{info.name}"] = _Tool(server, info)
    listed_any = len(listings) > len(not_connected) + len(unavailable)
    if unavailable and not listed_any:
        names = ", ".join(failure.server_name for failure in unavailable)
        raise McpServerFailed(names, unavailable[0].cause)
    unavailable_names = tuple(failure.server_name for failure in unavailable)

    shared = _Shared(
        request,
        earlier_texts(history),
        tools,
        gate,
        screen,
        tool_span,
        record_usage,
        tool_output_chars,
        history_tokens,
    )
    confirmation = _confirmation(tools, shared)
    read_only_names: list[BaseTool | str] = [
        name for name, tool in tools.items() if tool.info.read_only
    ]
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=step_limit),
        *([] if selection is None else [selection]),
        _Guard(shared),
        *([confirmation] if confirm_writes else []),
        _Run(shared),
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
    saved = await agent.aget_state(config)
    waiting = {interrupt.id: interrupt for interrupt in saved.interrupts}
    if decisions is None:
        if waiting:
            raise ConversationWaiting
        start: InputAgentState | Command[Any] = {
            "messages": [*close_unanswered_calls(history), HumanMessage(request)]
        }
    else:
        if not waiting:
            raise NothingToConfirm
        if set(decisions) != set(waiting):
            raise WrongDecisions
        shared.shown = set(tools)
        shared.flag_raised = _flag_before_pause(saved.values["messages"], waiting, tools)
        for interrupt in waiting.values():
            confirmation.interrupt_on.setdefault(interrupt.value["name"], _REVIEW)
        start = Command(
            resume={interrupt_id: {"type": choice} for interrupt_id, choice in decisions.items()}
        )
    try:
        output = await agent.ainvoke(start, config=config, version="v2")
    except BaseException:
        await _end_pause(agent, config)
        raise
    pending = tuple(_pending_call(interrupt, tools) for interrupt in output.interrupts)
    before = len(saved.values.get("messages", []))
    decided = _decided_calls(output.value["messages"][before:], waiting, tools)
    for call, choice in decided:
        if choice == "reject":
            shared.blocked.append(
                BlockedCall(call.server, call.tool, call.arguments, "the employee declined")
            )

    last = shared.last_response
    return AgentResult(
        "" if last is None or pending else last.text,
        tuple(shared.executed),
        tuple(shared.blocked),
        tuple(not_connected),
        unavailable_names,
        tuple(shared.usage),
        not pending and (last is None or bool(last.tool_calls)),
        last is not None and not last.tool_calls and _hit_output_limit(last.response_metadata),
        shared.flagged,
        tuple(changed_tools),
        pending,
        tuple(DecidedCall(call.server, call.tool, choice) for call, choice in decided),
    )


_REVIEW: InterruptOnConfig = {"allowed_decisions": ["approve", "reject"]}
_REJECTION = "User rejected the tool call"


def _confirmation(tools: dict[str, _Tool], shared: _Shared) -> HumanInTheLoopMiddleware:
    after_a_flag: InterruptOnConfig = {**_REVIEW, "when": lambda request: shared.flag_raised}
    interrupt_on: dict[str, bool | InterruptOnConfig] = {
        name: after_a_flag if tool.info.read_only else _REVIEW for name, tool in tools.items()
    }
    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on, interrupt_mode="per_call")


def _flag_before_pause(
    messages: list[AnyMessage], waiting: dict[str, Interrupt], tools: dict[str, _Tool]
) -> bool:
    asked = max(
        index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
    )
    if any(
        isinstance(message, ToolMessage) and is_flagged(str(message.content))
        for message in messages[asked:]
    ):
        return True
    return any(
        tools[interrupt.value["name"]].info.read_only
        for interrupt in waiting.values()
        if interrupt.value["name"] in tools
    )


def _pending_call(interrupt: Interrupt, tools: dict[str, _Tool]) -> PendingCall:
    name: str = interrupt.value["name"]
    waiting_tool = tools.get(name)
    server, tool = (
        ("", name) if waiting_tool is None else (waiting_tool.server.name, waiting_tool.info.name)
    )
    return PendingCall(interrupt.id, server, tool, interrupt.value["args"])


def _decided_calls(
    new_messages: list[AnyMessage], waiting: dict[str, Interrupt], tools: dict[str, _Tool]
) -> list[tuple[PendingCall, Decision]]:
    waiting_by_call = {interrupt.value["tool_call_id"]: interrupt for interrupt in waiting.values()}
    decided: list[tuple[PendingCall, Decision]] = []
    for message in new_messages:
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage) and message.tool_call_id in waiting_by_call:
            call = _pending_call(waiting_by_call[message.tool_call_id], tools)
            declined = message.status == "error" and str(message.content).startswith(_REJECTION)
            decided.append((call, "reject" if declined else "approve"))
    return decided


async def _end_pause(agent: Any, config: RunnableConfig) -> None:
    saved = await agent.aget_state(config)
    if saved.interrupts:
        closing = close_unanswered_calls(saved.values["messages"])
        await agent.aupdate_state(config, {"messages": closing}, as_node="tools")


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
