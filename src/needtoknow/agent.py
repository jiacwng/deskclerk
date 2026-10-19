# One employee's request answered by a chat model that calls the tools of the MCP servers the
# employee has connected. Tools that are not read-only pass a gate first; a refused call never runs.

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from needtoknow.connections import NotConnected, OAuthConnections, ToolInfo
from needtoknow.providers import Usage, usage_from
from needtoknow.servers import McpServer


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str


Gate = Callable[[str, str, str, dict[str, Any]], Awaitable[Verdict]]


async def allow_all(request: str, server: str, tool: str, arguments: dict[str, Any]) -> Verdict:
    return Verdict(True, "no write check is configured")


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

    @property
    def total_usage(self) -> Usage:
        return Usage(
            sum(call.input_tokens for call in self.usage),
            sum(call.output_tokens for call in self.usage),
            sum(call.cache_read_tokens for call in self.usage),
            sum(call.cache_write_tokens for call in self.usage),
        )


@dataclass(frozen=True)
class _Tool:
    server: McpServer
    info: ToolInfo


async def run_agent(
    user_id: str,
    request: str,
    servers: list[McpServer],
    connections: OAuthConnections,
    model: BaseChatModel,
    gate: Gate,
    step_limit: int,
) -> AgentResult:
    tools: dict[str, _Tool] = {}
    not_connected: list[str] = []
    for server in servers:
        try:
            listed = await connections.tools(user_id, server)
        except NotConnected:
            not_connected.append(server.name)
            continue
        for info in listed:
            tools[f"{server.name}__{info.name}"] = _Tool(server, info)

    chat_model: Any = model
    if tools:
        chat_model = model.bind_tools([_tool_schema(name, tool) for name, tool in tools.items()])

    messages: list[BaseMessage] = [HumanMessage(request)]
    executed: list[ExecutedCall] = []
    blocked: list[BlockedCall] = []
    usage: list[Usage] = []
    answer = ""
    step_limit_hit = True
    for _ in range(step_limit):
        response: AIMessage = await chat_model.ainvoke(messages)
        messages.append(response)
        usage.append(usage_from(response))
        answer = response.text
        if not response.tool_calls:
            step_limit_hit = False
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
                verdict = await gate(request, server_name, tool_name, arguments)
                if not verdict.allowed:
                    blocked.append(BlockedCall(server_name, tool_name, arguments, verdict.reason))
                    text = f"This call was blocked and not run: {verdict.reason}"
                    messages.append(
                        ToolMessage(text, tool_call_id=call["id"] or "", status="error")
                    )
                    continue
            try:
                result = await connections.call(user_id, tool.server, tool_name, arguments)
                result_text, is_error = result.text, result.is_error
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
