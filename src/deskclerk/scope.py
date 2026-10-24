# Picks the tools the employee's request needs before the agent's model sees any: one call to Jev
# asks a yes/no question per server, and the model is offered only the tools of the servers that
# pass. When Jev fails, the request has no user message, or no server passes, the model is offered
# every tool. With one server there is nothing to choose and nothing is asked.

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from deskclerk.jev import Jev, JevError

SERVER_QUESTION = "Does this request need the tools of the {server} server?"
SERVER_THRESHOLD = 0.10

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolEntry:
    name: str
    description: str
    server: str


@dataclass(frozen=True)
class Batch:
    state: dict[str, Any]
    questions: dict[str, str]
    tools_of: dict[str, tuple[str, ...]]


def server_of(tool_name: str) -> str:
    return tool_name.partition("__")[0]


def server_batch(request: str, entries: list[ToolEntry]) -> Batch:
    by_server: dict[str, list[ToolEntry]] = {}
    for entry in entries:
        by_server.setdefault(entry.server, []).append(entry)
    servers = [
        {
            "server": name,
            "tools": [{"name": e.name, "description": e.description} for e in members],
        }
        for name, members in by_server.items()
    ]
    questions = {name: SERVER_QUESTION.format(server=name) for name in by_server}
    tools_of = {name: tuple(entry.name for entry in members) for name, members in by_server.items()}
    return Batch({"request": request, "servers": servers}, questions, tools_of)


def kept_tools(batch: Batch, probabilities: dict[str, float]) -> set[str]:
    return {
        name
        for server, probability in probabilities.items()
        if probability >= SERVER_THRESHOLD
        for name in batch.tools_of[server]
    }


class ToolSelection(AgentMiddleware[AgentState[Any], None, Any]):
    def __init__(self, jev: Jev, on_tokens: Callable[[int], None] | None = None) -> None:
        super().__init__()
        self.jev = jev
        self.on_tokens = on_tokens
        self.asked = False
        self.kept: set[str] | None = None

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if not self.asked:
            self.asked = True
            self.kept = await self._select(request)
        if self.kept is None:
            return await handler(request)
        kept = self.kept
        offered = [tool for tool in request.tools if isinstance(tool, dict) or tool.name in kept]
        return await handler(request.override(tools=offered))

    async def _select(self, request: ModelRequest[Any]) -> set[str] | None:
        texts = [str(m.content) for m in request.messages if isinstance(m, HumanMessage)]
        if not texts:
            _log.warning("tool selection skipped, every tool offered: no user message")
            return None
        entries = [
            ToolEntry(tool.name, tool.description, server_of(tool.name))
            for tool in request.tools
            if isinstance(tool, BaseTool)
        ]
        batch = server_batch(texts[-1], entries)
        if len(batch.questions) < 2:
            _log.info("tool selection skipped, every tool offered: one server")
            return None
        try:
            probabilities = await asyncio.to_thread(
                self.jev.noul_many, batch.state, batch.questions, self.on_tokens
            )
        except JevError as error:
            _log.warning("tool selection failed, every tool offered: %s", error)
            return None
        kept = kept_tools(batch, probabilities)
        if not kept:
            _log.warning("tool selection kept no server, every tool offered")
        return kept or None


def jev_tool_selection(
    jev: Jev | None, enabled: bool, on_tokens: Callable[[int], None] | None = None
) -> ToolSelection | None:
    if jev is None or not enabled:
        return None
    return ToolSelection(jev, on_tokens)
