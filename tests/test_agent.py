# The agent loop against a scripted chat model and the real test MCP server: read tools run freely,
# write tools wait for the gate, a refused call never reaches the server, and the loop stops.

import itertools
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from cryptography.fernet import Fernet
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from deskclerk.agent import AgentResult, ProposedCall, Verdict, run_agent
from deskclerk.connections import OAuthConnections
from deskclerk.providers import Usage
from deskclerk.servers import McpServer
from mcp_server import RunningServer, running_mcp_server
from test_connections import REDIRECT, connect, make_connections, server_record


class ScriptedModel(GenericFakeChatModel):
    seen: list[list[BaseMessage]] = []
    bound: list[Any] = []

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        self.bound = list(tools)
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class ProviderGoneAfterScript(ScriptedModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super()._generate(*args, **kwargs)
        except StopIteration:
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=request) from None


def script(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies))


def calls(*tool_calls: tuple[str, dict[str, Any]], usage: UsageMetadata | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call-{index}", "type": "tool_call"}
            for index, (name, args) in enumerate(tool_calls)
        ],
        usage_metadata=usage,
    )


def answer(text: str, usage: UsageMetadata | None = None) -> AIMessage:
    return AIMessage(content=text, usage_metadata=usage)


class RecordingGate:
    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason
        self.asked: list[ProposedCall] = []

    async def __call__(self, call: ProposedCall) -> Verdict:
        self.asked.append(call)
        return Verdict(self.allowed, self.reason)


@pytest.fixture
def mcp_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


@pytest.fixture
async def connections(tmp_path: Path, mcp_server: RunningServer) -> OAuthConnections:
    connections = make_connections(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())
    await connect(connections, "ada", server_record(mcp_server))
    return connections


async def run(
    connections: OAuthConnections,
    servers: list[McpServer],
    model: BaseChatModel,
    gate: RecordingGate | None = None,
    step_limit: int = 5,
) -> AgentResult:
    return await run_agent(
        "ada",
        "save a note",
        servers,
        connections,
        model,
        gate or RecordingGate(True),
        step_limit,
        lambda usage: None,
    )


@pytest.mark.anyio
async def test_a_read_tool_runs_without_asking_the_gate(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    gate = RecordingGate(False, "no")
    model = script(calls(("notes__read_note", {})), answer("It says: the note"))
    result = await run(connections, [server_record(mcp_server)], model, gate)
    assert gate.asked == []
    assert [(call.tool, call.result, call.is_error) for call in result.executed] == [
        ("read_note", "the note", False)
    ]
    assert result.answer == "It says: the note"
    assert result.step_limit_hit is False
    bound = {tool["function"]["name"]: tool["function"] for tool in model.bound}
    assert set(bound) == {"notes__read_note", "notes__write_note", "notes__break_note"}
    assert bound["notes__write_note"]["parameters"]["required"] == ["text"]


@pytest.mark.anyio
async def test_an_allowed_write_asks_the_gate_and_reaches_the_server(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    gate = RecordingGate(True)
    model = script(calls(("notes__write_note", {"text": "hi"})), answer("done"))
    result = await run(connections, [server_record(mcp_server)], model, gate)
    [asked] = gate.asked
    assert (asked.request, asked.server, asked.tool.name, asked.arguments) == (
        "save a note",
        "notes",
        "write_note",
        {"text": "hi"},
    )
    assert asked.observed == ()
    assert [note[1] for note in mcp_server.notes] == ["hi"]
    assert result.executed[0].result == "wrote hi"
    assert result.blocked == ()


@pytest.mark.anyio
async def test_a_refused_write_never_reaches_the_server_and_the_model_is_told_why(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    gate = RecordingGate(False, "the request did not ask for a note")
    model = script(calls(("notes__write_note", {"text": "hi"})), answer("I could not save it"))
    result = await run(connections, [server_record(mcp_server)], model, gate)
    assert mcp_server.notes == []
    assert result.executed == ()
    assert [(call.tool, call.reason) for call in result.blocked] == [
        ("write_note", "the request did not ask for a note")
    ]
    reply = model.seen[1][-1]
    assert isinstance(reply, ToolMessage)
    assert "the request did not ask for a note" in str(reply.content)
    assert reply.status == "error"


@pytest.mark.anyio
async def test_a_server_the_employee_has_not_connected_is_skipped_and_reported(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    other = McpServer("other", mcp_server.url, None, None)
    model = script(answer("nothing to call"))
    result = await run(connections, [server_record(mcp_server), other], model)
    assert result.not_connected == ("other",)
    names = [tool["function"]["name"] for tool in model.bound]
    assert names and all(name.startswith("notes__") for name in names)


@pytest.mark.anyio
async def test_the_step_limit_stops_a_model_that_keeps_calling_tools(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    forever = itertools.repeat(calls(("notes__read_note", {})))
    model = ScriptedModel(messages=forever)
    result = await run(connections, [server_record(mcp_server)], model, step_limit=3)
    assert result.step_limit_hit is True
    assert len(result.executed) == 3
    assert len(result.usage) == 3


@pytest.mark.anyio
async def test_a_reply_cut_off_at_the_output_limit_ends_the_loop_and_is_reported(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    cut = AIMessage(content="The meeting is at", response_metadata={"stop_reason": "max_tokens"})
    model = script(cut, answer("never asked for"))
    result = await run(connections, [server_record(mcp_server)], model)
    assert result.cut_off is True
    assert result.step_limit_hit is False
    assert result.answer == "The meeting is at"
    assert len(result.usage) == 1


@pytest.mark.anyio
async def test_a_reply_that_ends_normally_is_not_cut_off(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    done = AIMessage(content="done", response_metadata={"stop_reason": "end_turn"})
    result = await run(connections, [server_record(mcp_server)], script(done))
    assert result.cut_off is False


@pytest.mark.anyio
async def test_usage_is_summed_with_cache_tokens(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    first = UsageMetadata(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        input_token_details={"cache_creation": 60},
    )
    second = UsageMetadata(
        input_tokens=150,
        output_tokens=20,
        total_tokens=170,
        input_token_details={"cache_read": 60},
    )
    model = script(calls(("notes__read_note", {}), usage=first), answer("ok", usage=second))
    result = await run(connections, [server_record(mcp_server)], model)
    total = result.total_usage
    assert (total.input_tokens, total.output_tokens) == (250, 30)
    assert (total.cache_read_tokens, total.cache_write_tokens) == (60, 60)
    assert len(result.usage) == 2


@pytest.mark.anyio
async def test_usage_billed_before_a_provider_failure_is_recorded(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    billed: list[Usage] = []
    first = UsageMetadata(input_tokens=7, output_tokens=3, total_tokens=10)
    model = ProviderGoneAfterScript(messages=iter([calls(("notes__read_note", {}), usage=first)]))
    with pytest.raises(anthropic.APIConnectionError):
        await run_agent(
            "ada",
            "read my note",
            [server_record(mcp_server)],
            connections,
            model,
            RecordingGate(True),
            5,
            billed.append,
        )
    assert [(usage.input_tokens, usage.output_tokens) for usage in billed] == [(7, 3)]


@pytest.mark.anyio
async def test_a_tool_error_goes_back_to_the_model_and_the_loop_continues(
    mcp_server: RunningServer, connections: OAuthConnections
) -> None:
    gate = RecordingGate(True)
    model = script(calls(("notes__break_note", {})), answer("the note is locked"))
    result = await run(connections, [server_record(mcp_server)], model, gate)
    assert result.executed[0].is_error is True
    assert "break_note" in result.executed[0].result
    reply = model.seen[1][-1]
    assert isinstance(reply, ToolMessage)
    assert reply.status == "error"
    assert result.answer == "the note is locked"


@pytest.mark.anyio
async def test_a_server_without_sign_in_works_without_connecting(tmp_path: Path) -> None:
    connections = make_connections(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())
    with running_mcp_server(REDIRECT, oauth=False) as open_server:
        server = McpServer("open", open_server.url, None, None, "none")
        gate = RecordingGate(True)
        model = script(
            calls(("open__read_note", {}), ("open__write_note", {"text": "x"})), answer("done")
        )
        result = await run(connections, [server], model, gate)
    assert result.not_connected == ()
    assert [call.result for call in result.executed] == ["the note", "wrote x"]
    assert [asked.tool.name for asked in gate.asked] == ["write_note"]
    assert gate.asked[0].observed == ("the note",)
