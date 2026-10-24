# The tool selection middleware with a fake Jev: which tools the model is offered, how Jev is
# asked, and that it is asked once per run. Two tests run it inside the agent against the test
# MCP server.

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from cryptography.fernet import Fernet
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from deskclerk.agent import AgentResult, run_agent
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.jev import Jev, JevError
from deskclerk.scope import (
    SERVER_QUESTION,
    SERVER_THRESHOLD,
    ToolSelection,
    jev_tool_selection,
)
from deskclerk.screen import pass_through
from deskclerk.servers import McpServer
from mcp_server import RunningServer, running_mcp_server
from test_agent import RecordingGate, ScriptedModel, answer, calls
from test_connections import REDIRECT, connect, make_connections, server_record

REQUEST = "move my 3pm meeting to Friday"


class FakeJev:
    def __init__(self, probability: Callable[[str], float] | Exception, tokens: int = 7) -> None:
        self.probability = probability
        self.tokens = tokens
        self.asked: list[tuple[Any, dict[str, str]]] = []

    def noul_many(
        self, state: Any, instructions: dict[str, str], on_tokens: Any = None
    ) -> dict[str, float]:
        self.asked.append((state, instructions))
        if isinstance(self.probability, Exception):
            raise self.probability
        if on_tokens is not None:
            on_tokens(self.tokens)
        return {name: self.probability(name) for name in instructions}


def as_jev(fake: FakeJev) -> Jev:
    return cast(Jev, fake)


def make_tool(server: str, name: str, description: str) -> BaseTool:
    async def call(**arguments: Any) -> str:
        return ""

    return StructuredTool(
        name=f"{server}__{name}",
        description=description,
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
    )


def make_tool_named(name: str) -> BaseTool:
    return make_tool("", name, "").model_copy(update={"name": name})


TOOLS = [
    make_tool("calendar", "create_event", "Creates an event"),
    make_tool("calendar", "delete_event", "Deletes an event"),
    make_tool("email", "send_email", "Sends an email"),
]
ALL_NAMES = [tool.name for tool in TOOLS]


async def offered(
    selection: ToolSelection,
    model_calls: int = 1,
    tools: list[BaseTool] | None = None,
    messages: list[Any] | None = None,
) -> list[list[str]]:
    seen: list[list[str]] = []

    async def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append([tool.name for tool in modified.tools if isinstance(tool, BaseTool)])
        return ModelResponse(result=[AIMessage("ok")])

    for _ in range(model_calls):
        request: ModelRequest[Any] = ModelRequest(
            model=ScriptedModel(messages=iter([])),
            messages=[HumanMessage(REQUEST)] if messages is None else messages,
            tools=list(TOOLS if tools is None else tools),
        )
        await selection.awrap_model_call(request, handler)
    return seen


def answers(by_name: dict[str, float]) -> Callable[[str], float]:
    def probability(name: str) -> float:
        return by_name.get(name, 0.0)

    return probability


@pytest.mark.anyio
async def test_per_server_keeps_every_tool_of_a_server_jev_says_yes_to() -> None:
    jev = FakeJev(answers({"calendar": SERVER_THRESHOLD}))
    assert await offered(ToolSelection(as_jev(jev))) == [
        ["calendar__create_event", "calendar__delete_event"]
    ]


@pytest.mark.anyio
async def test_per_server_asks_one_call_with_one_question_per_server() -> None:
    jev = FakeJev(answers({}))
    await offered(ToolSelection(as_jev(jev)))
    [(state, questions)] = jev.asked
    assert state["request"] == REQUEST
    assert state["servers"][0] == {
        "server": "calendar",
        "tools": [
            {"name": "calendar__create_event", "description": "Creates an event"},
            {"name": "calendar__delete_event", "description": "Deletes an event"},
        ],
    }
    assert questions == {
        "calendar": "Does this request need the tools of the calendar server?",
        "email": "Does this request need the tools of the email server?",
    }
    assert SERVER_QUESTION == "Does this request need the tools of the {server} server?"


@pytest.mark.anyio
async def test_per_server_with_one_server_asks_nothing_and_offers_every_tool() -> None:
    jev = FakeJev(answers({}))
    only_calendar = TOOLS[:2]
    seen = await offered(ToolSelection(as_jev(jev)), tools=only_calendar)
    assert seen == [[tool.name for tool in only_calendar]]
    assert jev.asked == []


@pytest.mark.anyio
async def test_jev_reads_every_employee_message_oldest_first() -> None:
    jev = FakeJev(answers({}))
    messages = [HumanMessage(REQUEST), AIMessage("Done."), HumanMessage("And for Bob?")]
    await offered(ToolSelection(as_jev(jev)), messages=messages)
    [(state, _)] = jev.asked
    assert state["request"] == f"{REQUEST}\n\nAnd for Bob?"


@pytest.mark.anyio
async def test_a_request_without_a_user_message_offers_every_tool() -> None:
    jev = FakeJev(answers({"email__send_email": 1.0}))
    seen = await offered(ToolSelection(as_jev(jev)), messages=[])
    assert seen == [ALL_NAMES]
    assert jev.asked == []


@pytest.mark.anyio
async def test_a_tool_whose_name_has_no_server_prefix_is_its_own_server() -> None:
    jev = FakeJev(answers({"lonely": 1.0}))
    tools = [*TOOLS, make_tool_named("lonely")]
    seen = await offered(ToolSelection(as_jev(jev)), tools=tools)
    assert seen == [["lonely"]]


@pytest.mark.anyio
async def test_every_tool_is_offered_when_jev_fails() -> None:
    selection = ToolSelection(as_jev(FakeJev(JevError("down"))))
    assert await offered(selection) == [ALL_NAMES]


@pytest.mark.anyio
async def test_every_tool_is_offered_when_no_server_passes() -> None:
    selection = ToolSelection(as_jev(FakeJev(answers({}))))
    assert await offered(selection) == [ALL_NAMES]


@pytest.mark.anyio
async def test_jev_is_asked_once_per_run_not_once_per_model_call() -> None:
    jev = FakeJev(answers({"email": 1.0}))
    seen = await offered(ToolSelection(as_jev(jev)), model_calls=3)
    assert seen == [["email__send_email"]] * 3
    assert len(jev.asked) == 1


@pytest.mark.anyio
async def test_the_tokens_jev_reads_are_reported() -> None:
    reported: list[int] = []
    selection = ToolSelection(as_jev(FakeJev(answers({}), tokens=7)), reported.append)
    await offered(selection)
    assert reported == [7]


def test_no_selection_without_jev_or_when_off() -> None:
    jev = as_jev(FakeJev(answers({})))
    assert jev_tool_selection(None, True) is None
    assert jev_tool_selection(jev, False) is None
    assert isinstance(jev_tool_selection(jev, True), ToolSelection)


def test_the_selection_is_off_by_default_and_reads_a_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings.model_fields["tool_selection"].default is False
    monkeypatch.setenv("DESKCLERK_TOOL_SELECTION", "true")
    assert Settings(audience="deskclerk-api").tool_selection is True


@pytest.fixture
def mcp_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


@pytest.fixture
def other_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


@pytest.fixture
async def connections(
    tmp_path: Path, mcp_server: RunningServer, other_server: RunningServer
) -> OAuthConnections:
    connections = make_connections(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())
    await connect(connections, "ada", server_record(mcp_server))
    await connect(connections, "ada", McpServer("extra", other_server.url, None, None))
    return connections


class RecordingModel(ScriptedModel):
    binds: list[list[str]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.binds.append([tool.name for tool in tools])
        return super().bind_tools(tools, **kwargs)


async def run_selected(
    connections: OAuthConnections,
    mcp_server: RunningServer,
    other_server: RunningServer,
    model: ScriptedModel,
    jev: FakeJev,
) -> AgentResult:
    return await run_agent(
        "ada",
        "save a note",
        "Be careful.",
        [server_record(mcp_server), McpServer("extra", other_server.url, None, None)],
        connections,
        connections.tokens,
        InMemorySaver(),
        "ada/test",
        [],
        model,
        RecordingGate(True),
        pass_through,
        5,
        lambda usage: None,
        selection=jev_tool_selection(as_jev(jev), True),
        confirm_writes=False,
    )


@pytest.mark.anyio
async def test_the_agent_model_is_offered_only_the_selected_servers_tools_on_every_call(
    mcp_server: RunningServer, other_server: RunningServer, connections: OAuthConnections
) -> None:
    jev = FakeJev(answers({"notes": 1.0}))
    model = RecordingModel(
        messages=iter([calls(("notes__read_note", {})), answer("It says: the note")])
    )
    model.binds = []
    result = await run_selected(connections, mcp_server, other_server, model, jev)
    assert all(name.startswith("notes__") for names in model.binds for name in names)
    assert len(model.binds) == 2 and model.binds[0]
    assert [call.result for call in result.executed] == ["the note"]
    assert len(jev.asked) == 1


@pytest.mark.anyio
async def test_a_tool_left_out_of_the_offer_is_refused_like_an_unknown_tool(
    mcp_server: RunningServer, other_server: RunningServer, connections: OAuthConnections
) -> None:
    jev = FakeJev(answers({"notes": 1.0}))
    model = RecordingModel(
        messages=iter([calls(("extra__write_note", {"text": "hi"})), answer("done")])
    )
    model.binds = []
    result = await run_selected(connections, mcp_server, other_server, model, jev)
    assert result.executed == ()
    assert other_server.notes == []
    reply = model.seen[-1][-1]
    assert isinstance(reply, ToolMessage)
    assert (reply.status, reply.content) == ("error", "There is no tool named extra__write_note.")
