# The write gate: schema first, then ids a destructive call names. One integration test runs a
# refused delete through the agent against the real test MCP server.

from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from langgraph.checkpoint.memory import InMemorySaver

from deskclerk.agent import ProposedCall, run_agent
from deskclerk.connections import ToolInfo
from deskclerk.conversations import open_checkpointer, saved_messages
from deskclerk.screen import pass_through
from deskclerk.writecheck import write_check
from mcp_server import running_mcp_server
from test_agent import answer, calls, script
from test_connections import REDIRECT, connect, make_connections, server_record

DELETE_SCHEMA = {
    "type": "object",
    "properties": {"note_id": {"type": "string"}},
    "required": ["note_id"],
}
WRITE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "owner_id": {"type": "string"}},
    "required": ["text"],
}


PAID_SCHEMA = {"type": "object", "properties": {"paid": {"type": "string"}}}
UUID_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string", "format": "uuid"}},
}


def proposed(
    schema: dict[str, Any],
    arguments: dict[str, Any],
    destructive: bool = True,
    request: str = "tidy my notes",
    observed: tuple[str, ...] = (),
) -> ProposedCall:
    tool = ToolInfo("delete_note", "Deletes a note", schema, False, destructive)
    return ProposedCall(request, "notes", tool, arguments, observed)


@pytest.mark.anyio
async def test_arguments_that_break_the_schema_are_refused() -> None:
    verdict = await write_check()(proposed(DELETE_SCHEMA, {"note_id": 7}))
    assert not verdict.allowed
    assert "schema check" in verdict.reason


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("call", "allowed", "reasons"),
    [
        pytest.param(
            proposed(DELETE_SCHEMA, {"note_id": "n-7"}, observed=("n-1 groceries",)),
            False,
            ("grounding check", "n-7"),
            id="a-destructive-call-on-an-id-it-has-not-seen-is-refused",
        ),
        pytest.param(
            proposed(DELETE_SCHEMA, {"note_id": "n-7"}, observed=("n-1", "n-7 old")),
            True,
            (),
            id="an-id-seen-in-an-observed-result-is-allowed",
        ),
        pytest.param(
            proposed(DELETE_SCHEMA, {"note_id": "n-7"}, request="delete note n-7"),
            True,
            (),
            id="an-id-seen-in-the-request-is-allowed",
        ),
        pytest.param(
            proposed(DELETE_SCHEMA, {"note_id": "n-1"}, observed=("n-12 groceries",)),
            False,
            (),
            id="an-id-only-seen-inside-a-longer-id-is-refused",
        ),
        pytest.param(
            proposed(PAID_SCHEMA, {"paid": "yes"}),
            True,
            (),
            id="a-name-that-merely-ends-in-id-is-not-an-identifier",
        ),
        pytest.param(
            proposed(UUID_SCHEMA, {"target": "0b1f"}),
            False,
            ("grounding check",),
            id="a-uuid-argument-is-an-identifier-whatever-its-name",
        ),
        pytest.param(
            proposed(WRITE_SCHEMA, {"text": "hi", "owner_id": "u-9"}, destructive=False),
            True,
            (),
            id="a-non-destructive-write-is-not-grounding-checked",
        ),
        pytest.param(
            proposed(WRITE_SCHEMA, {"text": "hi"}),
            True,
            (),
            id="a-valid-write-passes",
        ),
        pytest.param(
            proposed(WRITE_SCHEMA, {}),
            False,
            (),
            id="a-write-missing-a-required-field-is-refused",
        ),
    ],
)
async def test_the_write_check_allows_a_call_only_when_its_ids_are_grounded(
    call: ProposedCall, allowed: bool, reasons: tuple[str, ...]
) -> None:
    verdict = await write_check()(call)
    assert verdict.allowed is allowed
    for reason in reasons:
        assert reason in verdict.reason


@pytest.mark.anyio
async def test_a_delete_of_an_unseen_id_never_reaches_the_server(tmp_path: Path) -> None:
    connections = make_connections(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())
    with running_mcp_server(REDIRECT, delete_tool=True) as running:
        server = server_record(running)
        await connect(connections, "ada", server)
        model = script(calls(("notes__delete_note", {"note_id": "n-7"})), answer("could not"))
        result = await run_agent(
            "ada",
            "clean up my notes",
            "system",
            [server],
            connections,
            connections.tokens,
            InMemorySaver(),
            "ada/test",
            [],
            model,
            write_check(),
            pass_through,
            5,
            lambda usage: None,
        )
    assert running.deleted == []
    assert result.executed == ()
    assert "grounding check" in result.blocked[0].reason


@pytest.mark.anyio
async def test_a_delete_may_name_an_id_the_employee_gave_in_an_earlier_turn(tmp_path: Path) -> None:
    connections = make_connections(tmp_path / "tokens.sqlite", Fernet.generate_key().decode())
    with running_mcp_server(REDIRECT, delete_tool=True) as running:
        server = server_record(running)
        await connect(connections, "ada", server)
        async with open_checkpointer(connections.tokens.path, connections.tokens.fernet) as saver:
            for request, model in (
                ("note n-7 is the old one", script(answer("Noted."))),
                (
                    "delete it",
                    script(calls(("notes__delete_note", {"note_id": "n-7"})), answer("deleted")),
                ),
            ):
                result = await run_agent(
                    "ada",
                    request,
                    "system",
                    [server],
                    connections,
                    connections.tokens,
                    saver,
                    "ada/test",
                    await saved_messages(saver, "ada/test"),
                    model,
                    write_check(),
                    pass_through,
                    5,
                    lambda usage: None,
                )
    assert running.deleted == ["n-7"]
    assert result.blocked == ()
