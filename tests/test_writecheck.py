# The write gate: schema first, then ids a destructive call names. One integration test runs a
# refused delete through the agent against the real test MCP server.

from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet

from deskclerk.agent import ProposedCall, run_agent
from deskclerk.connections import ToolInfo
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
async def test_a_destructive_call_on_an_id_it_has_not_seen_is_refused() -> None:
    call = proposed(DELETE_SCHEMA, {"note_id": "n-7"}, observed=("n-1 groceries",))
    verdict = await write_check()(call)
    assert not verdict.allowed
    assert "grounding check" in verdict.reason
    assert "n-7" in verdict.reason


@pytest.mark.anyio
async def test_an_id_seen_in_an_observed_result_or_in_the_request_is_allowed() -> None:
    gate = write_check()
    in_result = proposed(DELETE_SCHEMA, {"note_id": "n-7"}, observed=("n-1", "n-7 old"))
    in_request = proposed(DELETE_SCHEMA, {"note_id": "n-7"}, request="delete note n-7")
    assert (await gate(in_result)).allowed
    assert (await gate(in_request)).allowed


@pytest.mark.anyio
async def test_an_id_only_seen_inside_a_longer_id_is_refused() -> None:
    call = proposed(DELETE_SCHEMA, {"note_id": "n-1"}, observed=("n-12 groceries",))
    assert not (await write_check()(call)).allowed


@pytest.mark.anyio
async def test_a_name_that_merely_ends_in_id_is_not_an_identifier() -> None:
    schema = {"type": "object", "properties": {"paid": {"type": "string"}}}
    assert (await write_check()(proposed(schema, {"paid": "yes"}))).allowed


@pytest.mark.anyio
async def test_a_uuid_argument_is_an_identifier_whatever_its_name() -> None:
    schema = {"type": "object", "properties": {"target": {"type": "string", "format": "uuid"}}}
    verdict = await write_check()(proposed(schema, {"target": "0b1f"}))
    assert not verdict.allowed
    assert "grounding check" in verdict.reason


@pytest.mark.anyio
async def test_a_non_destructive_write_is_not_grounding_checked() -> None:
    call = proposed(WRITE_SCHEMA, {"text": "hi", "owner_id": "u-9"}, destructive=False)
    assert (await write_check()(call)).allowed


@pytest.mark.anyio
async def test_a_valid_write_passes_and_a_missing_field_does_not() -> None:
    gate = write_check()
    assert (await gate(proposed(WRITE_SCHEMA, {"text": "hi"}))).allowed
    assert not (await gate(proposed(WRITE_SCHEMA, {}))).allowed


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
            model,
            write_check(),
            pass_through,
            5,
            lambda usage: None,
        )
    assert running.deleted == []
    assert result.executed == ()
    assert "grounding check" in result.blocked[0].reason
