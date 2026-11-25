# The write gate: schema first, then ids a destructive call names. One integration test runs a
# refused delete through the agent against the real test MCP server.

from collections.abc import AsyncIterator
from typing import Any

import pytest
from cryptography.fernet import Fernet

from deskclerk.agent import ProposedCall
from deskclerk.connections import OAuthConnections, ToolInfo
from deskclerk.conversations import open_checkpointer
from deskclerk.database import Database
from deskclerk.writecheck import write_check
from mcp_server import running_mcp_server
from test_agent import answer, calls, run_approving, script
from test_connections import REDIRECT, connect, server_record, serving_connections

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
BULK_DELETE_SCHEMA = {
    "type": "object",
    "properties": {"note_ids": {"type": "array", "items": {"type": "string"}}},
}
NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {"type": "object", "properties": {"user_id": {"type": "string"}}},
    },
}
UUID_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "targets": {"type": "array", "items": {"type": "string", "format": "uuid"}},
    },
}
BROKEN_REF_SCHEMA = {
    "type": "object",
    "properties": {"note_id": {"$ref": "#/$defs/missing"}},
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
        pytest.param(
            proposed(BULK_DELETE_SCHEMA, {"note_ids": ["n-1", "invented"]}, observed=("n-1",)),
            False,
            ("grounding check", "note_ids[1]=invented"),
            id="an-ungrounded-id-inside-a-list-is-refused",
        ),
        pytest.param(
            proposed(BULK_DELETE_SCHEMA, {"note_ids": ["n-1", "n-2"]}, observed=("n-1 n-2",)),
            True,
            (),
            id="grounded-ids-inside-a-list-are-allowed",
        ),
        pytest.param(
            proposed(NESTED_SCHEMA, {"owner": {"user_id": "u-9"}}),
            False,
            ("grounding check", "owner.user_id=u-9"),
            id="an-ungrounded-id-inside-a-nested-object-is-refused",
        ),
        pytest.param(
            proposed(UUID_LIST_SCHEMA, {"targets": ["0b1f"]}),
            False,
            ("grounding check", "targets[0]=0b1f"),
            id="a-uuid-string-inside-an-array-is-an-identifier",
        ),
        pytest.param(
            proposed(BROKEN_REF_SCHEMA, {"note_id": "n-1"}, observed=("n-1",)),
            False,
            ("schema check", "cannot be resolved"),
            id="a-schema-with-an-unresolvable-reference-is-refused",
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


@pytest.fixture
async def connections(database: Database) -> AsyncIterator[OAuthConnections]:
    async with serving_connections(database, Fernet.generate_key().decode()) as running:
        yield running


@pytest.mark.anyio
async def test_a_delete_of_an_unseen_id_never_reaches_the_server(
    connections: OAuthConnections,
) -> None:
    with running_mcp_server(REDIRECT, delete_tool=True) as running:
        server = server_record(running)
        await connect(connections, "ada", server)
        model = script(calls(("notes__delete_note", {"note_id": "n-7"})), answer("could not"))
        result = await run_approving(
            connections, [server], model, write_check(), request="clean up my notes"
        )
    assert running.deleted == []
    assert result.executed == ()
    assert "grounding check" in result.blocked[0].reason


@pytest.mark.anyio
async def test_a_delete_may_name_an_id_the_employee_gave_in_an_earlier_turn(
    connections: OAuthConnections,
) -> None:
    with running_mcp_server(REDIRECT, delete_tool=True) as running:
        server = server_record(running)
        await connect(connections, "ada", server)
        saver = await open_checkpointer(connections.tokens.database, connections.tokens.fernet)
        for request, model in (
            ("note n-7 is the old one", script(answer("Noted."))),
            (
                "delete it",
                script(calls(("notes__delete_note", {"note_id": "n-7"})), answer("deleted")),
            ),
        ):
            result = await run_approving(
                connections, [server], model, write_check(), checkpointer=saver, request=request
            )
    assert running.deleted == ["n-7"]
    assert result.blocked == ()
