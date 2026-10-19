# The write gate: schema first, then ids a destructive call names, then Jev. A fake stands in for
# Jev, the paid API, and records what it was asked. One integration test runs a refused delete
# through the agent against the real test MCP server; one llm test asks the real Jev.

from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from cryptography.fernet import Fernet
from typesafe_sdk import TypeSafeBadRequestError, TypeSafeClient

from deskclerk.agent import ProposedCall, run_agent
from deskclerk.connections import ToolInfo
from deskclerk.jev import Jev, JevError, JevInputTooLong, jev_from_environment
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


class FakeJev:
    def __init__(
        self, probability: float | None = None, error: str = "", most_results: int | None = None
    ) -> None:
        self.probability = probability
        self.error = error
        self.most_results = most_results
        self.asked: list[tuple[Any, str]] = []

    def noul(self, state: Any, instructions: str) -> float:
        self.asked.append((state, instructions))
        if self.most_results is not None and len(state["earlier_results"]) > self.most_results:
            raise JevInputTooLong("too long")
        if self.error:
            raise JevError(self.error)
        assert self.probability is not None
        return self.probability


def as_jev(fake: FakeJev) -> Jev:
    return cast(Jev, fake)


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
async def test_arguments_that_break_the_schema_are_refused_before_jev_is_asked() -> None:
    jev = FakeJev(0.99)
    verdict = await write_check(as_jev(jev))(proposed(DELETE_SCHEMA, {"note_id": 7}))
    assert not verdict.allowed
    assert "schema check" in verdict.reason
    assert jev.asked == []


@pytest.mark.anyio
async def test_a_destructive_call_on_an_id_it_has_not_seen_is_refused() -> None:
    jev = FakeJev(0.99)
    call = proposed(DELETE_SCHEMA, {"note_id": "n-7"}, observed=("n-1 groceries",))
    verdict = await write_check(as_jev(jev))(call)
    assert not verdict.allowed
    assert "grounding check" in verdict.reason
    assert "n-7" in verdict.reason
    assert jev.asked == []


@pytest.mark.anyio
async def test_an_id_seen_in_an_observed_result_or_in_the_request_is_allowed() -> None:
    gate = write_check(None)
    in_result = proposed(DELETE_SCHEMA, {"note_id": "n-7"}, observed=("n-1", "n-7 old"))
    in_request = proposed(DELETE_SCHEMA, {"note_id": "n-7"}, request="delete note n-7")
    assert (await gate(in_result)).allowed
    assert (await gate(in_request)).allowed


@pytest.mark.anyio
async def test_an_id_only_seen_inside_a_longer_id_is_refused() -> None:
    call = proposed(DELETE_SCHEMA, {"note_id": "n-1"}, observed=("n-12 groceries",))
    assert not (await write_check(None)(call)).allowed


@pytest.mark.anyio
async def test_a_name_that_merely_ends_in_id_is_not_an_identifier() -> None:
    schema = {"type": "object", "properties": {"paid": {"type": "string"}}}
    assert (await write_check(None)(proposed(schema, {"paid": "yes"}))).allowed


@pytest.mark.anyio
async def test_a_uuid_argument_is_an_identifier_whatever_its_name() -> None:
    schema = {"type": "object", "properties": {"target": {"type": "string", "format": "uuid"}}}
    verdict = await write_check(None)(proposed(schema, {"target": "0b1f"}))
    assert not verdict.allowed
    assert "grounding check" in verdict.reason


@pytest.mark.anyio
async def test_a_non_destructive_write_is_not_grounding_checked() -> None:
    call = proposed(WRITE_SCHEMA, {"text": "hi", "owner_id": "u-9"}, destructive=False)
    assert (await write_check(None)(call)).allowed


@pytest.mark.anyio
async def test_jev_below_one_half_refuses_and_the_reason_names_jev() -> None:
    jev = FakeJev(0.49)
    verdict = await write_check(as_jev(jev))(proposed(WRITE_SCHEMA, {"text": "hi"}))
    assert not verdict.allowed
    assert "Jev" in verdict.reason


@pytest.mark.anyio
async def test_jev_at_one_half_or_more_allows() -> None:
    verdict = await write_check(as_jev(FakeJev(0.5)))(proposed(WRITE_SCHEMA, {"text": "hi"}))
    assert verdict.allowed


@pytest.mark.anyio
async def test_a_jev_error_fails_closed() -> None:
    jev = FakeJev(error="Jev spent too much")
    verdict = await write_check(as_jev(jev))(proposed(WRITE_SCHEMA, {"text": "hi"}))
    assert not verdict.allowed
    assert "Jev" in verdict.reason
    assert "spent too much" in verdict.reason


def earlier_results_asked(jev: FakeJev) -> list[list[str]]:
    return [state["earlier_results"] for state, _ in jev.asked]


@pytest.mark.anyio
async def test_a_state_jev_refuses_as_too_long_is_asked_again_with_the_newer_half() -> None:
    jev = FakeJev(0.9, most_results=1)
    observed = ("r1", "r2", "r3", "r4", "r5")
    verdict = await write_check(as_jev(jev))(
        proposed(WRITE_SCHEMA, {"text": "hi"}, observed=observed)
    )
    assert earlier_results_asked(jev) == [["r1", "r2", "r3", "r4", "r5"], ["r4", "r5"], ["r5"]]
    assert verdict.allowed


@pytest.mark.anyio
async def test_a_state_always_too_long_ends_with_no_earlier_results_and_then_fails_closed() -> None:
    jev = FakeJev(0.9, most_results=-1)
    observed = ("r1", "r2", "r3")
    verdict = await write_check(as_jev(jev))(
        proposed(WRITE_SCHEMA, {"text": "hi"}, observed=observed)
    )
    assert earlier_results_asked(jev) == [["r1", "r2", "r3"], ["r3"], []]
    assert not verdict.allowed
    assert "failed closed" in verdict.reason


@pytest.mark.anyio
async def test_the_call_is_judged_alone_once_the_earlier_results_are_dropped() -> None:
    jev = FakeJev(0.2, most_results=0)
    call = proposed(WRITE_SCHEMA, {"text": "hi"}, observed=("r1", "r2"))
    verdict = await write_check(as_jev(jev))(call)
    assert earlier_results_asked(jev) == [["r1", "r2"], ["r2"], []]
    assert not verdict.allowed
    assert "does not match" in verdict.reason


@pytest.mark.anyio
async def test_another_jev_error_is_not_retried() -> None:
    jev = FakeJev(error="Jev failed: 500")
    call = proposed(WRITE_SCHEMA, {"text": "hi"}, observed=("r1", "r2"))
    verdict = await write_check(as_jev(jev))(call)
    assert len(jev.asked) == 1
    assert not verdict.allowed


class RefusingClient:
    def __init__(self, body: object) -> None:
        self.body = body

    def system_one(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeSafeBadRequestError(400, self.body, httpx2.Headers())


def test_a_400_max_tokens_exceeded_from_typesafe_is_jev_input_too_long() -> None:
    body = {"detail": {"code": "max_tokens_exceeded"}}
    jev = Jev(cast(TypeSafeClient, RefusingClient(body)), 1.0)
    with pytest.raises(JevInputTooLong):
        jev.noul({}, "q")


def test_any_other_400_from_typesafe_is_a_plain_jev_error() -> None:
    jev = Jev(cast(TypeSafeClient, RefusingClient({"detail": "bad state"})), 1.0)
    with pytest.raises(JevError) as raised:
        jev.noul({}, "q")
    assert not isinstance(raised.value, JevInputTooLong)


@pytest.mark.anyio
async def test_without_jev_only_the_deterministic_checks_run() -> None:
    gate = write_check(None)
    assert (await gate(proposed(WRITE_SCHEMA, {"text": "hi"}))).allowed
    assert not (await gate(proposed(WRITE_SCHEMA, {}))).allowed


@pytest.mark.anyio
async def test_jev_is_asked_the_question_about_exactly_the_call() -> None:
    jev = FakeJev(0.9)
    call = proposed(WRITE_SCHEMA, {"text": "hi"}, request="save hi", observed=("n-1 groceries",))
    await write_check(as_jev(jev))(call)
    [(state, instructions)] = jev.asked
    assert state == {
        "request": "save hi",
        "tool": "delete_note",
        "description": "Deletes a note",
        "arguments": {"text": "hi"},
        "earlier_results": ["n-1 groceries"],
    }
    assert "without doing anything they did not ask for" in instructions


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
            write_check(None),
            5,
            lambda usage: None,
        )
    assert running.deleted == []
    assert result.executed == ()
    assert "grounding check" in result.blocked[0].reason


@pytest.mark.llm
@pytest.mark.anyio
async def test_the_real_jev_separates_a_matching_call_from_a_mismatched_one() -> None:
    jev = jev_from_environment(budget_usd=0.05)
    if jev is None:
        pytest.skip("no TYPESAFE_API_KEY")
    gate = write_check(jev)
    matching = proposed(WRITE_SCHEMA, {"text": "buy milk"}, False, "save a note saying buy milk")
    mismatched = proposed(WRITE_SCHEMA, {"text": "buy milk"}, False, "what is the weather today")
    assert (await gate(matching)).allowed
    assert not (await gate(mismatched)).allowed
