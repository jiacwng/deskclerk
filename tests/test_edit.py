# Edit and resend: a request sent earlier is replaced and the turns after it leave the conversation,
# the model reads only the new line of history, and the refusals and the audit line are checked.

import logging
from collections.abc import Iterator

import httpx2
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage

from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from mcp_server import RunningServer, running_mcp_server
from test_agent import answer, calls, script
from test_api import (
    Sign,
    as_user,
    ask,
    ask_in,
    audit_lines,
    connect,
    make_settings,
    notes,
    web,
)
from test_connections import REDIRECT
from test_stream import STREAM, events_of

MISSING = "0d6b2c3e-9d5e-4c27-8f3e-3c1d2b4a5e6f"


@pytest.fixture
def notes_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


def edit(
    client: TestClient, headers: dict[str, str], conversation: object, index: int, request: str
) -> httpx2.Response:
    body = {"request": request, "conversation": conversation, "edit_of": index}
    return client.post("/ask", json=body, headers=headers)


def human_texts(messages: list[BaseMessage]) -> list[str]:
    return [str(message.content) for message in messages if message.type == "human"]


def test_editing_request_0_of_two_turns_leaves_one_turn_with_the_new_request_and_answer(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        edited = edit(site.client, ada, conversation, 0, "first, fixed")
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
        ask_in(site, ada, conversation, "third")
        later = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
    assert edited.status_code == 200
    assert turns == [{"request": "first, fixed", "answer": "a3"}]
    assert later == [
        {"request": "first, fixed", "answer": "a3"},
        {"request": "third", "answer": "a4"},
    ]


def test_the_model_receives_the_new_request_and_not_the_replaced_turns(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        edit(site.client, ada, conversation, 0, "first, fixed")
        ask_in(site, ada, conversation, "third")
    assert human_texts(model.seen[2]) == ["first, fixed"]
    assert [m.content for m in model.seen[2] if m.type == "ai"] == []
    assert human_texts(model.seen[3]) == ["first, fixed", "third"]
    assert [m.content for m in model.seen[3] if m.type == "ai"] == ["a3"]


def test_editing_the_second_request_keeps_the_first_turn_and_forks_after_its_answer(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        ask_in(site, ada, conversation, "third")
        edit(site.client, ada, conversation, 1, "second, fixed")
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
    assert turns == [
        {"request": "first", "answer": "a1"},
        {"request": "second, fixed", "answer": "a4"},
    ]
    assert human_texts(model.seen[3]) == ["first", "second, fixed"]


def test_an_edit_of_a_request_that_does_not_exist_gives_422_and_runs_nothing(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        past_the_end = edit(site.client, ada, conversation, 1, "x")
        negative = edit(site.client, ada, conversation, -1, "x")
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
    assert past_the_end.status_code == 422
    assert past_the_end.json() == {"detail": "no such request to edit"}
    assert negative.status_code == 422
    assert len(model.seen) == 1
    assert turns == [{"request": "first", "answer": "a1"}]


def test_editing_another_employees_or_an_unknown_conversation_gives_404(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"))
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        taken = edit(site.client, grace, conversation, 0, "mine now")
        unknown = edit(site.client, ada, MISSING, 0, "x")
        without = site.client.post("/ask", json={"request": "x", "edit_of": 0}, headers=ada)
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
    assert (taken.status_code, unknown.status_code, without.status_code) == (404, 404, 404)
    assert len(model.seen) == 1
    assert turns == [{"request": "first", "answer": "a1"}]


def test_an_edit_while_calls_wait_for_a_decision_gives_409(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), calls(("notes__write_note", {"text": "hi"})))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        connect(site, ada)
        conversation = ask(site, ada, "first").json()["conversation"]
        paused = ask_in(site, ada, conversation, "save a note").json()
        refused = edit(site.client, ada, conversation, 0, "first, fixed")
        reloaded = site.client.get(f"/conversations/{conversation}", headers=ada).json()
    assert len(paused["pending"]) == 1
    assert refused.status_code == 409
    assert refused.json() == {"detail": "this conversation waits for a confirmation"}
    assert [t["request"] for t in reloaded["turns"]] == ["first", "save a note"]
    assert len(reloaded["pending"]) == 1


def test_a_streaming_edit_streams_the_new_answer_and_replaces_the_later_turns(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("the fixed answer"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        body = {"request": "first, fixed", "conversation": conversation, "edit_of": 0}
        streamed = site.client.post("/ask", json=body, headers={**ada, **STREAM})
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
    found = events_of(streamed.text)
    deltas = [data["text"] for name, data in found if name == "answer"]
    assert found[-1][0] == "done"
    assert "".join(deltas) == found[-1][1]["answer"] == "the fixed answer"
    assert turns == [{"request": "first, fixed", "answer": "the fixed answer"}]
    assert human_texts(model.seen[2]) == ["first, fixed"]


def test_the_audit_line_of_an_edit_records_the_index_and_other_asks_have_none(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"))
    ada = as_user(sign_token, "ada")
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        edit(site.client, ada, conversation, 1, "second, fixed")
        edit(site.client, ada, conversation, 5, "nowhere")
    lines = audit_lines(caplog)
    assert [line.get("edit_of") for line in lines] == [None, None, 1, 5]
    assert [line["outcome"] for line in lines] == ["answered", "answered", "answered", "wrong_edit"]


def test_editing_the_first_request_retitles_the_conversation(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        edit(site.client, ada, conversation, 1, "second, fixed")
        kept = site.client.get("/conversations", headers=ada).json()[0]["title"]
        edit(site.client, ada, conversation, 0, "first, fixed")
        retitled = site.client.get("/conversations", headers=ada).json()[0]["title"]
    assert (kept, retitled) == ("first", "first, fixed")
