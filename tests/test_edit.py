# Edit and resend: a request sent earlier is replaced and the turns after it leave the conversation,
# the model reads only the new line of history, and the refusals and the audit line are checked.

import logging
from collections.abc import Iterator

import httpx2
import psycopg
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.conversations import FernetCipher
from deskclerk.database import Database
from mcp_server import RunningServer, running_mcp_server
from test_agent import ProviderGoneAfterScript, answer, calls, script
from test_api import (
    Sign,
    as_user,
    ask,
    ask_in,
    audit_lines,
    confirm,
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


def stored_text(database: Database, key: str, user: str, conversation: object) -> str:
    serde = EncryptedSerializer(FernetCipher(Fernet(key)))
    thread = f"{user}/{conversation}"
    found: list[str] = []
    with psycopg.connect(database.url) as connection:
        for checkpoint_id, checkpoint, metadata in connection.execute(
            "SELECT checkpoint_id, checkpoint::text, metadata::text "
            "FROM checkpoints WHERE thread_id = %s",
            (thread,),
        ):
            found += ["CHK " + checkpoint_id + checkpoint + metadata]
        for channel, version, kind, blob in connection.execute(
            "SELECT channel, version, type, blob FROM checkpoint_blobs "
            "WHERE thread_id = %s AND blob IS NOT NULL",
            (thread,),
        ):
            found.append(f"BLOB {channel} {version} " + str(serde.loads_typed((kind, bytes(blob)))))
        for checkpoint_id, kind, blob in connection.execute(
            "SELECT checkpoint_id, type, blob FROM checkpoint_writes WHERE thread_id = %s",
            (thread,),
        ):
            found.append(f"WRITE {checkpoint_id} " + str(serde.loads_typed((kind, bytes(blob)))))
    return "\x00".join(found)


def test_after_an_edit_no_stored_row_holds_the_replaced_request_and_the_kept_turns_load(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"), answer("a5"))
    ada = as_user(sign_token, "ada")
    key = Fernet.generate_key().decode()
    with web(
        make_settings(token_settings),
        [notes(notes_server)],
        model,
        token_signing_key,
        database,
        key=key,
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second old request")
        ask_in(site, ada, conversation, "third old request")
        before = stored_text(database, key, "ada", conversation)
        edit(site.client, ada, conversation, 1, "second new request")
        after = stored_text(database, key, "ada", conversation)
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
        next_ask = ask_in(site, ada, conversation, "fourth")
    assert "second old request" in before
    assert "third old request" in before
    assert "second old request" not in after
    assert "third old request" not in after
    assert turns == [
        {"request": "first", "answer": "a1"},
        {"request": "second new request", "answer": "a4"},
    ]
    assert next_ask.json()["answer"] == "a5"


def test_a_failed_edit_keeps_the_old_branch_until_the_next_successful_run(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = ProviderGoneAfterScript(messages=iter([answer("a1"), answer("a2")]))
    ada = as_user(sign_token, "ada")
    key = Fernet.generate_key().decode()
    with web(
        make_settings(token_settings),
        [notes(notes_server)],
        model,
        token_signing_key,
        database,
        key=key,
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second old request")
        failed = edit(site.client, ada, conversation, 1, "second new request")
        kept = stored_text(database, key, "ada", conversation)
        model.messages = iter([answer("a4")])
        ask_in(site, ada, conversation, "third")
        cleaned = stored_text(database, key, "ada", conversation)
    assert failed.status_code == 502
    assert "second old request" in kept
    assert "second old request" not in cleaned


def test_an_edit_leaves_another_conversation_of_the_same_employee_untouched(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("b1"), answer("b2"), answer("a2"))
    ada = as_user(sign_token, "ada")
    key = Fernet.generate_key().decode()
    with web(
        make_settings(token_settings),
        [notes(notes_server)],
        model,
        token_signing_key,
        database,
        key=key,
    ) as site:
        one = ask(site, ada, "one first").json()["conversation"]
        other = ask(site, ada, "other first").json()["conversation"]
        ask_in(site, ada, other, "other second")
        before = stored_text(database, key, "ada", other)
        edit(site.client, ada, one, 0, "one first, fixed")
        after = stored_text(database, key, "ada", other)
        turns = site.client.get(f"/conversations/{other}", headers=ada).json()["turns"]
    assert after == before
    assert [t["request"] for t in turns] == ["other first", "other second"]


def test_editing_request_2_and_then_request_0_forks_from_kept_ancestors(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"), answer("a5"))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        ask_in(site, ada, conversation, "third")
        first_edit = edit(site.client, ada, conversation, 2, "third, fixed")
        after_first = site.client.get(f"/conversations/{conversation}", headers=ada).json()
        second_edit = edit(site.client, ada, conversation, 0, "first, fixed")
        after_second = site.client.get(f"/conversations/{conversation}", headers=ada).json()
    assert (first_edit.status_code, second_edit.status_code) == (200, 200)
    assert [t["request"] for t in after_first["turns"]] == ["first", "second", "third, fixed"]
    assert after_second["turns"] == [{"request": "first, fixed", "answer": "a5"}]
    assert human_texts(model.seen[3]) == ["first", "second", "third, fixed"]
    assert human_texts(model.seen[4]) == ["first, fixed"]


def test_editing_request_1_twice_in_a_row_keeps_the_first_turn(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(answer("a1"), answer("a2"), answer("a3"), answer("a4"), answer("a5"))
    ada = as_user(sign_token, "ada")
    key = Fernet.generate_key().decode()
    with web(
        make_settings(token_settings),
        [notes(notes_server)],
        model,
        token_signing_key,
        database,
        key=key,
    ) as site:
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        ask_in(site, ada, conversation, "third")
        first_edit = edit(site.client, ada, conversation, 1, "second v2")
        second_edit = edit(site.client, ada, conversation, 1, "second v3")
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
        stored = stored_text(database, key, "ada", conversation)
    assert (first_edit.status_code, second_edit.status_code) == (200, 200)
    assert turns == [
        {"request": "first", "answer": "a1"},
        {"request": "second v3", "answer": "a5"},
    ]
    assert human_texts(model.seen[3]) == ["first", "second v2"]
    assert human_texts(model.seen[4]) == ["first", "second v3"]
    assert "second v2" not in stored
    assert "third" not in stored


def test_a_pause_after_an_edit_keeps_its_pending_call_until_it_is_confirmed(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(
        answer("a1"),
        answer("a2"),
        answer("a3"),
        calls(("notes__write_note", {"text": "hi"})),
        answer("saved"),
    )
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        connect(site, ada)
        conversation = ask(site, ada, "first").json()["conversation"]
        ask_in(site, ada, conversation, "second")
        edit(site.client, ada, conversation, 0, "first, fixed")
        paused = ask_in(site, ada, conversation, "save a note").json()
        waiting = site.client.get(f"/conversations/{conversation}", headers=ada).json()
        decisions = {call["id"]: "approve" for call in paused["pending"]}
        done = confirm(site, ada, conversation, decisions)
        turns = site.client.get(f"/conversations/{conversation}", headers=ada).json()["turns"]
    assert len(paused["pending"]) == 1
    assert len(waiting["pending"]) == 1
    assert done.status_code == 200
    assert done.json()["answer"] == "saved"
    assert [note[1] for note in notes_server.notes] == ["hi"]
    assert turns[-1] == {"request": "save a note", "answer": "saved"}
