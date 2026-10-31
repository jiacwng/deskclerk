# Streamed answers: the same ask as Server-Sent Events, the events in order, the last event equal
# to the JSON reply, errors that carry the calls that ran, and a client that goes away mid-run.

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import httpx
import httpx2
import pytest
import uvicorn
from cryptography.fernet import Fernet
from langchain_core.language_models import BaseChatModel

from deskclerk.api import create_app
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.conversations import close_unanswered_calls, open_checkpointer, saved_messages
from deskclerk.conversations import thread_id as thread_of
from deskclerk.database import Database
from deskclerk.servers import McpServer
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore
from deskclerk.writecheck import write_check
from mcp_server import RunningServer, running_mcp_server
from test_agent import ProviderGoneAfterScript, answer, calls, script
from test_api import (
    Sign,
    SlowModel,
    Web,
    as_user,
    ask,
    audit_lines,
    connect,
    make_settings,
    notes,
    open_notes,
    stored_catalog,
    web,
)
from test_connections import REDIRECT

STREAM = {"Accept": "text/event-stream"}


@pytest.fixture
def notes_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


def events_of(text: str) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for block in text.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if lines:
            found.append((lines[0].removeprefix("event: "), json.loads(lines[1][len("data: ") :])))
    return found


def stream(site: Web, headers: dict[str, str], request: str = "read it") -> httpx2.Response:
    return site.client.post("/ask", json={"request": request}, headers={**headers, **STREAM})


def test_the_stream_gives_the_call_then_the_answer_then_the_whole_reply(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    model = script(calls(("open__read_note", {})), answer("hello there my friend"))
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with web(
            make_settings(token_settings), [open_notes(running)], model, token_signing_key, database
        ) as site:
            response = stream(site, as_user(sign_token, "ada"))
    assert response.headers["content-type"].startswith("text/event-stream")
    found = events_of(response.text)
    names = [name for name, _ in found]
    assert names[:2] == ["call", "call"]
    assert names[-1] == "done"
    assert set(names[2:-1]) == {"answer"}
    started, ended = found[0][1], found[1][1]
    assert (started["server"], started["tool"], started["status"]) == (
        "open",
        "read_note",
        "running",
    )
    assert (ended["id"], ended["status"]) == (started["id"], "ok")
    deltas = [data["text"] for name, data in found if name == "answer"]
    assert len(deltas) > 1
    assert "".join(deltas) == found[-1][1]["answer"] == "hello there my friend"
    assert found[-1][1]["executed"][0]["result"] == "the note"
    assert "the note" not in "".join(deltas)


def test_the_last_event_equals_the_json_reply_for_the_same_scripted_model(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    def scripted() -> BaseChatModel:
        return script(calls(("open__read_note", {})), answer("done reading"))

    ada = as_user(sign_token, "ada")
    settings = make_settings(token_settings)
    key = Fernet.generate_key().decode()
    with running_mcp_server(REDIRECT, oauth=False) as running:
        servers = [open_notes(running)]
        with web(settings, servers, scripted(), token_signing_key, database, key=key) as site:
            plain = ask(site, ada, "read it")
        with web(settings, servers, scripted(), token_signing_key, database, key=key) as site:
            streamed = events_of(stream(site, ada).text)[-1]
    assert plain.headers["content-type"] == "application/json"
    assert streamed[0] == "done"
    assert streamed[1] | {"conversation": ""} == plain.json() | {"conversation": ""}
    assert streamed[1]["executed"][0]["result"] == "the note"


def test_a_call_that_waits_for_the_employee_arrives_as_pending_and_the_reply_lists_it(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(calls(("notes__write_note", {"text": "hi"})))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        connect(site, ada)
        found = events_of(stream(site, ada, "save a note").text)
    assert [name for name, _ in found] == ["pending", "done"]
    waiting = found[0][1]
    assert (waiting["server"], waiting["tool"], waiting["arguments"]) == (
        "notes",
        "write_note",
        {"text": "hi"},
    )
    assert found[1][1]["pending"] == [waiting]
    assert notes_server.notes == []


def test_an_error_event_carries_the_status_the_detail_and_the_calls_that_ran(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = ProviderGoneAfterScript(messages=iter([calls(("notes__write_note", {"text": "hi"}))]))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        connect(site, ada)
        paused = ask(site, ada, "save a note").json()
        decisions = {call["id"]: "approve" for call in paused["pending"]}
        body = {"conversation": paused["conversation"], "decisions": decisions}
        response = site.client.post("/ask/confirm", json=body, headers={**ada, **STREAM})
    found = events_of(response.text)
    assert [name for name, _ in found] == ["call", "call", "error"]
    failure = found[-1][1]
    assert failure["status"] == 502
    assert failure["detail"] == "the model did not answer (APIConnectionError)"
    [shown] = failure["calls"]
    assert (shown["tool"], shown["arguments"], shown["is_error"]) == (
        "write_note",
        {"text": "hi"},
        False,
    )
    assert [note[1] for note in notes_server.notes] == ["hi"]


def test_a_refused_streaming_ask_gets_the_status_and_body_of_the_json_refusal(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    with web(make_settings(token_settings), [], script(), token_signing_key, database) as site:
        ada = as_user(sign_token, "ada")
        body = {"request": "hi", "conversation": "00000000-0000-4000-8000-000000000000"}
        plain = site.client.post("/ask", json=body, headers=ada)
        streamed = site.client.post("/ask", json=body, headers={**ada, **STREAM})
    assert plain.status_code == streamed.status_code == 404
    assert streamed.headers["content-type"] == "application/json"
    assert streamed.json() == plain.json() == {"detail": "no such conversation"}


def test_a_rate_limited_streaming_ask_gets_429_with_retry_after(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, requests_per_minute=1)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        ada = as_user(sign_token, "ada")
        assert stream(site, ada).status_code == 200
        refused = stream(site, ada)
    assert refused.status_code == 429
    assert refused.headers["retry-after"] == "60"
    assert refused.headers["content-type"] == "application/json"
    assert refused.json() == {"detail": "at most 1 asks a minute"}


def test_a_streaming_ask_on_a_conversation_that_waits_gets_409(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(calls(("notes__write_note", {"text": "hi"})))
    ada = as_user(sign_token, "ada")
    with web(
        make_settings(token_settings), [notes(notes_server)], model, token_signing_key, database
    ) as site:
        connect(site, ada)
        paused = ask(site, ada, "save a note").json()
        body = {"request": "more", "conversation": paused["conversation"]}
        refused = site.client.post("/ask", json=body, headers={**ada, **STREAM})
    assert refused.status_code == 409
    assert refused.json() == {"detail": "this conversation waits for a confirmation"}


def test_a_client_without_the_accept_header_still_gets_one_json_reply(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    model = script(answer("plain"))
    with web(make_settings(token_settings), [], model, token_signing_key, database) as site:
        response = ask(site, as_user(sign_token, "ada"))
    assert response.headers["content-type"] == "application/json"
    assert response.json()["answer"] == "plain"


def test_a_comment_line_keeps_a_quiet_stream_alive(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deskclerk.api.STREAM_PING_SECONDS", 0.05)
    model = SlowModel(messages=iter([answer("slow")]))
    with web(make_settings(token_settings), [], model, token_signing_key, database) as site:
        response = stream(site, as_user(sign_token, "ada"))
    assert ": ping" in response.text
    assert events_of(response.text)[-1][0] == "done"


@contextmanager
def listening(
    settings: Settings,
    servers: list[McpServer],
    model: BaseChatModel,
    signing_key: SigningKeyLookup,
    database: Database,
    key: str,
) -> Iterator[str]:
    tokens = TokenStore(database, key)
    app = create_app(
        settings,
        database,
        stored_catalog(database, servers),
        OAuthConnections(tokens, REDIRECT, "deskclerk test", 600),
        tokens,
        SpendingStore(database),
        SessionStore(database),
        model,
        write_check(),
        None,
        signing_key,
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join()


def unanswered_calls(database_url: str, key: str, user: str, conversation: UUID) -> int:
    async def read() -> int:
        other = Database(database_url)
        try:
            async with other.serving():
                saver = await open_checkpointer(other, Fernet(key))
                history = await saved_messages(saver, thread_of(user, conversation))
                return len(close_unanswered_calls(history))
        finally:
            other.close()

    return asyncio.run(read())


def test_a_client_that_disconnects_mid_run_frees_the_employee_and_leaves_no_open_call(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    model = script(calls(("open__pause", {"seconds": 8})), answer("second ask answered"))
    ada = as_user(sign_token, "ada")
    key = Fernet.generate_key().decode()
    with running_mcp_server(REDIRECT, oauth=False, heavy_tools=True) as running:
        with listening(
            make_settings(token_settings),
            [open_notes(running)],
            model,
            token_signing_key,
            database,
            key,
        ) as base:
            with httpx.Client(base_url=base, timeout=10) as client:
                with client.stream(
                    "POST", "/ask", json={"request": "rest"}, headers={**ada, **STREAM}
                ) as first:
                    conversation = first.headers["x-conversation"]
                    for line in first.iter_lines():
                        if line == "event: call":
                            break
                second = client.post("/ask", json={"request": "again"}, headers=ada, timeout=4)
    assert second.status_code == 200
    assert second.json()["answer"] == "second ask answered"
    [cancelled] = [line for line in audit_lines(caplog) if line["outcome"] == "cancelled"]
    assert cancelled["conversation"] == conversation
    assert unanswered_calls(database.url, key, "ada", UUID(conversation)) == 0
