# Erasing an employee removes their tokens, conversations and spending rows from the real database
# and nothing of another employee whose id starts the same way; the command logs one audit line
# with the counts and exits with the reason when the token key is wrong.

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from deskclerk import __main__ as entry
from deskclerk.auth import Identity
from deskclerk.conversations import delete_user_threads, open_checkpointer, thread_id
from deskclerk.database import Database
from deskclerk.providers import Usage
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore

KEY = Fernet.generate_key().decode()


async def save_thread(
    saver: AsyncPostgresSaver, user_id: str, checkpoint_id: str | None = None
) -> str:
    thread = thread_id(user_id, uuid4())
    config: Any = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    if checkpoint_id is not None:
        checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = {"messages": ["hello"]}
    checkpoint["channel_versions"] = {"messages": "1"}
    saved = await saver.aput(config, checkpoint, {"source": "input", "step": 0}, {"messages": "1"})
    await saver.aput_writes(saved, [("messages", "hello")], "task")
    return thread


async def save_threads(database: Database, *user_ids: str) -> list[str]:
    async with database.serving():
        saver = await open_checkpointer(database, Fernet(KEY))
        return [await save_thread(saver, user_id) for user_id in user_ids]


def seed(database: Database) -> None:
    tokens = TokenStore(database, KEY)
    spending = SpendingStore(database)
    sessions = SessionStore(database)
    for user in ("ana", "anabel"):
        for kind in ("tokens", "client", "tool_pins"):
            tokens.put(user, "notes", kind, "secret")
        sessions.create(Identity(user, "deskclerk-api", user), 3600)
        spending.add(user, "2026-10", "model", Usage(1, 1, 0, 0), 0.5)
    asyncio.run(save_threads(database, "ana", "ana", "anabel", "anabel"))


def rows(database: Database, query: str) -> list[tuple[Any, ...]]:
    with database.pool.connection() as connection:
        return connection.execute(query).fetchall()


def erase(monkeypatch: pytest.MonkeyPatch, user_id: str, key: str = KEY) -> None:
    monkeypatch.setattr(entry, "log_to_stdout", lambda: None)
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY", key)
    entry.main(["erase", user_id])


def test_erase_removes_only_that_employee_in_all_three_places(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(database)

    erase(monkeypatch, "ana")

    assert rows(database, "SELECT DISTINCT user_id FROM secrets") == [("anabel",)]
    assert rows(database, "SELECT DISTINCT user_id FROM sessions") == [("anabel",)]
    assert rows(database, "SELECT DISTINCT user_id FROM costs") == [("anabel",)]
    for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        query = f"SELECT DISTINCT substr(thread_id, 1, 7) FROM {table}"  # noqa: S608
        assert rows(database, query) == [("anabel/",)]
    assert rows(database, "SELECT COUNT(DISTINCT thread_id) FROM checkpoints") == [(2,)]


def test_erase_keeps_an_employee_whose_id_extends_the_erased_id_after_a_slash(
    database: Database,
) -> None:
    kept = asyncio.run(save_threads(database, "a", "a/b"))[1]

    assert delete_user_threads(database, "a") == 1

    assert rows(database, "SELECT DISTINCT thread_id FROM checkpoints") == [(kept,)]
    assert rows(database, "SELECT DISTINCT thread_id FROM checkpoint_writes") == [(kept,)]


def test_erase_logs_one_audit_line_with_the_counts_and_no_content(
    database: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    seed(database)

    erase(monkeypatch, "ana")

    [record] = [r for r in caplog.records if r.name == "deskclerk.audit"]
    line = json.loads(record.getMessage())
    assert (line["entry"], line["user"]) == ("erase", "ana")
    assert (line["secrets"], line["sessions"], line["threads"], line["costs"]) == (3, 1, 2, 1)
    assert set(line) == {"time", "user", "entry", "secrets", "sessions", "threads", "costs"}


def test_erase_with_the_wrong_token_key_exits_with_an_error(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(database)
    with pytest.raises(SystemExit) as stopped:
        erase(monkeypatch, "ana", Fernet.generate_key().decode())

    assert "not encrypted with this token key" in str(stopped.value)
    assert rows(database, "SELECT COUNT(*) FROM secrets") == [(6,)]
