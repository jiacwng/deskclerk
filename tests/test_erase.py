# Erasing an employee removes their tokens, conversations and spending rows from the real files
# and nothing of another employee whose id starts the same way; the command logs one
# audit line with the counts and exits with the reason when the token key is wrong.

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from langgraph.checkpoint.base import empty_checkpoint

from deskclerk import __main__ as entry
from deskclerk.conversations import open_checkpointer, thread_id
from deskclerk.providers import Usage
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenStore

KEY = Fernet.generate_key().decode()


async def save_thread(path: Path, user_id: str, checkpoint_id: str | None = None) -> str:
    thread = thread_id(user_id, uuid4())
    checkpoint = empty_checkpoint()
    if checkpoint_id is not None:
        checkpoint["id"] = checkpoint_id
    config: Any = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
    async with open_checkpointer(path, Fernet(KEY)) as saver:
        saved = await saver.aput(config, checkpoint, {"source": "input", "step": 0}, {})
        await saver.aput_writes(saved, [("messages", "hello")], "task")
    return thread


def seed(directory: Path) -> Path:
    path = directory / "tokens.sqlite"
    tokens = TokenStore(path, KEY)
    spending = SpendingStore(directory / "spending.sqlite")
    for user in ("ana", "anabel"):
        for kind in ("tokens", "client", "tool_pins"):
            tokens.put(user, "notes", kind, "secret")
        spending.add(user, "2026-10", "model", Usage(1, 1, 0, 0), 0.5)
        asyncio.run(save_thread(path, user))
        asyncio.run(save_thread(path, user))
    return path


def rows(path: Path, query: str) -> list[tuple[Any, ...]]:
    database = sqlite3.connect(path)
    try:
        return database.execute(query).fetchall()
    finally:
        database.close()


def erase(monkeypatch: pytest.MonkeyPatch, directory: Path, user_id: str, key: str = KEY) -> None:
    monkeypatch.setattr(entry, "log_to_stdout", lambda: None)
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY", key)
    monkeypatch.setenv("DESKCLERK_TOKEN_FILE", str(directory / "tokens.sqlite"))
    entry.main(["erase", user_id])


def test_erase_removes_only_that_employee_in_all_three_places(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = seed(tmp_path)

    erase(monkeypatch, tmp_path, "ana")

    spending = tmp_path / "spending.sqlite"
    assert rows(path, "SELECT DISTINCT user_id FROM secrets") == [("anabel",)]
    assert rows(spending, "SELECT DISTINCT user_id FROM costs") == [("anabel",)]
    assert rows(path, "SELECT DISTINCT substr(thread_id, 1, 7) FROM checkpoints") == [("anabel/",)]
    assert rows(path, "SELECT DISTINCT substr(thread_id, 1, 7) FROM writes") == [("anabel/",)]
    assert rows(path, "SELECT COUNT(DISTINCT thread_id) FROM checkpoints") == [(2,)]


def test_erase_logs_one_audit_line_with_the_counts_and_no_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    seed(tmp_path)

    erase(monkeypatch, tmp_path, "ana")

    [record] = [r for r in caplog.records if r.name == "deskclerk.audit"]
    line = json.loads(record.getMessage())
    assert (line["entry"], line["user"]) == ("erase", "ana")
    assert (line["secrets"], line["threads"], line["costs"]) == (3, 2, 1)
    assert set(line) == {"time", "user", "entry", "secrets", "threads", "costs"}


def test_erase_with_the_wrong_token_key_exits_with_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        erase(monkeypatch, tmp_path, "ana", Fernet.generate_key().decode())

    assert "not encrypted with this token key" in str(stopped.value)
    assert rows(tmp_path / "tokens.sqlite", "SELECT COUNT(*) FROM secrets") == [(6,)]
