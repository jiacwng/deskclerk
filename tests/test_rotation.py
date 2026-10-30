# A comma separated token key list decrypts with every key and encrypts with the first; rotate-key
# re-encrypts secrets, the key check and the checkpoints so the older keys can be dropped.

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from langgraph.checkpoint.base import empty_checkpoint

from deskclerk import __main__ as entry
from deskclerk.conversations import open_checkpointer, saved_messages, thread_id
from deskclerk.database import Database
from deskclerk.rotation import rotate_key
from deskclerk.tokens import TokenKeyError, TokenStore, multi_fernet

OLD = Fernet.generate_key().decode()
NEW = Fernet.generate_key().decode()


def test_a_value_written_under_the_old_key_reads_with_both_keys(database: Database) -> None:
    TokenStore(database, OLD).put("ada", "notes", "tokens", "secret")
    assert TokenStore(database, f"{NEW},{OLD}").get("ada", "notes", "tokens") == "secret"


def test_rotation_moves_every_secret_to_the_newest_key(database: Database) -> None:
    TokenStore(database, OLD).put("ada", "notes", "tokens", "secret")
    both = TokenStore(database, f"{NEW},{OLD}")
    rotated = {item.table: item.rows for item in rotate_key(database, both.fernet)}
    assert rotated["secrets"] == 1
    assert rotated["key_check"] == 1
    assert TokenStore(database, NEW).get("ada", "notes", "tokens") == "secret"
    with pytest.raises(TokenKeyError, match="not encrypted with this token key"):
        TokenStore(database, OLD)


def test_a_malformed_second_key_names_its_position(database: Database) -> None:
    with pytest.raises(TokenKeyError, match="token key 2 is not a Fernet key"):
        TokenStore(database, f"{NEW},short")


def test_one_key_still_works_alone(database: Database) -> None:
    store = TokenStore(database, OLD)
    store.put("ada", "notes", "tokens", "secret")
    assert TokenStore(database, OLD).get("ada", "notes", "tokens") == "secret"


async def save_thread(database: Database, key: str) -> str:
    thread = thread_id("ada", uuid4())
    config: Any = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": ["hello"]}
    checkpoint["channel_versions"] = {"messages": "1"}
    async with database.serving():
        saver = await open_checkpointer(database, multi_fernet(key))
        saved = await saver.aput(
            config, checkpoint, {"source": "input", "step": 0}, {"messages": "1"}
        )
        await saver.aput_writes(saved, [("messages", "hello")], "task")
    return thread


async def read_thread(database: Database, key: str, thread: str) -> object:
    async with database.serving():
        saver = await open_checkpointer(database, multi_fernet(key))
        return await saved_messages(saver, thread)


@pytest.mark.anyio
async def test_a_checkpoint_survives_rotation_and_loads_with_the_newest_key_alone(
    database: Database,
) -> None:
    thread = await save_thread(database, OLD)
    rotated = {item.table: item.rows for item in rotate_key(database, multi_fernet(f"{NEW},{OLD}"))}
    assert rotated["checkpoint_blobs"] > 0
    assert rotated["checkpoint_writes"] == 1
    assert await read_thread(database, NEW, thread) == ["hello"]
    with pytest.raises(Exception, match="InvalidToken|^$"):
        await read_thread(database, OLD, thread)


def test_the_command_rotates_and_prints_the_counts(
    database: Database, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    TokenStore(database, OLD).put("ada", "notes", "tokens", "secret")
    asyncio.run(save_thread(database, OLD))
    monkeypatch.setenv("DESKCLERK_TOKEN_KEY", f"{NEW},{OLD}")
    entry.main(["rotate-key"])
    assert "rotated 1 rows in secrets" in capsys.readouterr().out
    assert TokenStore(database, NEW).get("ada", "notes", "tokens") == "secret"
