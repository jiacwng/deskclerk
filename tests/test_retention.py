# Retention deletes a conversation whose newest checkpoint is older than the retention period and
# keeps a newer one; zero days keeps everything.

import asyncio
import logging
import time

import psycopg
import pytest
from cryptography.fernet import Fernet
from langgraph.checkpoint.base.id import UUID

from deskclerk import retention
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.conversations import open_checkpointer, record_turn
from deskclerk.database import Database
from deskclerk.retention import (
    TICKS_PER_SECOND,
    UUID_EPOCH_TICKS,
    delete_old_threads,
    delete_old_threads_daily,
)
from test_agent import script
from test_api import make_settings, web
from test_erase import KEY, rows, save_thread

DAY = 86400


def checkpoint_id_aged(days: int) -> str:
    ticks = int((time.time() - days * DAY) * TICKS_PER_SECOND) + UUID_EPOCH_TICKS
    value = ((ticks >> 12) & 0xFFFFFFFFFFFF) << 80 | (ticks & 0xFFF) << 64
    return str(UUID(int=value, version=6))


async def save_and_sweep(
    database: Database, days: int, *saved: tuple[str, str | None]
) -> tuple[list[str], int]:
    async with database.serving():
        saver = await open_checkpointer(database, Fernet(KEY))
        threads = [await save_thread(saver, user, aged) for user, aged in saved]
        return threads, await delete_old_threads(database, saver, days)


def test_an_old_thread_is_deleted_and_a_new_one_is_kept(database: Database) -> None:
    threads, deleted = asyncio.run(
        save_and_sweep(database, 30, ("ana", checkpoint_id_aged(40)), ("ana", None))
    )

    assert deleted == 1

    assert rows(database, "SELECT DISTINCT thread_id FROM checkpoints") == [(threads[1],)]
    assert rows(database, "SELECT DISTINCT thread_id FROM checkpoint_writes") == [(threads[1],)]


def test_an_old_thread_loses_its_conversation_row_and_a_new_one_keeps_it(
    database: Database,
) -> None:
    async def run() -> list[str]:
        async with database.serving():
            saver = await open_checkpointer(database, Fernet(KEY))
            threads = [
                await save_thread(saver, "ana", checkpoint_id_aged(40)),
                await save_thread(saver, "ana"),
            ]
            for thread in threads:
                await record_turn(database, "ana", UUID(thread.split("/")[1]), "hello")
            await delete_old_threads(database, saver, 30)
        return threads

    threads = asyncio.run(run())

    assert rows(database, "SELECT id::text FROM conversations") == [(threads[1].split("/")[1],)]


def test_a_younger_thread_survives(database: Database) -> None:
    _, deleted = asyncio.run(save_and_sweep(database, 30, ("ana", checkpoint_id_aged(10))))

    assert deleted == 0
    assert rows(database, "SELECT COUNT(*) FROM checkpoints") == [(1,)]


def start_server(
    days: int, token_settings: Settings, signing_key: SigningKeyLookup, database: Database
) -> None:
    settings = make_settings(token_settings, retention_days=days)
    with web(settings, [], script(), signing_key, database, key=KEY):
        pass


def test_the_server_deletes_old_threads_at_startup_and_zero_keeps_them(
    token_settings: Settings, token_signing_key: SigningKeyLookup, database: Database
) -> None:
    asyncio.run(save_and_sweep(database, 100_000, ("ana", checkpoint_id_aged(40))))

    start_server(0, token_settings, token_signing_key, database)
    assert rows(database, "SELECT COUNT(*) FROM checkpoints") == [(1,)]

    start_server(30, token_settings, token_signing_key, database)
    assert rows(database, "SELECT COUNT(*) FROM checkpoints") == [(0,)]
    assert rows(database, "SELECT COUNT(*) FROM checkpoint_writes") == [(0,)]


def test_a_failed_daily_sweep_is_logged_and_the_next_one_runs(
    database: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs = 0
    sleeps = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 2:
            raise asyncio.CancelledError

    async def failing_first(database: Database, checkpointer: object, days: int) -> int:
        nonlocal runs
        runs += 1
        if runs == 1:
            raise psycopg.OperationalError("connection is lost")
        return 0

    async def run() -> None:
        async with database.serving():
            saver = await open_checkpointer(database, Fernet(KEY))
            await delete_old_threads_daily(database, saver, 30)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(retention, "delete_old_threads", failing_first)
    caplog.set_level(logging.ERROR, logger="deskclerk.retention")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert runs == 2
    assert "connection is lost" in caplog.text
