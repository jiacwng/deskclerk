# Retention deletes a conversation whose newest checkpoint is older than the retention period and
# keeps a newer one; zero days keeps everything.

import asyncio
import logging
import sqlite3
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from langgraph.checkpoint.base.id import UUID

from deskclerk import retention
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.conversations import open_checkpointer
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


async def sweep(path: Path, days: int) -> int:
    async with open_checkpointer(path, Fernet(KEY)) as saver:
        return await delete_old_threads(saver, days)


def test_an_old_thread_is_deleted_and_a_new_one_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "tokens.sqlite"
    asyncio.run(save_thread(path, "ana", checkpoint_id_aged(40)))
    new = asyncio.run(save_thread(path, "ana"))

    assert asyncio.run(sweep(path, 30)) == 1

    assert rows(path, "SELECT DISTINCT thread_id FROM checkpoints") == [(new,)]
    assert rows(path, "SELECT DISTINCT thread_id FROM writes") == [(new,)]


def test_a_younger_thread_survives(tmp_path: Path) -> None:
    path = tmp_path / "tokens.sqlite"
    asyncio.run(save_thread(path, "ana", checkpoint_id_aged(10)))

    assert asyncio.run(sweep(path, 30)) == 0
    assert rows(path, "SELECT COUNT(*) FROM checkpoints") == [(1,)]


def start_server(
    days: int, token_settings: Settings, signing_key: SigningKeyLookup, path: Path
) -> None:
    settings = make_settings(token_settings, retention_days=days)
    with web(settings, [], script(), signing_key, path, key=KEY):
        pass


def test_the_server_deletes_old_threads_at_startup_and_zero_keeps_them(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    path = tmp_path / "tokens.sqlite"
    asyncio.run(save_thread(path, "ana", checkpoint_id_aged(40)))

    start_server(0, token_settings, token_signing_key, tmp_path)
    assert rows(path, "SELECT COUNT(*) FROM checkpoints") == [(1,)]

    start_server(30, token_settings, token_signing_key, tmp_path)
    assert rows(path, "SELECT COUNT(*) FROM checkpoints") == [(0,)]
    assert rows(path, "SELECT COUNT(*) FROM writes") == [(0,)]


def test_a_failed_daily_sweep_is_logged_and_the_next_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runs = 0
    sleeps = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 2:
            raise asyncio.CancelledError

    async def failing_first(checkpointer: object, days: int) -> int:
        nonlocal runs
        runs += 1
        if runs == 1:
            raise sqlite3.OperationalError("database is locked")
        return 0

    async def run() -> None:
        async with open_checkpointer(tmp_path / "tokens.sqlite", Fernet(KEY)) as saver:
            await delete_old_threads_daily(saver, 30)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(retention, "delete_old_threads", failing_first)
    caplog.set_level(logging.ERROR, logger="deskclerk.retention")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert runs == 2
    assert "database is locked" in caplog.text
