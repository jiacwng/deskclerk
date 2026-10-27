# The one place that opens Postgres, where all state lives so several copies of deskclerk can run
# behind a load balancer: a pool for the stores that run in worker threads, a pool for the
# checkpointer and the async queries, and a pool whose connections hold one employee's ask lock.

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, LiteralString

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

CONNECT_TIMEOUT_SECONDS = 30
SYNC_POOL_SIZE = 10
ASYNC_POOL_SIZE = 10
LOCK_POOL_SIZE = 20
SCHEMA_LOCK_KEY = 0x6465736B
SCHEMA_LOCK_POLL_SECONDS = 0.05
CHECKPOINTER_CONNECTION: dict[str, Any] = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}


def lock_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.pool = ConnectionPool(url, min_size=1, max_size=SYNC_POOL_SIZE, open=False)
        self.pool.open(wait=True, timeout=CONNECT_TIMEOUT_SECONDS)

    def _make_async_pools(self) -> None:
        self.async_pool = AsyncConnectionPool[AsyncConnection[DictRow]](
            self.url,
            min_size=1,
            max_size=ASYNC_POOL_SIZE,
            kwargs=CHECKPOINTER_CONNECTION,
            open=False,
        )
        self.lock_pool = AsyncConnectionPool[AsyncConnection[Any]](
            self.url, min_size=1, max_size=LOCK_POOL_SIZE, kwargs={"autocommit": True}, open=False
        )

    def close(self) -> None:
        self.pool.close()

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        self._make_async_pools()
        await self.async_pool.open(wait=True, timeout=CONNECT_TIMEOUT_SECONDS)
        await self.lock_pool.open(wait=True, timeout=CONNECT_TIMEOUT_SECONDS)
        try:
            yield
        finally:
            await self.lock_pool.close()
            await self.async_pool.close()

    @asynccontextmanager
    async def user_lock(self, user_id: str) -> AsyncIterator[None]:
        async with self.lock_pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key(user_id),))
            yield

    @asynccontextmanager
    async def schema_lock(self) -> AsyncIterator[None]:
        async with await AsyncConnection.connect(self.url, autocommit=True) as connection:
            while True:
                cursor = await connection.execute(
                    "SELECT pg_try_advisory_lock(%s)", (SCHEMA_LOCK_KEY,)
                )
                row = await cursor.fetchone()
                if row is not None and row[0]:
                    break
                await asyncio.sleep(SCHEMA_LOCK_POLL_SECONDS)
            yield

    def create_tables(self, *statements: LiteralString) -> None:
        with self.pool.connection() as connection:
            while True:
                row = connection.execute(
                    "SELECT pg_try_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,)
                ).fetchone()
                if row is not None and row[0]:
                    break
                time.sleep(SCHEMA_LOCK_POLL_SECONDS)
            for statement in statements:
                connection.execute(statement)
