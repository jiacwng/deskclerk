# Conversation retention: a thread whose newest checkpoint is older than the retention period is
# deleted with its row in the conversations table. A checkpoint id is a uuid6, so its age is read
# from the id without decrypting anything.

import asyncio
import logging
import time

import psycopg
from langgraph.checkpoint.base.id import UUID
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from deskclerk.database import Database

_log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400
UUID_EPOCH_TICKS = 0x01B21DD213814000
TICKS_PER_SECOND = 10_000_000


def checkpoint_unix_time(checkpoint_id: str) -> float:
    return (UUID(hex=checkpoint_id).time - UUID_EPOCH_TICKS) / TICKS_PER_SECOND


async def delete_old_threads(
    database: Database, checkpointer: AsyncPostgresSaver, retention_days: int
) -> int:
    cutoff = time.time() - retention_days * SECONDS_PER_DAY
    async with database.async_pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT thread_id, MAX(checkpoint_id) AS newest FROM checkpoints GROUP BY thread_id"
        )
        newest = await cursor.fetchall()
    old = [row["thread_id"] for row in newest if checkpoint_unix_time(row["newest"]) < cutoff]
    for thread in old:
        await checkpointer.adelete_thread(thread)
    if old:
        async with database.async_pool.connection() as connection:
            await connection.execute(
                "DELETE FROM conversations WHERE user_id || '/' || id::text = ANY(%s)", (old,)
            )
    return len(old)


async def delete_old_threads_daily(
    database: Database, checkpointer: AsyncPostgresSaver, retention_days: int
) -> None:
    while True:
        await asyncio.sleep(SECONDS_PER_DAY)
        try:
            await delete_old_threads(database, checkpointer, retention_days)
        except psycopg.Error:
            _log.exception("retention sweep failed")
