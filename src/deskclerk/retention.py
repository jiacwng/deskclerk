# Conversation retention: a thread whose newest checkpoint is older than the retention period is
# deleted. A checkpoint id is a uuid6, so its age is read from the id without decrypting anything.

import asyncio
import time

from langgraph.checkpoint.base.id import UUID
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

SECONDS_PER_DAY = 86400
UUID_EPOCH_TICKS = 0x01B21DD213814000
TICKS_PER_SECOND = 10_000_000


def checkpoint_unix_time(checkpoint_id: str) -> float:
    return (UUID(hex=checkpoint_id).time - UUID_EPOCH_TICKS) / TICKS_PER_SECOND


async def delete_old_threads(checkpointer: AsyncSqliteSaver, retention_days: int) -> int:
    await checkpointer.setup()
    cutoff = time.time() - retention_days * SECONDS_PER_DAY
    async with (
        checkpointer.lock,
        checkpointer.conn.execute(
            "SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id"
        ) as cursor,
    ):
        newest = await cursor.fetchall()
    old = [thread for thread, last in newest if checkpoint_unix_time(last) < cutoff]
    for thread in old:
        await checkpointer.adelete_thread(thread)
    return len(old)


async def delete_old_threads_daily(checkpointer: AsyncSqliteSaver, retention_days: int) -> None:
    while True:
        await asyncio.sleep(SECONDS_PER_DAY)
        await delete_old_threads(checkpointer, retention_days)
