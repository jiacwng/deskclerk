# A per-employee request quota with a moving one-minute window over a Postgres table, so every copy
# of the app counts the same requests: one employee looping on requests cannot spend the shared
# model budget. DESKCLERK_REQUESTS_PER_MINUTE sets the rate.

import time

from deskclerk.database import Database, lock_key

WINDOW_SECONDS = 60


class Quota:
    def __init__(self, database: Database, per_minute: int) -> None:
        self._database = database
        self._per_minute = per_minute
        database.create_tables(
            "CREATE TABLE IF NOT EXISTS quota_hits ("
            "user_id TEXT NOT NULL, hit_at DOUBLE PRECISION NOT NULL)",
            "CREATE INDEX IF NOT EXISTS quota_hits_user ON quota_hits (user_id, hit_at)",
        )

    async def allows(self, user_id: str) -> bool:
        now = time.time()
        async with (
            self._database.async_pool.connection() as connection,
            connection.transaction(),
        ):
            await connection.execute(
                "SELECT pg_advisory_xact_lock(%s)", (lock_key("quota/" + user_id),)
            )
            await connection.execute(
                "DELETE FROM quota_hits WHERE user_id = %s AND hit_at <= %s",
                (user_id, now - WINDOW_SECONDS),
            )
            cursor = await connection.execute(
                "SELECT COUNT(*) AS hits FROM quota_hits WHERE user_id = %s", (user_id,)
            )
            row = await cursor.fetchone()
            if row is None or row["hits"] >= self._per_minute:
                return False
            await connection.execute(
                "INSERT INTO quota_hits (user_id, hit_at) VALUES (%s, %s)", (user_id, now)
            )
        return True
