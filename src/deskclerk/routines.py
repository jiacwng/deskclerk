# Scheduled requests. A routine is a request an employee saved with weekdays (Monday 0 to Sunday 6)
# and a time of day in their time zone. RoutineStore keeps them in Postgres. The Scheduler of every
# running copy claims one due routine at a time with FOR UPDATE SKIP LOCKED, moves it to its next
# occurrence in the same transaction, and only then runs it, so each occurrence runs once.

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, time, timedelta
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from starlette.concurrency import run_in_threadpool

from deskclerk.database import Database
from deskclerk.tokens import TokenStore

POLL_SECONDS = 30
WEEK_DAYS = 7

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Routine:
    id: UUID
    user_id: str
    request: str
    days: tuple[int, ...]
    time_of_day: time
    time_zone: str
    next_run_at: datetime
    paused: bool
    last_run_at: datetime | None
    last_outcome: str | None
    last_conversation: UUID | None


def next_occurrence(
    days: tuple[int, ...], time_of_day: time, zone: ZoneInfo, after: datetime
) -> datetime:
    local_after = after.astimezone(zone)
    for offset in range(WEEK_DAYS + 1):
        day = local_after.date() + timedelta(days=offset)
        if day.weekday() not in days:
            continue
        candidate = datetime.combine(day, time_of_day, tzinfo=zone).astimezone(UTC)
        if candidate > after:
            return candidate
    raise ValueError("a routine needs at least one weekday")


def _routine(row: tuple[Any, ...]) -> Routine:
    return Routine(row[0], row[1], row[2], tuple(row[3]), *row[4:])


def _following(routine: Routine, after: datetime) -> datetime:
    return next_occurrence(routine.days, routine.time_of_day, ZoneInfo(routine.time_zone), after)


class RoutineStore:
    def __init__(self, database: Database, tokens: TokenStore) -> None:
        self._pool = database.pool
        self._tokens = tokens

    def add(
        self, user_id: str, request: str, days: tuple[int, ...], at: time, zone: str, now: datetime
    ) -> Routine:
        routine_id = uuid4()
        due = next_occurrence(days, at, ZoneInfo(zone), now)
        with self._pool.connection() as connection:
            connection.execute(
                "INSERT INTO routines VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, NULL, NULL, NULL)",
                (routine_id, user_id, request, list(days), at, zone, due),
            )
        return Routine(routine_id, user_id, request, days, at, zone, due, False, None, None, None)

    def of_user(self, user_id: str) -> list[Routine]:
        with self._pool.connection() as connection:
            found = connection.execute(
                "SELECT * FROM routines WHERE user_id = %s ORDER BY next_run_at, id",
                (user_id,),
            ).fetchall()
        return [_routine(row) for row in found]

    def pause(self, user_id: str, routine_id: UUID) -> bool:
        with self._pool.connection() as connection:
            return (
                connection.execute(
                    "UPDATE routines SET paused = TRUE WHERE user_id = %s AND id = %s",
                    (user_id, routine_id),
                ).rowcount
                == 1
            )

    def resume(self, user_id: str, routine_id: UUID, now: datetime) -> bool:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT * FROM routines WHERE user_id = %s AND id = %s FOR UPDATE",
                (user_id, routine_id),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE routines SET paused = FALSE, next_run_at = %s WHERE id = %s",
                (_following(_routine(row), now), routine_id),
            )
        return True

    def delete(self, user_id: str, routine_id: UUID) -> bool:
        with self._pool.connection() as connection:
            return (
                connection.execute(
                    "DELETE FROM routines WHERE user_id = %s AND id = %s", (user_id, routine_id)
                ).rowcount
                == 1
            )

    def delete_user(self, user_id: str) -> int:
        self._tokens.delete_refresh(user_id)
        with self._pool.connection() as connection:
            return connection.execute(
                "DELETE FROM routines WHERE user_id = %s", (user_id,)
            ).rowcount

    def pause_user(self, user_id: str, outcome: str) -> int:
        with self._pool.connection() as connection:
            return connection.execute(
                "UPDATE routines SET paused = TRUE, last_outcome = %s WHERE user_id = %s",
                (outcome, user_id),
            ).rowcount

    def claim_due(self, now: datetime) -> Routine | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT * FROM routines WHERE NOT paused AND next_run_at <= %s "
                "ORDER BY next_run_at LIMIT 1 FOR UPDATE SKIP LOCKED",
                (now,),
            ).fetchone()
            if row is None:
                return None
            claimed = _routine(row)
            due = _following(claimed, now)
            connection.execute(
                "UPDATE routines SET next_run_at = %s WHERE id = %s", (due, claimed.id)
            )
        return replace(claimed, next_run_at=due)

    def record(
        self, routine_id: UUID, outcome: str, conversation: UUID | None, ran_at: datetime
    ) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                "UPDATE routines SET last_run_at = %s, last_outcome = %s, "
                "last_conversation = COALESCE(%s, last_conversation) WHERE id = %s",
                (ran_at, outcome, conversation, routine_id),
            )


class Scheduler:
    def __init__(self, store: RoutineStore, run: Callable[[Routine], Awaitable[None]]) -> None:
        self._store = store
        self._run = run

    async def run_due(self) -> None:
        while (
            routine := await run_in_threadpool(self._store.claim_due, datetime.now(UTC))
        ) is not None:
            try:
                await self._run(routine)
            except Exception:
                _log.exception("routine %s failed", routine.id)

    async def _poll(self) -> None:
        while True:
            try:
                await self.run_due()
            except Exception:
                _log.exception("the routine scheduler could not read the routines table")
            await asyncio.sleep(POLL_SECONDS)

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        polling = asyncio.create_task(self._poll())
        try:
            yield
        finally:
            polling.cancel()
            await asyncio.gather(polling, return_exceptions=True)
