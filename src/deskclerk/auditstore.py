# The audit log kept in Postgres: every line the audit log prints is also one row, with a few
# columns pulled from the line for search and a hash chain over the lines. A row's hash is the
# HMAC-SHA256, under the chain key that the database never holds, of the previous row's hash
# followed by its line, so a changed or removed row shows when the chain is walked. The checkpoint
# row holds the hash that the oldest kept row chains from.

import hashlib
import hmac
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from psycopg.rows import dict_row

from deskclerk.config import Settings
from deskclerk.database import Database, lock_key

CHAIN_LOCK = lock_key("audit_chain")
CALL_LISTS = ("executed", "blocked", "pending", "decided")
COLUMNS = "id, time, user_id AS user, entry, servers, tools, decisions, hash, line"
INSERT = (
    "INSERT INTO audit_log (time, line, user_id, entry, servers, tools, decisions, hash) "
    "VALUES (%(time)s, %(line)s, %(user_id)s, %(entry)s, %(servers)s, %(tools)s, %(decisions)s, "
    "%(hash)s)"
)
NEWEST_HASH = (
    "SELECT COALESCE((SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1), "
    "(SELECT hash FROM audit_checkpoint)) AS hash"
)


class MissingChainKey(RuntimeError):
    pass


def chain_key_of(settings: Settings) -> bytes:
    if settings.audit_chain_key is None or not settings.audit_chain_key.get_secret_value():
        raise MissingChainKey("DESKCLERK_AUDIT_CHAIN_KEY is not set; put it in .env")
    return settings.audit_chain_key.get_secret_value().encode()


@dataclass(frozen=True)
class AuditRow:
    id: int
    time: datetime
    user: str
    entry: str
    servers: list[str]
    tools: list[str]
    decisions: list[str]
    hash: str
    line: str


@dataclass(frozen=True)
class AuditFilter:
    employee: str | None = None
    entry: str | None = None
    server: str | None = None
    tool: str | None = None
    decision: str | None = None
    since: datetime | None = None
    until: datetime | None = None


@dataclass(frozen=True)
class AuditPage:
    rows: list[AuditRow]
    next_cursor: int | None


@dataclass(frozen=True)
class ChainReport:
    rows: int
    broken_at: int | None


def chain_hash(key: bytes, previous: str, line: str) -> str:
    return hmac.new(key, (previous + line).encode(), hashlib.sha256).hexdigest()


def _columns(line: str) -> dict[str, object]:
    fields = json.loads(line)
    calls = [call for name in CALL_LISTS for call in fields.get(name, [])]
    servers = {call["server"] for call in calls}
    if "server" in fields:
        servers.add(fields["server"])
    decisions = {call["decision"] for call in fields.get("decided", [])}
    if "outcome" in fields:
        decisions.add(fields["outcome"])
    return {
        "time": datetime.fromisoformat(fields["time"]),
        "user_id": fields["user"],
        "entry": fields["entry"],
        "servers": sorted(servers),
        "tools": sorted({call["tool"] for call in calls}),
        "decisions": sorted(decisions),
    }


def _where(found: AuditFilter) -> tuple[str, list[object]]:
    clauses: list[tuple[object, str]] = [
        (found.employee, "user_id = %s"),
        (found.entry, "entry = %s"),
        (found.server, "%s = ANY(servers)"),
        (found.tool, "%s = ANY(tools)"),
        (found.decision, "%s = ANY(decisions)"),
        (found.since, "time >= %s"),
        (found.until, "time < %s"),
    ]
    conditions = []
    values: list[object] = []
    for value, condition in clauses:
        if value is not None:
            conditions.append(condition)
            values.append(value)
    return " AND ".join(conditions) or "TRUE", values


class AuditStore:
    def __init__(self, database: Database, chain_key: bytes) -> None:
        self._pool = database.pool
        self._chain_key = chain_key

    def append(self, line: str) -> None:
        with self._pool.connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (CHAIN_LOCK,))
            [newest] = connection.cursor(row_factory=dict_row).execute(NEWEST_HASH).fetchall()
            columns = _columns(line) | {
                "line": line,
                "hash": chain_hash(self._chain_key, newest["hash"], line),
            }
            connection.execute(INSERT, columns)

    def search(self, found: AuditFilter, limit: int, cursor: int | None) -> AuditPage:
        where, values = _where(found)
        if cursor is not None:
            where += " AND id < %s"
            values.append(cursor)
        query = f"SELECT {COLUMNS} FROM audit_log WHERE {where} ORDER BY id DESC LIMIT %s"  # noqa: S608
        with self._pool.connection() as connection:
            cursor_rows = connection.cursor(row_factory=dict_row)
            found_rows = cursor_rows.execute(query, [*values, limit + 1]).fetchall()
        rows = [AuditRow(**row) for row in found_rows[:limit]]
        has_more = len(found_rows) > limit
        return AuditPage(rows, rows[-1].id if has_more else None)

    def rows(self, found: AuditFilter) -> Iterator[AuditRow]:
        where, values = _where(found)
        query = f"SELECT {COLUMNS} FROM audit_log WHERE {where} ORDER BY id"  # noqa: S608
        with self._pool.connection() as connection:
            with connection.cursor(name="audit_rows", row_factory=dict_row) as cursor:
                cursor.execute(query, values)
                for row in cursor:
                    yield AuditRow(**row)

    def verify(self) -> ChainReport:
        checked = 0
        with self._pool.connection() as connection:
            start = connection.cursor(row_factory=dict_row)
            [checkpoint] = start.execute("SELECT hash FROM audit_checkpoint").fetchall()
            previous = checkpoint["hash"]
            with connection.cursor(name="audit_walk", row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT id, line, hash, time, user_id, entry, servers, tools, decisions "
                    "FROM audit_log ORDER BY id"
                )
                for row in cursor:
                    if chain_hash(self._chain_key, previous, row["line"]) != row["hash"]:
                        return ChainReport(checked, row["id"])
                    columns = _columns(row["line"])
                    if any(row[name] != value for name, value in columns.items()):
                        return ChainReport(checked, row["id"])
                    previous = row["hash"]
                    checked += 1
        return ChainReport(checked, None)

    def delete_before(self, cutoff: datetime) -> int:
        with self._pool.connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (CHAIN_LOCK,))
            newest_old = (
                connection.cursor(row_factory=dict_row)
                .execute(
                    "SELECT id, hash FROM audit_log WHERE time < %s ORDER BY id DESC LIMIT 1",
                    (cutoff,),
                )
                .fetchone()
            )
            if newest_old is None:
                return 0
            deleted = connection.execute(
                "DELETE FROM audit_log WHERE id <= %s", (newest_old["id"],)
            ).rowcount
            connection.execute("UPDATE audit_checkpoint SET hash = %s", (newest_old["hash"],))
            return deleted
