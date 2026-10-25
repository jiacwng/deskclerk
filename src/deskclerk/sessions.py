# An SQLite table of the browser sessions that are open. The cookie holds a random id, the table
# holds only its SHA-256, and deleting a row ends the session at once.

import hashlib
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from deskclerk.auth import Identity


def _digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_id TEXT NOT NULL, "
                "name TEXT NOT NULL, sid TEXT, subject TEXT, expires_at REAL NOT NULL)"
            )

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self.path)
        try:
            with database:
                yield database
        finally:
            database.close()

    def create(self, identity: Identity, lifetime_seconds: int) -> str:
        session_id = secrets.token_urlsafe(32)
        with self._database() as database:
            database.execute(
                "INSERT INTO sessions "
                "(id_hash, user_id, client_id, name, sid, subject, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _digest(session_id),
                    identity.user_id,
                    identity.client_id,
                    identity.display_name,
                    identity.sid,
                    identity.subject,
                    time.time() + lifetime_seconds,
                ),
            )
        return session_id

    def lookup(self, session_id: str) -> Identity | None:
        with self._database() as database:
            row = database.execute(
                "SELECT user_id, client_id, name, expires_at FROM sessions WHERE id_hash = ?",
                (_digest(session_id),),
            ).fetchone()
            if row is None:
                return None
            if row[3] < time.time():
                database.execute("DELETE FROM sessions WHERE id_hash = ?", (_digest(session_id),))
                return None
        return Identity(user_id=row[0], client_id=row[1], display_name=row[2])

    def end(self, session_id: str) -> None:
        with self._database() as database:
            database.execute("DELETE FROM sessions WHERE id_hash = ?", (_digest(session_id),))

    def end_by_sid(self, sid: str) -> None:
        with self._database() as database:
            database.execute("DELETE FROM sessions WHERE sid = ?", (sid,))

    def end_by_subject(self, subject: str) -> None:
        with self._database() as database:
            database.execute("DELETE FROM sessions WHERE subject = ?", (subject,))
