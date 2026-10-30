# A Postgres table of the browser sessions that are open. The cookie holds a random id, the table
# holds only its SHA-256, and deleting a row ends the session at once.

import hashlib
import json
import secrets
import time

from deskclerk.auth import Identity
from deskclerk.database import Database


def _digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()


class SessionStore:
    def __init__(self, database: Database) -> None:
        self._pool = database.pool

    def create(self, identity: Identity, lifetime_seconds: int) -> str:
        session_id = secrets.token_urlsafe(32)
        with self._pool.connection() as connection:
            connection.execute(
                "INSERT INTO sessions "
                "(id_hash, user_id, client_id, name, sid, subject, groups, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    _digest(session_id),
                    identity.user_id,
                    identity.client_id,
                    identity.display_name,
                    identity.sid,
                    identity.subject,
                    json.dumps(sorted(identity.groups)),
                    time.time() + lifetime_seconds,
                ),
            )
        return session_id

    def lookup(self, session_id: str) -> Identity | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT user_id, client_id, name, groups, expires_at "
                "FROM sessions WHERE id_hash = %s",
                (_digest(session_id),),
            ).fetchone()
            if row is None:
                return None
            if row[4] < time.time():
                connection.execute(
                    "DELETE FROM sessions WHERE id_hash = %s", (_digest(session_id),)
                )
                return None
        return Identity(
            user_id=row[0],
            client_id=row[1],
            display_name=row[2],
            groups=frozenset(json.loads(row[3])),
        )

    def end(self, session_id: str) -> None:
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE id_hash = %s", (_digest(session_id),))

    def delete_user(self, user_id: str) -> int:
        with self._pool.connection() as connection:
            return connection.execute(
                "DELETE FROM sessions WHERE user_id = %s", (user_id,)
            ).rowcount

    def end_by_sid(self, sid: str) -> None:
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE sid = %s", (sid,))

    def end_by_subject(self, subject: str) -> None:
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE subject = %s", (subject,))
