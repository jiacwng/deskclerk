# The MCP servers an admin has set up, kept in a Postgres table, and the list of them that each
# copy of the app holds in memory. A write sends NOTIFY, every copy listens and reloads its list,
# so a change reaches all copies without a restart. Secrets stay references: the table holds the
# name of an environment variable, and the value is read when the list is loaded.

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from deskclerk.database import Database
from deskclerk.servers import McpServer, ServerEntry, ServersError, resolve_servers

CHANNEL = "servers_changed"
LISTEN_START_SECONDS = 10
RECONNECT_SECONDS = 1
INSERT = (
    "INSERT INTO servers (name, plain_http, url, client_id, client_secret_env, auth, tools, "
    "headers, client_cert, client_key, groups, always_allow, confirm_tools, trusted, title, "
    "description, examples, paused) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "%s, %s, %s, %s, %s)"
)
UPDATE = (
    "UPDATE servers SET plain_http = %s, url = %s, client_id = %s, client_secret_env = %s, "
    "auth = %s, tools = %s, headers = %s, client_cert = %s, client_key = %s, groups = %s, "
    "always_allow = %s, confirm_tools = %s, trusted = %s, title = %s, description = %s, "
    "examples = %s, paused = %s WHERE name = %s"
)
SELECT = (
    "SELECT name, plain_http, url, client_id, client_secret_env, auth, tools, headers, "
    "client_cert, client_key, groups, always_allow, confirm_tools, trusted, title, description, "
    "examples, paused FROM servers ORDER BY position"
)

_log = logging.getLogger(__name__)


class ServerExists(Exception):
    pass


class NoSuchServer(Exception):
    pass


def _row(entry: ServerEntry) -> list[object]:
    values = entry.model_dump()
    values["headers"] = Jsonb(values["headers"])
    return list(values.values())


class ServerStore:
    def __init__(self, database: Database) -> None:
        self._pool = database.pool

    def entries(self) -> list[ServerEntry]:
        with self._pool.connection() as connection:
            rows = connection.cursor(row_factory=dict_row).execute(SELECT).fetchall()
        return [ServerEntry.model_validate(row) for row in rows]

    def entry(self, name: str) -> ServerEntry:
        for entry in self.entries():
            if entry.name == name:
                return entry
        raise NoSuchServer(f"no server named {name}")

    def add(self, entry: ServerEntry) -> None:
        with self._pool.connection() as connection:
            try:
                connection.execute(INSERT, _row(entry))
            except psycopg.errors.UniqueViolation:
                raise ServerExists(f"a server named {entry.name} already exists") from None
            _announce(connection)

    def update(self, entry: ServerEntry) -> None:
        row = _row(entry)
        with self._pool.connection() as connection:
            changed = connection.execute(UPDATE, [*row[1:], entry.name]).rowcount
            if changed == 0:
                raise NoSuchServer(f"no server named {entry.name}")
            _announce(connection)

    def set_paused(self, name: str, paused: bool) -> None:
        with self._pool.connection() as connection:
            changed = connection.execute(
                "UPDATE servers SET paused = %s WHERE name = %s", (paused, name)
            ).rowcount
            if changed == 0:
                raise NoSuchServer(f"no server named {name}")
            _announce(connection)

    def remove(self, name: str) -> None:
        with self._pool.connection() as connection:
            removed = connection.execute("DELETE FROM servers WHERE name = %s", (name,)).rowcount
            if removed == 0:
                raise NoSuchServer(f"no server named {name}")
            _announce(connection)

    def replace_all(self, entries: list[ServerEntry]) -> None:
        with self._pool.connection() as connection:
            connection.execute("DELETE FROM servers")
            for entry in entries:
                connection.execute(INSERT, _row(entry))
            _announce(connection)


def _announce(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute("SELECT pg_notify(%s, '')", (CHANNEL,))


class ServerCatalog:
    def __init__(
        self, database: Database, store: ServerStore, environ: Mapping[str, str] = os.environ
    ) -> None:
        self._database = database
        self.store = store
        self._environ = environ
        self._listening = asyncio.Event()
        self.servers: tuple[McpServer, ...] = ()

    @property
    def offered(self) -> list[McpServer]:
        return [server for server in self.servers if not server.paused]

    def check(self, entry: ServerEntry) -> None:
        resolve_servers([entry], self._environ)

    def reload(self) -> None:
        self.servers = tuple(resolve_servers(self.store.entries(), self._environ))

    async def refresh(self) -> None:
        try:
            await asyncio.to_thread(self.reload)
        except ServersError as error:
            _log.error("the server list was not reloaded and the previous list stays: %s", error)

    @asynccontextmanager
    async def listening(self) -> AsyncIterator[None]:
        listener = asyncio.create_task(self._listen())
        try:
            await asyncio.wait_for(self._listening.wait(), LISTEN_START_SECONDS)
            yield
        finally:
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)

    async def _listen(self) -> None:
        while True:
            try:
                async with await AsyncConnection.connect(
                    self._database.url, autocommit=True
                ) as listener:
                    await listener.execute(f"LISTEN {CHANNEL}")
                    await self.refresh()
                    self._listening.set()
                    async for _ in listener.notifies():
                        await self.refresh()
            except psycopg.Error:
                _log.exception("the server list listener lost its database connection")
                self._listening.clear()
                await asyncio.sleep(RECONNECT_SECONDS)
