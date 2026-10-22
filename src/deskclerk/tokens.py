# An SQLite file holding each employee's OAuth tokens and pinned tool definitions for each MCP
# server, encrypted with Fernet.

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.shared.auth_utils import calculate_token_expiry
from pydantic import BaseModel

from deskclerk.servers import McpServer

Kind = Literal["tokens", "client", "tool_pins"]

KEY_CHECK_TEXT = b"deskclerk"


class TokenKeyError(Exception):
    pass


class _StoredTokens(BaseModel):
    token: OAuthToken
    expires_at: float | None


class TokenStore:
    def __init__(self, path: Path, key: str) -> None:
        try:
            self._fernet = Fernet(key)
        except ValueError:
            raise TokenKeyError("the token key is not a Fernet key") from None
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS secrets ("
                "user_id TEXT NOT NULL, server_name TEXT NOT NULL, kind TEXT NOT NULL, "
                "value BLOB NOT NULL, PRIMARY KEY (user_id, server_name, kind))"
            )
            database.execute("CREATE TABLE IF NOT EXISTS key_check (value BLOB NOT NULL)")
            row = database.execute("SELECT value FROM key_check").fetchone()
            if row is None:
                database.execute(
                    "INSERT INTO key_check (value) VALUES (?)",
                    (self._fernet.encrypt(KEY_CHECK_TEXT),),
                )
            else:
                self._decrypt(row[0])

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self._path)
        try:
            with database:
                yield database
        finally:
            database.close()

    def _decrypt(self, value: bytes) -> str:
        try:
            return self._fernet.decrypt(value).decode()
        except InvalidToken:
            raise TokenKeyError("the token file was not encrypted with this token key") from None

    def get(self, user_id: str, server_name: str, kind: Kind) -> str | None:
        with self._database() as database:
            row = database.execute(
                "SELECT value FROM secrets WHERE user_id = ? AND server_name = ? AND kind = ?",
                (user_id, server_name, kind),
            ).fetchone()
        if row is None:
            return None
        return self._decrypt(row[0])

    def put(self, user_id: str, server_name: str, kind: Kind, text: str) -> None:
        with self._database() as database:
            database.execute(
                "INSERT OR REPLACE INTO secrets (user_id, server_name, kind, value) "
                "VALUES (?, ?, ?, ?)",
                (user_id, server_name, kind, self._fernet.encrypt(text.encode())),
            )

    def delete_connection(self, user_id: str, server_name: str) -> None:
        with self._database() as database:
            database.execute(
                "DELETE FROM secrets WHERE user_id = ? AND server_name = ? "
                "AND kind IN ('tokens', 'tool_pins')",
                (user_id, server_name),
            )

    def storage_for(
        self, user_id: str, server: McpServer, redirect_url: str
    ) -> "ServerTokenStorage":
        return ServerTokenStorage(self, user_id, server, redirect_url)


class ServerTokenStorage:
    def __init__(
        self, store: TokenStore, user_id: str, server: McpServer, redirect_url: str
    ) -> None:
        self._store = store
        self._user_id = user_id
        self._server = server
        self._redirect_url = redirect_url

    def _stored_tokens(self) -> _StoredTokens | None:
        text = self._store.get(self._user_id, self._server.name, "tokens")
        if text is None:
            return None
        return _StoredTokens.model_validate_json(text)

    def token_expiry(self) -> float | None:
        stored = self._stored_tokens()
        return None if stored is None else stored.expires_at

    async def get_tokens(self) -> OAuthToken | None:
        stored = self._stored_tokens()
        return None if stored is None else stored.token

    async def set_tokens(self, tokens: OAuthToken) -> None:
        stored = _StoredTokens(token=tokens, expires_at=calculate_token_expiry(tokens.expires_in))
        self._store.put(self._user_id, self._server.name, "tokens", stored.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        if self._server.client_id is not None:
            method = "none" if self._server.client_secret is None else "client_secret_post"
            return OAuthClientInformationFull(
                client_id=self._server.client_id,
                client_secret=self._server.client_secret,
                redirect_uris=[self._redirect_url],
                token_endpoint_auth_method=method,
            )
        text = self._store.get(self._user_id, self._server.name, "client")
        if text is None:
            return None
        return OAuthClientInformationFull.model_validate_json(text)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        if self._server.client_id is not None:
            return
        self._store.put(self._user_id, self._server.name, "client", client_info.model_dump_json())
