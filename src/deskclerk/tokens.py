# A Postgres table holding each employee's OAuth tokens, client registrations, pinned tool
# definitions and always allowed tools for each MCP server, encrypted with Fernet.

from typing import Literal

from cryptography.fernet import Fernet, InvalidToken
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.shared.auth_utils import calculate_token_expiry
from pydantic import BaseModel

from deskclerk.database import Database
from deskclerk.servers import McpServer

Kind = Literal["tokens", "client", "tool_pins", "always_allowed", "tool_menu"]

KEY_CHECK_TEXT = b"deskclerk"


class TokenKeyError(Exception):
    pass


class _StoredTokens(BaseModel):
    token: OAuthToken
    expires_at: float | None


class TokenStore:
    def __init__(self, database: Database, key: str) -> None:
        try:
            self.fernet = Fernet(key)
        except ValueError:
            raise TokenKeyError("the token key is not a Fernet key") from None
        self.database = database
        database.create_tables(
            "CREATE TABLE IF NOT EXISTS secrets ("
            "user_id TEXT NOT NULL, server_name TEXT NOT NULL, kind TEXT NOT NULL, "
            "value BYTEA NOT NULL, PRIMARY KEY (user_id, server_name, kind))",
            "CREATE TABLE IF NOT EXISTS key_check ("
            "only_row BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_row), value BYTEA NOT NULL)",
        )
        with database.pool.connection() as connection:
            connection.execute(
                "INSERT INTO key_check (value) VALUES (%s) ON CONFLICT DO NOTHING",
                (self.fernet.encrypt(KEY_CHECK_TEXT),),
            )
            row = connection.execute("SELECT value FROM key_check").fetchone()
        if row is not None:
            self._decrypt(row[0])

    def check(self) -> None:
        with self.database.pool.connection() as connection:
            connection.execute("SELECT value FROM key_check").fetchone()

    def _decrypt(self, value: bytes) -> str:
        try:
            return self.fernet.decrypt(bytes(value)).decode()
        except InvalidToken:
            raise TokenKeyError(
                "the stored tokens were not encrypted with this token key"
            ) from None

    def get(self, user_id: str, server_name: str, kind: Kind) -> str | None:
        with self.database.pool.connection() as connection:
            row = connection.execute(
                "SELECT value FROM secrets WHERE user_id = %s AND server_name = %s AND kind = %s",
                (user_id, server_name, kind),
            ).fetchone()
        if row is None:
            return None
        return self._decrypt(row[0])

    def connected_counts(self) -> dict[str, int]:
        with self.database.pool.connection() as connection:
            found = connection.execute(
                "SELECT server_name, COUNT(DISTINCT user_id) FROM secrets "
                "WHERE kind = 'tokens' GROUP BY server_name"
            ).fetchall()
        return {str(name): int(count) for name, count in found}

    def put(self, user_id: str, server_name: str, kind: Kind, text: str) -> None:
        with self.database.pool.connection() as connection:
            connection.execute(
                "INSERT INTO secrets (user_id, server_name, kind, value) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (user_id, server_name, kind) DO UPDATE SET value = EXCLUDED.value",
                (user_id, server_name, kind, self.fernet.encrypt(text.encode())),
            )

    def delete_tokens(self, user_id: str, server_name: str) -> None:
        with self.database.pool.connection() as connection:
            connection.execute(
                "DELETE FROM secrets WHERE user_id = %s AND server_name = %s AND kind = 'tokens'",
                (user_id, server_name),
            )

    def delete_connection(self, user_id: str, server_name: str) -> None:
        with self.database.pool.connection() as connection:
            connection.execute(
                "DELETE FROM secrets WHERE user_id = %s AND server_name = %s "
                "AND kind IN ('tokens', 'tool_pins', 'always_allowed', 'tool_menu')",
                (user_id, server_name),
            )

    def delete_user(self, user_id: str) -> int:
        with self.database.pool.connection() as connection:
            return connection.execute("DELETE FROM secrets WHERE user_id = %s", (user_id,)).rowcount

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
