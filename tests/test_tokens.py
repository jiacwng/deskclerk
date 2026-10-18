# The token file keeps tokens encrypted, keyed by employee and server, readable only with the key.

from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from needtoknow.servers import McpServer
from needtoknow.tokens import TokenKeyError, TokenStore

REDIRECT = "http://localhost:8000/oauth/callback"
SERVER = McpServer("notes", "https://notes.test/mcp", None, None)
ACCESS = "access-token-value-1234567890"


@pytest.fixture
def key() -> str:
    return Fernet.generate_key().decode()


@pytest.mark.anyio
async def test_tokens_are_not_readable_in_the_file(tmp_path: Path, key: str) -> None:
    path = tmp_path / "tokens.sqlite"
    storage = TokenStore(path, key).storage_for("ada", SERVER, REDIRECT)
    await storage.set_tokens(OAuthToken(access_token=ACCESS, refresh_token="refresh-value-9876"))
    raw = path.read_bytes()
    assert ACCESS.encode() not in raw
    assert b"refresh-value-9876" not in raw
    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == ACCESS


@pytest.mark.anyio
async def test_a_new_store_on_the_same_file_and_key_reads_the_tokens(
    tmp_path: Path, key: str
) -> None:
    path = tmp_path / "tokens.sqlite"
    first = TokenStore(path, key).storage_for("ada", SERVER, REDIRECT)
    await first.set_tokens(OAuthToken(access_token=ACCESS, expires_in=3600))
    second = TokenStore(path, key).storage_for("ada", SERVER, REDIRECT)
    stored = await second.get_tokens()
    assert stored is not None
    assert stored.access_token == ACCESS
    assert second.token_expiry() is not None


def test_another_key_cannot_open_the_file(tmp_path: Path, key: str) -> None:
    path = tmp_path / "tokens.sqlite"
    TokenStore(path, key)
    with pytest.raises(TokenKeyError, match="not encrypted with this token key"):
        TokenStore(path, Fernet.generate_key().decode())


def test_a_key_that_is_not_a_fernet_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TokenKeyError, match="not a Fernet key"):
        TokenStore(tmp_path / "tokens.sqlite", "short")


@pytest.mark.anyio
async def test_each_employee_and_server_has_their_own_tokens(tmp_path: Path, key: str) -> None:
    store = TokenStore(tmp_path / "tokens.sqlite", key)
    other = McpServer("other", "https://other.test/mcp", None, None)
    await store.storage_for("ada", SERVER, REDIRECT).set_tokens(OAuthToken(access_token=ACCESS))
    assert await store.storage_for("grace", SERVER, REDIRECT).get_tokens() is None
    assert await store.storage_for("ada", other, REDIRECT).get_tokens() is None


@pytest.mark.anyio
async def test_deleting_tokens_removes_only_that_employees_tokens(tmp_path: Path, key: str) -> None:
    store = TokenStore(tmp_path / "tokens.sqlite", key)
    for user in ("ada", "grace"):
        await store.storage_for(user, SERVER, REDIRECT).set_tokens(OAuthToken(access_token=user))
    store.delete_tokens("ada", SERVER.name)
    assert await store.storage_for("ada", SERVER, REDIRECT).get_tokens() is None
    assert await store.storage_for("grace", SERVER, REDIRECT).get_tokens() is not None


@pytest.mark.anyio
async def test_a_registered_client_is_stored_for_one_employee(tmp_path: Path, key: str) -> None:
    store = TokenStore(tmp_path / "tokens.sqlite", key)
    info = OAuthClientInformationFull(client_id="dynamic", redirect_uris=[REDIRECT])
    await store.storage_for("ada", SERVER, REDIRECT).set_client_info(info)
    stored = await store.storage_for("ada", SERVER, REDIRECT).get_client_info()
    assert stored is not None
    assert stored.client_id == "dynamic"
    assert await store.storage_for("grace", SERVER, REDIRECT).get_client_info() is None


@pytest.mark.anyio
async def test_an_admin_registered_client_is_returned_and_never_stored(
    tmp_path: Path, key: str
) -> None:
    store = TokenStore(tmp_path / "tokens.sqlite", key)
    server = McpServer("notes", "https://notes.test/mcp", "preset", "preset-secret")
    storage = store.storage_for("ada", server, REDIRECT)
    await storage.set_client_info(OAuthClientInformationFull(client_id="other"))
    info = await storage.get_client_info()
    assert info is not None
    assert (info.client_id, info.client_secret) == ("preset", "preset-secret")
    assert info.token_endpoint_auth_method == "client_secret_post"
    assert store.get("ada", "notes", "client") is None
