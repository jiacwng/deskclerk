# An employee connects an MCP server through OAuth against a real local server, then lists and
# calls its tools; another employee, a restart, a stale flow and a disconnect behave as stated.

import asyncio
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from cryptography.fernet import Fernet
from mcp.client.auth import OAuthFlowError
from mcp.shared.auth import OAuthToken

from deskclerk.connections import (
    UPSTREAM_ERRORS,
    AlreadyConnected,
    FlowExpired,
    NoSignInNeeded,
    NotConnected,
    OAuthConnections,
    UnknownState,
    WrongUser,
    first_leaf,
)
from deskclerk.servers import McpServer
from deskclerk.tokens import TokenStore
from mcp_server import PRESET_CLIENT_ID, PRESET_CLIENT_SECRET, RunningServer, running_mcp_server
from tls_files import write_tls_files

REDIRECT = "http://localhost:8000/oauth/callback"


@pytest.fixture
def mcp_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


@pytest.fixture
def token_path(tmp_path: Path) -> Path:
    return tmp_path / "tokens.sqlite"


@pytest.fixture
def key() -> str:
    return Fernet.generate_key().decode()


def make_connections(path: Path, key: str, timeout: float = 600) -> OAuthConnections:
    return OAuthConnections(TokenStore(path, key), REDIRECT, "deskclerk test", timeout)


def server_record(running: RunningServer, preregistered: bool = False) -> McpServer:
    if preregistered:
        return McpServer("notes", running.url, PRESET_CLIENT_ID, PRESET_CLIENT_SECRET)
    return McpServer("notes", running.url, None, None)


async def approve(authorization_url: str) -> tuple[str, str]:
    async with httpx.AsyncClient() as browser:
        response = await browser.get(authorization_url)
    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith(REDIRECT)
    query = parse_qs(urlparse(location).query)
    return query["state"][0], query["code"][0]


async def connect(connections: OAuthConnections, user_id: str, server: McpServer) -> None:
    authorization_url = await connections.begin(user_id, server)
    state, code = await approve(authorization_url)
    await connections.complete(user_id, state, code)


@pytest.mark.anyio
async def test_connect_then_list_and_call_tools_as_that_employee(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    authorization_url = await connections.begin("ada", server)
    assert authorization_url.startswith(mcp_server.url.removesuffix("/mcp") + "/authorize")
    state, code = await approve(authorization_url)
    await connections.complete("ada", state, code)

    tools = await connections.tools("ada", server)
    assert {tool.name for tool in tools} == {"read_note", "write_note", "break_note"}
    result = await connections.call("ada", server, "write_note", {"text": "hello"})
    assert (result.text, result.is_error) == ("wrote hello", False)

    stored = await TokenStore(token_path, key).storage_for("ada", server, REDIRECT).get_tokens()
    assert stored is not None
    assert mcp_server.notes == [(stored.access_token, "hello")]
    assert mcp_server.provider.registrations == 1


@pytest.mark.anyio
async def test_another_employee_is_not_connected(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    await connect(connections, "ada", server)
    await connections.tools("ada", server)
    with pytest.raises(NotConnected):
        await connections.tools("grace", server)
    with pytest.raises(NotConnected):
        await connections.call("grace", server, "write_note", {"text": "x"})
    assert mcp_server.notes == []


@pytest.mark.anyio
async def test_a_restart_keeps_the_connection_and_the_stored_expiry_triggers_a_refresh(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    server = server_record(mcp_server)
    await connect(make_connections(token_path, key), "ada", server)

    store = TokenStore(token_path, key)
    storage = store.storage_for("ada", server, REDIRECT)
    old = await storage.get_tokens()
    assert old is not None
    mcp_server.provider.access_tokens.pop(old.access_token)
    await storage.set_tokens(old.model_copy(update={"expires_in": -10}))

    restarted = OAuthConnections(TokenStore(token_path, key), REDIRECT, "deskclerk test", 600)
    assert {tool.name for tool in await restarted.tools("ada", server)} == {
        "read_note",
        "write_note",
        "break_note",
    }
    assert mcp_server.provider.refreshes == 1
    new = await storage.get_tokens()
    assert new is not None
    assert new.access_token != old.access_token


@pytest.mark.anyio
async def test_tokens_the_server_no_longer_accepts_never_start_a_redirect(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    await connect(connections, "ada", server)
    storage = TokenStore(token_path, key).storage_for("ada", server, REDIRECT)
    await storage.set_tokens(OAuthToken(access_token="revoked", token_type="Bearer"))
    with pytest.raises(NotConnected):
        await connections.tools("ada", server)


@pytest.mark.anyio
async def test_a_state_that_was_never_issued_or_was_already_used_is_refused(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    with pytest.raises(UnknownState):
        await connections.complete("ada", "never-issued", "code")
    state, code = await approve(await connections.begin("ada", server_record(mcp_server)))
    await connections.complete("ada", state, code)
    with pytest.raises(UnknownState):
        await connections.complete("ada", state, code)


@pytest.mark.anyio
async def test_a_flow_finished_after_the_bound_has_expired_leaves_no_task_running(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    before = asyncio.all_tasks()
    connections = make_connections(token_path, key, timeout=0.2)
    server = server_record(mcp_server)
    authorization_url = await connections.begin("ada", server)
    assert asyncio.all_tasks() - before
    state, code = await approve(authorization_url)
    await asyncio.sleep(0.6)
    with pytest.raises(FlowExpired):
        await connections.complete("ada", state, code)
    with pytest.raises(NotConnected):
        await connections.tools("ada", server)
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.anyio
async def test_connecting_again_while_connected_is_reported(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    await connect(connections, "ada", server)
    with pytest.raises(AlreadyConnected):
        await connections.begin("ada", server)


@pytest.mark.anyio
async def test_a_registered_client_skips_dynamic_registration(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server, preregistered=True)
    await connect(connections, "ada", server)
    assert {tool.name for tool in await connections.tools("ada", server)} == {
        "read_note",
        "write_note",
        "break_note",
    }
    assert mcp_server.provider.registrations == 0
    assert TokenStore(token_path, key).get("ada", "notes", "client") is None


@pytest.mark.anyio
async def test_each_employees_calls_carry_their_own_token(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    await connect(connections, "ada", server)
    await connect(connections, "grace", server)
    await connections.call("ada", server, "write_note", {"text": "from ada"})
    await connections.call("grace", server, "write_note", {"text": "from grace"})

    store = TokenStore(token_path, key)
    ada = await store.storage_for("ada", server, REDIRECT).get_tokens()
    grace = await store.storage_for("grace", server, REDIRECT).get_tokens()
    assert ada is not None
    assert grace is not None
    assert ada.access_token != grace.access_token
    assert mcp_server.notes == [
        (ada.access_token, "from ada"),
        (grace.access_token, "from grace"),
    ]


@pytest.mark.anyio
async def test_the_client_names_the_server_as_the_token_resource(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    await connect(connections, "ada", server_record(mcp_server))
    assert mcp_server.provider.resources == [mcp_server.url]


@pytest.mark.anyio
async def test_another_employee_cannot_finish_a_connection_they_did_not_start(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    state, code = await approve(await connections.begin("ada", server))
    with pytest.raises(WrongUser):
        await connections.complete("grace", state, code)
    store = TokenStore(token_path, key)
    for user in ("ada", "grace"):
        assert await store.storage_for(user, server, REDIRECT).get_tokens() is None
    await connections.complete("ada", state, code)
    assert await store.storage_for("ada", server, REDIRECT).get_tokens() is not None


@pytest.mark.anyio
async def test_a_server_with_no_sign_in_lists_its_tools_with_their_hints(
    token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    with running_mcp_server(REDIRECT, oauth=False) as open_server:
        server = McpServer("open", open_server.url, None, None, "none")
        tools = {tool.name: tool for tool in await connections.tools("ada", server)}
        with pytest.raises(NoSignInNeeded):
            await connections.begin("ada", server)
    assert (tools["read_note"].read_only, tools["read_note"].destructive) == (True, False)
    assert (tools["write_note"].read_only, tools["write_note"].destructive) == (False, True)


@pytest.mark.anyio
async def test_a_server_that_names_its_issuer_on_the_redirect_connects_when_it_is_forwarded(
    token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    with running_mcp_server(REDIRECT, issuer_in_response=True) as issuing:
        server = server_record(issuing)
        authorization_url = await connections.begin("ada", server)
        async with httpx.AsyncClient() as browser:
            response = await browser.get(authorization_url)
        query = parse_qs(urlparse(response.headers["location"]).query)
        assert query["iss"] == [issuing.url.removesuffix("/mcp")]
        await connections.complete("ada", query["state"][0], query["code"][0], query["iss"][0])
        tools = await connections.tools("ada", server)
    assert {tool.name for tool in tools} == {"read_note", "write_note", "break_note"}


@pytest.mark.anyio
async def test_a_wrong_issuer_fails_the_connection_as_an_upstream_error_not_a_group(
    token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    with running_mcp_server(REDIRECT, issuer_in_response=True) as issuing:
        authorization_url = await connections.begin("ada", server_record(issuing))
        state, code = await approve(authorization_url)
        with pytest.raises(OAuthFlowError):
            await connections.complete("ada", state, code, "https://elsewhere.example")


@pytest.mark.anyio
async def test_content_that_is_not_text_is_named_as_left_out(token_path: Path, key: str) -> None:
    connections = make_connections(token_path, key)
    with running_mcp_server(REDIRECT, oauth=False, image_tool=True) as open_server:
        server = McpServer("open", open_server.url, None, None, "none")
        result = await connections.call("ada", server, "show_chart", {})
    assert result.text == "sales are up\n[image/png content omitted]"


@pytest.mark.anyio
async def test_a_header_from_the_server_entry_reaches_the_server(
    token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    with running_mcp_server(REDIRECT, oauth=False, required_header=("X-Api-Key", "k1")) as guarded:
        with_header = McpServer(
            "guarded", guarded.url, None, None, "none", None, (("X-Api-Key", "k1"),)
        )
        without_header = McpServer("guarded", guarded.url, None, None, "none")
        tools = await connections.tools("ada", with_header)
        result = await connections.call("ada", with_header, "read_note", {})
        with pytest.raises(ExceptionGroup) as refused:
            await connections.tools("ada", without_header)
    assert isinstance(first_leaf(refused.value), UPSTREAM_ERRORS)
    assert "read_note" in {tool.name for tool in tools}
    assert result.text == "the note"


@pytest.mark.anyio
async def test_a_client_certificate_lets_a_call_through_mutual_tls(
    token_path: Path, key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = write_tls_files(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(files.ca))
    connections = make_connections(token_path, key)
    with running_mcp_server(REDIRECT, oauth=False, tls=files) as secured:
        with_cert = McpServer(
            "secured",
            secured.url,
            None,
            None,
            "none",
            None,
            (),
            str(files.client_cert),
            str(files.client_key),
        )
        without_cert = McpServer("secured", secured.url, None, None, "none")
        result = await connections.call("ada", with_cert, "read_note", {})
        with pytest.raises(ExceptionGroup) as refused:
            await connections.tools("ada", without_cert)
    assert isinstance(first_leaf(refused.value), UPSTREAM_ERRORS)
    assert result.text == "the note"


CLIENT_URL = "https://deskclerk.example.com/oauth/client.json"


def url_connections(path: Path, key: str) -> OAuthConnections:
    return OAuthConnections(TokenStore(path, key), REDIRECT, "deskclerk test", 600, CLIENT_URL)


@pytest.mark.anyio
async def test_a_server_that_supports_client_id_metadata_documents_gets_the_url_and_no_registration(
    token_path: Path, key: str
) -> None:
    with running_mcp_server(REDIRECT, client_id_metadata=True) as running:
        connections = url_connections(token_path, key)
        server = server_record(running)
        authorization_url = await connections.begin("ada", server)
        assert parse_qs(urlparse(authorization_url).query)["client_id"] == [CLIENT_URL]
        state, code = await approve(authorization_url)
        await connections.complete("ada", state, code)
        assert {tool.name for tool in await connections.tools("ada", server)}
        assert running.provider.registrations == 0


@pytest.mark.anyio
async def test_a_server_without_support_still_registers_dynamically(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = url_connections(token_path, key)
    authorization_url = await connections.begin("ada", server_record(mcp_server))
    assert parse_qs(urlparse(authorization_url).query)["client_id"] != [CLIENT_URL]
    assert mcp_server.provider.registrations == 1


@pytest.mark.anyio
async def test_a_registered_client_wins_over_the_metadata_url(token_path: Path, key: str) -> None:
    with running_mcp_server(REDIRECT, client_id_metadata=True) as running:
        connections = url_connections(token_path, key)
        authorization_url = await connections.begin(
            "ada", server_record(running, preregistered=True)
        )
        assert parse_qs(urlparse(authorization_url).query)["client_id"] == [PRESET_CLIENT_ID]


@pytest.mark.anyio
async def test_cancelling_all_flows_of_an_employee_leaves_the_others_waiting(
    mcp_server: RunningServer, token_path: Path, key: str
) -> None:
    connections = make_connections(token_path, key)
    server = server_record(mcp_server)
    ada_state, ada_code = await approve(await connections.begin("ada", server))
    bob_state, bob_code = await approve(await connections.begin("bob", server))

    connections.cancel_all_for("ada")

    with pytest.raises(UnknownState):
        await connections.complete("ada", ada_state, ada_code)
    assert await connections.complete("bob", bob_state, bob_code) == server.name
