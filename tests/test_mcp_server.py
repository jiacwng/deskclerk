# The MCP server over Streamable HTTP, driven in process through an ASGI transport, against the
# real database and embedding model with tokens signed by the throwaway key from conftest.py.

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import psycopg
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from psycopg.rows import TupleRow
from starlette.applications import Starlette

from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings, load_settings
from needtoknow.documents import Document
from needtoknow.mcp_server import create_server
from needtoknow.retrieval import MAX_RESULTS

Sign = Callable[[dict[str, object]], str]

URL = "http://127.0.0.1:8001/mcp"
QUERY = "Who is helping with the Wharton and Rice projects for the Tiger Team?"
TIGER_TEAM = "enron-770963d7e2ceeb8d22d92a1ebfaba325068dfa242524cb4a8983dc0d1165f2ab"
ACCOUNTS_IN_DISTRICT_1 = "SELECT COUNT(*) FROM financial.account WHERE district_id = 1"
WRITE = "UPDATE financial.account SET frequency = 'x'"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def asgi_app(token_settings: Settings, token_signing_key: SigningKeyLookup) -> Starlette:
    return create_server(token_settings, token_signing_key).streamable_http_app()


@asynccontextmanager
async def _connect(app: Starlette, token: str) -> AsyncIterator[Client]:
    headers = {"Authorization": f"Bearer {token}"}
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), headers=headers) as http,
        Client(streamable_http_client(URL, http_client=http)) as client,
    ):
        yield client


async def _post(app: Starlette, headers: dict[str, str]) -> httpx2.Response:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as http:
        return await http.post(URL, json=message, headers=headers)


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-header"),
        pytest.param({"Authorization": "Basic anVsaWU6bmVlZHRva25vdw=="}, id="basic-scheme"),
        pytest.param({"Authorization": "Bearer"}, id="empty-bearer"),
    ],
)
async def test_a_request_without_a_token_is_refused(
    asgi_app: Starlette, headers: dict[str, str]
) -> None:
    response = await _post(asgi_app, headers)
    assert response.status_code == 401


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param({"exp": 1}, id="expired"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, id="wrong-issuer"),
        pytest.param({"typ": "ID"}, id="id-token"),
    ],
)
async def test_a_request_with_a_bad_token_is_refused(
    asgi_app: Starlette, sign_token: Sign, claims: dict[str, object]
) -> None:
    response = await _post(asgi_app, {"Authorization": f"Bearer {sign_token(claims)}"})
    assert response.status_code == 401


async def test_a_request_with_garbage_is_refused(asgi_app: Starlette) -> None:
    response = await _post(asgi_app, {"Authorization": "Bearer not-a-token"})
    assert response.status_code == 401


@pytest.mark.parametrize("user", ["kean-s", "kaminski-v"])
async def test_search_returns_only_documents_the_token_user_can_read(
    owner: psycopg.Connection[TupleRow],
    asgi_app: Starlette,
    sign_token: Sign,
    emails: dict[str, Document],
    user: str,
) -> None:
    token = sign_token({"preferred_username": user, "groups": []})

    async with _connect(asgi_app, token) as client:
        result = await client.call_tool("search_documents", {"query": QUERY, "k": MAX_RESULTS})

    assert not result.is_error
    assert result.structured_content is not None
    hits: list[dict[str, Any]] = result.structured_content["result"]
    found = {hit["doc_id"] for hit in hits}
    for doc_id in found:
        assert doc_id not in emails or f"user:{user}" in emails[doc_id].readers
    assert (TIGER_TEAM in found) == (user == "kaminski-v")
    for hit in hits:
        assert set(hit) == {"doc_id", "title", "text", "distance"}


@pytest.mark.parametrize("k", [0, MAX_RESULTS + 1])
async def test_k_out_of_range_is_a_tool_error(
    owner: psycopg.Connection[TupleRow], asgi_app: Starlette, sign_token: Sign, k: int
) -> None:
    async with _connect(asgi_app, sign_token({})) as client:
        result = await client.call_tool("search_documents", {"query": QUERY, "k": k})

    assert result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    assert content.text.endswith(f"k must be between 1 and {MAX_RESULTS}, got {k}")


def test_the_server_listens_on_localhost_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEEDTOKNOW_MCP_HOST", raising=False)
    assert load_settings().mcp_host == "127.0.0.1"

    monkeypatch.setenv("NEEDTOKNOW_MCP_HOST", "10.1.2.3")
    assert load_settings().mcp_host == "10.1.2.3"


async def test_the_server_offers_the_document_and_the_data_tools(
    asgi_app: Starlette, sign_token: Sign
) -> None:
    async with _connect(asgi_app, sign_token({})) as client:
        listing = await client.list_tools()

    names = {tool.name for tool in listing.tools}
    assert names == {"search_documents", "describe_data", "query_data"}


async def test_describe_data_names_the_bird_schemas(
    owner: psycopg.Connection[TupleRow], asgi_app: Starlette, sign_token: Sign
) -> None:
    async with _connect(asgi_app, sign_token({})) as client:
        result = await client.call_tool("describe_data", {})

    assert not result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    assert "financial" in content.text


@pytest.mark.parametrize(
    ("user", "group", "count"),
    [
        pytest.param("district-1", "district:1", "5", id="the-district-sees-its-rows"),
        pytest.param("county-sierra", "county:Sierra", "0", id="another-user-sees-none"),
    ],
)
async def test_query_data_returns_only_the_rows_the_token_user_can_read(
    owner: psycopg.Connection[TupleRow],
    asgi_app: Starlette,
    sign_token: Sign,
    user: str,
    group: str,
    count: str,
) -> None:
    token = sign_token({"preferred_username": user, "groups": [group]})

    async with _connect(asgi_app, token) as client:
        result = await client.call_tool("query_data", {"sql": ACCOUNTS_IN_DISTRICT_1})

    assert not result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    assert content.text.splitlines()[1] == count


async def test_a_write_through_query_data_is_a_tool_error(
    owner: psycopg.Connection[TupleRow], asgi_app: Starlette, sign_token: Sign
) -> None:
    token = sign_token({"preferred_username": "district-1", "groups": ["district:1"]})

    async with _connect(asgi_app, token) as client:
        result = await client.call_tool("query_data", {"sql": WRITE})

    assert result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    assert "The query failed" in content.text
    assert owner.execute("SELECT DISTINCT frequency FROM financial.account").fetchall() != [("x",)]
