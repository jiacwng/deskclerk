# The MCP gateway at /mcp against the real app on a localhost port and the real test MCP server,
# driven by the SDK's own client over streamable HTTP: sign-in discovery, which tools are listed,
# screened reads, writes that wait for the employee's answer through elicitation, and the audit.

import logging
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

import httpx
import httpx2
import pytest
import uvicorn
from cryptography.fernet import Fernet
from mcp.client import Client
from mcp.client.session import ElicitationFnT
from mcp.client.streamable_http import streamable_http_client
from mcp_types import CallToolResult, ElicitRequestParams, ElicitResult, TextContent
from prometheus_client import CollectorRegistry

from deskclerk.api import create_app
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.database import Database
from deskclerk.jev import PRICE_PER_INPUT_TOKEN, Jev
from deskclerk.providers import Usage
from deskclerk.screen import NOTICE
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore, month_of
from deskclerk.tokens import TokenStore
from deskclerk.writecheck import write_check
from mcp_server import RunningServer
from test_agent import answer, script
from test_api import (
    Sign,
    audit_lines,
    make_settings,
    notes,
    notes_server,  # noqa: F401
    open_notes,
    open_server,  # noqa: F401
    stored_catalog,
)
from test_connections import REDIRECT
from test_screen import FakeJev, as_jev

TOKEN_KEY = Fernet.generate_key().decode()


@dataclass
class Site:
    base: str
    tokens: TokenStore
    registry: CollectorRegistry


@contextmanager
def serving(
    settings: Settings,
    servers: list[Any],
    signing_key: SigningKeyLookup,
    database: Database,
    jev: Jev | None = None,
) -> Iterator[Site]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    tokens = TokenStore(database, TOKEN_KEY)
    registry = CollectorRegistry()
    app = create_app(
        settings.model_copy(update={"public_url": base}),
        database,
        stored_catalog(database, servers),
        OAuthConnections(tokens, REDIRECT, "deskclerk test", 600),
        tokens,
        SpendingStore(database),
        SessionStore(database),
        script(answer("unused")),
        write_check(),
        jev,
        signing_key,
        registry,
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield Site(base, tokens, registry)
    finally:
        server.should_exit = True
        thread.join()
        sock.close()


def token_of(sign_token: Sign, user: str, group: str = "staff") -> str:
    return sign_token({"sub": user, "preferred_username": user, "groups": [group]})


def connect(site: Site, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    started = httpx.post(f"{site.base}/servers/notes/connect", headers=headers)
    approval = httpx.get(started.json()["authorization_url"])
    query = parse_qs(urlparse(approval.headers["location"]).query)
    params: dict[str, str] = {
        name: query[name][0] for name in ("state", "code", "iss") if name in query
    }
    callback = httpx.get(f"{site.base}/oauth/callback", params=params, headers=headers)
    assert callback.status_code == 303


Answer = ElicitationFnT


def answering(
    asked: list[str], action: Literal["accept", "decline"] = "accept", approve: bool = True
) -> Answer:
    async def callback(context: Any, params: ElicitRequestParams) -> ElicitResult:
        asked.append(params.message)
        content = {"approve": approve} if action == "accept" else None
        return ElicitResult(action=action, content=content)

    return callback


@asynccontextmanager
async def mcp_client(
    site: Site, token: str, elicitation: Answer | None = None
) -> AsyncIterator[Client]:
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http:
        transport = streamable_http_client(f"{site.base}/mcp", http_client=http)
        async with Client(transport, mode="legacy", elicitation_callback=elicitation) as client:
            yield client


def text_of(result: CallToolResult) -> str:
    [block] = result.content
    assert isinstance(block, TextContent)
    return block.text


def test_a_caller_without_a_token_gets_401_and_the_address_of_the_resource_metadata(
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    with serving(settings, [notes(notes_server)], token_signing_key, database) as site:
        refused = httpx.post(f"{site.base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        metadata_url = f"{site.base}/.well-known/oauth-protected-resource/mcp"
        assert refused.status_code == 401
        assert f'resource_metadata="{metadata_url}"' in refused.headers["www-authenticate"]

        metadata = httpx.get(metadata_url).json()
        assert metadata["resource"].rstrip("/") == f"{site.base}/mcp"
        assert [url.rstrip("/") for url in metadata["authorization_servers"]] == [settings.issuer]

        spoiled = httpx.post(
            f"{site.base}/mcp",
            headers={"Authorization": "Bearer not-a-token"},
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        assert spoiled.status_code == 401


@pytest.mark.anyio
async def test_the_list_holds_only_the_allowed_tools_of_connected_servers_open_to_the_group(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    servers = [
        replace(notes(notes_server), tools=("read_note", "write_note"), groups=("staff",)),
        replace(open_notes(open_server), paused=True),
    ]
    ada, grace = token_of(sign_token, "ada"), token_of(sign_token, "grace")
    eve = token_of(sign_token, "eve", "others")
    with serving(settings, servers, token_signing_key, database) as site:
        connect(site, ada)
        async with mcp_client(site, ada) as client:
            listed = (await client.list_tools()).tools
        assert sorted(tool.name for tool in listed) == ["notes__read_note", "notes__write_note"]
        hints = {tool.name: tool.annotations for tool in listed}
        assert hints["notes__read_note"] is not None
        assert hints["notes__read_note"].read_only_hint is True
        assert hints["notes__write_note"] is not None
        assert hints["notes__write_note"].read_only_hint is False

        async with mcp_client(site, grace) as client:
            assert (await client.list_tools()).tools == []
            missed = await client.call_tool("notes__read_note", {})
        assert missed.is_error
        assert "notes" in text_of(missed)
        assert site.base in text_of(missed)

        async with mcp_client(site, eve) as client:
            assert (await client.list_tools()).tools == []


@pytest.mark.anyio
async def test_a_read_answers_with_the_screened_result(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = token_of(sign_token, "ada")
    with serving(settings, [notes(notes_server)], token_signing_key, database) as site:
        connect(site, ada)
        async with mcp_client(site, ada) as client:
            assert text_of(await client.call_tool("notes__read_note", {})) == "the note"
    with serving(
        settings, [notes(notes_server)], token_signing_key, database, as_jev(FakeJev(0.9))
    ) as site:
        async with mcp_client(site, ada) as client:
            flagged = text_of(await client.call_tool("notes__read_note", {}))
    assert flagged == f"{NOTICE}\nthe note"


@pytest.mark.anyio
async def test_a_write_waits_for_the_answer_and_runs_on_approve_only(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = token_of(sign_token, "ada")
    with serving(settings, [notes(notes_server)], token_signing_key, database) as site:
        connect(site, ada)
        asked: list[str] = []
        async with mcp_client(site, ada, answering(asked, approve=False)) as client:
            unticked = await client.call_tool("notes__write_note", {"text": "no"})
        async with mcp_client(site, ada, answering(asked, action="decline")) as client:
            declined = await client.call_tool("notes__write_note", {"text": "no"})
        assert notes_server.notes == []
        assert unticked.is_error and declined.is_error
        assert "declined" in text_of(declined)

        async with mcp_client(site, ada, answering(asked)) as client:
            approved = await client.call_tool("notes__write_note", {"text": "yes"})
        assert not approved.is_error
        assert text_of(approved) == "wrote yes"
        assert [text for _, text in notes_server.notes] == ["yes"]
        assert all("notes" in message and "write_note" in message for message in asked)
        assert '"text": "yes"' in asked[-1]


@pytest.mark.anyio
async def test_a_write_from_a_client_that_cannot_ask_is_refused(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = token_of(sign_token, "ada")
    with serving(settings, [notes(notes_server)], token_signing_key, database) as site:
        connect(site, ada)
        async with mcp_client(site, ada) as client:
            refused = await client.call_tool("notes__write_note", {"text": "x"})
            read = await client.call_tool("notes__read_note", {})
    assert refused.is_error
    assert "elicitation" in text_of(refused)
    assert notes_server.notes == []
    assert text_of(read) == "the note"


@pytest.mark.anyio
async def test_each_call_leaves_one_audit_line_and_the_counters_move(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, audit_text=True)
    ada = token_of(sign_token, "ada")
    with serving(settings, [notes(notes_server)], token_signing_key, database) as site:
        connect(site, ada)
        async with mcp_client(site, ada, answering([])) as client:
            await client.call_tool("notes__read_note", {})
            await client.call_tool("notes__write_note", {"text": "yes"})
            await client.call_tool("notes__write_note", {})
            await client.call_tool("notes__nothing", {})
        lines = [line for line in audit_lines(caplog) if line["entry"] == "mcp_call"]
        assert [(line["tool"], line["outcome"], line["decision"]) for line in lines] == [
            ("read_note", "ok", None),
            ("write_note", "ok", "approve"),
            ("write_note", "blocked", None),
            ("", "unknown_tool", None),
        ]
        assert lines[1]["arguments"] == {"text": "yes"}
        assert lines[1]["user"] == "ada"
        counted = site.registry.get_sample_value
        assert counted("deskclerk_gateway_calls_total", {"outcome": "ok"}) == 2
        assert counted("deskclerk_gateway_calls_total", {"outcome": "blocked"}) == 1
        assert counted("deskclerk_tool_calls_total", {"server": "notes", "outcome": "ok"}) == 2
        assert counted("deskclerk_confirmations_total", {"decision": "approve"}) == 1


@pytest.mark.anyio
async def test_calls_beyond_the_per_minute_quota_are_refused(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings, requests_per_minute=1)
    ada = token_of(sign_token, "ada")
    with serving(settings, [notes(notes_server)], token_signing_key, database) as site:
        connect(site, ada)
        async with mcp_client(site, ada) as client:
            first = await client.call_tool("notes__read_note", {})
            second = await client.call_tool("notes__read_note", {})
    assert not first.is_error
    assert second.is_error
    assert "a minute" in text_of(second)


@pytest.mark.anyio
async def test_a_call_by_an_employee_over_budget_is_refused_before_anything_runs(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, budget_usd=0.5)
    SpendingStore(database).add("ada", month_of(datetime.now(UTC)), "model", Usage(0, 0, 0, 0), 0.5)
    fake = FakeJev(0.0)
    ada = token_of(sign_token, "ada")
    with serving(
        settings, [notes(notes_server)], token_signing_key, database, as_jev(fake)
    ) as site:
        connect(site, ada)
        async with mcp_client(site, ada, answering([])) as client:
            read = await client.call_tool("notes__read_note", {})
            write = await client.call_tool("notes__write_note", {"text": "late"})
        for refused in (read, write):
            assert refused.is_error
            assert "DESKCLERK_BUDGET_USD" in text_of(refused)
        counted = site.registry.get_sample_value
        assert counted("deskclerk_gateway_calls_total", {"outcome": "over_budget"}) == 2
    assert notes_server.notes == []
    assert fake.asked == []
    lines = [line for line in audit_lines(caplog) if line["entry"] == "mcp_call"]
    assert [line["outcome"] for line in lines] == ["over_budget", "over_budget"]


@pytest.mark.anyio
async def test_the_injection_screen_on_a_call_is_charged_to_the_monthly_budget(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    fake = FakeJev(0.0, tokens=1_000_000)
    ada = token_of(sign_token, "ada")
    with serving(
        settings, [notes(notes_server)], token_signing_key, database, as_jev(fake)
    ) as site:
        connect(site, ada)
        async with mcp_client(site, ada) as client:
            await client.call_tool("notes__read_note", {})
    spent = SpendingStore(database).load("ada", month_of(datetime.now(UTC)))
    assert spent == pytest.approx(1_000_000 * PRICE_PER_INPUT_TOKEN)
