# The chat and admin pages in headless Chromium against the real app, the development Keycloak and
# a scripted model: sign-in, the approval card, inert markup, the flag badge and the admin tabs.

import socket
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Any

import httpx
import pytest
from cryptography.fernet import Fernet
from langchain_core.language_models import BaseChatModel
from playwright.sync_api import ConsoleMessage, Page, expect
from pydantic import SecretStr

from conftest import CHAIN_KEY
from deskclerk.auth import discover_jwks_url, jwks_signing_key
from deskclerk.config import Settings
from deskclerk.database import Database
from deskclerk.jev import Jev
from deskclerk.servers import McpServer
from mcp_server import READ_ONLY, running_mcp_server
from test_agent import answer, calls, script
from test_api import SESSION_KEY, open_notes
from test_connections import REDIRECT
from test_screen import FakeJev, as_jev
from test_stream import listening

pytestmark = [pytest.mark.browser, pytest.mark.keycloak]

KEYCLOAK = "http://localhost:8080"
CLIENT = "deskclerk-api"
PASSWORD = "deskclerk"  # noqa: S105
INJECTION = '<img src=x onerror="window.pwned=1"> and <script>window.pwned=1</script>'


def _free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _admin_headers(keycloak: httpx.Client) -> dict[str, str]:
    token = keycloak.post(
        f"{KEYCLOAK}/realms/master/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": "admin",
            "password": "admin",
        },
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _client_url(keycloak: httpx.Client, headers: dict[str, str]) -> str:
    listing = keycloak.get(
        f"{KEYCLOAK}/admin/realms/deskclerk/clients", params={"clientId": CLIENT}, headers=headers
    ).json()
    return f"{KEYCLOAK}/admin/realms/deskclerk/clients/{listing[0]['id']}"


@pytest.fixture
def port() -> Iterator[int]:
    free = _free_port()
    callback = f"http://127.0.0.1:{free}/login/callback"
    with httpx.Client() as keycloak:
        headers = _admin_headers(keycloak)
        url = _client_url(keycloak, headers)
        registered: dict[str, Any] = keycloak.get(url, headers=headers).json()
        original = list(registered["redirectUris"])
        keycloak.put(
            url, json=registered | {"redirectUris": [*original, callback]}, headers=headers
        )
        yield free
        keycloak.put(
            url, json=registered | {"redirectUris": original}, headers=_admin_headers(keycloak)
        )


def _settings(port: int) -> Settings:
    return Settings(
        session_key=SecretStr(SESSION_KEY),
        audit_chain_key=SecretStr(CHAIN_KEY.decode()),
        public_url=f"http://127.0.0.1:{port}",
        admin_group="research",
    )


def _sign_in(page: Page, base: str) -> None:
    page.goto(base)
    page.get_by_role("textbox", name="Username or email").fill("kaminski-v")
    page.get_by_label("Password", exact=True).fill(PASSWORD)
    page.get_by_role("button", name="Sign In").click()
    expect(page.get_by_role("button", name="Sign out")).to_be_visible()


def _ask(page: Page, request: str) -> None:
    page.get_by_label("Your request").fill(request)
    page.get_by_role("button", name="Send").click()


def _console_errors(page: Page) -> list[str]:
    errors: list[str] = []

    def remember(message: ConsoleMessage) -> None:
        if message.type == "error":
            errors.append(message.text)

    page.on("console", remember)
    page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


@contextmanager
def _app(
    port: int,
    database: Database,
    servers: list[McpServer],
    model: BaseChatModel,
    jev: Jev | None = None,
) -> Iterator[str]:
    settings = _settings(port)
    with httpx.Client() as discovery:
        signing_key = jwks_signing_key(discover_jwks_url(settings, discovery))
    key = Fernet.generate_key().decode()
    with listening(settings, servers, model, signing_key, database, key, jev, port) as base:
        yield base


def test_an_employee_signs_in_through_keycloak_and_the_page_loads_without_errors(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        errors = _console_errors(page)
        page.reload()
        heading = page.get_by_role("heading", name="What do you need from your connected apps?")
        expect(heading).to_be_visible()
    assert errors == []


def test_a_write_waits_in_an_approval_card_and_runs_when_approved(
    page: Page, port: int, database: Database
) -> None:
    model = script(
        calls(("open__write_note", {"text": "buy stamps"})), answer("The note is saved.")
    )
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model) as base:
            _sign_in(page, base)
            _ask(page, "Save a note")
            card = page.locator(".pending-card")
            expect(card).to_contain_text("write_note")
            expect(card).to_contain_text("buy stamps")
            card.get_by_role("button", name="Approve").click()
            expect(page.get_by_text("The note is saved.")).to_be_visible()
            expect(page.locator("details.call", has_text="write_note")).to_contain_text("Done")
        assert [text for _, text in running.notes] == ["buy stamps"]


def test_markup_from_a_tool_result_and_the_answer_stays_inert(
    page: Page, port: int, database: Database
) -> None:
    def payload_note() -> str:
        return INJECTION

    model = script(calls(("open__payload_note", {})), answer(f"The note says: {INJECTION}"))
    with running_mcp_server(REDIRECT, oauth=False) as running:
        running.mcp.add_tool(payload_note, annotations=READ_ONLY)
        with _app(port, database, [open_notes(running)], model) as base:
            _sign_in(page, base)
            _ask(page, "Read the payload note")
            row = page.locator("details.call", has_text="payload_note")
            expect(row).to_be_visible()
            row.locator("summary").click()
            expect(row.locator("pre").last).to_contain_text("onerror=")
            expect(page.get_by_text("The note says:")).to_be_visible()
            assert page.locator("#turns img, #turns script").count() == 0
            assert page.evaluate("window.pwned") is None


def test_a_flagged_result_shows_the_badge_and_the_next_read_asks_first(
    page: Page, port: int, database: Database
) -> None:
    model = script(
        calls(("open__read_note", {})), calls(("open__read_note", {})), answer("Both notes read.")
    )
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model, as_jev(FakeJev(0.9))) as base:
            _sign_in(page, base)
            _ask(page, "Read the note twice")
            expect(page.get_by_text("Flagged by screen").first).to_be_visible()
            card = page.locator(".pending-card")
            expect(card).to_contain_text("read_note")
            expect(card).to_contain_text("flagged as a possible prompt injection")
            card.get_by_role("button", name="Approve").click()
            expect(page.get_by_text("Both notes read.")).to_be_visible()


def test_the_admin_tabs_switch_and_the_audit_tab_checks_the_chain(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        page.goto(f"{base}/admin")
        servers_tab = page.get_by_role("tab", name="Servers")
        expect(servers_tab).to_have_attribute("aria-selected", "true")
        page.get_by_role("tab", name="People").click()
        expect(page.get_by_role("heading", name="Revoke an employee")).to_be_visible()
        expect(servers_tab).to_have_attribute("aria-selected", "false")
        page.get_by_role("tab", name="Audit log").click()
        page.get_by_role("button", name="Verify chain").click()
        expect(page.get_by_text("Chain verified")).to_be_visible()
