# The chat and admin pages in headless Chromium against the real app, the development Keycloak and
# a scripted model: sign-in, the approval card, inert markup, the flag badge and the admin page.

import json
import socket
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from cryptography.fernet import Fernet
from langchain_core.language_models import BaseChatModel
from playwright.sync_api import ConsoleMessage, Page, Route, expect
from pydantic import SecretStr

from conftest import CHAIN_KEY
from deskclerk.auditstore import AuditStore
from deskclerk.auth import discover_jwks_url, jwks_signing_key
from deskclerk.config import Settings
from deskclerk.database import Database
from deskclerk.jev import Jev
from deskclerk.providers import Usage
from deskclerk.servers import McpServer
from deskclerk.spending import SpendingStore, month_of
from mcp_server import READ_ONLY, running_mcp_server
from test_agent import ScriptedModel, answer, calls, script
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


class FailsOnTheSecondAsk(ScriptedModel):
    broken: bool = True

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        if len(self.seen) == 1 and self.broken:
            self.broken = False
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=request)
        return super()._generate(*args, **kwargs)


def test_retrying_a_failed_ask_replaces_the_failed_request_instead_of_sending_it_twice(
    page: Page, port: int, database: Database
) -> None:
    model = FailsOnTheSecondAsk(messages=iter([answer("one"), answer("two")]))
    with _app(port, database, [], model) as base:
        _sign_in(page, base)
        _ask(page, "first")
        expect(page.get_by_text("one", exact=True)).to_be_visible()
        _ask(page, "second")
        expect(page.locator(".turn-error")).to_contain_text("did not answer")
        page.get_by_role("button", name="Retry").click()
        expect(page.get_by_text("two", exact=True)).to_be_visible()
    asked = [m.content for m in model.seen[1] if m.type == "human"]
    assert asked == ["first", "second"]


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


def _server(name: str, state: str, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "url": f"http://{name}.example/mcp",
        "title": None,
        "description": None,
        "examples": [],
        "state": state,
        "needs_sign_in": True,
        "tools": tools or [],
        "always_allowed": [],
    }


def _server_listing(state: str) -> str:
    return json.dumps([_server("notes", state)])


def _fulfil_json(page: Page, pattern: str, body: Any) -> None:
    page.route(
        pattern,
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(body)
        ),
    )


def _conversation_listing(identifier: str, pending: list[Any]) -> str:
    return json.dumps({"conversation": identifier, "turns": [], "pending": pending})


def test_two_overlapping_loads_of_the_server_list_render_one_card(
    page: Page, port: int, database: Database
) -> None:
    held: list[Route] = []
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        page.route("**/servers", lambda route: held.append(route))
        page.evaluate("() => { loadServers(); loadServers(); }")
        while len(held) < 2:
            page.wait_for_timeout(50)
        older, newer = held
        newer.fulfill(
            status=200, content_type="application/json", body=_server_listing("connected")
        )
        expect(page.locator("#servers .server")).to_have_count(1)
        stale = json.loads(_server_listing("connected")) * 2
        older.fulfill(status=200, content_type="application/json", body=json.dumps(stale))
        page.wait_for_timeout(300)
        expect(page.locator("#servers .server")).to_have_count(1)


def test_a_server_without_sign_in_has_no_disconnect_button(
    page: Page, port: int, database: Database
) -> None:
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], script()) as base:
            _sign_in(page, base)
            card = page.locator("#servers .server")
            expect(card).to_contain_text("Connected")
            expect(card.get_by_role("button")).to_have_count(0)


def test_the_cards_follow_the_server_after_a_nothing_to_confirm_answer(
    page: Page, port: int, database: Database
) -> None:
    model = script(calls(("open__write_note", {"text": "buy stamps"})))
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model) as base:
            _sign_in(page, base)
            _ask(page, "Save a note")
            expect(page.locator(".pending-card")).to_have_count(1)
            identifier = page.evaluate("conversation")

            def nothing_to_confirm(route: Route) -> None:
                route.fulfill(
                    status=409,
                    content_type="application/json",
                    body=json.dumps({"detail": "this conversation has nothing to confirm"}),
                )

            def nothing_pending(route: Route) -> None:
                if route.request.method != "GET":
                    return route.fallback()
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=_conversation_listing(identifier, []),
                )

            page.route("**/ask/confirm", nothing_to_confirm)
            page.route("**/conversations/*", nothing_pending)
            page.locator(".pending-card").get_by_role("button", name="Approve").click()
            expect(page.locator("#error")).to_contain_text("nothing to confirm")
            expect(page.locator(".pending-card")).to_have_count(0)


def test_the_cards_stay_when_the_server_still_holds_the_call_after_a_502(
    page: Page, port: int, database: Database
) -> None:
    model = script(calls(("open__write_note", {"text": "buy stamps"})))
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model) as base:
            _sign_in(page, base)
            _ask(page, "Save a note")
            expect(page.locator(".pending-card")).to_have_count(1)
            page.route(
                "**/ask/confirm",
                lambda route: route.fulfill(
                    status=502, content_type="application/json", body='{"detail": "upstream"}'
                ),
            )
            page.locator(".pending-card").get_by_role("button", name="Approve").click()
            expect(page.locator("#error")).to_contain_text("did not answer")
            expect(page.locator(".pending-card")).to_have_count(1)


def test_a_deleted_conversation_starts_a_new_one_on_the_next_ask(
    page: Page, port: int, database: Database
) -> None:
    model = script(answer("First answer."), answer("Second answer."))
    with _app(port, database, [], model) as base:
        _sign_in(page, base)
        _ask(page, "First")
        expect(page.get_by_text("First answer.")).to_be_visible()
        page.wait_for_function("conversation !== null")
        page.route(
            "**/ask",
            lambda route: route.fulfill(
                status=404,
                content_type="application/json",
                body='{"detail": "no such conversation"}',
            ),
        )
        _ask(page, "Second")
        expect(page.locator("#error")).to_contain_text("That conversation was deleted")
        assert page.evaluate("conversation") is None
        assert "#c=" not in page.url
        expect(page.get_by_label("Your request")).to_have_value("Second")
        page.unroute("**/ask")
        page.get_by_role("button", name="Send").click()
        expect(page.get_by_text("Second answer.")).to_be_visible()


def test_the_empty_page_points_to_the_left_panel_when_no_app_is_connected(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        page.route(
            "**/servers",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=_server_listing("not_connected")
            ),
        )
        page.evaluate("loadServers()")
        hint = page.locator("#connect-hint")
        expect(hint).to_be_visible()
        expect(hint).to_contain_text("Connect an app in the left panel.")
        expect(hint.locator(".phone-only")).to_be_hidden()
        page.set_viewport_size({"width": 375, "height": 812})
        expect(hint.locator(".phone-only")).to_contain_text("Menu button")
        expect(page.get_by_role("button", name="Menu")).to_be_visible()
        page.route(
            "**/servers",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=_server_listing("connected")
            ),
        )
        page.evaluate("loadServers()")
        expect(hint).to_be_hidden()


def test_each_server_state_shows_its_label_and_its_action(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        states = ["connected", "not_connected", "sign_in_expired", "error"]
        _fulfil_json(
            page, "**/servers", [_server(f"s{n}", state) for n, state in enumerate(states)]
        )
        page.evaluate("loadServers()")
        cards = page.locator("#servers .server-foot")
        expect(cards).to_have_text(
            ["ConnectedDisconnect", "Not connectedConnect", "Sign-in expiredReconnect",
             "Not respondingDisconnect"]
        )  # fmt: skip
        expect(page.get_by_role("button", name="Reconnect s2")).to_be_visible()
        expect(page.locator("#rail-summary")).to_have_text("Connected apps: 1 of 4 connected")


def test_the_rail_meter_shows_spend_below_one_cent(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        page.evaluate("showBudget(0.0043, 2)")
        expect(page.locator("#budget")).to_have_text("This month: $0.0043 of $2.00")


def test_no_suggestion_is_made_up_from_a_tool_name(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        tools = [
            {"name": "git_status", "title": "", "read_only": True},
            {"name": "git_commit", "title": "", "read_only": False},
        ]
        _fulfil_json(page, "**/servers", [_server("git", "connected", tools)])
        page.evaluate("loadServers()")
        expect(page.locator("#chips .chip")).to_have_text(
            ["What can you do with git?", "List the tools git offers",
             "Give me an example of using git"]
        )  # fmt: skip


def test_only_an_admin_is_told_which_variable_turns_the_screen_on(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        banner = page.locator("#unscreened")
        expect(banner).to_contain_text("Set TYPESAFE_API_KEY")
        me = page.evaluate("fetch('/me').then((response) => response.json())")
        _fulfil_json(page, "**/me", me | {"admin": False})
        page.reload()
        expect(banner).to_have_text("Tool results are not screened for prompt injection.")


def test_a_long_conversation_title_wraps_to_two_lines(
    page: Page, port: int, database: Database
) -> None:
    title = "Summarise the open pull requests and the failing checks on the release branch"
    with _app(port, database, [], script()) as base:
        _sign_in(page, base)
        listed = [{"id": "c1", "title": title, "updated_at": datetime.now(UTC).isoformat()}]
        _fulfil_json(page, "**/conversations", listed)
        page.evaluate("loadConversations()")
        shown = page.locator(".conv-title")
        expect(shown).to_have_text(title)
        height = shown.evaluate("(node) => node.clientHeight")
        line = shown.evaluate("(node) => parseFloat(getComputedStyle(node).lineHeight)")
        assert round(height / line) == 2
        assert (page.locator(".conv-open").get_attribute("title") or "").startswith(title)


def _seed_people(database: Database) -> None:
    with database.pool.connection() as connection:
        connection.execute(
            "INSERT INTO employees (user_id, display_name, last_seen) "
            "VALUES ('6f1c0a52-aaaa-4bbb-8ccc-0123456789ab', 'Ada Lovelace', now())"
        )
        connection.execute(
            "INSERT INTO conversations (user_id, id, title, updated_at) "
            "VALUES ('6f1c0a52-aaaa-4bbb-8ccc-0123456789ab', gen_random_uuid(), 'x', now())"
        )
    ada = "6f1c0a52-aaaa-4bbb-8ccc-0123456789ab"
    month = month_of(datetime.now(UTC))
    SpendingStore(database).add(ada, month, "model", Usage(1, 1, 0, 0), 0.5)
    chain = AuditStore(database, CHAIN_KEY)
    for _ in range(3):
        fields = {
            "time": datetime.now(UTC).isoformat(),
            "user": ada,
            "entry": "ask",
            "text": "word " * 200,
        }
        chain.append(json.dumps(fields))


def test_the_admin_pages_show_each_employee_by_name_with_the_id_beside_it(
    page: Page, port: int, database: Database
) -> None:
    ada = "6f1c0a52-aaaa-4bbb-8ccc-0123456789ab"
    with _app(port, database, [], script()) as base:
        _seed_people(database)
        _sign_in(page, base)
        page.goto(f"{base}/admin#people")
        usage = page.locator("#usage tr").first
        expect(usage).to_contain_text("Ada Lovelace")
        expect(usage).to_contain_text(ada)
        expect(usage.locator("td.who")).to_have_attribute("title", ada)
        expect(page.locator("#recent tr").first).to_contain_text("Ada Lovelace")
        page.get_by_role("tab", name="Audit log").click()
        expect(page.locator("#audit tr").first).to_contain_text("Ada Lovelace")


def test_revoking_picks_a_known_employee_from_a_list_and_clears_their_stored_secrets(
    page: Page, port: int, database: Database
) -> None:
    ada = "6f1c0a52-aaaa-4bbb-8ccc-0123456789ab"
    with _app(port, database, [], script()) as base:
        _seed_people(database)
        with database.pool.connection() as connection:
            for kind in ("tokens", "client", "tool_pins", "always_allowed", "tool_menu"):
                connection.execute(
                    "INSERT INTO secrets (user_id, server_name, kind, value) "
                    "VALUES (%s, 'notes', %s, 'x')",
                    (ada, kind),
                )
        _sign_in(page, base)
        page.goto(f"{base}/admin#people")
        picker = page.get_by_label("Employee to revoke")
        expect(picker.locator("option")).to_contain_text(
            ["Choose an employee", f"Ada Lovelace ({ada})", "kaminski-v"]
        )
        assert page.locator("#revoke input[type=text]").count() == 0
        picker.select_option(ada)
        page.get_by_role("button", name="Revoke", exact=True).click()
        dialog = page.get_by_role("dialog")
        expect(dialog).to_contain_text("ends their browser sessions")
        dialog.get_by_role("button", name="Revoke employee").click()
        expect(page.locator("#revoked")).to_contain_text("5 stored secrets deleted")
        with database.pool.connection() as connection:
            left = connection.execute("SELECT COUNT(*) FROM secrets WHERE user_id = %s", (ada,))
            assert left.fetchone() == (0,)


def test_no_admin_tab_scrolls_the_page_sideways_on_a_phone(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _seed_people(database)
        page.set_viewport_size({"width": 375, "height": 812})
        _sign_in(page, base)
        for tab, rows in (
            ("servers", "#servers tr"),
            ("people", "#usage tr"),
            ("audit", "#audit tr"),
        ):
            page.goto(f"{base}/admin#{tab}")
            expect(page.locator(rows).first).to_be_visible()
            widths = page.evaluate(
                "[document.documentElement.scrollWidth, document.documentElement.clientWidth]"
            )
            assert widths[0] <= widths[1], (tab, widths)


def _open_admin(page: Page, base: str) -> None:
    _sign_in(page, base)
    page.on("dialog", lambda dialog: pytest.fail(f"a browser dialog opened: {dialog.message}"))
    page.goto(f"{base}/admin")


def _add_server(page: Page, name: str, url: str) -> None:
    page.get_by_role("button", name="Add a server").click()
    page.get_by_label("Name", exact=True).fill(name)
    page.get_by_label("URL", exact=True).fill(url)


def test_a_server_added_through_the_form_posts_the_entry_and_focus_returns_to_its_row(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _open_admin(page, base)
        _add_server(page, "wiki", "http://127.0.0.1:9/mcp")
        page.get_by_label("Sign-in").select_option("none")
        expect(page.get_by_label("Client secret variable")).to_be_hidden()
        page.get_by_label("Groups").fill("research, finance")
        page.get_by_label("Allowed tools").fill("read_note")
        page.get_by_label("Offer Always allow").uncheck()
        with page.expect_request("**/admin/servers") as posted:
            page.get_by_role("button", name="Save").click()
        assert posted.value.post_data_json == {
            "name": "wiki",
            "url": "http://127.0.0.1:9/mcp",
            "auth": "none",
            "groups": ["research", "finance"],
            "tools": ["read_note"],
            "always_allow": False,
        }
        edit = page.locator("#servers tr", has_text="wiki").get_by_role("button", name="Edit")
        expect(edit).to_be_focused()
        expect(page.locator("#editor")).to_be_hidden()


def test_a_shared_header_is_saved_as_a_header_and_a_refusal_shows_beside_the_form(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _open_admin(page, base)
        _add_server(page, "wiki", "http://127.0.0.1:9/mcp")
        page.get_by_label("Sign-in").select_option("header")
        expect(page.get_by_label("Client ID")).to_be_hidden()
        value = "Bearer ${env:DESKCLERK_SERVER_WIKI_TOKEN}"
        page.get_by_label("Header value").fill(value)
        with page.expect_request("**/admin/servers") as posted:
            page.get_by_role("button", name="Save").click()
        assert posted.value.post_data_json == {
            "name": "wiki",
            "url": "http://127.0.0.1:9/mcp",
            "auth": "none",
            "headers": {"Authorization": value},
        }
        expect(page.locator("#editor-error")).to_contain_text("which is not set")
        expect(page.locator("#error")).to_be_hidden()
        page.get_by_role("button", name="Edit as JSON").click()
        shown = json.loads(page.locator("#entry").input_value())
        assert shown == posted.value.post_data_json


def test_editing_keeps_the_fields_the_form_does_not_show_and_json_it_cannot_show_stays_json(
    page: Page, port: int, database: Database
) -> None:
    with _app(port, database, [], script()) as base:
        _open_admin(page, base)
        _add_server(page, "wiki", "http://127.0.0.1:9/mcp")
        page.get_by_label("Sign-in").select_option("none")
        page.get_by_role("button", name="Save").click()
        row = page.locator("#servers tr", has_text="wiki")
        row.get_by_role("button", name="Pause").click()
        expect(row.get_by_role("button", name="Resume")).to_be_focused()
        row.get_by_role("button", name="Edit").click()
        page.get_by_label("Title").fill("Wiki")
        with page.expect_request("**/admin/servers/wiki") as put:
            page.get_by_role("button", name="Save").click()
        sent = put.value.post_data_json
        assert sent is not None
        assert (sent["paused"], sent["title"]) == (True, "Wiki")
        row.get_by_role("button", name="Edit").click()
        page.get_by_role("button", name="Edit as JSON").click()
        two_headers: dict[str, Any] = {"name": "wiki", "url": "http://127.0.0.1:9/mcp"}
        two_headers |= {"auth": "none", "headers": {"A": "1", "B": "2"}}
        page.locator("#entry").fill(json.dumps(two_headers))
        page.get_by_role("button", name="Edit as a form").click()
        expect(page.locator("#editor-error")).to_contain_text("the form cannot show")
        expect(page.locator("#entry")).to_be_visible()


def test_a_write_after_the_session_lapsed_goes_to_the_sign_in_page(
    page: Page, port: int, database: Database
) -> None:
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], script()) as base:
            _open_admin(page, base)
            page.route("**/admin/servers/open/pause", lambda route: route.fulfill(status=401))
            with page.expect_request(lambda request: request.url == f"{base}/login?next=/admin"):
                page.get_by_role("button", name="Pause").click()


def test_removing_a_server_asks_in_a_dialog_that_says_what_is_deleted(
    page: Page, port: int, database: Database
) -> None:
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], script()) as base:
            _open_admin(page, base)
            remove = page.locator("#servers").get_by_role("button", name="Remove")
            remove.click()
            dialog = page.get_by_role("dialog")
            expect(dialog).to_contain_text("Remove open?")
            expect(dialog).to_contain_text("stored sign-in, always-allow choices and tool pins")
            dialog.get_by_role("button", name="Cancel").click()
            expect(remove).to_be_focused()
            remove.click()
            dialog.get_by_role("button", name="Remove server").click()
            expect(page.locator("#servers")).to_contain_text("Nothing yet.")
            expect(page.get_by_role("button", name="Add a server")).to_be_focused()


def test_the_servers_table_shows_whether_each_server_answers_and_which_tools_write(
    page: Page, port: int, database: Database
) -> None:
    gone = McpServer("gone", "http://127.0.0.1:9/mcp", None, None, "none")
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running), gone], script()) as base:
            _open_admin(page, base)
            reachable = page.locator("#servers tr", has_text="open")
            expect(reachable).to_contain_text("Reachable")
            reachable.locator("summary").click()
            expect(reachable.locator("li", has_text="write_note")).to_contain_text("write")
            expect(reachable.locator("li", has_text="read_note")).to_contain_text("read")
            unreachable = page.locator("#servers tr", has_text="gone")
            expect(unreachable).to_contain_text("Unreachable")
            expect(unreachable).to_contain_text("ConnectError")


def test_the_audit_log_shows_plain_labels_and_filters_by_an_outcome_from_a_list(
    page: Page, port: int, database: Database
) -> None:
    ada = "6f1c0a52-aaaa-4bbb-8ccc-0123456789ab"
    with _app(port, database, [], script()) as base:
        _seed_people(database)
        chain = AuditStore(database, CHAIN_KEY)
        time = datetime.now(UTC).isoformat()
        failed = {"time": time, "user": ada, "entry": "ask", "outcome": "upstream_error"}
        chain.append(json.dumps(failed))
        removed = {"time": time, "user": ada, "entry": "admin", "action": "remove_server"}
        chain.append(json.dumps(removed | {"target": "wiki"}))
        _open_admin(page, base)
        page.get_by_role("tab", name="Audit log").click()
        expect(page.locator("#audit tr").first).to_contain_text("removed server wiki")
        page.get_by_label("Outcome").select_option(label="A server or the model did not answer")
        page.get_by_role("button", name="Search").click()
        rows = page.locator("#audit tr")
        expect(rows).to_have_count(1)
        expect(rows.first).to_contain_text("Request")
        expect(rows.first.locator("td").nth(1)).to_contain_text("Ada Lovelace")
        expect(rows.first).to_contain_text("A server or the model did not answer")
        rows.first.get_by_text("Show").click()
        expect(rows.first.locator("pre")).to_contain_text("upstream_error")
