# The chat and admin pages in headless Chromium against the real app, the development Keycloak and
# a scripted model: sign-in, the approval card, inert markup, the flag badge and the admin tabs.

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


CARD_BEFORE_ANSWER = """(() => {
  const turn = document.querySelector(".turn");
  const answer = turn.querySelector(".answer");
  const order = turn.querySelector(".call").compareDocumentPosition(answer);
  return Boolean(order & Node.DOCUMENT_POSITION_FOLLOWING);
})()"""


def _watch_announcements(page: Page) -> None:
    page.evaluate(
        """() => {
          const region = document.getElementById("announce");
          window.heard = [];
          new MutationObserver(() => {
            if (region.textContent) window.heard.push(region.textContent);
          }).observe(region, { childList: true, characterData: true, subtree: true });
        }"""
    )


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
            _watch_announcements(page)
            _ask(page, "Save a note")
            card = page.locator(".pending-card")
            expect(card).to_contain_text("write_note")
            expect(card).to_contain_text("buy stamps")
            expect(card.locator(".call-head")).to_contain_text("Awaiting approval")
            expect(card.locator(".verdict")).to_have_count(0)
            card.get_by_role("button", name="Approve").click()
            expect(page.get_by_text("The note is saved.")).to_be_visible()
            shown = page.locator(".call")
            expect(shown).to_have_count(1)
            expect(shown.locator(".call-head")).to_contain_text("Done")
            assert page.evaluate(CARD_BEFORE_ANSWER)
            heard = page.evaluate("window.heard")
            assert "open write_note: Running" in heard
            assert "open write_note: Done" in heard
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
            row = page.locator(".call", has_text="payload_note")
            expect(row).to_be_visible()
            row.get_by_role("button", name="Show details").click()
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
            expect(page.get_by_text("Possible prompt injection", exact=True).first).to_be_visible()
            card = page.locator(".pending-card")
            expect(card).to_contain_text("read_note")
            expect(card).to_contain_text(
                "Possible prompt injection detected in the result of open read_note"
            )
            card.get_by_role("button", name="Show that result").click()
            source = page.locator(".call[data-flagged] .call-body")
            expect(source).to_be_visible()
            expect(source).to_contain_text("Result")
            card.get_by_role("button", name="Approve").click()
            expect(page.get_by_text("Both notes read.")).to_be_visible()


def test_reads_after_a_flag_are_approved_together_and_a_write_keeps_its_own_decision(
    page: Page, port: int, database: Database
) -> None:
    model = script(
        calls(("open__read_note", {})),
        calls(
            ("open__read_note", {}), ("open__read_note", {}), ("open__write_note", {"text": "x"})
        ),
        answer("Done reading."),
    )
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model, as_jev(FakeJev(0.9))) as base:
            _sign_in(page, base)
            _ask(page, "Read the notes")
            cards = page.locator(".pending-card")
            expect(cards).to_have_count(3)
            expect(cards.filter(has_text="Waiting for you")).to_have_count(3)
            expect(page.get_by_text("Not decided")).to_have_count(0)
            page.get_by_role("button", name="Approve all reads").click()
            expect(page.locator("#decided")).to_have_text("2 of 3 decided")
            expect(page.get_by_role("button", name="Continue")).to_be_disabled()
            cards.filter(has_text="write_note").get_by_role("button", name="Reject").click()
            page.get_by_role("button", name="Continue").click()
            expect(page.get_by_text("Done reading.")).to_be_visible()
            expect(page.locator(".call", has_text="read_note")).to_have_count(3)
            expect(page.locator(".call", has_text="write_note")).to_contain_text("Denied")
        assert running.notes == []


def test_reject_all_denies_every_waiting_call_in_one_action(
    page: Page, port: int, database: Database
) -> None:
    model = script(
        calls(("open__read_note", {})),
        calls(("open__read_note", {}), ("open__read_note", {})),
        answer("Nothing more read."),
    )
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model, as_jev(FakeJev(0.9))) as base:
            _sign_in(page, base)
            _ask(page, "Read the notes")
            expect(page.locator(".pending-card")).to_have_count(2)
            page.get_by_role("button", name="Reject all").click()
            expect(page.get_by_text("Nothing more read.")).to_be_visible()
            denied = page.locator(".call[data-state=denied]")
            expect(denied).to_have_count(2)


def test_always_allow_sits_apart_and_asks_for_a_confirmation_that_names_the_tool(
    page: Page, port: int, database: Database
) -> None:
    model = script(calls(("open__append_note", {"text": "milk"})), answer("Appended."))
    with running_mcp_server(REDIRECT, oauth=False, append_tool=True) as running:
        with _app(port, database, [open_notes(running)], model) as base:
            _sign_in(page, base)
            _ask(page, "Append milk")
            card = page.locator(".pending-card")
            choices = card.get_by_role("group", name="Decision for open append_note")
            expect(choices.get_by_role("button")).to_have_count(3)
            card.get_by_role("button", name="Always allow this tool…").click()
            prompt = card.get_by_role("group", name="Confirm always allow")
            expect(prompt).to_contain_text("Always allow open append_note?")
            assert running.notes == []
            with page.expect_request("**/ask/confirm") as sent:
                prompt.get_by_role("button", name="Always allow append_note").click()
            body = sent.value.post_data_json
            assert body is not None
            assert list(body["decisions"].values()) == ["approve_always"]
            expect(page.get_by_text("Appended.")).to_be_visible()
        assert [text for _, text in running.notes] == ["milk"]


def test_the_request_box_locks_while_calls_wait_and_a_chip_jumps_to_the_card(
    page: Page, port: int, database: Database
) -> None:
    long_answer = "\n\n".join(f"Line {number}" for number in range(80))
    model = script(answer(long_answer), calls(("open__write_note", {"text": "x"})))
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model) as base:
            page.set_viewport_size({"width": 900, "height": 600})
            _sign_in(page, base)
            _ask(page, "First")
            expect(page.get_by_text("Line 79")).to_be_visible()
            _ask(page, "Save a note")
            expect(page.locator(".pending-card")).to_have_count(1)
            expect(page.get_by_label("Your request")).not_to_be_editable()
            expect(page.locator("#locked")).to_contain_text("Approve or reject the waiting calls")
            chip = page.get_by_role("button", name="Jump to approval")
            expect(chip).to_be_hidden()
            page.evaluate("el('messages').scrollTo({ top: 0 })")
            expect(chip).to_be_visible()
            chip.click()
            approve = page.locator(".pending-card").get_by_role("button", name="Approve")
            expect(approve).to_be_focused()
            expect(chip).to_be_hidden()


def test_a_rejection_with_a_reason_sends_the_reason_to_the_model(
    page: Page, port: int, database: Database
) -> None:
    model = script(calls(("open__write_note", {"text": "x"})), answer("Not saved."))
    with running_mcp_server(REDIRECT, oauth=False) as running:
        with _app(port, database, [open_notes(running)], model) as base:
            _sign_in(page, base)
            _ask(page, "Save a note")
            card = page.locator(".pending-card")
            card.get_by_role("button", name="Reject").click()
            card.get_by_label("Reason for rejecting (optional, sent to the assistant)").fill(
                "wrong note"
            )
            card.get_by_role("button", name="Send rejection").click()
            expect(page.get_by_text("Not saved.")).to_be_visible()
            shown = page.locator(".call")
            expect(shown.locator(".call-head")).to_contain_text("Denied")
            shown.get_by_role("button", name="Show details").click()
            expect(shown).to_contain_text("You rejected this call: wrong note")
        assert running.notes == []
    told = [message for message in model.seen[-1] if message.type == "tool"]
    assert "wrong note" in str(told[-1].content)


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


def _server_listing(connected: bool, needs_sign_in: bool = True) -> str:
    return json.dumps(
        [
            {
                "name": "notes",
                "url": "http://notes.example/mcp",
                "title": None,
                "description": None,
                "examples": [],
                "connected": connected,
                "needs_sign_in": needs_sign_in,
                "tools": [],
                "always_allowed": [],
            }
        ]
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
        newer.fulfill(status=200, content_type="application/json", body=_server_listing(True))
        expect(page.locator("#servers .server")).to_have_count(1)
        stale = json.loads(_server_listing(True)) * 2
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
                status=200, content_type="application/json", body=_server_listing(False)
            ),
        )
        page.evaluate("loadServers()")
        hint = page.locator("#connect-hint")
        expect(hint).to_be_visible()
        expect(hint).to_contain_text("Connect an app in the left panel.")
        expect(hint.locator(".phone-only")).to_be_hidden()
        page.set_viewport_size({"width": 375, "height": 812})
        expect(hint.locator(".phone-only")).to_contain_text("Conversations button")
        page.route(
            "**/servers",
            lambda route: route.fulfill(
                status=200, content_type="application/json", body=_server_listing(True)
            ),
        )
        page.evaluate("loadServers()")
        expect(hint).to_be_hidden()


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
        page.once("dialog", lambda dialog: dialog.accept())
        page.get_by_role("button", name="Revoke", exact=True).click()
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
