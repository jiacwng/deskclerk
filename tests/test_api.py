# The web front against the real test MCP server, with tokens signed by the throwaway key from
# conftest.py and a scripted chat model: who may call it, the per-employee limits, the audit lines
# and the cookie protections. One test signs in through the development Keycloak.

import base64
import hashlib
import hmac
import html
import itertools
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import httpx2
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from pydantic import SecretStr

from deskclerk import cookies
from deskclerk.api import (
    BROWSER_AUTH_REFUSED,
    CSRF_HEADER,
    CSRF_VALUE,
    LOGIN_COOKIE,
    SESSION_COOKIE,
    MissingSessionKey,
    create_app,
)
from deskclerk.auth import SigningKeyLookup, discover_jwks_url, jwks_signing_key
from deskclerk.config import Settings
from deskclerk.connections import OAuthConnections
from deskclerk.instructions import system_prompt
from deskclerk.jev import PRICE_PER_INPUT_TOKEN, Jev
from deskclerk.screen import WITHHELD
from deskclerk.servers import McpServer
from deskclerk.tokens import TokenStore
from deskclerk.tracing import AskTrace, Tracing
from deskclerk.writecheck import write_check
from mcp_server import RunningServer, running_mcp_server
from test_agent import ProviderGoneAfterScript, ScriptedModel, answer, calls, script
from test_connections import REDIRECT
from test_screen import FakeJev, as_jev

SESSION_KEY = "a-session-key-of-at-least-32-characters"
AUDIT_KEY = "an-audit-key"
Sign = Callable[[dict[str, object]], str]


@dataclass
class Web:
    client: TestClient
    tokens: TokenStore


@pytest.fixture
def notes_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


@pytest.fixture
def open_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT, oauth=False) as running:
        yield running


def make_settings(token_settings: Settings, **changes: object) -> Settings:
    return token_settings.model_copy(update={"session_key": SecretStr(SESSION_KEY), **changes})


@contextmanager
def web(
    settings: Settings,
    servers: list[McpServer],
    model: BaseChatModel,
    signing_key: SigningKeyLookup,
    path: Path,
    idp_transport: httpx.AsyncBaseTransport | None = None,
    key: str | None = None,
    jev: Jev | None = None,
    tracing: Tracing | None = None,
) -> Iterator[Web]:
    tokens = TokenStore(path / "tokens.sqlite", key or Fernet.generate_key().decode())
    connections = OAuthConnections(tokens, REDIRECT, "deskclerk test", 600)
    app = create_app(
        settings,
        servers,
        connections,
        tokens,
        model,
        write_check(),
        jev,
        signing_key,
        tracing,
        idp_transport=idp_transport,
    )
    with TestClient(app) as client:
        yield Web(client, tokens)


class FakeIdp:
    def __init__(self, issuer: str, access_token: str) -> None:
        self.issuer = issuer
        self.access_token = access_token
        self.discovery_calls = 0
        self.token_calls: list[dict[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/openid-configuration"):
            self.discovery_calls += 1
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": f"{self.issuer}/auth",
                    "token_endpoint": f"{self.issuer}/token",
                },
            )
        assert request.url.path.endswith("/token")
        form = parse_qs(request.content.decode())
        self.token_calls.append({name: values[0] for name, values in form.items()})
        return httpx.Response(200, json={"access_token": self.access_token})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


class SlowModel(ScriptedModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(0.3)
        return super()._generate(*args, **kwargs)


def begin_sign_in(site: Web) -> tuple[str, str, str]:
    start = site.client.get("/login", follow_redirects=False)
    assert start.status_code == 302
    query = parse_qs(urlparse(start.headers["location"]).query)
    pending = cookies.unsign(SESSION_KEY.encode(), "login", site.client.cookies[LOGIN_COOKIE])
    assert pending is not None
    return query["state"][0], pending["verifier"], query["code_challenge"][0]


def notes(running: RunningServer) -> McpServer:
    return McpServer("notes", running.url, None, None)


def open_notes(running: RunningServer) -> McpServer:
    return McpServer("open", running.url, None, None, "none")


def as_user(sign_token: Sign, user: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {sign_token({'preferred_username': user})}"}


def start_connect(site: Web, headers: dict[str, str], name: str = "notes") -> dict[str, str]:
    response = site.client.post(f"/servers/{name}/connect", headers=headers)
    assert response.status_code == 200
    approval = httpx.get(response.json()["authorization_url"])
    query = parse_qs(urlparse(approval.headers["location"]).query)
    return {"state": query["state"][0], "code": query["code"][0]}


def connect(site: Web, headers: dict[str, str]) -> None:
    callback = site.client.get(
        "/oauth/callback", params=start_connect(site, headers), headers=headers,
        follow_redirects=False,
    )  # fmt: skip
    assert (callback.status_code, callback.headers["location"]) == (303, "/")


def ask(site: Web, headers: dict[str, str], request: str = "save a note") -> httpx2.Response:
    return site.client.post("/ask", json={"request": request}, headers=headers)


def audit_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "deskclerk.audit"]


def session_cookie(user: str) -> str:
    payload = {
        "user": user,
        "client": "deskclerk-api",
    }
    return cookies.sign(SESSION_KEY.encode(), "session", payload, 3600)


def test_the_app_needs_a_session_key(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = token_settings.model_copy(update={"session_key": None})
    with pytest.raises(MissingSessionKey):
        with web(settings, [], script(), token_signing_key, tmp_path):
            pass


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/me"),
        ("GET", "/servers"),
        ("POST", "/servers/notes/connect"),
        ("POST", "/servers/notes/disconnect"),
        ("GET", "/oauth/callback?state=x&code=y"),
        ("POST", "/logout"),
    ],
)
def test_requests_without_a_session_or_token_are_refused(
    method: str,
    path: str,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    with web(settings, [], script(), token_signing_key, tmp_path) as site:
        assert site.client.request(method, path).status_code == 401
        assert site.client.post("/ask", json={"request": "hi"}).status_code == 401


def test_a_bad_bearer_token_is_refused_and_a_good_one_names_the_employee(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    with web(settings, [], script(), token_signing_key, tmp_path) as site:
        for authorization in ["Bearer garbage", "Basic YWRhOnB3", "Bearer"]:
            refused = site.client.get("/me", headers={"Authorization": authorization})
            assert refused.status_code == 401
        other_service = sign_token({"aud": "another-service"})
        refused = site.client.get("/me", headers={"Authorization": f"Bearer {other_service}"})
        assert refused.status_code == 401
        accepted = site.client.get("/me", headers=as_user(sign_token, "ada"))
        assert accepted.json() == {"user": "ada"}


def test_a_callback_without_the_sign_in_cookie_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    with web(settings, [], script(), token_signing_key, tmp_path) as site:
        response = site.client.get("/login/callback?code=x&state=y", follow_redirects=False)
        assert response.status_code == 400


def test_connecting_is_finished_by_the_employee_who_started_it(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(settings, [notes(notes_server)], script(), token_signing_key, tmp_path) as site:
        started = start_connect(site, ada)
        stolen = site.client.get(
            "/oauth/callback", params=started, headers=grace, follow_redirects=False
        )
        assert stolen.status_code == 403
        assert site.tokens.get("grace", "notes", "tokens") is None

        finished = site.client.get(
            "/oauth/callback", params=started, headers=ada, follow_redirects=False
        )
        assert finished.status_code == 303
        assert site.tokens.get("ada", "notes", "tokens") is not None

        again = site.client.get(
            "/oauth/callback", params=started, headers=ada, follow_redirects=False
        )
        assert again.status_code == 400


def test_connect_errors_are_clear(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    open_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    servers = [notes(notes_server), open_notes(open_server)]
    ada = as_user(sign_token, "ada")
    with web(settings, servers, script(), token_signing_key, tmp_path) as site:
        assert site.client.post("/servers/nowhere/connect", headers=ada).status_code == 404
        assert site.client.post("/servers/open/connect", headers=ada).status_code == 400
        connect(site, ada)
        assert site.client.post("/servers/notes/connect", headers=ada).status_code == 409


def test_servers_show_whether_this_employee_is_connected(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    open_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    servers = [notes(notes_server), open_notes(open_server)]
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")

    def connected(headers: dict[str, str]) -> dict[str, bool]:
        listing = site.client.get("/servers", headers=headers).json()
        return {entry["name"]: entry["connected"] for entry in listing}

    with web(settings, servers, script(), token_signing_key, tmp_path) as site:
        assert connected(ada) == {"notes": False, "open": True}
        connect(site, ada)
        assert connected(ada) == {"notes": True, "open": True}
        assert connected(grace) == {"notes": False, "open": True}

        assert site.client.post("/servers/notes/disconnect", headers=ada).status_code == 204
        assert connected(ada) == {"notes": False, "open": True}


def test_an_ask_runs_the_agent_and_the_server_sees_the_write_under_that_employee_s_token(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    model = script(
        calls(("notes__write_note", {"text": "hi"})), answer("saved"), answer("nothing to do")
    )
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(settings, [notes(notes_server)], model, token_signing_key, tmp_path) as site:
        connect(site, ada)
        response = ask(site, ada)
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "saved"
        assert [(c["server"], c["tool"], c["result"], c["is_error"]) for c in body["executed"]] == [
            ("notes", "write_note", "wrote hi", False)
        ]
        assert (body["blocked"], body["not_connected"], body["step_limit_hit"]) == ([], [], False)

        stored = json.loads(site.tokens.get("ada", "notes", "tokens") or "")
        assert notes_server.notes == [(stored["token"]["access_token"], "hi")]

        other = ask(site, grace).json()
        assert other["not_connected"] == ["notes"]
        assert len(notes_server.notes) == 1


def test_an_ask_gives_the_model_the_instructions_with_today_s_utc_date(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    model = script(answer("ok"))
    ada = as_user(sign_token, "ada")
    days_around_the_call = {datetime.now(UTC).date()}
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        assert ask(site, ada).status_code == 200
    days_around_the_call.add(datetime.now(UTC).date())
    sent = model.seen[0][0]
    assert isinstance(sent, SystemMessage)
    assert sent.content in {system_prompt(day) for day in days_around_the_call}


class RecordingTracing(Tracing):
    def __init__(self) -> None:
        super().__init__(None, "test", True, None)
        self.users: list[str] = []
        self.withheld: list[int] = []

    @contextmanager
    def ask(self, user_id: str) -> Iterator[AskTrace]:
        self.users.append(user_id)
        trace = AskTrace({}, SimpleNamespace(update=self._record), None)
        yield trace

    def _record(self, metadata: dict[str, int]) -> None:
        self.withheld.append(metadata["withheld"])


def test_an_ask_is_traced_under_the_employee_with_the_withheld_count(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    tracing = RecordingTracing()
    ada = as_user(sign_token, "ada")
    with web(
        settings, [], script(answer("ok")), token_signing_key, tmp_path, tracing=tracing
    ) as site:
        assert ask(site, ada).status_code == 200
    assert (tracing.users, tracing.withheld) == (["ada"], [0])


def test_an_employee_is_told_when_the_reply_was_cut_off_at_the_output_limit(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    cut = AIMessage(content="The meeting is at", response_metadata={"stop_reason": "max_tokens"})
    ada = as_user(sign_token, "ada")
    with web(settings, [], script(cut), token_signing_key, tmp_path) as site:
        body = ask(site, ada).json()
        page = site.client.get("/").text
    assert (body["answer"], body["cut_off"]) == ("The meeting is at", True)
    assert "cut off at the output limit" in page


def test_asks_beyond_the_per_minute_quota_get_429_for_that_employee_only(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings, requests_per_minute=2)
    model = ScriptedModel(messages=itertools.repeat(answer("ok")))
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        assert [ask(site, ada).status_code for _ in range(2)] == [200, 200]
        refused = ask(site, ada)
        assert refused.status_code == 429
        assert refused.headers["retry-after"] == "60"
        assert ask(site, grace).status_code == 200


def test_an_employee_who_spent_the_cap_is_stopped_with_402_and_others_are_not(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings, budget_usd=0.05)
    million_tokens = UsageMetadata(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000)
    model = ScriptedModel(messages=itertools.repeat(answer("ok", usage=million_tokens)))
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        first = ask(site, ada)
        assert first.status_code == 200
        assert first.json()["cost_usd"] == pytest.approx(0.10)
        stopped = ask(site, ada)
        assert stopped.status_code == 402
        assert "DESKCLERK_BUDGET_USD" in stopped.json()["detail"]
        assert ask(site, grace).status_code == 200


def test_tokens_billed_before_a_provider_failure_count_against_the_cap(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, budget_usd=0.05)
    million_tokens = UsageMetadata(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000)
    model = ProviderGoneAfterScript(
        messages=iter([calls(("notes__read_note", {}), usage=million_tokens)])
    )
    ada = as_user(sign_token, "ada")
    with web(settings, [notes(notes_server)], model, token_signing_key, tmp_path) as site:
        connect(site, ada)
        failed = ask(site, ada)
        stopped = ask(site, ada)
    assert (failed.status_code, stopped.status_code) == (502, 402)
    asks = [line for line in audit_lines(caplog) if line["entry"] == "ask"]
    assert [line["outcome"] for line in asks] == ["upstream_error", "over_budget"]
    assert [line["cost_usd"] for line in asks] == [pytest.approx(0.10), 0.0]


def test_a_withheld_result_is_reported_and_jev_tokens_count_against_the_cap(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, budget_usd=0.04)
    model = ScriptedModel(
        messages=itertools.cycle([calls(("notes__read_note", {})), answer("done")])
    )
    jev = FakeJev(0.9, tokens=1_000_000)
    ada = as_user(sign_token, "ada")
    with web(
        settings, [notes(notes_server)], model, token_signing_key, tmp_path, jev=as_jev(jev)
    ) as site:
        connect(site, ada)
        first = ask(site, ada)
        stopped = ask(site, ada)
    body = first.json()
    assert body["executed"][0]["result"] == WITHHELD
    assert body["withheld"] == 1
    assert body["cost_usd"] == pytest.approx(1_000_000 * PRICE_PER_INPUT_TOKEN)
    assert stopped.status_code == 402
    [line] = [line for line in audit_lines(caplog) if line.get("outcome") == "answered"]
    assert line["withheld"] == 1


def test_a_write_that_fails_the_write_check_never_reaches_the_server(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    model = script(calls(("notes__write_note", {})), answer("I could not save it"))
    ada = as_user(sign_token, "ada")
    with web(settings, [notes(notes_server)], model, token_signing_key, tmp_path) as site:
        connect(site, ada)
        body = ask(site, ada).json()
    assert notes_server.notes == []
    assert body["executed"] == []
    [blocked] = body["blocked"]
    assert blocked["tool"] == "write_note"
    assert blocked["reason"].startswith("schema check:")


def test_a_request_must_be_between_1_and_4000_characters(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    model = ScriptedModel(messages=itertools.repeat(answer("ok")))
    ada = as_user(sign_token, "ada")
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        assert ask(site, ada, "").status_code == 422
        assert ask(site, ada, "x" * 4001).status_code == 422
        assert ask(site, ada, "x" * 4000).status_code == 200


def test_the_audit_log_holds_no_argument_values_unless_audit_text_is_set(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, audit_key=SecretStr(AUDIT_KEY))
    model = script(calls(("notes__write_note", {"text": "secret-text"})), answer("saved"))
    ada = as_user(sign_token, "ada")
    with web(settings, [notes(notes_server)], model, token_signing_key, tmp_path) as site:
        connect(site, ada)
        ask(site, ada, "my private request")
        site.client.post("/servers/notes/disconnect", headers=ada)

    lines = audit_lines(caplog)
    assert [line["entry"] for line in lines] == ["connect", "ask", "disconnect"]
    assert lines[0]["server"] == lines[2]["server"] == "notes"
    line = lines[1]
    assert line["user"] == "ada"
    assert line["executed"] == [{"server": "notes", "tool": "write_note", "is_error": False}]
    expected = hmac.new(AUDIT_KEY.encode(), b"my private request", hashlib.sha256).hexdigest()
    assert (line["text_chars"], line["text_hmac"]) == (18, expected)
    assert "text" not in line
    assert "secret-text" not in caplog.text
    assert "my private request" not in caplog.text


def test_the_audit_log_holds_arguments_and_results_when_audit_text_is_set(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, audit_text=True)
    model = script(calls(("notes__write_note", {"text": "secret-text"})), answer("saved"))
    ada = as_user(sign_token, "ada")
    with web(settings, [notes(notes_server)], model, token_signing_key, tmp_path) as site:
        connect(site, ada)
        ask(site, ada, "my private request")

    line = audit_lines(caplog)[1]
    assert line["text"] == "my private request"
    assert line["executed"] == [
        {
            "server": "notes",
            "tool": "write_note",
            "is_error": False,
            "arguments": {"text": "secret-text"},
            "result": "wrote secret-text",
        }
    ]


def test_a_refused_ask_is_audited_with_its_outcome(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings, requests_per_minute=1)
    model = ScriptedModel(messages=itertools.repeat(answer("ok")))
    ada = as_user(sign_token, "ada")
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        ask(site, ada)
        ask(site, ada)
    assert [line["outcome"] for line in audit_lines(caplog)] == ["answered", "rate_limited"]


def test_a_cookie_post_needs_the_custom_header_and_a_bearer_post_does_not(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    model = ScriptedModel(messages=itertools.repeat(answer("ok")))
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        site.client.cookies.set(SESSION_COOKIE, session_cookie("ada"))
        assert site.client.get("/me").json() == {"user": "ada"}
        assert ask(site, {}).status_code == 403
        assert ask(site, {CSRF_HEADER: "something else"}).status_code == 403
        assert ask(site, {CSRF_HEADER: CSRF_VALUE}).status_code == 200
        assert site.client.post("/logout").status_code == 403

        site.client.cookies.clear()
        assert ask(site, as_user(sign_token, "ada")).status_code == 200


def test_a_forged_expired_or_misplaced_cookie_is_not_a_session(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    key = SESSION_KEY.encode()
    payload = {"user": "ada", "client": "c"}
    refused = [
        session_cookie("ada")[:-2] + "xx",
        cookies.sign(b"another key", "session", payload, 3600),
        cookies.sign(key, "session", payload, -10),
        cookies.sign(key, "login", payload, 3600),
        "",
    ]
    with web(settings, [], script(), token_signing_key, tmp_path) as site:
        for value in refused:
            site.client.cookies.set(SESSION_COOKIE, value)
            assert site.client.get("/me").status_code == 401
            site.client.cookies.clear()


def test_the_page_is_served_with_a_policy_that_allows_only_its_own_script(
    token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    with web(settings, [], script(), token_signing_key, tmp_path) as site:
        response = site.client.get("/")
    assert response.status_code == 200
    script_block = re.search(r"<script>(.*?)</script>", response.text, re.DOTALL)
    assert script_block is not None
    digest = hashlib.sha256(script_block.group(1).encode()).digest()
    policy = response.headers["content-security-policy"]
    assert f"script-src 'sha256-{base64.b64encode(digest).decode()}'" in policy
    assert "src=" not in response.text
    assert "innerHTML" not in response.text


@pytest.mark.keycloak
def test_an_employee_signs_in_through_keycloak_and_asks(tmp_path: Path) -> None:
    settings = Settings(session_key=SecretStr(SESSION_KEY))
    with httpx.Client() as discovery:
        signing_key = jwks_signing_key(discover_jwks_url(settings, discovery))
    model = script(answer("hello analyst"))
    with web(settings, [], model, signing_key, tmp_path) as site:
        site.client.follow_redirects = False
        start = site.client.get("/login")
        assert start.status_code == 302
        authorization_url = start.headers["location"]
        query = parse_qs(urlparse(authorization_url).query)
        assert query["code_challenge_method"] == ["S256"]

        with httpx.Client() as browser:
            login_page = browser.get(authorization_url)
            # Browsers send Keycloak's Secure cookies to localhost over http; httpx does not.
            for cookie in browser.cookies.jar:
                cookie.secure = False
            form = re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', login_page.text)
            assert form is not None
            submitted = browser.post(
                html.unescape(form.group(1)),
                data={"username": "analyst", "password": "deskclerk"},
            )
        assert submitted.status_code == 302
        landing = urlparse(submitted.headers["location"])
        assert f"{landing.scheme}://{landing.netloc}{landing.path}" == settings.sign_in_redirect_url

        forged = site.client.get(f"{landing.path}?code=x&state=forged")
        assert forged.status_code == 400
        assert site.client.get("/me").status_code == 401

        finished = site.client.get(f"{landing.path}?{landing.query}")
        assert (finished.status_code, finished.headers["location"]) == (303, "/")
        assert site.client.get("/me").json() == {"user": "analyst"}
        asked = ask(site, {CSRF_HEADER: CSRF_VALUE})
        assert asked.json()["answer"] == "hello analyst"


def test_a_wrong_state_with_a_valid_login_cookie_never_reaches_the_token_endpoint(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    idp = FakeIdp(settings.issuer, sign_token({}))
    with web(settings, [], script(), token_signing_key, tmp_path, idp.transport) as site:
        begin_sign_in(site)
        forged = site.client.get("/login/callback?code=c&state=forged", follow_redirects=False)
        assert forged.status_code == 400
        assert idp.token_calls == []
        assert site.client.get("/me").status_code == 401


def test_the_code_verifier_sent_to_the_token_endpoint_is_the_one_in_the_login_cookie(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    idp = FakeIdp(settings.issuer, sign_token({}))
    with web(settings, [], script(), token_signing_key, tmp_path, idp.transport) as site:
        state, verifier, challenge = begin_sign_in(site)
        done = site.client.get(f"/login/callback?code=c&state={state}", follow_redirects=False)
        assert done.status_code == 303
    assert len(idp.token_calls) == 1
    sent = idp.token_calls[0]
    assert (sent["code"], sent["code_verifier"]) == ("c", verifier)
    digest = hashlib.sha256(verifier.encode()).digest()
    assert challenge == base64.urlsafe_b64encode(digest).decode().rstrip("=")


def test_the_session_ends_when_the_access_token_does(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(token_settings, session_hours=8)
    idp = FakeIdp(settings.issuer, sign_token({"exp": int(time.time()) + 5}))
    with web(settings, [], script(), token_signing_key, tmp_path, idp.transport) as site:
        state, _, _ = begin_sign_in(site)
        site.client.get(f"/login/callback?code=c&state={state}", follow_redirects=False)
        assert site.client.get("/me").json() == {"user": "district-1"}
        now = time.time()
        monkeypatch.setattr(cookies, "time", SimpleNamespace(time=lambda: now + 10))
        assert site.client.get("/me").status_code == 401


def test_a_refused_access_token_gives_a_fixed_message_on_the_browser_path(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    idp = FakeIdp(settings.issuer, sign_token({"aud": "another-service"}))
    with web(settings, [], script(), token_signing_key, tmp_path, idp.transport) as site:
        state, _, _ = begin_sign_in(site)
        refused = site.client.get(f"/login/callback?code=c&state={state}", follow_redirects=False)
    assert (refused.status_code, refused.json()["detail"]) == (401, BROWSER_AUTH_REFUSED)


def test_the_discovery_document_is_fetched_once_per_process(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings)
    idp = FakeIdp(settings.issuer, sign_token({}))
    with web(settings, [], script(), token_signing_key, tmp_path, idp.transport) as site:
        state, _, _ = begin_sign_in(site)
        begin_sign_in(site)
        site.client.get(f"/login/callback?code=c&state={state}", follow_redirects=False)
    assert idp.discovery_calls == 1


def test_one_ask_at_a_time_per_employee_so_the_cap_cannot_be_raced(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup, tmp_path: Path
) -> None:
    settings = make_settings(token_settings, budget_usd=0.05)
    million_tokens = UsageMetadata(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000)
    model = SlowModel(messages=itertools.repeat(answer("ok", usage=million_tokens)))
    ada = as_user(sign_token, "ada")
    statuses: list[int] = []
    with web(settings, [], model, token_signing_key, tmp_path) as site:
        threads = [
            threading.Thread(target=lambda: statuses.append(ask(site, ada).status_code))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert sorted(statuses) == [200, 402]


def test_an_mcp_server_that_stopped_gives_502_and_an_upstream_error_line(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    key = Fernet.generate_key().decode()
    with running_mcp_server(REDIRECT) as running:
        with web(
            settings, [notes(running)], script(), token_signing_key, tmp_path, key=key
        ) as site:
            connect(site, ada)
        stopped = notes(running)
    model = script(answer("x"))
    with web(settings, [stopped], model, token_signing_key, tmp_path, key=key) as site:
        response = ask(site, ada)
    assert response.status_code == 502
    outcomes = [line["outcome"] for line in audit_lines(caplog) if line["entry"] == "ask"]
    assert outcomes == ["upstream_error"]


def test_a_server_that_refuses_the_sign_in_cancels_the_pending_connection(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    with web(settings, [notes(notes_server)], script(), token_signing_key, tmp_path) as site:
        started = start_connect(site, ada)
        refused = site.client.get(
            "/oauth/callback",
            params={"state": started["state"], "error": "access_denied"},
            headers=ada,
            follow_redirects=False,
        )
        assert refused.status_code == 400
        late = site.client.get(
            "/oauth/callback", params=started, headers=ada, follow_redirects=False
        )
        assert late.status_code == 400
        assert site.tokens.get("ada", "notes", "tokens") is None


def test_disconnecting_cancels_a_pending_connection_to_that_server(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    tmp_path: Path,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    with web(settings, [notes(notes_server)], script(), token_signing_key, tmp_path) as site:
        started = start_connect(site, ada)
        assert site.client.post("/servers/notes/disconnect", headers=ada).status_code == 204
        late = site.client.get(
            "/oauth/callback", params=started, headers=ada, follow_redirects=False
        )
        assert late.status_code == 400
        assert site.tokens.get("ada", "notes", "tokens") is None
