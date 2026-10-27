# A server that lists groups is hidden, refused and never offered to an employee outside them, for
# a bearer caller and for a browser session alike.

import logging
from collections.abc import Callable
from typing import Any

import pytest

from deskclerk.api import CSRF_HEADER, CSRF_VALUE, SESSION_COOKIE
from deskclerk.auth import Identity, SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from deskclerk.servers import McpServer
from mcp_server import RunningServer
from test_agent import answer, script
from test_api import (
    FakeIdp,
    Sign,
    Web,
    ask,
    begin_sign_in,
    finish_sign_in,
    make_settings,
    notes_server,  # noqa: F401
    open_server,  # noqa: F401
    web,
)

Caller = Callable[[Web, list[str] | None], dict[str, str]]


def servers_for(open_running: RunningServer, signing_in: RunningServer) -> list[McpServer]:
    finance = ("finance",)
    return [
        McpServer("public", open_running.url, None, None, "none"),
        McpServer("ledger", open_running.url, None, None, "none", groups=finance),
        McpServer("vault", signing_in.url, None, None, "oauth", groups=finance),
    ]


@pytest.fixture(params=["bearer", "cookie"])
def caller(request: pytest.FixtureRequest, sign_token: Sign) -> Caller:
    def make(site: Web, groups: list[str] | None) -> dict[str, str]:
        if request.param == "bearer":
            claims: dict[str, object] = {"groups": groups}
            return {"Authorization": f"Bearer {sign_token(claims)}"}
        identity = Identity("ada", "deskclerk-api", "ada", groups=frozenset(groups or []))
        site.client.cookies.set(SESSION_COOKIE, site.sessions.create(identity, 3600))
        return {CSRF_HEADER: CSRF_VALUE}

    return make


def offered(model: Any) -> set[str]:
    return {tool["name"] if isinstance(tool, dict) else tool.name for tool in model.bound}


def names(site: Web, headers: dict[str, str]) -> list[str]:
    return [entry["name"] for entry in site.client.get("/servers", headers=headers).json()]


def test_a_server_outside_the_employee_s_groups_is_hidden_refused_and_never_offered(
    caller: Caller,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    model = script(answer("ok"))
    settings = make_settings(token_settings)
    with web(
        settings, servers_for(open_server, notes_server), model, token_signing_key, database
    ) as site:
        outsider = caller(site, ["hr"])
        assert names(site, outsider) == ["public"]
        for action in ("connect", "disconnect"):
            response = site.client.post(f"/servers/vault/{action}", headers=outsider)
            assert response.status_code == 404
        assert ask(site, outsider).status_code == 200
        assert {name.split("__")[0] for name in offered(model)} == {"public"}


def test_a_server_is_offered_to_a_member_of_one_of_its_groups(
    caller: Caller,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    model = script(answer("ok"))
    settings = make_settings(token_settings)
    with web(
        settings, servers_for(open_server, notes_server), model, token_signing_key, database
    ) as site:
        member = caller(site, ["hr", "finance"])
        assert names(site, member) == ["public", "ledger", "vault"]
        assert site.client.post("/servers/vault/connect", headers=member).status_code == 200
        assert ask(site, member).status_code == 200
        assert {name.split("__")[0] for name in offered(model)} == {"public", "ledger"}


def test_an_employee_with_no_groups_claim_sees_only_the_servers_without_groups(
    caller: Caller,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    with web(
        settings, servers_for(open_server, notes_server), script(), token_signing_key, database
    ) as site:
        assert names(site, caller(site, None)) == ["public"]


def test_the_claim_that_lists_the_groups_is_set_by_the_groups_claim_setting(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings, groups_claim="roles")
    claims: dict[str, object] = {"roles": ["finance"], "groups": None}
    headers = {"Authorization": f"Bearer {sign_token(claims)}"}
    with web(
        settings, servers_for(open_server, notes_server), script(), token_signing_key, database
    ) as site:
        assert names(site, headers) == ["public", "ledger", "vault"]


def overage_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "overage" in r.getMessage()]


def test_a_bearer_token_with_group_overage_gives_no_groups_and_one_warning(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="deskclerk.auth")
    settings = make_settings(token_settings)
    claims: dict[str, object] = {"_claim_names": {"groups": "src1"}}
    headers = {"Authorization": f"Bearer {sign_token(claims)}"}
    with web(
        settings, servers_for(open_server, notes_server), script(), token_signing_key, database
    ) as site:
        assert names(site, headers) == ["public"]
    assert len(overage_warnings(caplog)) == 1


def test_a_browser_sign_in_with_group_overage_gives_no_groups_and_one_warning(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="deskclerk.auth")
    settings = make_settings(token_settings)
    idp = FakeIdp(settings.issuer)
    servers = servers_for(open_server, notes_server)
    with web(settings, servers, script(), token_signing_key, database, idp.transport) as site:
        state, _, _ = begin_sign_in(site)
        claims: dict[str, object] = {"_claim_names": {"groups": "src1"}}
        finish_sign_in(site, idp, sign_token, state, claims)
        assert names(site, {}) == ["public"]
    assert len(overage_warnings(caplog)) == 1


def test_the_groups_of_a_browser_sign_in_are_kept_with_its_session(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    open_server: RunningServer,  # noqa: F811
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    idp = FakeIdp(settings.issuer)
    servers = servers_for(open_server, notes_server)
    with web(settings, servers, script(), token_signing_key, database, idp.transport) as site:
        state, _, _ = begin_sign_in(site)
        finish_sign_in(site, idp, sign_token, state, {"groups": ["finance"]})
        assert names(site, {}) == ["public", "ledger", "vault"]
