# The admin routes that change things: members of the admin group add, edit, pause, resume and
# remove a server and revoke an employee, each change is audited, a non-member is refused, a
# browser session must send the request header, and an invalid entry is refused with the entry
# check's message.

from typing import Any

import pytest

from deskclerk.api import CSRF_HEADER, CSRF_VALUE, SESSION_COOKIE
from deskclerk.auth import Identity, SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from test_admin import GROUP, bearer
from test_agent import answer, script
from test_api import Sign, as_user, audit_lines, make_settings, session_cookie, web

WIKI = {"name": "wiki", "url": "http://127.0.0.1:9/mcp", "auth": "none"}
WRITES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("POST", "/admin/servers", WIKI),
    ("PUT", "/admin/servers/wiki", WIKI),
    ("POST", "/admin/servers/wiki/pause", None),
    ("POST", "/admin/servers/wiki/resume", None),
    ("DELETE", "/admin/servers/wiki", None),
    ("POST", "/admin/employees/revoke", {"user_id": "ana"}),
]


@pytest.mark.parametrize(("method", "path", "body"), WRITES)
def test_a_non_admin_is_refused_every_write(
    method: str,
    path: str,
    body: dict[str, Any] | None,
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        outsider = bearer(sign_token, "bob", ["finance"])
        refused = site.client.request(method, path, json=body, headers=outsider)
        stored = site.catalog.store.entries()
    assert refused.status_code == 403
    assert stored == []


def test_a_browser_session_needs_the_request_header_for_a_write(
    token_settings: Settings, token_signing_key: SigningKeyLookup, database: Database
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        root = Identity("root", "deskclerk-api", "root", groups=frozenset({GROUP}))
        site.client.cookies.set(SESSION_COOKIE, site.sessions.create(root, 3600))
        without = site.client.post("/admin/servers", json=WIKI)
        header = {CSRF_HEADER: CSRF_VALUE}
        with_header = site.client.post("/admin/servers", json=WIKI, headers=header)
    assert (without.status_code, with_header.status_code) == (403, 204)


def test_an_admin_adds_edits_and_removes_a_server_and_each_change_is_audited(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    caplog.set_level("INFO", logger="deskclerk.audit")
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])
        added = site.client.post("/admin/servers", json=WIKI, headers=admin)
        again = site.client.post("/admin/servers", json=WIKI, headers=admin)
        changed = WIKI | {"title": "Wiki", "tools": ["read"]}
        edited = site.client.put("/admin/servers/wiki", json=changed, headers=admin)
        fetched = site.client.get("/admin/servers/wiki", headers=admin)
        summary = site.client.get("/admin/summary", headers=admin)
        removed = site.client.delete("/admin/servers/wiki", headers=admin)
        missing = site.client.delete("/admin/servers/wiki", headers=admin)
        listed = site.catalog.servers
    statuses = [r.status_code for r in (added, again, edited, removed, missing)]
    assert statuses == [204, 409, 204, 204, 404]
    assert fetched.json()["title"] == "Wiki"
    rows = summary.json()["servers"]
    assert [(row["name"], row["allowlist_size"]) for row in rows] == [("wiki", 1)]
    assert listed == ()
    actions = [
        (line["user"], line["action"], line["target"])
        for line in audit_lines(caplog)
        if line["entry"] == "admin"
    ]
    assert actions == [
        ("root", "add_server", "wiki"),
        ("root", "edit_server", "wiki"),
        ("root", "remove_server", "wiki"),
    ]


def test_an_invalid_entry_is_refused_with_the_message_of_the_entry_check(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])
        insecure = {"name": "wiki", "url": "http://wiki.example.com/mcp"}
        unnamed = {"name": "a b", "url": "https://wiki.test/mcp"}
        no_secret = {
            "name": "crm",
            "url": "https://crm.test/mcp",
            "client_id": "c",
            "client_secret_env": "NOT_SET_ANYWHERE",
        }
        refused = [
            site.client.post("/admin/servers", json=entry, headers=admin)
            for entry in (insecure, unnamed, no_secret)
        ]
        renamed = site.client.put("/admin/servers/other", json=WIKI, headers=admin)
        stored = site.catalog.store.entries()
    assert [r.status_code for r in refused] == [422, 422, 422]
    assert "must be https" in refused[0].text
    assert "must use only letters" in refused[1].text
    assert "NOT_SET_ANYWHERE" in refused[2].text
    assert renamed.status_code == 422
    assert stored == []


def test_a_paused_server_is_not_offered_until_it_is_resumed(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])
        ada = as_user(sign_token, "ada")
        site.client.post("/admin/servers", json=WIKI, headers=admin)

        def employee_sees() -> list[str]:
            return [row["name"] for row in site.client.get("/servers", headers=ada).json()]

        before = employee_sees()
        paused = site.client.post("/admin/servers/wiki/pause", headers=admin)
        hidden = employee_sees()
        connect = site.client.post("/servers/wiki/connect", headers=ada)
        offered = [server.name for server in site.catalog.offered]
        row = site.client.get("/admin/summary", headers=admin).json()["servers"][0]
        resumed = site.client.post("/admin/servers/wiki/resume", headers=admin)
        after = employee_sees()
    assert (before, hidden, after) == (["wiki"], [], ["wiki"])
    assert (paused.status_code, resumed.status_code, connect.status_code) == (204, 204, 404)
    assert offered == []
    assert row["paused"] is True


def test_revoking_an_employee_deletes_their_tokens_and_ends_their_sessions(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    caplog.set_level("INFO", logger="deskclerk.audit")
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        site.tokens.put("ana", "notes", "tokens", "t")
        site.tokens.put("bob", "notes", "tokens", "t")
        site.client.cookies.set(SESSION_COOKIE, session_cookie(site, "ana"))
        before = site.client.get("/me")
        revoked = site.client.post(
            "/admin/employees/revoke",
            json={"user_id": "ana"},
            headers=bearer(sign_token, "root", [GROUP]),
        )
        after = site.client.get("/me")
        ana_token = site.tokens.get("ana", "notes", "tokens")
        bob_token = site.tokens.get("bob", "notes", "tokens")
    assert before.status_code == 200
    assert (revoked.status_code, revoked.json()) == (200, {"tokens": 1, "sessions": 1})
    assert after.status_code == 401
    assert (ana_token, bob_token) == (None, "t")
    revoked_lines = [line for line in audit_lines(caplog) if line["entry"] == "admin"]
    assert [(line["action"], line["target"]) for line in revoked_lines] == [
        ("revoke_employee", "ana")
    ]
