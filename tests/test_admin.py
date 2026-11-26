# The admin page and its summary exist only when DESKCLERK_ADMIN_GROUP is set, open only to members
# of that group, show servers, this month's usage and the latest conversations from stored data.
# The main page learns from /me whether to show its Admin link. The routes that change servers and
# revoke an employee are in test_admin_servers.py. Employees are stored by display name when they
# sign in or call with a token, the admin pages show the name, and revoking deletes every stored
# secret of the employee.

import json
import re
from datetime import UTC, datetime
from importlib import resources

import pytest

from conftest import CHAIN_KEY
from deskclerk.auditstore import AuditStore
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from deskclerk.providers import Usage
from deskclerk.routes.health import content_security_policy
from deskclerk.servers import McpServer
from deskclerk.spending import SpendingStore, month_of
from deskclerk.tokens import Kind
from test_agent import answer, script
from test_api import (
    FakeIdp,
    Sign,
    begin_sign_in,
    finish_sign_in,
    make_settings,
    web,
)

GROUP = "it-admins"


def bearer(sign_token: Sign, user: str, groups: list[str]) -> dict[str, str]:
    claims: dict[str, object] = {"sub": user, "preferred_username": user, "groups": groups}
    return {"Authorization": f"Bearer {sign_token(claims)}"}


def add_conversation(database: Database, user: str, title: str, minutes_ago: int) -> None:
    with database.pool.connection() as connection:
        connection.execute(
            "INSERT INTO conversations (user_id, id, title, updated_at) "
            "VALUES (%s, gen_random_uuid(), %s, now() - make_interval(mins => %s))",
            (user, title, minutes_ago),
        )


def test_without_the_setting_the_admin_page_and_summary_do_not_exist(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=None)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        member = bearer(sign_token, "ada", [GROUP])
        for path in ("/admin", "/admin/summary"):
            assert site.client.get(path, headers=member).status_code == 404
            assert site.client.get(path).status_code == 404
        assert site.client.get("/me", headers=member).json()["admin"] is False


def test_a_signed_in_employee_outside_the_group_is_refused(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        for groups in ([], ["finance"]):
            outsider = bearer(sign_token, "bob", groups)
            assert site.client.get("/admin", headers=outsider).status_code == 403
            assert site.client.get("/admin/summary", headers=outsider).status_code == 403
            assert site.client.get("/me", headers=outsider).json()["admin"] is False


def test_someone_not_signed_in_is_sent_to_sign_in(
    token_settings: Settings, token_signing_key: SigningKeyLookup, database: Database
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        page = site.client.get("/admin", follow_redirects=False)
        assert page.status_code == 302
        assert page.headers["location"] == "/login?next=/admin"
        assert site.client.get("/admin/summary").status_code == 401


def test_a_member_sees_the_servers_usage_and_latest_conversations_that_are_stored(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP, budget_usd=3.0, audit_text=True)
    servers = [
        McpServer("notes", "http://notes.test/mcp", None, None, "oauth", tools=("add", "list")),
        McpServer("open", "http://open.test/mcp", None, None, "none"),
    ]
    month = month_of(datetime.now(UTC))
    store = SpendingStore(database)
    store.add("ana", month, "model", Usage(1, 1, 0, 0), 0.25)
    store.add("ana", month, "jev", Usage(10, 0, 0, 0), 0.5)
    store.add("bob", month, "model", Usage(1, 1, 0, 0), 1.0)
    store.add("bob", "2020-01", "model", Usage(1, 1, 0, 0), 9.0)
    with web(settings, servers, script(answer("ok")), token_signing_key, database) as site:
        site.tokens.put("ana", "notes", "tokens", "t")
        site.tokens.put("bob", "notes", "tokens", "t")
        site.tokens.put("bob", "notes", "client", "c")
        add_conversation(database, "bob", "older question", 10)
        add_conversation(database, "ana", "newer question", 1)
        admin = bearer(sign_token, "root", [GROUP])

        page = site.client.get("/admin", headers=admin)
        summary = site.client.get("/admin/summary", headers=admin)

    assert page.status_code == 200
    assert "deskclerk admin" in page.text
    assert summary.status_code == 200
    body = summary.json()
    assert body["month"] == month
    assert body["servers"] == [
        {
            "name": "notes",
            "url": "http://notes.test/mcp",
            "auth": "oauth",
            "has_headers": False,
            "trusted": False,
            "paused": False,
            "employees_connected": 2,
            "allowlist_size": 2,
        },
        {
            "name": "open",
            "url": "http://open.test/mcp",
            "auth": "none",
            "has_headers": False,
            "trusted": False,
            "paused": False,
            "employees_connected": None,
            "allowlist_size": None,
        },
    ]
    assert body["usage"] == [
        {"user_id": "bob", "name": None, "calls": 1, "spent_usd": 1.0, "budget_usd": 3.0},
        {"user_id": "ana", "name": None, "calls": 2, "spent_usd": 0.75, "budget_usd": 3.0},
    ]
    assert [(row["user_id"], row["title"]) for row in body["recent"]] == [
        ("ana", "newer question"),
        ("bob", "older question"),
    ]


def test_recent_activity_is_the_latest_fifty(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP, audit_text=True)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        for number in range(55):
            add_conversation(database, "ana", f"question {number}", number)
        recent = site.client.get("/admin/summary", headers=bearer(sign_token, "r", [GROUP]))
    titles = [row["title"] for row in recent.json()["recent"]]
    assert len(titles) == 50
    assert titles[0] == "question 0"


def test_the_page_and_the_summary_accept_only_get(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            for path in ("/admin", "/admin/summary"):
                assert site.client.request(method, path, headers=admin).status_code == 405


def test_the_me_flag_is_true_only_for_members(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        member = site.client.get("/me", headers=bearer(sign_token, "root", ["a", GROUP]))
        outsider = site.client.get("/me", headers=bearer(sign_token, "bob", ["a"]))
    assert member.json()["admin"] is True
    assert outsider.json()["admin"] is False


def test_the_admin_page_has_its_own_hash_policy_and_no_inline_handlers(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        page = site.client.get("/admin", headers=bearer(sign_token, "root", [GROUP]))
    assert page.headers["content-security-policy"] == content_security_policy(page.text, {})
    assert re.search(r"\s(on[a-z]+|style)\s*=", page.text) is None
    assert "default-src 'none'" in page.headers["content-security-policy"]


@pytest.mark.parametrize(("audit_text", "shown"), [(False, None), (True, "secret plan")])
def test_titles_are_shown_only_when_request_text_is_allowed_in_the_audit(
    audit_text: bool,
    shown: str | None,
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP, audit_text=audit_text)
    with web(settings, [], script(answer("ok")), token_signing_key, database) as site:
        add_conversation(database, "ana", "secret plan", 1)
        recent = site.client.get("/admin/summary", headers=bearer(sign_token, "r", [GROUP]))
    assert [(row["user_id"], row["title"]) for row in recent.json()["recent"]] == [("ana", shown)]
    assert audit_text or "secret plan" not in recent.text


def admin_page() -> str:
    return resources.files("deskclerk").joinpath("static/admin.html").read_text(encoding="utf-8")


def test_the_flagged_filter_checkbox_sits_inline_before_its_label() -> None:
    page = admin_page()
    assert '<label class="check"><input id="audit-flagged" type="checkbox">' in page
    assert ".filters label.check { display: flex; align-items: center;" in page


def test_each_audit_row_shows_whether_a_result_was_flagged() -> None:
    page = admin_page()
    assert "<th>Outcome</th><th>Flagged</th><th>Raw line</th>" in page
    assert "flagCell(r.flagged)" in page
    assert 'fill("audit", rows, 8)' in page


def named(sign_token: Sign, user_id: str, name: str) -> dict[str, str]:
    claims: dict[str, object] = {"sub": user_id, "preferred_username": name, "groups": [GROUP]}
    return {"Authorization": f"Bearer {sign_token(claims)}"}


def stored_employees(database: Database) -> list[tuple[str, str]]:
    with database.pool.connection() as connection:
        found = connection.execute(
            "SELECT user_id, display_name FROM employees ORDER BY user_id"
        ).fetchall()
    return [(str(user), str(name)) for user, name in found]


def test_browser_sign_in_and_a_bearer_call_store_the_display_name(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    idp = FakeIdp(settings.issuer)
    with web(settings, [], script(), token_signing_key, database, idp.transport) as site:
        state, _, _ = begin_sign_in(site)
        finish_sign_in(
            site, idp, sign_token, state, {"sub": "u-1", "preferred_username": "Ada Lovelace"}
        )
        site.client.cookies.clear()
        site.client.get("/me", headers=named(sign_token, "u-2", "Grace Hopper"))
    assert stored_employees(database) == [("u-1", "Ada Lovelace"), ("u-2", "Grace Hopper")]


def test_a_bearer_caller_is_written_once_and_again_only_when_the_name_changes(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    def last_seen() -> object:
        with database.pool.connection() as connection:
            row = connection.execute("SELECT last_seen FROM employees").fetchone()
        return None if row is None else row[0]

    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(), token_signing_key, database) as site:
        site.client.get("/me", headers=named(sign_token, "u-1", "Ada"))
        first = last_seen()
        for _ in range(3):
            site.client.get("/me", headers=named(sign_token, "u-1", "Ada"))
        assert last_seen() == first
        site.client.get("/me", headers=named(sign_token, "u-1", "Ada Lovelace"))
        assert last_seen() != first
    assert stored_employees(database) == [("u-1", "Ada Lovelace")]


def test_the_summary_and_the_audit_search_give_each_known_employee_a_name(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP, audit_text=True)
    month = month_of(datetime.now(UTC))
    SpendingStore(database).add("u-1", month, "model", Usage(1, 1, 0, 0), 0.25)
    SpendingStore(database).add("ghost", month, "model", Usage(1, 1, 0, 0), 0.1)
    add_conversation(database, "u-2", "a question", 1)
    chain = AuditStore(database, CHAIN_KEY)
    for user in ("u-1", "ghost"):
        fields = {"time": datetime.now(UTC).isoformat(), "user": user, "entry": "ask"}
        chain.append(json.dumps(fields))
    with web(settings, [], script(), token_signing_key, database) as site:
        site.client.get("/me", headers=named(sign_token, "u-1", "Ada Lovelace"))
        admin = named(sign_token, "u-2", "Grace Hopper")
        summary = site.client.get("/admin/summary", headers=admin).json()
        audit = site.client.get("/admin/audit", headers=admin).json()
        download = site.client.get("/admin/audit/download", headers=admin)

    assert {row["user_id"]: row["name"] for row in summary["usage"]} == {
        "u-1": "Ada Lovelace",
        "ghost": None,
    }
    assert [(row["user_id"], row["name"]) for row in summary["recent"]] == [("u-2", "Grace Hopper")]
    assert summary["employees"] == [
        {"user_id": "u-1", "display_name": "Ada Lovelace"},
        {"user_id": "u-2", "display_name": "Grace Hopper"},
    ]
    assert {row["user"]: row["name"] for row in audit["rows"]} == {
        "u-1": "Ada Lovelace",
        "ghost": None,
    }
    assert all("name" not in row for row in map(json.loads, download.text.splitlines()))


def test_revoking_an_employee_deletes_every_kind_of_stored_secret_and_keeps_conversations(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    kinds: tuple[Kind, ...] = ("tokens", "client", "tool_pins", "always_allowed", "tool_menu")
    with web(settings, [], script(), token_signing_key, database) as site:
        for kind in kinds:
            site.tokens.put("ana", "notes", kind, "secret")
            site.tokens.put("ana", "wiki", kind, "secret")
            site.tokens.put("bob", "notes", kind, "secret")
        add_conversation(database, "ana", "kept", 1)
        revoked = site.client.post(
            "/admin/employees/revoke",
            json={"user_id": "ana"},
            headers=bearer(sign_token, "root", [GROUP]),
        )
        with database.pool.connection() as connection:
            left = connection.execute("SELECT DISTINCT user_id FROM secrets").fetchall()
            conversations = connection.execute("SELECT COUNT(*) FROM conversations").fetchone()
    assert revoked.json()["secrets"] == 10
    assert left == [("bob",)]
    assert conversations == (1,)
