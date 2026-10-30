# The admin page and its summary exist only when DESKCLERK_ADMIN_GROUP is set, open only to members
# of that group, show servers, this month's usage and the latest conversations from stored data,
# and change nothing. The main page learns from /me whether to show its Admin link.

import re
from datetime import UTC, datetime

import pytest

from deskclerk.api import _content_security_policy
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from deskclerk.providers import Usage
from deskclerk.servers import McpServer
from deskclerk.spending import SpendingStore, month_of
from test_agent import answer, script
from test_api import Sign, make_settings, web

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
        assert page.headers["location"] == "/login"
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
            "employees_connected": 2,
            "allowlist_size": 2,
        },
        {
            "name": "open",
            "url": "http://open.test/mcp",
            "auth": "none",
            "employees_connected": None,
            "allowlist_size": None,
        },
    ]
    assert body["usage"] == [
        {"user_id": "bob", "calls": 1, "spent_usd": 1.0, "budget_usd": 3.0},
        {"user_id": "ana", "calls": 2, "spent_usd": 0.75, "budget_usd": 3.0},
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


def test_the_admin_page_cannot_change_anything(
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
    assert page.headers["content-security-policy"] == _content_security_policy(page.text, {})
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
