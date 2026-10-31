# Two copies of the app on one Postgres database, as behind a load balancer: they share an
# employee's session, conversation and budget, a second ask of one employee waits for the first,
# the request quota counts both, an OAuth callback that lands on the other copy still completes
# the flow, a server an admin changes on one copy reaches the other without a restart, and audit
# lines written by both at once leave one intact hash chain.

import itertools
import threading
import time
from collections.abc import Callable, Iterator

import pytest
from langchain_core.messages.ai import UsageMetadata

from conftest import DATABASE_URL
from deskclerk.api import SESSION_COOKIE
from deskclerk.auditstore import AuditStore
from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from mcp_server import RunningServer
from test_admin import GROUP, bearer
from test_agent import ScriptedModel, answer, script
from test_api import (
    Sign,
    Web,
    as_user,
    ask,
    make_settings,
    notes,
    notes_server,  # noqa: F401
    session_cookie,
    start_connect,
    web,
)
from test_erase import KEY
from test_scim import HeldModel


@pytest.fixture
def other_database(database: Database) -> Iterator[Database]:
    opened = Database(DATABASE_URL)
    yield opened
    opened.close()


def test_two_instances_share_an_employees_session(
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings)
    with (
        web(settings, [], script(), token_signing_key, database, key=KEY) as first,
        web(settings, [], script(), token_signing_key, other_database, key=KEY) as second,
    ):
        second.client.cookies.set(SESSION_COOKIE, session_cookie(first, "ada"))
        seen = second.client.get("/me")
    assert (seen.status_code, seen.json()["user"]) == (200, "ada")


def test_two_instances_share_a_conversation(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    with (
        web(settings, [], script(answer("the answer")), token_signing_key, database, key=KEY) as a,
        web(settings, [], script(), token_signing_key, other_database, key=KEY) as b,
    ):
        asked = ask(a, ada, "what is the plan")
        conversation = asked.json()["conversation"]
        reloaded = b.client.get(f"/conversations/{conversation}", headers=ada)
    assert reloaded.status_code == 200
    assert reloaded.json()["turns"] == [{"request": "what is the plan", "answer": "the answer"}]


def test_two_instances_count_one_budget(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings, budget_usd=0.012)
    ninety_thousand_tokens = UsageMetadata(
        input_tokens=90_000, output_tokens=0, total_tokens=90_000
    )
    model = ScriptedModel(messages=itertools.repeat(answer("ok", usage=ninety_thousand_tokens)))
    ada = as_user(sign_token, "ada")
    with (
        web(settings, [], model, token_signing_key, database, key=KEY) as a,
        web(settings, [], model, token_signing_key, other_database, key=KEY) as b,
    ):
        statuses = [ask(b, ada).status_code, ask(a, ada).status_code, ask(b, ada).status_code]
    assert statuses == [200, 200, 402]


def test_a_second_ask_of_one_employee_waits_for_the_first_across_instances(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings)
    held = HeldModel(messages=iter([answer("one")]))
    held.entered, held.release = threading.Event(), threading.Event()
    ada = as_user(sign_token, "ada")
    statuses: dict[str, int] = {}
    with (
        web(settings, [], held, token_signing_key, database, key=KEY) as a,
        web(settings, [], script(answer("two")), token_signing_key, other_database, key=KEY) as b,
    ):
        first = threading.Thread(target=lambda: statuses.update(first=ask(a, ada).status_code))
        first.start()
        assert held.entered.wait(10)
        second = threading.Thread(target=lambda: statuses.update(second=ask(b, ada).status_code))
        second.start()
        time.sleep(1)
        waiting = second.is_alive()
        held.release.set()
        first.join(10)
        second.join(10)
    assert waiting
    assert statuses == {"first": 200, "second": 200}


def test_the_rate_limit_counts_across_instances(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings, requests_per_minute=2)
    model = ScriptedModel(messages=itertools.repeat(answer("ok")))
    ada = as_user(sign_token, "ada")
    with (
        web(settings, [], model, token_signing_key, database, key=KEY) as a,
        web(settings, [], model, token_signing_key, other_database, key=KEY) as b,
    ):
        statuses = [ask(a, ada).status_code, ask(b, ada).status_code, ask(a, ada).status_code]
        refused_elsewhere = ask(b, ada)
    assert statuses == [200, 200, 429]
    assert refused_elsewhere.status_code == 429


def test_an_oauth_callback_received_by_another_instance_completes_the_flow(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,  # noqa: F811
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    servers = [notes(notes_server)]
    with (
        web(settings, servers, script(), token_signing_key, database, key=KEY) as a,
        web(settings, servers, script(), token_signing_key, other_database, key=KEY) as b,
    ):
        callback = start_connect(a, ada)
        finished = b.client.get(
            "/oauth/callback", params=callback, headers=ada, follow_redirects=False
        )
        connected = a.tokens.get("ada", "notes", "tokens") is not None
    assert (finished.status_code, finished.headers["location"]) == (303, "/")
    assert connected


def wait_until(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return False


def test_a_server_changed_on_one_instance_reaches_the_other_without_a_restart(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    admin = bearer(sign_token, "root", [GROUP])
    ada = as_user(sign_token, "ada")
    wiki = {"name": "wiki", "url": "http://127.0.0.1:9/mcp", "auth": "none"}
    with (
        web(settings, [], script(), token_signing_key, database, key=KEY) as a,
        web(settings, [], script(), token_signing_key, other_database, key=KEY) as b,
    ):

        def b_offers() -> list[str]:
            return [row["name"] for row in b.client.get("/servers", headers=ada).json()]

        added = a.client.post("/admin/servers", json=wiki, headers=admin)
        seen_added = wait_until(lambda: b_offers() == ["wiki"])
        paused = a.client.post("/admin/servers/wiki/pause", headers=admin)
        seen_paused = wait_until(lambda: b_offers() == [])
    assert (added.status_code, paused.status_code) == (204, 204)
    assert seen_added
    assert seen_paused


def test_audit_lines_written_by_two_instances_at_once_leave_an_intact_chain(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    other_database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    admin = bearer(sign_token, "root", [GROUP])

    def revoke_many(site: Web, prefix: str) -> None:
        for number in range(15):
            body = {"user_id": f"{prefix}{number}"}
            assert site.client.post("/admin/employees/revoke", json=body, headers=admin).is_success

    with (
        web(settings, [], script(), token_signing_key, database, key=KEY) as a,
        web(settings, [], script(), token_signing_key, other_database, key=KEY) as b,
    ):
        threads = [
            threading.Thread(target=revoke_many, args=(a, "a")),
            threading.Thread(target=revoke_many, args=(b, "b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        report = AuditStore(database).verify()
    assert (report.rows, report.broken_at) == (30, None)
