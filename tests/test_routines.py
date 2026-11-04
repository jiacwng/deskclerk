# Scheduled requests against the real stores: a due routine runs as the employee through a
# refreshed login, every write in it waits for approval even when always allowed, a refused refresh
# pauses the employee's routines, one occurrence runs once across copies, and erasing, revoking or
# deactivating an employee deletes their routines and refresh token. One test uses the development
# Keycloak.

import asyncio
import html
import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, time
from importlib import resources
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import httpx
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from conftest import CHAIN_KEY
from deskclerk.api import CSRF_HEADER, CSRF_VALUE
from deskclerk.auth import SigningKeyLookup, discover_jwks_url, jwks_signing_key
from deskclerk.config import Settings
from deskclerk.database import Database
from deskclerk.rotation import rotate_key
from deskclerk.routines import Routine, RoutineStore, Scheduler, next_occurrence
from deskclerk.tokens import LOGIN, TokenStore
from mcp_server import RunningServer, running_mcp_server
from test_admin import GROUP, bearer
from test_agent import ScriptedModel, answer, calls, script
from test_api import (
    SESSION_KEY,
    FakeIdp,
    Sign,
    Web,
    as_user,
    ask,
    audit_lines,
    begin_sign_in,
    confirm,
    connect,
    finish_sign_in,
    make_settings,
    notes,
    web,
)
from test_connections import REDIRECT
from test_erase import KEY, erase, rows
from test_scim import AUTH, USERS, provision, scim_site

EVERY_DAY = list(range(7))
PARIS = ZoneInfo("Europe/Paris")


class OfflineIdp(FakeIdp):
    def __init__(self, issuer: str) -> None:
        super().__init__(issuer)
        self.refreshed_id_token = ""
        self.refused = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/openid-configuration"):
            return super().handle(request)
        form = {name: values[0] for name, values in parse_qs(request.content.decode()).items()}
        self.token_calls.append(form)
        if form["grant_type"] == "authorization_code":
            body = {"id_token": self.id_token, "refresh_token": "rt-1"}
            return httpx.Response(200, json={"access_token": "opaque"} | body)
        if self.refused:
            return httpx.Response(400, json={"error": "invalid_grant"})
        body = {"id_token": self.refreshed_id_token, "refresh_token": "rt-2"}
        return httpx.Response(200, json={"access_token": "opaque"} | body)


@pytest.fixture
def notes_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT) as running:
        yield running


@pytest.fixture
def append_server() -> Iterator[RunningServer]:
    with running_mcp_server(REDIRECT, append_tool=True) as running:
        yield running


@contextmanager
def routines_web(
    token_settings: Settings,
    signing_key: SigningKeyLookup,
    database: Database,
    sign_token: Sign,
    model: ScriptedModel,
    servers: list[Any],
    **changes: object,
) -> Iterator[tuple[Web, OfflineIdp]]:
    settings = make_settings(token_settings, **changes)
    idp = OfflineIdp(settings.issuer)
    idp.refreshed_id_token = sign_token({"sub": "ada", "preferred_username": "ada"})
    with web(settings, servers, model, signing_key, database, idp.transport) as site:
        site.tokens.put("ada", LOGIN, "refresh", "rt-1")
        yield site, idp


def add_routine(
    site: Web, headers: dict[str, str], request: str = "read my note", at: str = "09:00"
) -> dict[str, Any]:
    body = {"request": request, "days": EVERY_DAY, "time": at, "time_zone": "Europe/Paris"}
    response = site.client.post("/routines", json=body, headers=headers)
    assert response.status_code == 201, response.text
    return cast(dict[str, Any], response.json())


def make_due(database: Database, routine_id: str) -> None:
    with database.pool.connection() as connection:
        connection.execute(
            "UPDATE routines SET next_run_at = now() - interval '1 minute' WHERE id = %s",
            (routine_id,),
        )


def run_due(site: Web) -> None:
    assert site.client.portal is not None
    scheduler = cast(Any, site.client.app).state.scheduler
    site.client.portal.call(scheduler.run_due)


def listed(site: Web, headers: dict[str, str]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], site.client.get("/routines", headers=headers).json())


def test_a_due_routine_runs_as_the_employee_and_its_read_result_reaches_the_conversation(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    model = script(calls(("notes__read_note", {})), answer("the note says hi"))
    ada = as_user(sign_token, "ada")
    with routines_web(
        token_settings, token_signing_key, database, sign_token, model, [notes(notes_server)]
    ) as (site, idp):
        connect(site, ada)
        routine = add_routine(site, ada)
        make_due(database, routine["id"])
        run_due(site)
        [row] = listed(site, ada)
        conversation = site.client.get(f"/conversations/{row['last_conversation']}", headers=ada)
        metrics = site.client.get("/metrics").text
        stored = site.tokens.get("ada", LOGIN, "refresh")
    assert row["last_outcome"] == "answered"
    assert datetime.fromisoformat(row["next_run_at"]) > datetime.now(UTC)
    assert conversation.json()["turns"] == [
        {"request": "read my note", "answer": "the note says hi"}
    ]
    assert "the note" in str(model.seen[1][-1].content)
    assert [call["grant_type"] for call in idp.token_calls] == ["refresh_token"]
    assert idp.token_calls[0]["scope"] == "openid profile email offline_access"
    assert stored == "rt-2"
    assert 'deskclerk_routine_runs_total{outcome="answered"} 1.0' in metrics
    [line] = [line for line in audit_lines(caplog) if line["entry"] == "routine"]
    assert (line["user"], line["outcome"]) == ("ada", "answered")


def test_a_write_in_a_routine_waits_and_ask_confirm_resumes_it(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
) -> None:
    model = script(calls(("notes__write_note", {"text": "hi"})), answer("saved"))
    ada = as_user(sign_token, "ada")
    with routines_web(
        token_settings, token_signing_key, database, sign_token, model, [notes(notes_server)]
    ) as (site, _):
        connect(site, ada)
        routine = add_routine(site, ada, "save a note")
        make_due(database, routine["id"])
        run_due(site)
        [row] = listed(site, ada)
        waiting = site.client.get(f"/conversations/{row['last_conversation']}", headers=ada).json()
        marked = site.client.get("/conversations", headers=ada).json()
        assert notes_server.notes == []
        [pending] = waiting["pending"]
        done = confirm(site, ada, row["last_conversation"], {pending["id"]: "approve"}).json()
        cleared = site.client.get("/conversations", headers=ada).json()
    assert row["last_outcome"] == "awaiting_confirmation"
    assert [item["waiting"] for item in marked] == [True]
    assert [item["waiting"] for item in cleared] == [False]
    assert done["answer"] == "saved"
    assert [note[1] for note in notes_server.notes] == ["hi"]


def test_an_always_allowed_tool_still_waits_in_a_routine(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    append_server: RunningServer,
    database: Database,
) -> None:
    append_server.redescribe_append_note("Appends a line to the notes. Nothing is removed.")
    model = script(
        calls(("notes__append_note", {"text": "one"})),
        answer("appended"),
        calls(("notes__append_note", {"text": "two"})),
        answer("appended again"),
    )
    ada = as_user(sign_token, "ada")
    with routines_web(
        token_settings, token_signing_key, database, sign_token, model, [notes(append_server)]
    ) as (site, _):
        connect(site, ada)
        paused = ask(site, ada, "append one").json()
        [first] = paused["pending"]
        always = confirm(site, ada, paused["conversation"], {first["id"]: "approve_always"})
        assert always.json()["answer"] == "appended"
        routine = add_routine(site, ada, "append two")
        make_due(database, routine["id"])
        run_due(site)
        [row] = listed(site, ada)
        waiting = site.client.get(f"/conversations/{row['last_conversation']}", headers=ada).json()
    assert row["last_outcome"] == "awaiting_confirmation"
    assert [call["arguments"] for call in waiting["pending"]] == [{"text": "two"}]
    assert [note[1] for note in append_server.notes] == ["one"]


def test_a_refresh_the_login_server_refuses_pauses_all_the_employees_routines(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    model = script(answer("never asked"))
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with routines_web(token_settings, token_signing_key, database, sign_token, model, []) as (
        site,
        idp,
    ):
        site.tokens.put("grace", LOGIN, "refresh", "rt-g")
        due = add_routine(site, ada)
        add_routine(site, ada, "another one")
        add_routine(site, grace)
        make_due(database, due["id"])
        idp.refused = True
        run_due(site)
        ada_rows, grace_rows = listed(site, ada), listed(site, grace)
    assert [(r["paused"], r["last_outcome"]) for r in ada_rows] == [(True, "sign_in_needed")] * 2
    assert [(r["paused"], r["last_outcome"]) for r in grace_rows] == [(False, None)]
    assert model.seen == []
    [line] = [line for line in audit_lines(caplog) if line["entry"] == "routine_paused"]
    assert (line["user"], line["outcome"]) == ("ada", "sign_in_needed")


def test_a_refreshed_token_that_names_another_employee_pauses_the_routines(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    model = script(answer("never asked"))
    ada = as_user(sign_token, "ada")
    with routines_web(token_settings, token_signing_key, database, sign_token, model, []) as (
        site,
        idp,
    ):
        idp.refreshed_id_token = sign_token({"sub": "grace", "preferred_username": "grace"})
        routine = add_routine(site, ada)
        make_due(database, routine["id"])
        run_due(site)
        [row] = listed(site, ada)
    assert (row["paused"], row["last_outcome"]) == (True, "sign_in_needed")
    assert model.seen == []


def test_a_routine_is_skipped_while_its_last_conversation_waits_for_approval(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    notes_server: RunningServer,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    model = script(calls(("notes__write_note", {"text": "hi"})))
    ada = as_user(sign_token, "ada")
    with routines_web(
        token_settings, token_signing_key, database, sign_token, model, [notes(notes_server)]
    ) as (site, _):
        connect(site, ada)
        routine = add_routine(site, ada, "save a note")
        make_due(database, routine["id"])
        run_due(site)
        [first] = listed(site, ada)
        make_due(database, routine["id"])
        run_due(site)
        [second] = listed(site, ada)
    assert second["last_outcome"] == "previous_run_waiting"
    assert second["last_conversation"] == first["last_conversation"]
    assert len(model.seen) == 1
    [line] = [line for line in audit_lines(caplog) if line["entry"] == "routine_skipped"]
    assert line["outcome"] == "previous_run_waiting"


def test_a_routine_over_the_monthly_budget_is_skipped_and_audited(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="deskclerk.audit")
    model = script(answer("never asked"))
    ada = as_user(sign_token, "ada")
    with routines_web(
        token_settings, token_signing_key, database, sign_token, model, [], budget_usd=0.0
    ) as (site, _):
        routine = add_routine(site, ada)
        make_due(database, routine["id"])
        run_due(site)
        [row] = listed(site, ada)
    assert row["last_outcome"] == "over_budget"
    assert model.seen == []
    [line] = [line for line in audit_lines(caplog) if line["entry"] == "routine"]
    assert line["outcome"] == "over_budget"


def test_two_schedulers_claiming_one_due_row_run_it_once(database: Database) -> None:
    store = RoutineStore(database, TokenStore(database, KEY))
    now = datetime.now(UTC)
    routine = store.add("ada", "read", tuple(EVERY_DAY), time(9, 0), "UTC", now)
    make_due(database, str(routine.id))
    ran: list[Routine] = []

    async def run(claimed: Routine) -> None:
        await asyncio.sleep(0.2)
        ran.append(claimed)

    async def both() -> None:
        await asyncio.wait_for(
            asyncio.gather(Scheduler(store, run).run_due(), Scheduler(store, run).run_due()), 10
        )

    asyncio.run(both())
    assert [claimed.id for claimed in ran] == [routine.id]
    assert ran[0].next_run_at > now


def test_a_claim_skips_a_row_another_transaction_holds(database: Database) -> None:
    store = RoutineStore(database, TokenStore(database, KEY))
    routine = store.add("ada", "read", tuple(EVERY_DAY), time(9, 0), "UTC", datetime.now(UTC))
    make_due(database, str(routine.id))
    claimed: list[Routine | None] = []
    with database.pool.connection() as holder:
        holder.execute("SELECT id FROM routines FOR UPDATE")
        thread = threading.Thread(target=lambda: claimed.append(store.claim_due(datetime.now(UTC))))
        thread.start()
        thread.join(5)
        blocked = thread.is_alive()
    thread.join()
    assert not blocked
    assert claimed == [None]


def test_the_next_occurrence_follows_the_wall_clock_across_a_daylight_saving_change() -> None:
    daily = tuple(EVERY_DAY)
    before = datetime(2026, 3, 28, 9, 0, 1, tzinfo=PARIS).astimezone(UTC)
    spring = next_occurrence(daily, time(9, 0), PARIS, before)
    assert spring == datetime(2026, 3, 29, 7, 0, tzinfo=UTC)
    assert next_occurrence(daily, time(9, 0), PARIS, spring) == datetime(
        2026, 3, 30, 7, 0, tzinfo=UTC
    )
    last_summer_day = datetime(2026, 10, 24, 9, 0, 1, tzinfo=PARIS).astimezone(UTC)
    autumn = next_occurrence(daily, time(9, 0), PARIS, last_summer_day)
    assert autumn == datetime(2026, 10, 25, 8, 0, tzinfo=UTC)


def test_the_next_occurrence_skips_days_that_are_not_chosen() -> None:
    friday_noon = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
    weekdays = (0, 1, 2, 3, 4)
    assert next_occurrence(weekdays, time(9, 0), PARIS, friday_noon) == datetime(
        2026, 11, 9, 8, 0, tzinfo=UTC
    )
    assert next_occurrence((4,), time(9, 0), PARIS, friday_noon) == datetime(
        2026, 11, 13, 8, 0, tzinfo=UTC
    )


def seed_ana_and_anabel(database: Database, tokens: TokenStore) -> RoutineStore:
    store = RoutineStore(database, tokens)
    for user in ("ana", "anabel"):
        tokens.put(user, LOGIN, "refresh", "rt")
        store.add(user, "read", tuple(EVERY_DAY), time(9, 0), "UTC", datetime.now(UTC))
    return store


def employee_routines(database: Database, user: str) -> tuple[int, int]:
    with database.pool.connection() as connection:
        found = connection.execute(
            "SELECT COUNT(*) FROM routines WHERE user_id = %s", (user,)
        ).fetchone()
        token = connection.execute(
            "SELECT COUNT(*) FROM secrets WHERE user_id = %s AND kind = 'refresh'", (user,)
        ).fetchone()
    assert found is not None and token is not None
    return found[0], token[0]


def test_erasing_an_employee_deletes_their_routines_and_refresh_token(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_ana_and_anabel(database, TokenStore(database, KEY))
    erase(monkeypatch, "ana")
    assert employee_routines(database, "ana") == (0, 0)
    assert employee_routines(database, "anabel") == (1, 1)


def test_revoking_an_employee_deletes_their_routines_and_refresh_token(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings, admin_group=GROUP)
    admin = bearer(sign_token, "admin", [GROUP])
    with web(settings, [], script(), token_signing_key, database, key=KEY) as site:
        seed_ana_and_anabel(database, site.tokens)
        revoked = site.client.post(
            "/admin/employees/revoke", json={"user_id": "ana"}, headers=admin
        )
    assert revoked.status_code == 200
    assert employee_routines(database, "ana") == (0, 0)
    assert employee_routines(database, "anabel") == (1, 1)


def test_deactivating_through_scim_deletes_their_routines_and_refresh_token(
    token_settings: Settings, token_signing_key: SigningKeyLookup, database: Database
) -> None:
    with scim_site(token_settings, token_signing_key, database) as site:
        seed_ana_and_anabel(database, site.tokens)
        user_id = provision(site.client)
        operation = {"op": "Replace", "path": "active", "value": False}
        patched = site.client.patch(
            f"{USERS}/{user_id}", headers=AUTH, json={"Operations": [operation]}
        )
    assert patched.status_code == 200
    assert employee_routines(database, "ana") == (0, 0)
    assert employee_routines(database, "anabel") == (1, 1)


def test_rotate_key_keeps_the_refresh_token_readable_with_the_newest_key_alone(
    database: Database,
) -> None:
    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    TokenStore(database, old).put("ada", LOGIN, "refresh", "rt-1")
    rotate_key(database, TokenStore(database, f"{new},{old}").fernet)
    assert TokenStore(database, new).get("ada", LOGIN, "refresh") == "rt-1"


def test_signing_in_stores_the_refresh_token_encrypted_and_replaces_the_old_one(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    idp = OfflineIdp(settings.issuer)
    with web(settings, [], script(), token_signing_key, database, idp.transport) as site:
        site.tokens.put("district-1", LOGIN, "refresh", "stale")
        state, _, _ = begin_sign_in(site)
        done = finish_sign_in(site, idp, sign_token, state, {"scope": None, "typ": "ID"})
        stored = site.tokens.get("district-1", LOGIN, "refresh")
        raw = rows(database, "SELECT value FROM secrets WHERE kind = 'refresh'")
    assert done.status_code == 303
    assert stored == "rt-1"
    assert b"rt-1" not in bytes(raw[0][0])


def test_only_the_routines_sign_in_asks_for_offline_access_and_me_says_whether_it_was_given(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    idp = OfflineIdp(settings.issuer)
    ada = as_user(sign_token, "ada")
    with web(settings, [], script(), token_signing_key, database, idp.transport) as site:
        normal = site.client.get("/login", follow_redirects=False)
        routines = site.client.get("/login?offline=1", follow_redirects=False)
        before = site.client.get("/me", headers=ada).json()["offline_access"]
        site.tokens.put("ada", LOGIN, "refresh", "rt-1")
        after = site.client.get("/me", headers=ada).json()["offline_access"]
    scopes = [
        parse_qs(urlparse(response.headers["location"]).query)["scope"]
        for response in (normal, routines)
    ]
    assert scopes == [["openid profile email"], ["openid profile email offline_access"]]
    assert (before, after) == (False, True)


def test_saving_a_routine_without_a_refresh_token_is_refused_naming_offline_access(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    grace = as_user(sign_token, "grace")
    body = {"request": "read", "days": EVERY_DAY, "time": "09:00", "time_zone": "Europe/Paris"}
    with routines_web(token_settings, token_signing_key, database, sign_token, script(), []) as (
        site,
        _,
    ):
        refused = site.client.post("/routines", json=body, headers=grace)
    assert refused.status_code == 409
    assert "offline_access" in refused.json()["detail"]
    assert "Allow routines while you are away" in refused.json()["detail"]


def test_the_routine_routes_validate_the_schedule_and_keep_routines_private(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    good = {"request": "read", "days": [4, 0, 4], "time": "09:30", "time_zone": "Europe/Paris"}
    with routines_web(token_settings, token_signing_key, database, sign_token, script(), []) as (
        site,
        _,
    ):
        site.tokens.put("grace", LOGIN, "refresh", "rt-g")
        wrong_zone = site.client.post(
            "/routines", json=good | {"time_zone": "Mars/Base"}, headers=ada
        )
        no_days = site.client.post("/routines", json=good | {"days": []}, headers=ada)
        bad_day = site.client.post("/routines", json=good | {"days": [7]}, headers=ada)
        bad_time = site.client.post("/routines", json=good | {"time": "late"}, headers=ada)
        anonymous = site.client.get("/routines")
        created = site.client.post("/routines", json=good, headers=ada)
        routine_id = created.json()["id"]
        grace_view = listed(site, grace)
        grace_pause = site.client.post(f"/routines/{routine_id}/pause", headers=grace)
        grace_delete = site.client.delete(f"/routines/{routine_id}", headers=grace)
        paused = site.client.post(f"/routines/{routine_id}/pause", headers=ada)
        make_due(database, routine_id)
        resumed = site.client.post(f"/routines/{routine_id}/resume", headers=ada)
        [row] = listed(site, ada)
        deleted = site.client.delete(f"/routines/{routine_id}", headers=ada)
        gone = listed(site, ada)
    assert [r.status_code for r in (wrong_zone, no_days, bad_day, bad_time)] == [422] * 4
    assert anonymous.status_code == 401
    assert created.status_code == 201
    assert (created.json()["days"], created.json()["time"]) == ([0, 4], "09:30")
    assert grace_view == []
    assert (grace_pause.status_code, grace_delete.status_code) == (404, 404)
    assert (paused.status_code, resumed.status_code, deleted.status_code) == (204, 204, 204)
    assert row["paused"] is False
    assert datetime.fromisoformat(row["next_run_at"]) > datetime.now(UTC)
    assert gone == []


def test_the_page_has_a_routines_panel_and_marks_a_conversation_that_waits() -> None:
    page = resources.files("deskclerk").joinpath("static/index.html").read_text(encoding="utf-8")
    assert 'id="routine-form"' in page
    assert 'id="routine-list"' in page
    assert "Open last conversation" in page
    assert "item.waiting" in page
    assert 'href="/login?offline=1"' in page
    assert "Allow routines while you are away" in page


def keycloak_admin_token(browser: httpx.Client, base: str) -> dict[str, str]:
    form = {"grant_type": "password", "client_id": "admin-cli", "username": "admin"}
    response = browser.post(
        f"{base}/realms/master/protocol/openid-connect/token",
        data=form | {"password": "admin"},
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.mark.keycloak
def test_a_routine_runs_through_keycloak_and_pauses_once_the_user_is_disabled(
    database: Database,
) -> None:
    settings = Settings(
        session_key=SecretStr(SESSION_KEY), audit_chain_key=SecretStr(CHAIN_KEY.decode())
    )
    with httpx.Client() as discovery:
        signing_key = jwks_signing_key(discover_jwks_url(settings, discovery))
    model = script(answer("hello from the routine"))
    headers = {CSRF_HEADER: CSRF_VALUE}
    with web(settings, [], model, signing_key, database) as site, httpx.Client() as browser:
        site.client.follow_redirects = False
        authorization_url = site.client.get("/login?offline=1").headers["location"]
        assert "offline_access" in parse_qs(urlparse(authorization_url).query)["scope"][0]
        login_page = browser.get(authorization_url)
        for cookie in browser.cookies.jar:
            cookie.secure = False
        form = re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', login_page.text)
        assert form is not None
        submitted = browser.post(
            html.unescape(form.group(1)),
            data={"username": "dasovich-j", "password": "deskclerk"},
        )
        landing = urlparse(submitted.headers["location"])
        site.client.get(f"{landing.path}?{landing.query}")
        keycloak = settings.issuer.split("/realms/")[0]
        admin = keycloak_admin_token(browser, keycloak)
        users = f"{keycloak}/admin/realms/deskclerk/users"
        [found] = browser.get(
            users, params={"username": "dasovich-j", "exact": "true"}, headers=admin
        ).json()
        try:
            body = {"request": "say hello", "days": EVERY_DAY, "time": "09:00", "time_zone": "UTC"}
            created = site.client.post("/routines", json=body, headers=headers)
            assert created.status_code == 201, created.text
            make_due(database, created.json()["id"])
            run_due(site)
            [ran] = site.client.get("/routines").json()
            browser.put(f"{users}/{found['id']}", json={"enabled": False}, headers=admin)
            make_due(database, created.json()["id"])
            run_due(site)
            [after] = site.client.get("/routines").json()
        finally:
            browser.put(f"{users}/{found['id']}", json={"enabled": True}, headers=admin)
    assert ran["last_outcome"] == "answered"
    assert (after["paused"], after["last_outcome"]) == (True, "sign_in_needed")
    assert len(model.seen) == 1
