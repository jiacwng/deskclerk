# The conversation list: an employee sees only their own conversations, newest first, titled with
# the first request; deleting one removes its thread and row; erase and retention remove the rows.

import itertools
import threading
import time

import pytest

from deskclerk.api import CSRF_HEADER, CSRF_VALUE, SESSION_COOKIE
from deskclerk.auth import Identity, SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.conversations import TITLE_CHARS, title_of
from deskclerk.database import Database
from test_agent import ScriptedModel, answer
from test_api import Sign, as_user, ask, ask_in, make_settings, web
from test_erase import KEY, erase, rows
from test_scim import HeldModel


def replying() -> ScriptedModel:
    return ScriptedModel(messages=itertools.repeat(answer("ok")))


def test_the_list_holds_only_the_callers_conversations_newest_first_with_a_cut_title(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    long_request = "  plan   the\nquarterly " + "review " * 20
    with web(settings, [], replying(), token_signing_key, database, key=KEY) as site:
        older = ask(site, ada, "first one").json()["conversation"]
        newer = ask(site, ada, long_request).json()["conversation"]
        ask(site, grace, "grace only")
        ask_in(site, ada, older, "bump the older one")
        listed = site.client.get("/conversations", headers=ada)
        anonymous = site.client.get("/conversations")
    assert anonymous.status_code == 401
    assert [item["id"] for item in listed.json()] == [older, newer]
    assert [item["title"] for item in listed.json()] == [
        "first one",
        title_of(long_request),
    ]
    assert title_of(long_request) == ("plan the quarterly " + "review " * 20)[:TITLE_CHARS] + "…"
    assert listed.json()[0]["updated_at"] > listed.json()[1]["updated_at"]


def test_deleting_another_employees_or_an_unknown_conversation_gives_404(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada, grace = as_user(sign_token, "ada"), as_user(sign_token, "grace")
    with web(settings, [], replying(), token_signing_key, database, key=KEY) as site:
        conversation = ask(site, ada, "mine").json()["conversation"]
        taken = site.client.delete(f"/conversations/{conversation}", headers=grace)
        unknown = site.client.delete(
            "/conversations/0d6b2c3e-9d5e-4c27-8f3e-3c1d2b4a5e6f", headers=ada
        )
        still_there = site.client.get(f"/conversations/{conversation}", headers=ada)
    assert (taken.status_code, unknown.status_code, still_there.status_code) == (404, 404, 200)
    assert taken.json() == unknown.json() == {"detail": "no such conversation"}


def test_deleting_a_conversation_removes_its_thread_and_row(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    with web(settings, [], replying(), token_signing_key, database, key=KEY) as site:
        gone = ask(site, ada, "delete me").json()["conversation"]
        kept = ask(site, ada, "keep me").json()["conversation"]
        deleted = site.client.delete(f"/conversations/{gone}", headers=ada)
        reload_gone = site.client.get(f"/conversations/{gone}", headers=ada)
        reload_kept = site.client.get(f"/conversations/{kept}", headers=ada)
        listed = site.client.get("/conversations", headers=ada).json()
        again = site.client.delete(f"/conversations/{gone}", headers=ada)
    assert (deleted.status_code, reload_gone.status_code, reload_kept.status_code) == (
        204,
        404,
        200,
    )
    assert again.status_code == 404
    assert [item["id"] for item in listed] == [kept]
    assert rows(database, "SELECT COUNT(*) FROM conversations") == [(1,)]
    assert rows(database, "SELECT DISTINCT thread_id FROM checkpoints") == [(f"ada/{kept}",)]


def test_a_signed_in_browser_must_send_the_posting_header_to_delete(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    with web(settings, [], replying(), token_signing_key, database, key=KEY) as site:
        conversation = ask(site, ada, "mine").json()["conversation"]
        site.client.cookies.set(
            SESSION_COOKIE, site.sessions.create(Identity("ada", "deskclerk-api", "ada"), 3600)
        )
        path = f"/conversations/{conversation}"
        refused = site.client.delete(path)
        allowed = site.client.delete(path, headers={CSRF_HEADER: CSRF_VALUE})
    assert (refused.status_code, allowed.status_code) == (403, 204)


def test_erasing_an_employee_removes_their_rows_and_keeps_another_whose_id_extends_it(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(token_settings)
    with web(settings, [], replying(), token_signing_key, database, key=KEY) as site:
        for user in ("ana", "ana/b"):
            ask(site, as_user(sign_token, user), "hello")
    assert rows(database, "SELECT user_id FROM conversations ORDER BY user_id") == [
        ("ana",),
        ("ana/b",),
    ]

    erase(monkeypatch, "ana")

    assert rows(database, "SELECT user_id FROM conversations") == [("ana/b",)]


def test_a_delete_waits_for_an_ask_of_the_same_employee_in_flight(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    settings = make_settings(token_settings)
    ada = as_user(sign_token, "ada")
    held = HeldModel(messages=iter([answer("one"), answer("two")]))
    held.entered, held.release = threading.Event(), threading.Event()
    statuses: dict[str, int] = {}
    with web(settings, [], held, token_signing_key, database, key=KEY) as site:
        held.release.set()
        conversation = ask(site, ada, "start").json()["conversation"]
        held.release.clear()
        held.entered.clear()
        asking = threading.Thread(
            target=lambda: statuses.update(ask=ask_in(site, ada, conversation, "more").status_code)
        )
        asking.start()
        assert held.entered.wait(10)
        deleting = threading.Thread(
            target=lambda: statuses.update(
                delete=site.client.delete(f"/conversations/{conversation}", headers=ada).status_code
            )
        )
        deleting.start()
        time.sleep(1)
        waiting = deleting.is_alive()
        held.release.set()
        asking.join(10)
        deleting.join(10)
    assert waiting
    assert statuses == {"ask": 200, "delete": 204}
    assert rows(database, "SELECT COUNT(*) FROM conversations") == [(0,)]
