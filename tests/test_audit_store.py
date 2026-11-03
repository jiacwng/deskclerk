# Every audit line is also stored as a row that matches the line printed, the rows form a hash chain
# that verify-audit walks and that a changed or removed row breaks, retention deletes old rows and
# keeps the chain verifiable, and the admin routes search the rows, page through them and download
# them as JSON lines for members of the admin group only.

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from conftest import CHAIN_KEY
from deskclerk import __main__ as entry
from deskclerk.audit import Audit
from deskclerk.auditstore import AuditFilter, AuditStore, chain_hash
from deskclerk.auth import Identity, SigningKeyLookup
from deskclerk.config import Settings
from deskclerk.database import Database
from test_admin import GROUP, bearer
from test_agent import answer, script
from test_api import Sign, Web, as_user, ask, make_settings, web
from test_erase import KEY, erase, rows

NOW = datetime.now(UTC)
ZERO_HASH = "0" * 64


def line(user: str = "ana", kind: str = "ask", when: datetime = NOW, **fields: object) -> str:
    base = {"time": when.isoformat(timespec="milliseconds"), "user": user, "entry": kind}
    return json.dumps(base | fields)


def calls(**fields: str) -> list[dict[str, str]]:
    return [{"server": "notes", "tool": "add"} | fields]


def stored_lines(store: AuditStore, **filters: Any) -> list[str]:
    return [row.line for row in store.rows(AuditFilter(**filters))]


def test_every_audit_call_is_stored_as_the_line_it_printed(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="deskclerk.audit")
    store = AuditStore(database, CHAIN_KEY)
    audit = Audit(False, None, store)
    who = Identity("ana", "deskclerk-api", "ana")

    audit.asked(who, None, "hello", time.monotonic(), "answered", None, 0.0)
    audit.connection(who, "connect", "notes")
    audit.connection(who, "disconnect", "notes")
    audit.admin_action(who, "pause_server", "notes")
    audit.erased("ana", 1, 2, 3, 4)

    printed = [r.getMessage() for r in caplog.records if r.name == "deskclerk.audit"]
    assert len(printed) == 5
    assert stored_lines(store) == printed
    assert store.verify().rows == 5


def test_an_ask_through_the_app_is_stored_as_the_line_it_printed(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="deskclerk.audit")
    with web(
        make_settings(token_settings), [], script(answer("ok")), token_signing_key, database
    ) as site:
        ask(site, as_user(sign_token, "ada"))
    printed = [r.getMessage() for r in caplog.records if r.name == "deskclerk.audit"]
    store = AuditStore(database, CHAIN_KEY)
    assert len(printed) == 1
    assert stored_lines(store) == printed
    [stored] = store.rows(AuditFilter())
    assert (stored.user, stored.entry, stored.decisions) == ("ada", "ask", ["answered"])


def test_the_search_columns_come_from_the_line(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    store.append(
        line(
            executed=calls(),
            blocked=[{"server": "wiki", "tool": "delete"}],
            pending=calls(),
            decided=[{"server": "notes", "tool": "add", "decision": "approve"}],
            outcome="awaiting_confirmation",
        )
    )
    store.append(line(kind="connect", server="wiki"))
    first, second = store.rows(AuditFilter())
    assert first.servers == ["notes", "wiki"]
    assert first.tools == ["add", "delete"]
    assert first.decisions == ["approve", "awaiting_confirmation"]
    assert (second.servers, second.tools, second.decisions) == (["wiki"], [], [])


def test_the_first_row_chains_from_the_zero_hash(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    text = line()
    store.append(text)
    [only] = store.rows(AuditFilter())
    assert only.hash == chain_hash(CHAIN_KEY, ZERO_HASH, text)


def seed_five(store: AuditStore) -> None:
    for number in range(5):
        store.append(line(user=f"user{number}"))


def test_verify_audit_reports_an_intact_chain_and_its_row_count(
    database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_five(AuditStore(database, CHAIN_KEY))
    entry.main(["verify-audit"])
    assert "the audit chain is intact, 5 rows" in capsys.readouterr().out


def test_verify_audit_on_an_empty_log_is_intact(
    database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    entry.main(["verify-audit"])
    assert "the audit chain is intact, 0 rows" in capsys.readouterr().out


def test_an_edited_row_is_named_and_the_command_exits_with_one(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    seed_five(store)
    third = rows(database, "SELECT id FROM audit_log ORDER BY id OFFSET 2 LIMIT 1")[0][0]
    with database.pool.connection() as connection:
        connection.execute(
            "UPDATE audit_log SET line = replace(line, 'user2', 'mallory') WHERE id = %s", (third,)
        )

    assert store.verify().broken_at == third
    with pytest.raises(SystemExit) as stopped:
        entry.main(["verify-audit"])
    assert stopped.value.code == f"deskclerk: the audit chain breaks at row {third}"


def edit_row_and_recompute_the_chain(
    database: Database, edited: int, rehash: Callable[[str, str], str]
) -> None:
    with database.pool.connection() as connection:
        stored = connection.execute("SELECT id, line, hash FROM audit_log ORDER BY id").fetchall()
        previous = stored[edited - 1][2]
        for position, (row_id, text, _) in enumerate(stored):
            if position < edited:
                continue
            if position == edited:
                text = text.replace("user1", "mallory")
                connection.execute(
                    "UPDATE audit_log SET user_id = 'mallory' WHERE id = %s", (row_id,)
                )
            previous = rehash(previous, text)
            connection.execute(
                "UPDATE audit_log SET line = %s, hash = %s WHERE id = %s", (text, previous, row_id)
            )


def plain_sha256(previous: str, text: str) -> str:
    return hashlib.sha256((previous + text).encode()).hexdigest()


def test_a_row_edited_with_the_chain_recomputed_without_the_key_is_named(
    database: Database,
) -> None:
    store = AuditStore(database, CHAIN_KEY)
    seed_five(store)
    ids = [row[0] for row in rows(database, "SELECT id FROM audit_log ORDER BY id")]

    edit_row_and_recompute_the_chain(database, 1, plain_sha256)

    assert store.verify().broken_at == ids[1]


def test_a_chain_recomputed_under_another_key_is_named(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    seed_five(store)
    ids = [row[0] for row in rows(database, "SELECT id FROM audit_log ORDER BY id")]

    edit_row_and_recompute_the_chain(
        database, 1, lambda previous, text: chain_hash(b"another-key", previous, text)
    )

    assert store.verify().broken_at == ids[1]
    assert AuditStore(database, b"another-key").verify().broken_at == ids[0]


def test_verify_audit_without_a_key_exits_with_two_and_makes_no_file(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed_five(AuditStore(database, CHAIN_KEY))
    monkeypatch.delenv("DESKCLERK_AUDIT_CHAIN_KEY")
    with pytest.raises(SystemExit) as stopped:
        entry.main(["verify-audit"])
    assert stopped.value.code == 2
    assert "no audit chain key" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_verify_audit_reads_the_key_file_in_the_data_directory(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed_five(AuditStore(database, CHAIN_KEY))
    monkeypatch.delenv("DESKCLERK_AUDIT_CHAIN_KEY")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "audit_chain_key").write_bytes(CHAIN_KEY + b"\n")
    entry.main(["verify-audit"])
    assert "the audit chain is intact, 5 rows" in capsys.readouterr().out


def test_a_search_column_edited_by_hand_is_named(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    seed_five(store)
    second = rows(database, "SELECT id FROM audit_log ORDER BY id OFFSET 1 LIMIT 1")[0][0]
    with database.pool.connection() as connection:
        connection.execute("UPDATE audit_log SET user_id = 'mallory' WHERE id = %s", (second,))

    assert store.verify().broken_at == second


def test_a_deleted_middle_row_is_named_at_the_row_after_the_gap(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    seed_five(store)
    ids = [row[0] for row in rows(database, "SELECT id FROM audit_log ORDER BY id")]
    with database.pool.connection() as connection:
        connection.execute("DELETE FROM audit_log WHERE id = %s", (ids[2],))

    assert store.verify().broken_at == ids[3]
    with pytest.raises(SystemExit):
        entry.main(["verify-audit"])


def test_retention_deletes_old_rows_and_the_chain_still_verifies(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    for days in (50, 45, 40):
        store.append(line(when=NOW - timedelta(days=days)))
    for days in (5, 1):
        store.append(line(when=NOW - timedelta(days=days)))
    old_hash = rows(database, "SELECT hash FROM audit_log ORDER BY id OFFSET 2 LIMIT 1")[0][0]

    deleted = store.delete_before(NOW - timedelta(days=30))

    assert deleted == 3
    assert rows(database, "SELECT hash FROM audit_checkpoint") == [(old_hash,)]
    assert store.verify().rows == 2
    store.append(line())
    report = store.verify()
    assert (report.rows, report.broken_at) == (3, None)


def test_retention_that_deletes_everything_still_lets_the_chain_continue(
    database: Database,
) -> None:
    store = AuditStore(database, CHAIN_KEY)
    seed_five(store)
    assert store.delete_before(NOW + timedelta(days=1)) == 5
    store.append(line())
    report = store.verify()
    assert (report.rows, report.broken_at) == (1, None)


def test_a_row_edited_after_retention_is_still_caught(database: Database) -> None:
    store = AuditStore(database, CHAIN_KEY)
    store.append(line(when=NOW - timedelta(days=40)))
    for number in range(3):
        store.append(line(user=f"user{number}"))
    store.delete_before(NOW - timedelta(days=30))
    with database.pool.connection() as connection:
        connection.execute("UPDATE audit_log SET line = replace(line, 'user0', 'mallory')")
    assert store.verify().broken_at is not None


def test_the_server_deletes_old_audit_rows_at_startup_and_zero_keeps_them(
    token_settings: Settings, token_signing_key: SigningKeyLookup, database: Database
) -> None:
    store = AuditStore(database, CHAIN_KEY)
    store.append(line(when=NOW - timedelta(days=40)))
    store.append(line())

    with web(
        make_settings(token_settings, retention_days=0),
        [],
        script(),
        token_signing_key,
        database,
        key=KEY,
    ):
        pass
    assert len(stored_lines(store)) == 2

    with web(
        make_settings(token_settings, retention_days=30),
        [],
        script(),
        token_signing_key,
        database,
        key=KEY,
    ):
        pass
    assert len(stored_lines(store)) == 1
    assert store.verify().broken_at is None


def test_an_erased_employee_keeps_their_audit_rows_and_the_chain_holds(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AuditStore(database, CHAIN_KEY)
    store.append(line(user="ana", text_chars=5))
    erase(monkeypatch, "ana")
    assert [row.entry for row in store.rows(AuditFilter(employee="ana"))] == ["ask", "erase"]
    assert store.verify().rows == 2


def seed_for_search(store: AuditStore) -> None:
    store.append(line("ana", "ask", NOW - timedelta(hours=5), executed=calls(), outcome="answered"))
    store.append(line("bob", "ask", NOW - timedelta(hours=4), blocked=calls(tool="drop")))
    store.append(
        line(
            "ana",
            "confirm",
            NOW - timedelta(hours=3),
            decided=calls(decision="reject"),
            outcome="answered",
        )
    )
    store.append(line("ana", "connect", NOW - timedelta(hours=2), server="wiki"))
    store.append(line("root", "admin", NOW - timedelta(hours=1), action="pause_server"))


def search(site: Web, admin: dict[str, str], **params: str | int) -> dict[str, Any]:
    response = site.client.get("/admin/audit", params=params, headers=admin)
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def minutes_ago(minutes: int) -> str:
    return (NOW - timedelta(minutes=minutes)).isoformat()


def test_search_filters_by_each_column_and_by_time_range(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    seed_for_search(AuditStore(database, CHAIN_KEY))
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])

        def found(**params: str | int) -> list[tuple[str, str]]:
            return [(r["user"], r["entry"]) for r in search(site, admin, **params)["rows"]]

        assert len(found()) == 5
        assert found(employee="ana") == [("ana", "connect"), ("ana", "confirm"), ("ana", "ask")]
        assert found(entry="admin") == [("root", "admin")]
        assert found(server="wiki") == [("ana", "connect")]
        assert found(server="notes") == [("ana", "confirm"), ("bob", "ask"), ("ana", "ask")]
        assert found(tool="drop") == [("bob", "ask")]
        assert found(decision="reject") == [("ana", "confirm")]
        assert found(decision="answered") == [("ana", "confirm"), ("ana", "ask")]
        assert found(employee="ana", server="notes") == [("ana", "confirm"), ("ana", "ask")]
        assert found(since=minutes_ago(250), until=minutes_ago(150)) == [
            ("ana", "confirm"),
            ("bob", "ask"),
        ]
        assert found(employee="nobody") == []


def test_search_pages_newest_first_with_a_cursor(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    seed_for_search(AuditStore(database, CHAIN_KEY))
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])
        first = search(site, admin, limit=2)
        second = search(site, admin, limit=2, cursor=first["next_cursor"])
        third = search(site, admin, limit=2, cursor=second["next_cursor"])
        too_many = site.client.get("/admin/audit", params={"limit": 201}, headers=admin)
    order = [r["entry"] for page in (first, second, third) for r in page["rows"]]
    assert order == ["admin", "connect", "confirm", "ask", "ask"]
    assert [len(p["rows"]) for p in (first, second, third)] == [2, 2, 1]
    assert third["next_cursor"] is None
    assert too_many.status_code == 422


def test_a_non_admin_is_refused_the_search_and_the_download(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    seed_for_search(AuditStore(database, CHAIN_KEY))
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(), token_signing_key, database) as site:
        outsider = bearer(sign_token, "bob", ["finance"])
        for path in ("/admin/audit", "/admin/audit/download"):
            assert site.client.get(path, headers=outsider).status_code == 403
            assert site.client.get(path).status_code == 401


def test_the_download_returns_the_filtered_rows_as_json_lines(
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    database: Database,
) -> None:
    seed_for_search(AuditStore(database, CHAIN_KEY))
    settings = make_settings(token_settings, admin_group=GROUP)
    with web(settings, [], script(), token_signing_key, database) as site:
        admin = bearer(sign_token, "root", [GROUP])
        everything = site.client.get("/admin/audit/download", headers=admin)
        anas = site.client.get("/admin/audit/download", params={"employee": "ana"}, headers=admin)
    assert everything.headers["content-type"].startswith("application/x-ndjson")
    assert "attachment" in everything.headers["content-disposition"]
    parsed = [json.loads(text) for text in everything.text.splitlines()]
    assert [row["entry"] for row in parsed] == ["ask", "ask", "confirm", "connect", "admin"]
    assert json.loads(parsed[0]["line"])["user"] == "ana"
    assert {row["user"] for row in map(json.loads, anas.text.splitlines())} == {"ana"}
    assert len(anas.text.splitlines()) == 3
