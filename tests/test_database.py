# Row-level security on the documents against the real database: each user sees exactly the
# documents open to them, nothing is visible without an identity, and the app role cannot write.

import psycopg
import pytest
from psycopg import sql
from psycopg.abc import Query
from psycopg.pq import TransactionStatus
from psycopg.rows import TupleRow

from needtoknow.auth import EVERYONE, principals
from needtoknow.db import APP_ROLE, as_user
from needtoknow.documents import Document
from needtoknow.provision import OWNER_ROLE

Connection = psycopg.Connection[TupleRow]

KAMINSKI = principals("kaminski-v", [])
KEAN = principals("kean-s", [])


def _ids(connection: Connection, query: Query) -> set[str]:
    return {row[0] for row in connection.execute(query).fetchall()}


def _count(connection: Connection, query: Query) -> int:
    row = connection.execute(query).fetchone()
    assert row is not None
    return int(row[0])


def test_each_user_sees_exactly_their_documents(
    app: Connection, owner: Connection, emails: dict[str, Document]
) -> None:
    rows = owner.execute("SELECT doc_id FROM doc_access WHERE principal = %s", [EVERYONE])
    for_everyone = {row[0] for row in rows}
    custodians = {reader for email in emails.values() for reader in email.readers}
    assert for_everyone
    assert len(custodians) == 3

    for custodian in custodians:
        own = {doc_id for doc_id, email in emails.items() if custodian in email.readers}
        with as_user(app, principals(custodian.removeprefix("user:"), [])):
            assert _ids(app, "SELECT id FROM documents") == own | for_everyone
            assert _ids(app, "SELECT doc_id FROM chunks") == own | for_everyone


@pytest.mark.parametrize("table", ["documents", "doc_access", "chunks"])
def test_no_identity_sees_no_rows(app: Connection, owner: Connection, table: str) -> None:
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
    assert _count(owner, query) > 0
    assert _count(app, query) == 0


def test_app_role_is_not_privileged_and_owns_nothing(app: Connection) -> None:
    role = app.execute(
        "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    owned = _count(
        app,
        "SELECT count(*) FROM pg_class "
        "WHERE relowner = (SELECT oid FROM pg_roles WHERE rolname = current_user)",
    )
    assert role == (APP_ROLE, False, False)
    assert owned == 0


def test_every_table_has_row_level_security(owner: Connection) -> None:
    tables = owner.execute(
        "SELECT relname, relowner::regrole::text, relrowsecurity FROM pg_class "
        "WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' ORDER BY relname"
    ).fetchall()
    assert tables == [
        ("chunks", OWNER_ROLE, True),
        ("doc_access", OWNER_ROLE, True),
        ("documents", OWNER_ROLE, True),
        ("schema_info", OWNER_ROLE, True),
    ]


def test_identity_ends_with_the_transaction(app: Connection) -> None:
    with as_user(app, KAMINSKI):
        assert _count(app, "SELECT count(*) FROM documents") > 0
    assert _count(app, "SELECT count(*) FROM documents") == 0


def test_identity_ends_when_the_transaction_fails(app: Connection) -> None:
    with pytest.raises(psycopg.errors.DivisionByZero), as_user(app, KAMINSKI):
        app.execute("SELECT 1 / 0")
    assert app.info.transaction_status == TransactionStatus.IDLE
    assert _count(app, "SELECT count(*) FROM documents") == 0


def test_identity_cannot_be_set_inside_an_open_transaction(app: Connection) -> None:
    with (
        as_user(app, KAMINSKI),
        pytest.raises(RuntimeError, match="no open transaction"),
        as_user(app, KEAN),
    ):
        pass


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (id, title, body) VALUES ('x', 'x', 'x')",
        "UPDATE documents SET title = 'x'",
        "DELETE FROM doc_access",
        "TRUNCATE chunks",
    ],
)
def test_app_cannot_write(app: Connection, statement: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege), as_user(app, KAMINSKI):
        app.execute(statement)


def test_access_rows_show_only_the_askers_principals(app: Connection) -> None:
    with as_user(app, KAMINSKI):
        visible = _ids(app, "SELECT principal FROM doc_access")
    assert visible
    assert visible <= KAMINSKI


@pytest.mark.parametrize("principal", ["group:everyone,user:kean-s", ""])
def test_malformed_principal_is_refused(app: Connection, principal: str) -> None:
    with (
        pytest.raises(ValueError, match="empty or contains a comma"),
        as_user(app, ["user:kaminski-v", principal]),
    ):
        pass
    assert app.info.transaction_status == TransactionStatus.IDLE
