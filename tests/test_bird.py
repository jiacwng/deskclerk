# The BIRD loader on the slice in tests/data: each database in its own schema with BIRD's
# descriptions as comments and documents, and row-level security that shows a district user only
# that district's banking rows and a county user only that county's schools, through query_data.

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from conftest import MINIDEV
from needtoknow.auth import principals
from needtoknow.bird import DUMP, TABLES, BirdError, load_bird, read_databases
from needtoknow.config import Settings, load_provision_settings, load_settings
from needtoknow.data import describe_data, query_data
from needtoknow.db import as_user
from needtoknow.provision import connect_admin
from needtoknow.retrieval import search

Connection = psycopg.Connection[TupleRow]

PRAGUE = principals("district-1", ["district:1"])
JIHLAVA = principals("district-58", ["district:58"])
SIERRA = principals("county-sierra", ["county:Sierra"])
ANALYST = principals("analyst", ["analyst"])
NOBODY: frozenset[str] = frozenset()
SET_CONFIG = "pg_catalog.set_config(text, text, boolean)"

# What each financial table holds for one district, read by the owner, whom no policy filters.
BY_DISTRICT = {
    "account": "SELECT count(*) FROM financial.account WHERE district_id = %s",
    "loan": "SELECT count(*) FROM financial.loan JOIN financial.account USING (account_id) "
    "WHERE district_id = %s",
    "trans": "SELECT count(*) FROM financial.trans JOIN financial.account USING (account_id) "
    "WHERE district_id = %s",
    "order": 'SELECT count(*) FROM financial."order" JOIN financial.account USING (account_id) '
    "WHERE district_id = %s",
    "disp": "SELECT count(*) FROM financial.disp JOIN financial.account USING (account_id) "
    "WHERE district_id = %s",
    "card": "SELECT count(*) FROM financial.card JOIN financial.disp USING (disp_id) "
    "JOIN financial.account USING (account_id) WHERE account.district_id = %s",
    "client": "SELECT count(DISTINCT client.client_id) FROM financial.client "
    "JOIN financial.disp USING (client_id) JOIN financial.account USING (account_id) "
    "WHERE account.district_id = %s",
}
BY_COUNTY = {
    "schools": "SELECT count(*) FROM california_schools.schools WHERE county = %s",
    "frpm": 'SELECT count(*) FROM california_schools.frpm WHERE "County Name" = %s',
    "satscores": "SELECT count(*) FROM california_schools.satscores WHERE cname = %s",
}


def _count_all(schema: str, table: str) -> str:
    return sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(schema, table)).as_string()


def _count(connection: Connection, query: str, parameters: list[Any] | None = None) -> int:
    row = connection.execute(query, parameters).fetchone()
    assert row is not None
    return int(row[0])


def _rows(output: str) -> list[str]:
    lines = output.splitlines()
    assert lines[-1].startswith("("), output
    return lines[1:-1]


def test_each_database_gets_its_own_schema_with_its_tables(owner: Connection) -> None:
    rows = owner.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema IN ('financial', 'california_schools', 'bird_dump') "
        "ORDER BY 1, 2"
    ).fetchall()

    assert rows == [
        ("california_schools", "frpm"),
        ("california_schools", "satscores"),
        ("california_schools", "schools"),
        ("financial", "account"),
        ("financial", "card"),
        ("financial", "client"),
        ("financial", "disp"),
        ("financial", "district"),
        ("financial", "loan"),
        ("financial", "order"),
        ("financial", "trans"),
    ]
    assert _count(owner, "SELECT count(*) FROM financial.district") == 77
    assert _count(owner, "SELECT count(*) FROM financial.trans") == 200


def test_describe_data_without_a_schema_lists_each_database(app: Connection) -> None:
    lines = describe_data(app).splitlines()

    assert "california_schools (3 tables)" in lines
    assert "financial (8 tables)" in lines
    assert lines[1].startswith("public (")


def test_describe_data_with_a_schema_shows_birds_descriptions(app: Connection) -> None:
    financial = describe_data(app, "financial")
    schools = describe_data(app, "california_schools")

    assert (
        "financial.account (table): Table account of the BIRD database financial.\n"
        "  account_id bigint: account id; the id of the account\n"
        "  district_id bigint: location of branch\n"
    ) in financial
    assert 'financial."order" (table)' in financial
    assert "california_schools" not in financial
    assert '  "County Name" text: County Code' in schools


def test_birds_descriptions_are_documents_everyone_finds(app: Connection) -> None:
    hits = search(app, principals("kaminski-v", []), "average salary in the district", 5)
    assert "bird-financial-district" in [hit.doc_id for hit in hits]


@pytest.mark.parametrize("table", list(BY_DISTRICT))
@pytest.mark.parametrize(("district", "asker"), [(1, PRAGUE), (58, JIHLAVA)])
def test_a_district_user_sees_only_their_districts_rows(
    app: Connection, owner: Connection, table: str, district: int, asker: frozenset[str]
) -> None:
    expected = _count(owner, BY_DISTRICT[table], [district])
    total = _count(owner, _count_all("financial", table))

    output = query_data(app, asker, _count_all("financial", table)).text

    assert 0 < expected < total
    assert _rows(output) == [str(expected)]


def test_a_district_user_reads_no_other_district(app: Connection) -> None:
    output = query_data(app, JIHLAVA, "SELECT DISTINCT district_id FROM financial.account").text
    assert _rows(output) == ["58"]


@pytest.mark.parametrize("table", list(BY_COUNTY))
def test_a_county_user_sees_only_their_countys_rows(
    app: Connection, owner: Connection, table: str
) -> None:
    expected = _count(owner, BY_COUNTY[table], ["Sierra"])
    total = _count(owner, _count_all("california_schools", table))

    output = query_data(app, SIERRA, _count_all("california_schools", table)).text

    assert 0 < expected < total
    assert _rows(output) == [str(expected)]


RESTRICTED = [("financial", name) for name in BY_DISTRICT] + [
    ("california_schools", name) for name in BY_COUNTY
]


@pytest.mark.parametrize(("schema", "table"), RESTRICTED)
def test_the_analyst_sees_every_row(
    app: Connection, owner: Connection, schema: str, table: str
) -> None:
    total = _count(owner, _count_all(schema, table))
    assert _rows(query_data(app, ANALYST, _count_all(schema, table)).text) == [str(total)]


@pytest.mark.parametrize(("schema", "table"), RESTRICTED)
@pytest.mark.parametrize("asker", [NOBODY, principals("kaminski-v", [])], ids=["none", "custodian"])
def test_without_a_district_or_county_the_rows_are_hidden(
    app: Connection, schema: str, table: str, asker: frozenset[str]
) -> None:
    assert _rows(query_data(app, asker, _count_all(schema, table)).text) == ["0"]


def test_a_table_without_a_policy_is_readable_by_all(app: Connection) -> None:
    output = query_data(app, NOBODY, "SELECT count(*) FROM financial.district").text
    assert _rows(output) == ["77"]


def test_a_county_user_sees_no_bank_rows_and_a_district_user_no_schools(
    app: Connection,
) -> None:
    banks = query_data(app, SIERRA, "SELECT count(*) FROM financial.account").text
    schools = query_data(app, PRAGUE, "SELECT count(*) FROM california_schools.schools").text
    assert _rows(banks) == ["0"]
    assert _rows(schools) == ["0"]


def test_the_district_and_county_tables_are_read_only(app: Connection) -> None:
    output = query_data(
        app,
        PRAGUE,
        "WITH gone AS (DELETE FROM financial.account RETURNING account_id) SELECT * FROM gone",
    ).text
    assert output.startswith("The query failed: WITH clause containing a data-modifying")
    with pytest.raises(psycopg.errors.InsufficientPrivilege), as_user(app, PRAGUE):
        app.execute("UPDATE financial.account SET frequency = 'x'")


@contextmanager
def _set_config_granted(settings: Settings) -> Iterator[None]:
    with connect_admin(settings, load_provision_settings()) as admin:
        admin.execute(f"GRANT EXECUTE ON FUNCTION {SET_CONFIG} TO needtoknow_app")
        try:
            yield
        finally:
            admin.execute(f"REVOKE EXECUTE ON FUNCTION {SET_CONFIG} FROM needtoknow_app")


def test_a_naive_executor_lets_injected_sql_become_another_district(
    app: Connection,
) -> None:
    districts = "SELECT DISTINCT district_id FROM financial.account ORDER BY 1"
    escalation = f"SELECT set_config('app.principals', 'group:district:1', true); {districts}"
    with _set_config_granted(load_settings()), as_user(app, JIHLAVA), app.cursor() as cur:
        cur.execute(escalation)
        while cur.nextset():
            pass
        leaked = [row[0] for row in cur.fetchall()]

    refused = query_data(app, JIHLAVA, escalation).text

    assert leaked == [1]
    assert refused.startswith("The query failed: syntax error")
    single = f"{districts.removesuffix(' ORDER BY 1')} WHERE set_config('app.principals', "
    single += "'group:district:1', true) IS NOT NULL"
    assert query_data(app, JIHLAVA, single).text.startswith(
        "The query failed: permission denied for function set_config"
    )


def test_a_database_name_that_is_not_a_plain_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "dev_tables.json"
    path.write_text('[{"db_id": "public", "table_names_original": ["documents"]}]')
    with pytest.raises(BirdError, match="'public' cannot be a schema name"):
        read_databases(path)


@pytest.mark.parametrize(
    ("line", "refused"),
    [
        ("GRANT SELECT ON public.account TO PUBLIC;", "GRANT SELECT ON public.account"),
        (
            "CREATE INDEX ON public.account (account_id); "
            "GRANT SELECT ON public.account TO PUBLIC;",
            "CREATE INDEX ON public.account",
        ),
    ],
    ids=["other-statement", "second-statement-on-one-line"],
)
def test_the_dump_may_hold_only_single_table_data_and_index_statements(
    owner: Connection, tmp_path: Path, line: str, refused: str
) -> None:
    minidev = tmp_path / "minidev"
    shutil.copytree(MINIDEV, minidev)
    dump = minidev / DUMP
    dump.write_text(dump.read_text(encoding="utf-8") + f"\n{line}\n", encoding="utf-8")
    accounts = _count(owner, "SELECT count(*) FROM financial.account")

    with pytest.raises(BirdError, match=f"does not run: {re.escape(refused)}"):
        load_bird(owner, minidev)

    assert read_databases(minidev / TABLES) == read_databases(MINIDEV / TABLES)
    assert _count(owner, "SELECT count(*) FROM financial.account") == accounts
