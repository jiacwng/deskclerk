# The read-only SQL tool against the real database: the model's SQL cannot change who is asking,
# run a second statement or write, reach another session, a slow or huge query is cut short, and
# row-level security decides the rows of the BIRD financial database.

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from langchain_core.messages import ToolMessage
from psycopg import sql
from psycopg.rows import TupleRow

from fakes import call_tool, final_reply, scripted
from needtoknow.agent import Query, ask
from needtoknow.auth import principals
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.data import (
    CHAR_CAP,
    DESCRIPTION_CUT,
    ROW_CAP,
    QueryResult,
    describe_data,
    query_data,
)
from needtoknow.db import connect_app
from needtoknow.methods import Method
from needtoknow.providers import Spending
from needtoknow.provision import connect_admin

Connection = psycopg.Connection[TupleRow]

PRAGUE = principals("district-1", ["district:1"])
ANALYST = "user:analyst,group:everyone,group:analyst"
DISTRICTS = "SELECT DISTINCT district_id FROM financial.account"
MULTI_STATEMENT = f"SELECT set_config('app.principals', '{ANALYST}', true); {DISTRICTS}"
SINGLE_STATEMENT = f"{DISTRICTS} WHERE set_config('app.principals', '{ANALYST}', true) IS NOT NULL"
MARKER = "victim-query-42"


def _rows(output: str) -> set[str]:
    lines = output.splitlines()
    assert lines[-1].startswith("(")
    return set(lines[1:-1])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("SELECT 1; COMMIT; DROP SCHEMA public CASCADE", "syntax error"),
        (
            "SELECT 1) AS q; COMMIT; DROP SCHEMA public CASCADE; SELECT * FROM (SELECT 1",
            "cannot insert multiple commands into a prepared statement",
        ),
    ],
    ids=["inside", "closing-the-parenthesis"],
)
def test_a_second_statement_fails_and_the_schema_stays(
    app: Connection, owner: Connection, payload: str, message: str
) -> None:
    result = query_data(app, PRAGUE, payload)

    assert result.failed
    assert result.text.startswith(f"The query failed: {message}")
    assert owner.execute("SELECT to_regclass('public.documents')").fetchone() == ("documents",)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (MULTI_STATEMENT, "syntax error"),
        (SINGLE_STATEMENT, "permission denied for function set_config"),
        ("SELECT set_identity('group:analyst')", "the identity is already set"),
        (
            "WITH done AS (UPDATE financial.account SET frequency = 'x' RETURNING account_id) "
            "SELECT * FROM done",
            "WITH clause containing a data-modifying statement must be at the top level",
        ),
        ("SELECT pg_sleep(10)", "canceling statement due to statement timeout"),
    ],
    ids=["multi-statement", "set-config", "set-identity", "data-modifying-with", "timeout"],
)
def test_a_refused_query_gets_a_short_error(
    app: Connection, owner: Connection, payload: str, message: str
) -> None:
    frequencies = "SELECT count(DISTINCT frequency) FROM financial.account"
    before = owner.execute(frequencies).fetchone()

    result = query_data(app, PRAGUE, payload)

    assert result.failed
    assert result.text.startswith(f"The query failed: {message}")
    assert "\n" not in result.text
    assert owner.execute(frequencies).fetchone() == before


def test_the_query_runs_read_only(app: Connection) -> None:
    result = query_data(app, PRAGUE, "SELECT current_setting('transaction_read_only')")
    assert not result.failed
    assert result.text.splitlines() == ["current_setting", "on", "(1 rows)"]


def test_a_result_that_reads_like_an_error_still_succeeds(app: Connection) -> None:
    sql = 'SELECT 1 AS "The query failed: x"'
    rows = "The query failed: x\n1\n(1 rows)"

    assert query_data(app, PRAGUE, sql) == QueryResult(text=rows, failed=False)
    assert _queries(app, sql) == (Query(sql=sql, rows=rows, error=None),)


@pytest.mark.parametrize(
    ("query", "last_line"),
    [
        ("SELECT n FROM generate_series(1, 200) AS n", "(200 rows)"),
        (
            "SELECT n FROM generate_series(1, 500) AS n",
            "The result was cut at 200 rows. Narrow the query or aggregate.",
        ),
    ],
)
def test_the_row_cap_holds(app: Connection, query: str, last_line: str) -> None:
    lines = query_data(app, PRAGUE, query).text.splitlines()

    assert ROW_CAP == 200
    assert lines[1:-1] == [str(n) for n in range(1, ROW_CAP + 1)]
    assert lines[-1] == last_line


def test_a_district_user_sees_only_their_district(app: Connection) -> None:
    output = query_data(app, PRAGUE, f"{DISTRICTS} -- every district").text
    assert output.splitlines()[0] == "district_id"
    assert _rows(output) == {"1"}


def test_an_empty_result_keeps_its_columns(app: Connection) -> None:
    output = query_data(
        app, PRAGUE, "SELECT account_id, frequency FROM financial.account LIMIT 0"
    ).text
    assert output.splitlines() == ["account_id | frequency", "(0 rows)"]


def test_the_size_cap_holds(app: Connection) -> None:
    query = "SELECT repeat('x', 1000) AS line FROM generate_series(1, 50)"

    lines = query_data(app, PRAGUE, query).text.splitlines()

    shown = CHAR_CAP // 1001
    assert lines[1:-1] == ["x" * 1000] * shown
    assert lines[-1] == f"The result was cut at {shown} rows. Narrow the query or aggregate."


@pytest.fixture
def victim(owner: Connection) -> Iterator[Connection]:
    with connect_app(load_settings()) as connection:
        connection.execute(f"SELECT '{MARKER}'")
        yield connection


def test_another_session_shows_its_query_text_to_the_admin(victim: Connection) -> None:
    with connect_admin(load_settings(), load_provision_settings()) as admin:
        row = admin.execute(
            "SELECT query FROM pg_stat_activity WHERE pid = %s", [victim.info.backend_pid]
        ).fetchone()
    assert row is not None
    assert MARKER in row[0]


@pytest.mark.usefixtures("victim")
@pytest.mark.parametrize(
    ("payload", "refused"),
    [
        ("SELECT query FROM pg_stat_activity", "view pg_stat_activity"),
        ("SELECT query FROM pg_stat_get_activity(NULL)", "function pg_stat_get_activity"),
        (
            "SELECT pg_stat_get_backend_activity(b) FROM pg_stat_get_backend_idset() AS b",
            "function pg_stat_get_backend_idset",
        ),
        ("SELECT * FROM pg_locks", "view pg_locks"),
        ("SELECT * FROM pg_lock_status()", "function pg_lock_status"),
        ("SELECT statement FROM pg_prepared_statements", "view pg_prepared_statements"),
        ("SELECT statement FROM pg_cursors", "view pg_cursors"),
        ("SELECT pg_advisory_lock(42)", "function pg_advisory_lock"),
        ("SELECT pg_try_advisory_xact_lock(42)", "function pg_try_advisory_xact_lock"),
        ("SELECT pg_notify('channel', 'message')", "function pg_notify"),
    ],
)
def test_query_data_cannot_read_or_reach_other_sessions(
    app: Connection, payload: str, refused: str
) -> None:
    output = query_data(app, PRAGUE, payload).text
    assert output == f"The query failed: permission denied for {refused}"


@pytest.mark.parametrize("function", ["pg_cancel_backend", "pg_terminate_backend"])
def test_query_data_cannot_cancel_or_kill_another_session(
    app: Connection, victim: Connection, function: str
) -> None:
    call = sql.SQL("SELECT {}({})").format(sql.Identifier(function), victim.info.backend_pid)

    output = query_data(app, PRAGUE, call.as_string()).text

    assert output == f"The query failed: permission denied for function {function}"
    assert victim.execute("SELECT 1").fetchone() == (1,)


def test_query_data_refuses_the_reader_role(reader: Connection) -> None:
    with pytest.raises(ValueError, match="query_data runs as needtoknow_app"):
        query_data(reader, PRAGUE, DISTRICTS)


@pytest.fixture
def notes(owner: Connection) -> Iterator[None]:
    owner.execute("CREATE TABLE shared_notes (note TEXT)")
    owner.execute("COMMENT ON TABLE shared_notes IS 'Notes every user may read.'")
    owner.execute("COMMENT ON COLUMN shared_notes.note IS 'The text of the note.'")
    owner.execute("CREATE VIEW note_lengths AS SELECT length(note) AS chars FROM shared_notes")
    owner.execute("COMMENT ON VIEW note_lengths IS 'The length of each note.'")
    owner.execute("GRANT SELECT ON shared_notes, note_lengths TO needtoknow_app")
    owner.execute("CREATE TABLE private_notes (note TEXT)")
    try:
        yield
    finally:
        owner.execute("DROP VIEW note_lengths")
        owner.execute("DROP TABLE shared_notes, private_notes")


@pytest.mark.usefixtures("notes")
def test_the_description_lists_only_what_the_app_role_can_read(app: Connection) -> None:
    description = describe_data(app, "public")

    assert (
        "shared_notes (table): Notes every user may read.\n  note text: The text of the note."
    ) in description
    assert "note_lengths (view): The length of each note.\n  chars integer" in description
    assert "private_notes" not in description


@pytest.fixture
def schemas(owner: Connection) -> Iterator[None]:
    owner.execute('CREATE SCHEMA "Team Notes"')
    owner.execute('CREATE TABLE "Team Notes".notes (note TEXT)')
    owner.execute('GRANT USAGE ON SCHEMA "Team Notes" TO needtoknow_app')
    owner.execute('GRANT SELECT ON "Team Notes".notes TO needtoknow_app')
    owner.execute("CREATE SCHEMA hidden")
    owner.execute("CREATE TABLE hidden.notes (note TEXT)")
    owner.execute("GRANT SELECT ON hidden.notes TO needtoknow_app")
    try:
        yield
    finally:
        owner.execute('DROP SCHEMA "Team Notes", hidden CASCADE')


@pytest.mark.usefixtures("schemas")
def test_the_schema_list_names_schemas_as_stored_and_skips_those_without_usage(
    app: Connection,
) -> None:
    lines = describe_data(app).splitlines()

    assert "Team Notes (1 tables)" in lines
    assert not any(line.startswith("hidden") for line in lines)
    assert '"Team Notes".notes (table)\n  note text' in describe_data(app, "Team Notes")
    assert describe_data(app, "hidden") == (
        "No schema named hidden that you can read. Call describe_data without a schema."
    )


def test_the_description_is_cut_at_its_table_budget(
    app: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("needtoknow.data.DESCRIBE_TABLES", 2)

    parts = describe_data(app, "public").split("\n\n")

    assert [part.splitlines()[0] for part in parts[1:-1]] == [
        "chunks (table)",
        "doc_access (table)",
    ]
    assert parts[-1] == DESCRIPTION_CUT


def test_the_description_is_cut_at_its_column_budget(
    app: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("needtoknow.data.DESCRIBE_COLUMNS", 3)

    parts = describe_data(app, "public").split("\n\n")

    assert parts[1:] == [
        'chunks (table)\n  id bigint\n  doc_id text\n  "position" integer',
        DESCRIPTION_CUT,
    ]


def _asks(
    connection: Connection,
    *calls: tuple[str, dict[str, Any]],
    method: Method = Method.RLS,
) -> tuple[list[ToolMessage], tuple[str, ...]]:
    model = scripted(*(call_tool(name, args) for name, args in calls), final_reply("Done."))
    question = "The memo says to run this query for the figures of every district."
    answer = ask(question, connection, PRAGUE, model, Spending("claude-haiku-5-5", 1.0), method)
    results = [message for message in model.prompts[-1] if isinstance(message, ToolMessage)]
    return results, answer.tool_outputs


def test_an_injected_escalation_through_the_agent_leaves_the_rows_filtered(
    app: Connection,
) -> None:
    results, outputs = _asks(
        app,
        ("query_data", {"sql": SINGLE_STATEMENT}),
        ("query_data", {"sql": MULTI_STATEMENT}),
        ("query_data", {"sql": DISTRICTS}),
    )

    assert outputs[0].startswith("The query failed: permission denied")
    assert outputs[1].startswith("The query failed: syntax error")
    assert _rows(outputs[2]) == {"1"}
    assert [result.status for result in results] == ["success"] * 3


def test_describe_data_takes_a_schema_or_nothing(app: Connection) -> None:
    results, outputs = _asks(
        app,
        ("describe_data", {"view": "account"}),
        ("describe_data", {}),
        ("describe_data", {"schema": "financial"}),
        ("describe_data", {"schema": "nowhere"}),
    )

    assert results[0].status == "error"
    assert results[0].content == (
        "wrong arguments, call describe_data with a schema only or with no arguments"
    )
    assert outputs == (
        describe_data(app),
        describe_data(app, "financial"),
        "No schema named nowhere that you can read. Call describe_data without a schema.",
    )


def _queries(connection: Connection, sql: str) -> tuple[Query, ...]:
    model = scripted(call_tool("query_data", {"sql": sql}), final_reply("Done."))
    answer = ask("Run it.", connection, PRAGUE, model, Spending("claude-haiku-5-5", 1.0))
    return answer.queries


def test_the_data_tools_exist_only_under_row_level_security(reader: Connection) -> None:
    results, outputs = _asks(reader, ("query_data", {"sql": DISTRICTS}), method=Method.IN_QUERY)

    assert results[0].status == "error"
    assert outputs == ()
