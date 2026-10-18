# A company's own database for the SQL tools: data-setup runs again without change, leaves the
# built-ins it takes from the app role to the roles the company had before, the agent's SQL tools
# read it as the employee asking while documents stay in the needtoknow database, and a table
# there that the app role reads without a policy stops startup.

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from fakes import HAIKU, call_tool, final_reply, scripted
from needtoknow import db
from needtoknow.agent import ask, employee_tools
from needtoknow.auth import principals
from needtoknow.config import Settings, company_database, load_provision_settings
from needtoknow.db import APP_ROLE, Pool, connect
from needtoknow.exposure import DATA_NOT_SET_UP, Exposed, check_startup
from needtoknow.providers import Spending
from needtoknow.provision import connect_admin, prepare_company_database

SALES = "SELECT team, amount FROM sales ORDER BY team"

_GRANTS = """
SELECT
    has_function_privilege(%(role)s, 'pg_catalog.set_config(text, text, boolean)', 'EXECUTE'),
    has_function_privilege(%(role)s, 'pg_catalog.pg_cancel_backend(integer)', 'EXECUTE'),
    has_function_privilege(%(role)s, 'pg_catalog.pg_advisory_lock(bigint)', 'EXECUTE'),
    has_table_privilege(%(role)s, 'pg_catalog.pg_stat_activity', 'SELECT')
"""

_IDENTITY_OWNER = """
SELECT r.rolname, r.rolcanlogin
FROM pg_proc p JOIN pg_roles r ON r.oid = p.proowner
WHERE p.oid = 'public.set_identity(text)'::regprocedure
"""


def _admin(company: Settings) -> psycopg.Connection[TupleRow]:
    return connect_admin(company_database(company), load_provision_settings())


def _state(company: Settings) -> tuple[object, ...]:
    with _admin(company) as admin:
        app = admin.execute(_GRANTS, {"role": APP_ROLE}).fetchone()
        before = admin.execute(_GRANTS, {"role": f"{company.data_name}_reporting"}).fetchone()
        owner = admin.execute(_IDENTITY_OWNER).fetchone()
    return app, before, owner


@pytest.fixture
def data_pool(app_pool: Pool, company: Settings) -> Iterator[Pool]:
    with db.data_pool(company, app_pool, size=1) as pool:
        yield pool


def test_data_setup_runs_again_and_changes_nothing(company: Settings) -> None:
    first = _state(company)
    with _admin(company) as admin:
        prepare_company_database(admin, company, load_provision_settings())

    assert _state(company) == first
    assert first[2] == ("needtoknow_data_owner", False)


def test_a_role_the_company_had_before_keeps_the_built_ins(company: Settings) -> None:
    app, before, _ = _state(company)

    assert app == (False, False, False, False)
    assert before == (True, True, True, True)


def test_the_app_role_cannot_rewrite_its_identity_or_watch_other_sessions(
    company: Settings,
) -> None:
    data = company_database(company)
    with connect(data, APP_ROLE, data.app_password) as app:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app.execute("SELECT set_config('app.principals', 'group:north', false)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app.execute("SELECT query FROM pg_stat_activity")


def test_each_employee_reads_only_their_rows_of_the_company_database(
    app_pool: Pool, data_pool: Pool
) -> None:
    rows = {}
    for group in ("north", "south"):
        model = scripted(call_tool("query_data", {"sql": SALES}), final_reply("Done."))
        employee = principals("ana", [group])
        tools = employee_tools(app_pool, employee, None, data_pool)
        [query] = ask("What did we sell?", tools, model, Spending(HAIKU, 1.0)).queries
        rows[group] = query.rows

    assert rows == {
        "north": "team | amount\ngroup:north | 100\n(1 rows)",
        "south": "team | amount\ngroup:south | 200\n(1 rows)",
    }


def test_documents_stay_in_the_needtoknow_database(app_pool: Pool, data_pool: Pool) -> None:
    model = scripted(call_tool("search_documents", {"query": "Tiger Team"}), final_reply("Done."))
    tools = employee_tools(app_pool, principals("kaminski-v", []), None, data_pool)

    answer = ask("Who helps the Tiger Team?", tools, model, Spending(HAIKU, 1.0))

    assert answer.tool_outputs[0].startswith("[enron-")


@pytest.mark.parametrize("table", ["salaries", "schema_info"])
def test_a_table_the_app_role_reads_without_a_policy_stops_startup(
    company: Settings, table: str
) -> None:
    check_startup(company)
    name = sql.Identifier(table)
    with _admin(company) as admin:
        admin.execute(sql.SQL("CREATE TABLE {} (name TEXT, amount INTEGER)").format(name))
        admin.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(name, sql.Identifier(APP_ROLE)))
        try:
            with pytest.raises(Exposed, match=f"public.{table}: row-level security is off"):
                check_startup(company)
        finally:
            admin.execute(sql.SQL("DROP TABLE {}").format(name))


def test_an_unreachable_company_database_stops_startup(company: Settings) -> None:
    unreachable = company.model_copy(update={"data_port": 1})

    with pytest.raises(SystemExit, match=DATA_NOT_SET_UP):
        check_startup(unreachable)
