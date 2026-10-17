# The startup check against the real database: the loaded data passes with the shipped list of
# public relations, and each way of reading past row-level security is reported once a throwaway
# relation, role or grant creates it. The API and the MCP server refuse to start while it fails.

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from fakes import scripted
from needtoknow import exposure
from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup
from needtoknow.config import PUBLIC_RELATIONS, Settings, load_provision_settings, load_settings
from needtoknow.db import APP_ROLE, connect
from needtoknow.exposure import Exposed, find_exposures
from needtoknow.mcp_server import create_server
from needtoknow.provision import connect_admin

Connection = psycopg.Connection[TupleRow]
SCHEMA = "exposure_test"
PROBE_PASSWORD = "probe"


@pytest.fixture
def admin(owner: Connection) -> Iterator[Connection]:
    with connect_admin(load_settings(), load_provision_settings()) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(SCHEMA)))
        connection.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(SCHEMA), sql.Identifier(APP_ROLE)
            )
        )
        try:
            yield connection
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(SCHEMA)))


# A login role of its own stands in for the app role where the test changes the role itself, so
# the real app role, which other databases on this server share, is never altered.
@pytest.fixture
def probe(admin: Connection) -> Iterator[tuple[str, Connection]]:
    settings = load_settings()
    name = f"{settings.db_name}_probe"
    group = f"{settings.db_name}_probe_owner"
    admin.execute(
        sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(name), sql.Literal(PROBE_PASSWORD)
        )
    )
    admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(group)))
    admin.execute(
        sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
            sql.Identifier(SCHEMA), sql.Identifier(name)
        )
    )
    try:
        with connect(settings, name, PROBE_PASSWORD) as connection:
            yield name, connection
    finally:
        for role in (name, group):
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


def _readable(admin: Connection, ddl: str, relation: str) -> None:
    admin.execute(ddl)
    admin.execute(
        sql.SQL("GRANT SELECT ON {}.{} TO {}").format(
            sql.Identifier(SCHEMA), sql.Identifier(relation), sql.Identifier(APP_ROLE)
        )
    )


def _guarded(admin: Connection) -> None:
    _readable(
        admin,
        "CREATE TABLE exposure_test.guarded (owner TEXT);"
        "ALTER TABLE exposure_test.guarded ENABLE ROW LEVEL SECURITY;"
        "CREATE POLICY own_rows ON exposure_test.guarded FOR SELECT TO needtoknow_app "
        "USING (owner = current_setting('app.principals', true))",
        "guarded",
    )


def test_the_loaded_data_passes_with_the_shipped_public_list(app: Connection) -> None:
    assert find_exposures(app, PUBLIC_RELATIONS) == []


def test_the_public_list_is_what_the_bird_data_needs(app: Connection) -> None:
    assert find_exposures(app, set()) == [
        "financial.district: row-level security is off",
        "public.schema_info: a policy lets the app role read every row",
    ]


def test_a_table_without_row_level_security_is_reported(app: Connection, admin: Connection) -> None:
    _readable(admin, "CREATE TABLE exposure_test.plain (a INT)", "plain")
    _guarded(admin)

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.plain: row-level security is off"
    ]
    assert find_exposures(app, PUBLIC_RELATIONS | {"exposure_test.plain"}) == []
    assert find_exposures(app, PUBLIC_RELATIONS | {SCHEMA}) == []


def test_a_table_the_app_cannot_select_is_not_reported(app: Connection, admin: Connection) -> None:
    admin.execute("CREATE TABLE exposure_test.hidden (a INT)")

    assert find_exposures(app, PUBLIC_RELATIONS) == []


def test_a_partition_without_row_level_security_is_reported(
    app: Connection, admin: Connection
) -> None:
    _readable(
        admin,
        "CREATE TABLE exposure_test.parted (a INT) PARTITION BY RANGE (a);"
        "ALTER TABLE exposure_test.parted ENABLE ROW LEVEL SECURITY;"
        "CREATE TABLE exposure_test.part PARTITION OF exposure_test.parted "
        "FOR VALUES FROM (0) TO (10)",
        "parted",
    )
    admin.execute(f"GRANT SELECT ON exposure_test.part TO {APP_ROLE}")

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.part: row-level security is off"
    ]


def test_a_policy_that_lets_every_row_through_is_reported(
    app: Connection, admin: Connection
) -> None:
    _readable(
        admin,
        "CREATE TABLE exposure_test.open (a INT);"
        "ALTER TABLE exposure_test.open ENABLE ROW LEVEL SECURITY;"
        "CREATE POLICY everyone ON exposure_test.open FOR SELECT USING (true)",
        "open",
    )

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.open: a policy lets the app role read every row"
    ]


def test_a_view_that_reads_as_its_owner_is_reported(app: Connection, admin: Connection) -> None:
    _guarded(admin)
    _readable(
        admin, "CREATE VIEW exposure_test.definer AS SELECT * FROM exposure_test.guarded", "definer"
    )
    _readable(
        admin,
        "CREATE VIEW exposure_test.invoker WITH (security_invoker = on) "
        "AS SELECT * FROM exposure_test.guarded",
        "invoker",
    )

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.definer: the view reads as its owner; set security_invoker = true"
    ]


def test_an_invoker_view_over_an_unprotected_table_is_reported(
    app: Connection, admin: Connection
) -> None:
    admin.execute("CREATE TABLE exposure_test.plain (a INT)")
    _readable(
        admin,
        "CREATE VIEW exposure_test.invoker WITH (security_invoker) "
        "AS SELECT * FROM exposure_test.plain",
        "invoker",
    )

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.invoker: it reads exposure_test.plain, which is not protected"
    ]
    assert find_exposures(app, PUBLIC_RELATIONS | {"exposure_test.plain"}) == []


def test_a_materialized_view_is_reported_unless_listed(app: Connection, admin: Connection) -> None:
    _guarded(admin)
    _readable(
        admin,
        "CREATE MATERIALIZED VIEW exposure_test.copied AS SELECT * FROM exposure_test.guarded",
        "copied",
    )

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.copied: a materialized view cannot have row-level security"
    ]
    assert find_exposures(app, PUBLIC_RELATIONS | {"exposure_test.copied"}) == []


def test_a_foreign_table_is_reported_unless_listed(app: Connection, admin: Connection) -> None:
    admin.execute("CREATE EXTENSION postgres_fdw")
    try:
        admin.execute("CREATE SERVER exposure_test_server FOREIGN DATA WRAPPER postgres_fdw")
        _readable(
            admin,
            "CREATE FOREIGN TABLE exposure_test.remote (a INT) SERVER exposure_test_server",
            "remote",
        )

        assert find_exposures(app, PUBLIC_RELATIONS) == [
            "exposure_test.remote: a foreign table cannot have row-level security"
        ]
        assert find_exposures(app, PUBLIC_RELATIONS | {"exposure_test.remote"}) == []
    finally:
        admin.execute("DROP EXTENSION postgres_fdw CASCADE")


def test_a_security_definer_function_is_reported(app: Connection, admin: Connection) -> None:
    admin.execute(
        "CREATE FUNCTION exposure_test.everything() RETURNS BIGINT LANGUAGE sql SECURITY DEFINER "
        "AS 'SELECT count(*) FROM documents'"
    )

    assert find_exposures(app, PUBLIC_RELATIONS) == [
        "exposure_test.everything(): a SECURITY DEFINER function the app role can run"
    ]


def test_a_table_the_role_owns_is_reported_until_forced(
    probe: tuple[str, Connection], admin: Connection
) -> None:
    name, connection = probe
    admin.execute(
        f"CREATE TABLE exposure_test.mine (a INT);"
        f"ALTER TABLE exposure_test.mine ENABLE ROW LEVEL SECURITY;"
        f'ALTER TABLE exposure_test.mine OWNER TO "{name}"'
    )

    assert find_exposures(connection, set()) == [
        "exposure_test.mine: the app role owns it, so its policies do not apply"
    ]

    admin.execute("ALTER TABLE exposure_test.mine FORCE ROW LEVEL SECURITY")
    assert find_exposures(connection, set()) == []


def test_a_table_owned_by_a_role_the_role_belongs_to_is_reported(
    probe: tuple[str, Connection], admin: Connection
) -> None:
    name, connection = probe
    group = f"{name}_owner"
    admin.execute(
        f"CREATE TABLE exposure_test.theirs (a INT);"
        f"ALTER TABLE exposure_test.theirs ENABLE ROW LEVEL SECURITY;"
        f'ALTER TABLE exposure_test.theirs OWNER TO "{group}";'
        f'GRANT SELECT ON exposure_test.theirs TO "{name}";'
        f'GRANT "{group}" TO "{name}"'
    )

    assert find_exposures(connection, set()) == [
        "exposure_test.theirs: the app role owns it, so its policies do not apply"
    ]


def test_a_role_that_bypasses_row_level_security_is_reported(
    probe: tuple[str, Connection], admin: Connection
) -> None:
    name, connection = probe
    admin.execute(f'ALTER ROLE "{name}" BYPASSRLS')

    assert find_exposures(connection, set()) == [
        "the app role has BYPASSRLS, so no policy applies to it"
    ]


def test_a_role_that_can_run_set_config_is_reported(
    probe: tuple[str, Connection], admin: Connection
) -> None:
    name, connection = probe
    admin.execute(
        f'GRANT EXECUTE ON FUNCTION pg_catalog.set_config(text, text, boolean) TO "{name}"'
    )

    try:
        assert find_exposures(connection, set()) == [
            "pg_catalog.set_config(text,text,boolean): the app role can run it and "
            "rewrite the identity the policies read"
        ]
    finally:
        admin.execute(
            f'REVOKE EXECUTE ON FUNCTION pg_catalog.set_config(text, text, boolean) FROM "{name}"'
        )


def test_a_superuser_is_reported(admin: Connection) -> None:
    assert "the app role is a superuser, so no policy applies to it" in find_exposures(
        admin, PUBLIC_RELATIONS
    )


def test_the_api_and_the_mcp_server_refuse_to_start(
    admin: Connection, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    _readable(admin, "CREATE TABLE exposure_test.plain (a INT)", "plain")

    with pytest.raises(Exposed, match="NEEDTOKNOW_PUBLIC_RELATIONS:\n  exposure_test.plain: row"):
        create_app(token_settings, token_signing_key, scripted())
    with pytest.raises(Exposed, match="exposure_test.plain"):
        create_server(token_settings, token_signing_key, None)


def test_the_command_fails_while_a_relation_is_exposed(
    admin: Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    exposure.main()
    assert "only protected or public relations" in capsys.readouterr().out

    _readable(admin, "CREATE TABLE exposure_test.plain (a INT)", "plain")
    with pytest.raises(SystemExit, match="exposure_test.plain: row-level security is off"):
        exposure.main()
