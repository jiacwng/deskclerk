# The baseline migration builds the schema the modules used to create themselves, and several
# copies of the app starting at once on an empty database all succeed.

from concurrent.futures import ThreadPoolExecutor

import psycopg
from psycopg.rows import tuple_row

from conftest import DATABASE_URL
from deskclerk.database import Database, sqlalchemy_url

CREATE_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS oauth_flows ("
    "state TEXT PRIMARY KEY, user_id TEXT NOT NULL, server_name TEXT NOT NULL, "
    "deadline DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS oauth_callbacks ("
    "state TEXT PRIMARY KEY, code TEXT, iss TEXT, error TEXT, "
    "created_at DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS conversations ("
    "user_id TEXT NOT NULL, id UUID NOT NULL, title TEXT NOT NULL, "
    "updated_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (user_id, id))",
    "CREATE TABLE IF NOT EXISTS quota_hits ("
    "user_id TEXT NOT NULL, hit_at DOUBLE PRECISION NOT NULL)",
    "CREATE INDEX IF NOT EXISTS quota_hits_user ON quota_hits (user_id, hit_at)",
    "CREATE TABLE IF NOT EXISTS scim_users ("
    "position BIGSERIAL, id TEXT PRIMARY KEY, external_id TEXT NOT NULL UNIQUE, "
    "user_name TEXT NOT NULL, active BOOLEAN NOT NULL)",
    "CREATE UNIQUE INDEX IF NOT EXISTS scim_users_user_name ON scim_users (lower(user_name))",
    "CREATE TABLE IF NOT EXISTS sessions ("
    "id_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_id TEXT NOT NULL, "
    "name TEXT NOT NULL, sid TEXT, subject TEXT, groups TEXT NOT NULL, "
    "expires_at DOUBLE PRECISION NOT NULL)",
    "CREATE TABLE IF NOT EXISTS costs ("
    "user_id TEXT NOT NULL, month TEXT NOT NULL, source TEXT NOT NULL, "
    "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
    "cache_read_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL, "
    "usd DOUBLE PRECISION NOT NULL)",
    "CREATE INDEX IF NOT EXISTS costs_user_month ON costs (user_id, month)",
    "CREATE TABLE IF NOT EXISTS secrets ("
    "user_id TEXT NOT NULL, server_name TEXT NOT NULL, kind TEXT NOT NULL, "
    "value BYTEA NOT NULL, PRIMARY KEY (user_id, server_name, kind))",
    "CREATE TABLE IF NOT EXISTS key_check ("
    "only_row BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_row), value BYTEA NOT NULL)",
]
SCRATCH = "legacy_schema"


def describe(connection: psycopg.Connection[tuple[object, ...]], schema: str) -> list[object]:
    tables = sorted(
        str(row[0])
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name <> 'alembic_version'",
            (schema,),
        )
    )
    columns = connection.execute(
        "SELECT table_name, column_name, ordinal_position, data_type, is_nullable, "
        "replace(column_default, %s, '') FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = ANY(%s) ORDER BY 1, 3",
        (schema + ".", schema, tables),
    ).fetchall()
    indexes = connection.execute(
        "SELECT tablename, indexname, replace(indexdef, %s, '') FROM pg_indexes "
        "WHERE schemaname = %s AND tablename = ANY(%s) ORDER BY 1, 2",
        (schema + ".", schema, tables),
    ).fetchall()
    constraints = connection.execute(
        "SELECT c.relname, k.conname, pg_get_constraintdef(k.oid) FROM pg_constraint k "
        "JOIN pg_class c ON c.oid = k.conrelid "
        "WHERE c.relnamespace = %s::regnamespace AND c.relname = ANY(%s) ORDER BY 1, 2",
        (schema, tables),
    ).fetchall()
    return [tables, columns, indexes, constraints]


def test_the_baseline_builds_the_schema_the_modules_used_to_create(database: Database) -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True, row_factory=tuple_row) as connection:
        connection.execute(f"DROP SCHEMA IF EXISTS {SCRATCH} CASCADE")
        connection.execute(f"CREATE SCHEMA {SCRATCH}")
        connection.execute(f"SET search_path TO {SCRATCH}")
        for statement in CREATE_STATEMENTS:
            connection.execute(statement.encode())
        connection.execute("SET search_path TO public")
        old = describe(connection, SCRATCH)
        new = describe(connection, "public")
        connection.execute(f"DROP SCHEMA {SCRATCH} CASCADE")
    assert len(old[0]) == 9  # type: ignore[arg-type]
    assert new == old


def test_migrating_again_changes_nothing(database: Database) -> None:
    database.migrate()
    with psycopg.connect(DATABASE_URL) as connection:
        versions = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    assert versions == [("0001",)]


def test_copies_starting_at_once_on_an_empty_database_all_succeed(database: Database) -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")

    def start() -> None:
        Database(DATABASE_URL).close()

    with ThreadPoolExecutor(max_workers=6) as executor:
        for future in [executor.submit(start) for _ in range(6)]:
            future.result()
    with psycopg.connect(DATABASE_URL) as connection:
        tables = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchone()
    assert tables == (10,)


def test_a_keyword_connection_string_becomes_a_sqlalchemy_url() -> None:
    url = sqlalchemy_url("host=db port=5433 dbname=app user=u password=p@ss sslmode=require")
    assert url.render_as_string(hide_password=False) == (
        "postgresql+psycopg://u:p%40ss@db:5433/app?sslmode=require"
    )
