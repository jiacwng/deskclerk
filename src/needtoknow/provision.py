# Sets up the database: the admin creates the owner, app and reader roles and takes the
# dangerous built-ins away from them, and the owner creates the tables. Only ingestion and the
# tests run this; the API never does. The reader role sees every row, so it can log in only when
# the evaluation asks for it. The built-ins are taken from every role of the database, so
# needtoknow needs a database of its own and refuses one that holds tables it did not create.

from importlib.resources import files
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.config import ProvisionSettings, Settings
from needtoknow.db import APP_ROLE, READER_ROLE, connect
from needtoknow.schema import SchemaMismatch, built_with, current

OWNER_ROLE = "needtoknow_owner"
SCHEMA = files("needtoknow").joinpath("sql/schema.sql")

_CREATE_ROLES = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'needtoknow_owner') THEN
        CREATE ROLE needtoknow_owner;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'needtoknow_app') THEN
        CREATE ROLE needtoknow_app;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'needtoknow_reader') THEN
        CREATE ROLE needtoknow_reader;
    END IF;
END
$$
"""

# SQL the model writes runs as the app role, on a connection other employees' requests share.
# These built-ins would let it rewrite the identity the policies read (set_config, which only the
# owner keeps, for set_identity), cancel or kill other sessions, read their SQL, prepared
# statements, cursors or locks, notify their listeners, or hold an advisory lock across requests.
_REVOKE_FROM_PUBLIC = r"""
DO $$
DECLARE
    target REGPROCEDURE;
BEGIN
    FOR target IN
        SELECT oid::regprocedure
        FROM pg_proc
        WHERE pronamespace = 'pg_catalog'::regnamespace
          AND (
            proname IN (
                'set_config', 'pg_cancel_backend', 'pg_terminate_backend',
                'pg_stat_get_activity', 'pg_lock_status', 'pg_blocking_pids',
                'pg_safe_snapshot_blocking_pids', 'pg_prepared_statement', 'pg_cursor',
                'pg_notify'
            )
            OR proname LIKE 'pg\_stat\_get\_backend\_%'
            OR proname LIKE 'pg\_%advisory\_%'
          )
    LOOP
        EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC', target);
    END LOOP;
END
$$;
REVOKE SELECT ON pg_catalog.pg_stat_activity, pg_catalog.pg_locks,
    pg_catalog.pg_prepared_statements, pg_catalog.pg_cursors FROM PUBLIC;
"""


_RELATIONS_OF_OTHERS = r"""
SELECT n.nspname || '.' || c.relname
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND n.nspname <> 'information_schema' AND n.nspname NOT LIKE 'pg\_%'
  AND c.relowner IS DISTINCT FROM (SELECT oid FROM pg_roles WHERE rolname = 'needtoknow_owner')
ORDER BY 1
"""


class SharedDatabase(RuntimeError):
    pass


# Every request holds one app connection, so a flood of requests or one heavy query must not
# take the server's connections, memory or disk from everyone else.
_APP_LIMITS = """
ALTER ROLE needtoknow_app CONNECTION LIMIT 50;
ALTER ROLE needtoknow_app SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE needtoknow_app SET work_mem = '16MB';
ALTER ROLE needtoknow_app SET temp_file_limit = '256MB';
"""


def connect_admin(settings: Settings, provision: ProvisionSettings) -> psycopg.Connection[TupleRow]:
    return connect(settings, provision.admin_user, provision.admin_password)


def connect_owner(settings: Settings, provision: ProvisionSettings) -> psycopg.Connection[TupleRow]:
    connection = connect(settings, OWNER_ROLE, provision.owner_password)
    register_vector(connection)
    return connection


def prepare_database(
    admin: psycopg.Connection[Any],
    settings: Settings,
    provision: ProvisionSettings,
    evaluation: bool,
) -> None:
    provision.check_for(settings.db_host)
    others = [row[0] for row in admin.execute(_RELATIONS_OF_OTHERS)]
    if others:
        raise SharedDatabase(
            f"needtoknow needs a database of its own, and {settings.db_name} holds relations "
            f"it did not create: {', '.join(others)}"
        )
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(_CREATE_ROLES)
    for role, password, login in (
        (OWNER_ROLE, provision.owner_password, True),
        (APP_ROLE, settings.app_password, True),
        (READER_ROLE, provision.reader_password, evaluation),
    ):
        # ALTER ROLE takes no bind parameters, so the password is quoted by psycopg instead.
        admin.execute(
            sql.SQL("ALTER ROLE {} {} NOSUPERUSER NOBYPASSRLS PASSWORD {}").format(
                sql.Identifier(role),
                sql.SQL("LOGIN" if login else "NOLOGIN"),
                sql.Literal(password),
            )
        )
    admin.execute(_APP_LIMITS)
    admin.execute(
        sql.SQL("GRANT CREATE ON DATABASE {} TO needtoknow_owner").format(
            sql.Identifier(settings.db_name)
        )
    )
    admin.execute("GRANT USAGE, CREATE ON SCHEMA public TO needtoknow_owner")
    admin.execute("GRANT USAGE ON SCHEMA public TO needtoknow_app")
    admin.execute("GRANT USAGE ON SCHEMA public TO needtoknow_reader")
    admin.execute(_REVOKE_FROM_PUBLIC)
    admin.execute(
        "GRANT EXECUTE ON FUNCTION pg_catalog.set_config(text, text, boolean) TO needtoknow_owner"
    )


# The tables are created on first use and kept while their version matches the code's. Older
# tables that hold documents are dropped only when replace says so.
def create_schema(owner: psycopg.Connection[Any], replace: bool) -> None:
    built = built_with(owner)
    code = current()
    if not replace:
        if built is not None and built.version == code.version:
            return
        stored = _stored_documents(owner)
        if stored:
            found = "no version" if built is None else f"schema version {built.version}"
            raise SchemaMismatch(
                f"the database holds {stored} documents in tables with {found}, and this code "
                f"needs schema version {code.version}: python -m needtoknow.ingest "
                "--replace-schema reindex drops them and creates the tables again, then load the "
                "documents again"
            )
    with owner.transaction():
        owner.execute(SCHEMA.read_text(encoding="utf-8"))
        owner.execute(
            "INSERT INTO schema_info (version, embedding_model, chunk_chars) VALUES (%s, %s, %s)",
            [code.version, code.embedding_model, code.chunk_chars],
        )


def _stored_documents(owner: psycopg.Connection[Any]) -> int:
    if owner.execute("SELECT to_regclass('public.documents')").fetchone() == (None,):
        return 0
    row = owner.execute("SELECT count(*) FROM documents").fetchone()
    return 0 if row is None else int(row[0])
