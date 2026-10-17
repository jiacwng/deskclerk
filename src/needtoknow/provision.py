# Sets up the database: the admin creates the owner, app and reader roles and takes the
# dangerous built-ins away from them, and the owner creates the tables. Only ingestion and the
# tests run this; the API never does. The reader role sees every row, so it can log in only when
# the evaluation asks for it.

from importlib.resources import files
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.config import ProvisionSettings, Settings
from needtoknow.db import APP_ROLE, READER_ROLE, connect

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


def create_schema(owner: psycopg.Connection[Any]) -> None:
    with owner.transaction():
        owner.execute(SCHEMA.read_text(encoding="utf-8"))
