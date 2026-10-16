# PostgreSQL access for the API: connecting as the app role and running queries as one employee,
# so that row-level security filters every read. Creating roles and tables is in provision.py.

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.pq import TransactionStatus
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from needtoknow.config import ProvisionSettings, Settings

APP_ROLE = "needtoknow_app"
READER_ROLE = "needtoknow_reader"


def connect(settings: Settings, user: str, password: str) -> psycopg.Connection[TupleRow]:
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=user,
        password=password,
        autocommit=True,
    )


def connect_app(settings: Settings) -> psycopg.Connection[TupleRow]:
    connection = connect(settings, APP_ROLE, settings.app_password)
    register_vector(connection)
    return connection


def connect_reader(
    settings: Settings, provision: ProvisionSettings
) -> psycopg.Connection[TupleRow]:
    connection = connect(settings, READER_ROLE, provision.reader_password)
    register_vector(connection)
    return connection


Pool = ConnectionPool[psycopg.Connection[TupleRow]]


# The API, the MCP server and the agent's tools borrow a connection from a bounded pool for one
# transaction at a time. A connection goes back idle, so the identity set_identity gave it has
# already ended.
def app_pool(settings: Settings, size: int) -> Pool:
    return _pool(settings, APP_ROLE, settings.app_password, size)


def reader_pool(settings: Settings, provision: ProvisionSettings, size: int) -> Pool:
    return _pool(settings, READER_ROLE, provision.reader_password, size)


def _pool(settings: Settings, user: str, password: str, size: int) -> Pool:
    return ConnectionPool(
        kwargs={
            "host": settings.db_host,
            "port": settings.db_port,
            "dbname": settings.db_name,
            "user": user,
            "password": password,
            "autocommit": True,
        },
        configure=register_vector,
        min_size=1,
        max_size=size,
        check=ConnectionPool.check_connection,
        open=True,
    )


@contextmanager
def as_user(app: psycopg.Connection[Any], principals: Iterable[str]) -> Iterator[None]:
    names = sorted(principals)
    for name in names:
        if not name or "," in name:
            raise ValueError(f"principal {name!r} is empty or contains a comma")
    # Inside an open transaction this block would only be a savepoint, and the identity would
    # outlive it until the outer transaction ends.
    if app.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("as_user needs a connection with no open transaction")
    with app.transaction():
        app.execute("SELECT set_identity(%s)", [",".join(names)])
        yield
