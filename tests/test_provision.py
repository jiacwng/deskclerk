# Provisioning takes built-ins away from every role of its database, so it refuses a database
# that already holds another application's tables. The reader role's grants and policies exist
# only when the evaluation asks for them.

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.config import load_provision_settings, load_settings
from needtoknow.provision import (
    SharedDatabase,
    connect_admin,
    connect_owner,
    create_schema,
    prepare_database,
)

_READER_POLICIES = """
SELECT polname FROM pg_policy
WHERE 'needtoknow_reader'::regrole = ANY (polroles)
ORDER BY polname
"""


def test_a_database_with_tables_of_another_application_is_refused(
    owner: psycopg.Connection[TupleRow],
) -> None:
    settings = load_settings()
    provision = load_provision_settings()
    with connect_admin(settings, provision) as admin:
        admin.execute("CREATE SCHEMA other_app; CREATE TABLE other_app.orders (id INT)")
        try:
            with pytest.raises(
                SharedDatabase, match="holds relations it did not create: other_app.orders"
            ):
                prepare_database(admin, settings, provision, evaluation=True)
        finally:
            admin.execute("DROP SCHEMA other_app CASCADE")


def test_a_database_with_only_its_own_tables_is_prepared_again(
    owner: psycopg.Connection[TupleRow],
) -> None:
    settings = load_settings()
    provision = load_provision_settings()
    with connect_admin(settings, provision) as admin:
        prepare_database(admin, settings, provision, evaluation=True)


# A database of its own, because the tables of the shared one carry the evaluation's policies.
def test_the_reader_policies_are_created_only_for_the_evaluation(
    owner: psycopg.Connection[TupleRow],
) -> None:
    settings = load_settings()
    provision = load_provision_settings()
    product = settings.model_copy(update={"db_name": f"{settings.db_name}_product"})
    name = sql.Identifier(product.db_name)
    with connect_admin(settings, provision) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(name))
    try:
        with connect_admin(product, provision) as admin:
            prepare_database(admin, product, provision, evaluation=True)
        with connect_owner(product, provision) as connection:
            create_schema(connection, replace=False, evaluation=False)
            without = [row[0] for row in connection.execute(_READER_POLICIES)]
            granted = connection.execute(
                "SELECT has_table_privilege('needtoknow_reader', 'documents', 'SELECT')"
            ).fetchone()
            for _ in range(2):
                create_schema(connection, replace=False, evaluation=True)
            with_evaluation = [row[0] for row in connection.execute(_READER_POLICIES)]
    finally:
        with connect_admin(settings, provision) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))

    assert without == []
    assert granted == (False,)
    assert with_evaluation == ["chunks_reader", "doc_access_reader", "documents_reader"]
