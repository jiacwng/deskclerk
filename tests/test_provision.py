# Provisioning takes built-ins away from every role of its database, so it refuses a database
# that already holds another application's tables.

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.config import load_provision_settings, load_settings
from needtoknow.provision import SharedDatabase, connect_admin, prepare_database


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
