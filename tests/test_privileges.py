# The functions the app role can run, compared with a reviewed list: a Postgres upgrade or a
# provisioning change that lets the model's SQL reach a new function with side effects fails here
# until someone has looked at it.

from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

REVIEWED = Path(__file__).parent / "data" / "app_volatile_functions.txt"

_VOLATILE = """
SELECT DISTINCT p.proname FROM pg_proc p
WHERE p.pronamespace = 'pg_catalog'::regnamespace AND p.provolatile = 'v' AND p.prokind = 'f'
  AND has_function_privilege(p.oid, 'EXECUTE')
"""


def test_the_app_role_runs_only_reviewed_volatile_functions(
    app: psycopg.Connection[TupleRow],
) -> None:
    lines = REVIEWED.read_text(encoding="utf-8").splitlines()
    reviewed = {line for line in lines if line and not line.startswith("#")}

    runnable = {row[0] for row in app.execute(_VOLATILE).fetchall()}

    assert runnable - reviewed == set()
    assert "set_config" not in runnable
    assert "pg_terminate_backend" not in runnable
