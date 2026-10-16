# The agent's read-only SQL: a list of the schemas the app role can read, a description of one
# schema's tables and views named as SQL must write them, and a SELECT that runs as the
# employee asking, so row-level security decides the rows. The SQL is never filtered, only
# wrapped in a subquery, sent as one statement, read-only and under a timeout, a row cap and a
# size cap.

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from needtoknow.db import APP_ROLE, as_user

ROW_CAP = 200
CHAR_CAP = 16_000
TIMEOUT_MS = 3000


# The caps keep a huge result out of the model's context. The evaluation raises them, because a
# benchmark answer the agent never sees is scored as a wrong answer.
@dataclass(frozen=True)
class Caps:
    rows: int
    chars: int


DEFAULT_CAPS = Caps(rows=ROW_CAP, chars=CHAR_CAP)

DESCRIBE_TABLES = 40
DESCRIBE_COLUMNS = 300
DESCRIPTION_CUT = "The list was cut. Query information_schema.columns for the rest."

_SCHEMAS = """
SELECT n.nspname, obj_description(n.oid, 'pg_namespace'), count(*)
FROM pg_namespace n
JOIN pg_class c ON c.relnamespace = n.oid
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND has_schema_privilege(n.oid, 'USAGE')
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND has_any_column_privilege(c.oid, 'SELECT')
GROUP BY n.oid, n.nspname
ORDER BY n.nspname <> 'public', n.nspname
"""

_COLUMNS = """
SELECT
    CASE n.nspname
        WHEN 'public' THEN quote_ident(c.relname)
        ELSE quote_ident(n.nspname) || '.' || quote_ident(c.relname)
    END,
    CASE c.relkind WHEN 'v' THEN 'view' WHEN 'm' THEN 'materialized view' ELSE 'table' END,
    obj_description(c.oid, 'pg_class'),
    quote_ident(a.attname),
    format_type(a.atttypid, a.atttypmod),
    col_description(c.oid, a.attnum)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE n.nspname = %s
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND has_schema_privilege(n.oid, 'USAGE')
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND has_column_privilege(c.oid, a.attnum, 'SELECT')
ORDER BY c.relname, a.attnum
LIMIT %s
"""


def describe_data(app: psycopg.Connection[Any], schema: str | None = None) -> str:
    if schema is None:
        return _describe_schemas(app)
    rows = app.execute(_COLUMNS, [schema, DESCRIBE_COLUMNS + 1]).fetchall()
    if not rows:
        return f"No schema named {schema} that you can read. Call describe_data without a schema."
    cut = len(rows) > DESCRIBE_COLUMNS
    tables: dict[str, list[str]] = {}
    for table, kind, table_comment, column, data_type, column_comment in rows[:DESCRIBE_COLUMNS]:
        if table not in tables:
            if len(tables) == DESCRIBE_TABLES:
                cut = True
                break
            tables[table] = [f"{table} ({kind})" + (f": {table_comment}" if table_comment else "")]
        line = f"  {column} {data_type}" + (f": {column_comment}" if column_comment else "")
        tables[table].append(line)
    parts = ["Tables and views you can query. Each returns only rows the employee may see."]
    parts += ["\n".join(lines) for lines in tables.values()]
    if cut:
        parts.append(DESCRIPTION_CUT)
    return "\n\n".join(parts)


def _describe_schemas(app: psycopg.Connection[Any]) -> str:
    lines = ["Schemas you can query, with the number of tables and views each holds."]
    for schema, comment, tables in app.execute(_SCHEMAS).fetchall():
        lines.append(f"{schema} ({tables} tables)" + (f": {comment}" if comment else ""))
    return "\n".join(lines)


@dataclass(frozen=True)
class QueryResult:
    text: str
    failed: bool


# The newlines keep a trailing -- comment from swallowing the closing parenthesis. Both
# statements go through the extended query protocol, which refuses more than one statement even
# when the SQL closes the parenthesis itself. The first reads only the column names; the second
# renames the columns by position and cuts every value inside PostgreSQL, and streams one row at
# a time, so neither a long result nor one huge value is ever held in memory.
def query_data(
    app: psycopg.Connection[Any],
    principals: Iterable[str],
    query: str,
    caps: Caps = DEFAULT_CAPS,
) -> QueryResult:
    if app.info.user != APP_ROLE:
        raise ValueError(f"query_data runs as {APP_ROLE}, not as {app.info.user}")
    columns = sql.SQL("SELECT * FROM (\n{}\n) AS q LIMIT 0").format(sql.SQL(query))
    lines: list[str] = []
    cut = False
    try:
        with as_user(app, principals), app.cursor() as cursor:
            app.execute("SET TRANSACTION READ ONLY")
            app.execute(f"SET LOCAL statement_timeout = {TIMEOUT_MS}")
            cursor.execute(columns, prepare=True)
            names = [column.name for column in cursor.description or []]
            lines.append(" | ".join(names)[: caps.chars])
            size = len(lines[0])
            for row in cursor.stream(_cut_values(query, len(names), caps)):
                room = caps.chars - size
                line = " | ".join(_text(value)[: room + 1] for value in row)
                size += len(line) + 1
                if len(lines) > caps.rows or size > caps.chars:
                    cut = True
                    break
                lines.append(line)
    except psycopg.Error as error:
        message = error.diag.message_primary or "the database refused it"
        return QueryResult(text=f"The query failed: {message}", failed=True)
    except UnicodeEncodeError:
        return QueryResult(text="The query failed: the query is not valid UTF-8.", failed=True)

    shown = len(lines) - 1
    if cut:
        lines.append(f"The result was cut at {shown} rows. Narrow the query or aggregate.")
    else:
        lines.append(f"({shown} rows)")
    return QueryResult(text="\n".join(lines), failed=False)


def _cut_values(query: str, width: int, caps: Caps) -> sql.Composed:
    aliases = [sql.Identifier(f"c{index}") for index in range(1, width + 1)]
    values = [
        sql.SQL("left({}::text, {})").format(alias, sql.Literal(caps.chars + 1))
        for alias in aliases
    ]
    renamed = sql.SQL("({})").format(sql.SQL(", ").join(aliases)) if aliases else sql.SQL("")
    return sql.SQL("SELECT {} FROM (\n{}\n) AS q{} LIMIT {}").format(
        sql.SQL(", ").join(values), sql.SQL(query), renamed, sql.Literal(caps.rows + 1)
    )


def _text(value: object) -> str:
    return "NULL" if value is None else str(value)
