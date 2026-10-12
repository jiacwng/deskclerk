# Loads BIRD Mini-Dev: its PostgreSQL dump of 11 databases, one schema per database, BIRD's
# column descriptions as comments and as documents everyone may read, and the row-level policies
# in sql/bird_policies.sql. The dump puts every table in public, so it is loaded into a staging
# schema and each table then moves to the schema of the database dev_tables.json lists it under.

import csv
import io
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import IO, Any

import psycopg
from psycopg import sql

from needtoknow.auth import EVERYONE
from needtoknow.db import APP_ROLE
from needtoknow.documents import Document, write_documents

DUMP = Path("MINIDEV_postgresql") / "BIRD_dev.sql"
TABLES = Path("MINIDEV") / "dev_tables.json"
DESCRIPTIONS = Path("MINIDEV") / "dev_databases"
POLICIES = files("needtoknow").joinpath("sql/bird_policies.sql")
STAGING = "bird_dump"
ID_PREFIX = "bird-"

# The dump is untrusted, so only single statements of these kinds run and any other stops the load.
# Session settings and ownership are skipped: the owner role owns everything it creates.
_RUN = (
    "CREATE TABLE public.",
    "CREATE SEQUENCE public.",
    "ALTER SEQUENCE public.",
    "ALTER TABLE ONLY public.",
    "CREATE INDEX ",
    "CREATE UNIQUE INDEX ",
    "SELECT pg_catalog.setval(",
    "COPY public.",
)
_SKIP = ("SET ", "SELECT pg_catalog.set_config(")
_OWNER = re.compile(r"\sOWNER TO \S+;$")
_PUBLIC = re.compile(r"\bpublic\.")
_DATABASE = re.compile(r"[a-z][a-z0-9_]*")


class BirdError(ValueError):
    pass


@dataclass(frozen=True)
class Loaded:
    tables: int
    rows: int
    comments: int
    documents: int


def read_databases(path: Path) -> dict[str, list[str]]:
    databases: dict[str, list[str]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        name = entry["db_id"]
        if not _DATABASE.fullmatch(name) or name in ("public", STAGING):
            raise BirdError(f"{path.name}: {name!r} cannot be a schema name")
        databases[name] = [table.lower() for table in entry["table_names_original"]]
    return databases


def load_bird(owner: psycopg.Connection[Any], minidev: Path) -> Loaded:
    databases = read_databases(minidev / TABLES)
    with owner.transaction():
        for name in [STAGING, *databases]:
            owner.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))
        owner.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(STAGING)))
        rows = _load_dump(owner, minidev / DUMP)
        _move_tables(owner, databases)
        documents, comments = _describe(owner, minidev / DESCRIPTIONS, databases)
        owner.execute(POLICIES.read_text(encoding="utf-8"))
        owner.execute("DELETE FROM chunks WHERE doc_id LIKE 'bird-%'")
        owner.execute("DELETE FROM doc_access WHERE doc_id LIKE 'bird-%'")
        owner.execute("DELETE FROM documents WHERE id LIKE 'bird-%'")
        write_documents(owner, documents)
    return Loaded(
        tables=sum(len(tables) for tables in databases.values()),
        rows=rows,
        comments=comments,
        documents=len(documents),
    )


def _move_tables(owner: psycopg.Connection[Any], databases: dict[str, list[str]]) -> None:
    app = sql.Identifier(APP_ROLE)
    for database, tables in databases.items():
        schema = sql.Identifier(database)
        owner.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        for table in tables:
            owner.execute(
                sql.SQL("ALTER TABLE {}.{} SET SCHEMA {}").format(
                    sql.Identifier(STAGING), sql.Identifier(table), schema
                )
            )
        owner.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, app))
        owner.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(schema, app))
    owner.execute(sql.SQL("DROP SCHEMA {}").format(sql.Identifier(STAGING)))


def _describe(
    owner: psycopg.Connection[Any], root: Path, databases: dict[str, list[str]]
) -> tuple[list[Document], int]:
    documents = []
    comments = 0
    for database, tables in databases.items():
        for path in sorted((root / database / "database_description").glob("*.csv")):
            table = path.stem.lower()
            if table not in tables:
                raise BirdError(f"{database}/{path.name}: no table {table} in {database}")
            lines = _comment(owner, database, path)
            comments += len(lines)
            documents.append(
                Document(
                    id=ID_PREFIX + f"{database}-{table}".replace("_", "-"),
                    title=f"{database}.{table}",
                    body="\n\n".join(lines),
                    readers=frozenset({EVERYONE}),
                )
            )
    return documents, comments


def _load_dump(owner: psycopg.Connection[Any], path: Path) -> int:
    rows = 0
    lines: list[str] = []
    with path.open("rb") as dump:
        for raw in dump:
            line = raw.decode("utf-8")
            if not lines and (not line.strip() or line.startswith("--")):
                continue
            lines.append(line)
            if not line.rstrip().endswith(";"):
                continue
            statement = "".join(lines).strip()
            lines = []
            if statement.startswith(_SKIP) or _OWNER.search(statement):
                continue
            if not statement.startswith(_RUN) or statement.count(";") != 1:
                first = statement.splitlines()[0]
                raise BirdError(f"the dump holds a statement this loader does not run: {first}")
            statement = _PUBLIC.sub(f"{STAGING}.", statement)
            if statement.startswith("COPY "):
                rows += _copy(owner, statement, dump)
            else:
                owner.execute(statement)
    return rows


def _copy(owner: psycopg.Connection[Any], statement: str, dump: IO[bytes]) -> int:
    rows = 0
    with owner.cursor() as cursor, cursor.copy(statement) as copy:
        for line in dump:
            if line.rstrip(b"\r\n") == b"\\.":
                return rows
            copy.write(line)
            rows += 1
    raise BirdError("the dump ends inside a COPY block")


def _comment(owner: psycopg.Connection[Any], database: str, path: Path) -> list[str]:
    table = path.stem.lower()
    columns = {}
    for (name,) in owner.execute(
        "SELECT attname FROM pg_attribute "
        "WHERE attrelid = to_regclass(format('%%I.%%I', %s::text, %s::text)) AND attnum > 0 "
        "AND NOT attisdropped",
        [database, table],
    ):
        columns[name.lower()] = name
    target = sql.SQL("{}.{}").format(sql.Identifier(database), sql.Identifier(table))
    owner.execute(
        sql.SQL("COMMENT ON TABLE {} IS {}").format(
            target, sql.Literal(f"Table {path.stem} of the BIRD database {database}.")
        )
    )
    lines = []
    for row in csv.DictReader(io.StringIO(_read_text(path))):
        column = columns.get((row.get("original_column_name") or "").strip().lower())
        description = _description(row)
        if column is None or not description:
            continue
        owner.execute(
            sql.SQL("COMMENT ON COLUMN {}.{} IS {}").format(
                target, sql.Identifier(column), sql.Literal(description)
            )
        )
        lines.append(f"{column}: {description}")
    return lines


def _description(row: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ("column_name", "column_description", "value_description"):
        text = " ".join((row.get(key) or "").split())
        if text and text.lower() not in [part.lower() for part in parts]:
            parts.append(text)
    return "; ".join(parts)


# A few of the description files are written in Windows-1252 rather than UTF-8.
def _read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")
