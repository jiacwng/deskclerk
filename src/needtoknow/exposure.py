# The startup check: the API and the MCP server refuse to start while the app role can read a
# relation that row-level security does not protect, or holds a privilege that gets around the
# policies. It follows Supabase's rls_disabled_in_public lint and runs as the app role, so the
# privilege functions answer for that role. A relation every employee may read goes on
# NEEDTOKNOW_PUBLIC_RELATIONS. `python -m needtoknow.exposure` runs the same check for CI.

import sys
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import psycopg

from needtoknow.config import Settings, load_settings
from needtoknow.db import connect_app
from needtoknow.schema import check_schema

_RELATIONS = r"""
SELECT c.oid, n.nspname, c.relname, c.relkind,
    has_schema_privilege(n.oid, 'USAGE') AND has_any_column_privilege(c.oid, 'SELECT'),
    c.relrowsecurity,
    c.relforcerowsecurity,
    pg_has_role(c.relowner, 'MEMBER'),
    EXISTS (
        SELECT FROM pg_options_to_table(c.reloptions)
        WHERE option_name = 'security_invoker' AND option_value::boolean
    ),
    EXISTS (
        SELECT FROM pg_policy p
        WHERE p.polrelid = c.oid AND p.polpermissive AND p.polcmd IN ('r', '*')
          AND pg_get_expr(p.polqual, p.polrelid) = 'true'
          AND EXISTS (
            SELECT FROM unnest(p.polroles) AS r (role)
            WHERE CASE WHEN r.role = 0 THEN true ELSE pg_has_role(r.role, 'MEMBER') END
          )
    )
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND n.nspname <> 'information_schema' AND n.nspname NOT LIKE 'pg\_%'
"""

# The relations each view's query reads, from the dependencies of its rewrite rule.
_VIEW_BASES = """
SELECT DISTINCT r.ev_class, d.refobjid
FROM pg_rewrite r
JOIN pg_depend d ON d.classid = 'pg_rewrite'::regclass AND d.objid = r.oid
WHERE d.refclassid = 'pg_class'::regclass AND d.refobjid <> r.ev_class
"""

_SUPERUSER = "SELECT rolsuper FROM pg_roles WHERE rolname = current_user"
_BYPASS = "SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user"
_SET_CONFIG = (
    "SELECT has_function_privilege('pg_catalog.set_config(text, text, boolean)', 'EXECUTE')"
)

_DEFINER_FUNCTIONS = """
SELECT p.oid::regprocedure::text
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.prosecdef AND n.nspname <> 'pg_catalog' AND has_function_privilege(p.oid, 'EXECUTE')
  AND p.oid IS DISTINCT FROM to_regprocedure('public.set_identity(text)')
ORDER BY 1
"""


SCHEMA_INFO = "public.schema_info"
NOT_PROVISIONED = (
    "the database is not provisioned or unreachable: run the load "
    "(docker compose -f compose.prod.yaml run --rm load) first"
)


class Exposed(RuntimeError):
    pass


@dataclass(frozen=True)
class Relation:
    name: str
    schema: str
    kind: str
    readable: bool
    row_security: bool
    forced: bool
    owned: bool
    invoker: bool
    open_policy: bool


def find_exposures(connection: psycopg.Connection[Any], public: Collection[str]) -> list[str]:
    problems = _role_problems(connection)
    relations: dict[int, Relation] = {}
    for row in connection.execute(_RELATIONS):
        relations[row[0]] = Relation(
            name=f"{row[1]}.{row[2]}",
            schema=row[1],
            kind=row[3],
            readable=row[4],
            row_security=row[5],
            forced=row[6],
            owned=row[7],
            invoker=row[8],
            open_policy=row[9],
        )
    bases: dict[int, list[int]] = {}
    for view, base in connection.execute(_VIEW_BASES):
        bases.setdefault(view, []).append(base)
    for oid, relation in sorted(relations.items(), key=lambda item: item[1].name):
        if not relation.readable or _is_public(relation, public):
            continue
        for reason in _reasons(oid, relations, bases, public):
            problems.append(f"{relation.name}: {reason}")
    return problems


def check_exposure(settings: Settings) -> None:
    with connect_app(settings) as connection:
        problems = find_exposures(connection, settings.public_relations)
    if problems:
        raise Exposed(
            "the app role can read data that row-level security does not protect; protect it, "
            "or list what every employee may read in NEEDTOKNOW_PUBLIC_RELATIONS:\n  "
            + "\n  ".join(problems)
        )


def check_startup(settings: Settings) -> None:
    try:
        check_exposure(settings)
        check_schema(settings)
    except psycopg.OperationalError:
        sys.exit(NOT_PROVISIONED)


def _role_problems(connection: psycopg.Connection[Any]) -> list[str]:
    problems = []
    if connection.execute(_SUPERUSER).fetchone() == (True,):
        problems.append("the app role is a superuser, so no policy applies to it")
    if connection.execute(_BYPASS).fetchone() == (True,):
        problems.append("the app role has BYPASSRLS, so no policy applies to it")
    if connection.execute(_SET_CONFIG).fetchone() == (True,):
        problems.append(
            "pg_catalog.set_config(text,text,boolean): the app role can run it and "
            "rewrite the identity the policies read"
        )
    for (function,) in connection.execute(_DEFINER_FUNCTIONS):
        problems.append(f"{function}: a SECURITY DEFINER function the app role can run")
    return problems


def _is_public(relation: Relation, public: Collection[str]) -> bool:
    return relation.name == SCHEMA_INFO or relation.schema in public or relation.name in public


def _reasons(
    oid: int,
    relations: dict[int, Relation],
    bases: dict[int, list[int]],
    public: Collection[str],
) -> list[str]:
    relation = relations[oid]
    reasons = []
    if relation.kind in ("r", "p"):
        if not relation.row_security:
            reasons.append("row-level security is off")
        elif relation.owned and not relation.forced:
            reasons.append("the app role owns it, so its policies do not apply")
        if relation.open_policy:
            reasons.append("a policy lets the app role read every row")
    elif relation.kind == "v":
        if not relation.invoker:
            reasons.append("the view reads as its owner; set security_invoker = true")
        # A base missing from relations is a system catalog, which the app role reads anyway.
        names = []
        for base in bases.get(oid, []):
            if base in relations and not _is_public(relations[base], public):
                if _reasons(base, relations, bases, public):
                    names.append(relations[base].name)
        for name in sorted(names):
            reasons.append(f"it reads {name}, which is not protected")
    elif relation.kind == "m":
        reasons.append("a materialized view cannot have row-level security")
    else:
        reasons.append("a foreign table cannot have row-level security")
    return reasons


def main() -> None:
    try:
        check_exposure(load_settings())
    except Exposed as error:
        sys.exit(str(error))
    print("the app role reads only protected or public relations")


if __name__ == "__main__":
    main()
