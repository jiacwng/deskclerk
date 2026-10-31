# Runs the web front: python -m deskclerk reads the settings, the server list from Postgres and the
# token store, builds the model, and serves the app with uvicorn. python -m deskclerk erase
# <user_id> deletes one employee's tokens, conversations and spending rows instead. python -m
# deskclerk usage [--month YYYY-MM] prints each employee's spend for a month as tab separated
# lines. python -m deskclerk rotate-key re-encrypts every stored value with the newest key in
# DESKCLERK_TOKEN_KEY. python -m deskclerk import-servers [--if-empty] <file> replaces the server
# list with a TOML or mcpServers JSON file, or changes nothing with --if-empty when the list has
# servers, and python -m deskclerk export-servers prints it as TOML. python -m deskclerk
# verify-audit walks the audit log's hash chain and exits with 1 at the first row that does not
# match.

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psycopg
import uvicorn
from prometheus_client import CollectorRegistry
from pydantic import ValidationError

from deskclerk.api import MissingSessionKey, create_app
from deskclerk.audit import Audit, log_to_stdout
from deskclerk.auditstore import AuditStore
from deskclerk.auth import AuthUnavailable, discover_jwks_url, jwks_signing_key
from deskclerk.config import Settings, load_settings
from deskclerk.connections import OAuthConnections
from deskclerk.database import Database
from deskclerk.erasure import erase_user
from deskclerk.generated_keys import fill_missing_keys
from deskclerk.jev import jev_from_environment
from deskclerk.providers import MissingApiKey, ProviderNotInstalled, chat_model
from deskclerk.rotation import rotate_key
from deskclerk.secret_files import SecretFileError, read_secret_files
from deskclerk.servers import ServersError, read_entries, toml_text
from deskclerk.serverstore import ServerCatalog, ServerStore
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore, month_of
from deskclerk.tokens import TokenKeyError, TokenStore
from deskclerk.tracing import TracingNotInstalled, tracing_from_environment
from deskclerk.writecheck import write_check

FLOW_TIMEOUT_SECONDS = 600
MCP_CLIENT_NAME = "deskclerk"


def open_database(settings: Settings) -> Database:
    try:
        return Database(settings.database_url.get_secret_value())
    except psycopg.Error as error:
        sys.exit(f"deskclerk: cannot open the database (DESKCLERK_DATABASE_URL): {error}")


def erase(user_id: str) -> None:
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("DESKCLERK_TOKEN_KEY is not set; put it in .env")
        database = open_database(settings)
        tokens = TokenStore(database, settings.token_key.get_secret_value())
    except (ValidationError, TokenKeyError) as error:
        sys.exit(f"deskclerk: {error}")
    log_to_stdout()
    erased = erase_user(
        user_id,
        tokens,
        SessionStore(database),
        SpendingStore(database),
        Audit(settings.audit_text, settings.audit_key, AuditStore(database)),
    )
    database.close()
    print(
        f"deskclerk: erased {user_id}: {erased.secrets} secrets, {erased.sessions} sessions, "
        f"{erased.threads} conversations, {erased.costs} spending rows"
    )


def rotate() -> None:
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("DESKCLERK_TOKEN_KEY is not set; put it in .env")
        database = open_database(settings)
        tokens = TokenStore(database, settings.token_key.get_secret_value())
    except (ValidationError, TokenKeyError) as error:
        sys.exit(f"deskclerk: {error}")
    for rotated in rotate_key(database, tokens.fernet):
        print(f"deskclerk: rotated {rotated.rows} rows in {rotated.table}")
    database.close()


def import_servers(file: Path, if_empty: bool) -> None:
    try:
        settings = load_settings()
        entries = read_entries(file)
    except (ValidationError, ServersError) as error:
        sys.exit(f"deskclerk: {error}")
    database = open_database(settings)
    store = ServerStore(database)
    if if_empty and store.entries():
        print("deskclerk: the server list already has servers, so nothing was imported")
    else:
        store.replace_all(entries)
        print(f"deskclerk: imported {len(entries)} servers from {file}")
    database.close()


def export_servers() -> None:
    try:
        settings = load_settings()
    except ValidationError as error:
        sys.exit(f"deskclerk: {error}")
    database = open_database(settings)
    entries = ServerStore(database).entries()
    database.close()
    print(toml_text(entries), end="")


def verify_audit() -> None:
    try:
        settings = load_settings()
    except ValidationError as error:
        sys.exit(f"deskclerk: {error}")
    database = open_database(settings)
    report = AuditStore(database).verify()
    database.close()
    if report.broken_at is not None:
        sys.exit(f"deskclerk: the audit chain breaks at row {report.broken_at}")
    print(f"deskclerk: the audit chain is intact, {report.rows} rows")


def month_argument(text: str) -> str:
    try:
        return month_of(datetime.strptime(text, "%Y-%m"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a month like 2026-10") from None


def usage(month: str | None) -> None:
    try:
        settings = load_settings()
    except ValidationError as error:
        sys.exit(f"deskclerk: {error}")
    database = open_database(settings)
    rows = SpendingStore(database).report(month or month_of(datetime.now(UTC)))
    database.close()
    print("user_id\tmodel_usd\tjev_usd\ttotal_usd\tcalls")
    for row in rows:
        print(
            f"{row.user_id}\t{row.model_usd:.4f}\t{row.jev_usd:.4f}\t{row.total_usd:.4f}\t{row.calls}"
        )
    model = sum(row.model_usd for row in rows)
    jev = sum(row.jev_usd for row in rows)
    calls = sum(row.calls for row in rows)
    print(f"total\t{model:.4f}\t{jev:.4f}\t{model + jev:.4f}\t{calls}")


def main(argv: Sequence[str] = ()) -> None:
    parser = argparse.ArgumentParser(prog="deskclerk")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("erase", help="delete one employee's data").add_argument("user_id")
    commands.add_parser("usage", help="print each employee's spend for a month").add_argument(
        "--month", type=month_argument
    )
    commands.add_parser("rotate-key", help="re-encrypt everything with the newest token key")
    import_parser = commands.add_parser(
        "import-servers", help="replace the server list with a TOML or mcpServers JSON file"
    )
    import_parser.add_argument("file", type=Path)
    import_parser.add_argument(
        "--if-empty", action="store_true", help="change nothing when the list has servers"
    )
    commands.add_parser("export-servers", help="print the server list as TOML")
    commands.add_parser("verify-audit", help="check the audit log's hash chain")
    arguments = parser.parse_args(argv)
    try:
        read_secret_files()
        fill_missing_keys()
    except (SecretFileError, OSError) as error:
        sys.exit(f"deskclerk: {error}")
    if arguments.command == "erase":
        erase(arguments.user_id)
        return
    if arguments.command == "rotate-key":
        rotate()
        return
    if arguments.command == "usage":
        usage(arguments.month)
        return
    if arguments.command == "import-servers":
        import_servers(arguments.file, arguments.if_empty)
        return
    if arguments.command == "export-servers":
        export_servers()
        return
    if arguments.command == "verify-audit":
        verify_audit()
        return
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("DESKCLERK_TOKEN_KEY is not set; put it in .env")
        jev = jev_from_environment()
        if jev is None:
            print(
                "deskclerk: TYPESAFE_API_KEY is not set, "
                "so tool outputs are not screened for prompt injection",
                file=sys.stderr,
            )
        database = open_database(settings)
        catalog = ServerCatalog(database, ServerStore(database))
        catalog.reload()
        tokens = TokenStore(database, settings.token_key.get_secret_value())
        connections = OAuthConnections(
            tokens,
            settings.redirect_url,
            MCP_CLIENT_NAME,
            FLOW_TIMEOUT_SECONDS,
            settings.client_metadata_url,
        )
        with httpx.Client() as client:
            jwks_url = discover_jwks_url(settings, client)
        tracing = tracing_from_environment(settings.audit_text, settings.audit_key)
        if tracing is None:
            print(
                "deskclerk: LANGFUSE keys are not set, so requests are not traced",
                file=sys.stderr,
            )
        else:
            print(f"deskclerk: tracing requests to {tracing.host}", file=sys.stderr)
        app = create_app(
            settings,
            database,
            catalog,
            connections,
            tokens,
            SpendingStore(database),
            SessionStore(database),
            chat_model(settings),
            write_check(),
            jev,
            jwks_signing_key(jwks_url),
            CollectorRegistry(),
            tracing,
        )
    except (
        ValidationError,
        ServersError,
        TokenKeyError,
        MissingSessionKey,
        MissingApiKey,
        ProviderNotInstalled,
        AuthUnavailable,
        TracingNotInstalled,
        ValueError,
    ) as error:
        sys.exit(f"deskclerk: {error}")
    log_to_stdout()
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False)
    database.close()
    if tracing is not None:
        tracing.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
