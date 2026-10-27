# Runs the web front: python -m deskclerk reads the settings, the server list and the token
# store, builds the model, and serves the app with uvicorn. python -m deskclerk erase <user_id>
# deletes one employee's tokens, conversations and spending rows instead. python -m deskclerk usage
# [--month YYYY-MM] prints each employee's spend for a month as tab separated lines.

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import httpx
import psycopg
import uvicorn
from pydantic import ValidationError

from deskclerk.api import MissingSessionKey, create_app
from deskclerk.audit import Audit, log_to_stdout
from deskclerk.auth import AuthUnavailable, discover_jwks_url, jwks_signing_key
from deskclerk.config import Settings, load_settings
from deskclerk.connections import OAuthConnections
from deskclerk.database import Database
from deskclerk.erasure import erase_user
from deskclerk.jev import jev_from_environment
from deskclerk.providers import MissingApiKey, ProviderNotInstalled, chat_model
from deskclerk.secret_files import SecretFileError, read_secret_files
from deskclerk.servers import ServersError, load_servers
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore, month_of
from deskclerk.tokens import TokenKeyError, TokenStore
from deskclerk.tracing import TracingNotInstalled, tracing_from_environment
from deskclerk.writecheck import write_check

FLOW_TIMEOUT_SECONDS = 600
MCP_CLIENT_NAME = "deskclerk"


class UnscreenedRefused(Exception):
    pass


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
        Audit(settings.audit_text, settings.audit_key),
    )
    database.close()
    print(
        f"deskclerk: erased {user_id}: {erased.secrets} secrets, {erased.sessions} sessions, "
        f"{erased.threads} conversations, {erased.costs} spending rows"
    )


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
    arguments = parser.parse_args(argv)
    try:
        read_secret_files()
    except (SecretFileError, ServersError) as error:
        sys.exit(f"deskclerk: {error}")
    if arguments.command == "erase":
        erase(arguments.user_id)
        return
    if arguments.command == "usage":
        usage(arguments.month)
        return
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("DESKCLERK_TOKEN_KEY is not set; put it in .env")
        jev = jev_from_environment()
        if jev is None:
            if not settings.allow_unscreened:
                raise UnscreenedRefused(
                    "TYPESAFE_API_KEY is not set, so tool outputs cannot be screened for prompt "
                    "injection. Set it, or set DESKCLERK_ALLOW_UNSCREENED=true to start "
                    "without the screen."
                )
            print(
                "deskclerk: TYPESAFE_API_KEY is not set, "
                "so tool outputs are not screened for prompt injection",
                file=sys.stderr,
            )
        servers = load_servers(settings.servers_file)
        database = open_database(settings)
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
            servers,
            connections,
            tokens,
            SpendingStore(database),
            SessionStore(database),
            chat_model(settings),
            write_check(),
            jev,
            jwks_signing_key(jwks_url),
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
        UnscreenedRefused,
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
