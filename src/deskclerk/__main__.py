# Runs the web front: python -m deskclerk reads the settings, the server list and the token
# store, builds the model, and serves the app with uvicorn. python -m deskclerk erase <user_id>
# deletes one employee's tokens, conversations and spending rows instead.

import argparse
import sys
from collections.abc import Sequence

import httpx
import uvicorn
from pydantic import ValidationError

from deskclerk.api import MissingSessionKey, create_app
from deskclerk.audit import Audit, log_to_stdout
from deskclerk.auth import AuthUnavailable, discover_jwks_url, jwks_signing_key
from deskclerk.config import load_settings
from deskclerk.connections import OAuthConnections
from deskclerk.conversations import delete_user_threads
from deskclerk.jev import jev_from_environment
from deskclerk.providers import MissingApiKey, ProviderNotInstalled, chat_model
from deskclerk.secret_files import SecretFileError, read_secret_files
from deskclerk.servers import ServersError, load_servers
from deskclerk.sessions import SessionStore
from deskclerk.spending import SpendingStore
from deskclerk.tokens import TokenKeyError, TokenStore
from deskclerk.tracing import TracingNotInstalled, tracing_from_environment
from deskclerk.writecheck import write_check

FLOW_TIMEOUT_SECONDS = 600
MCP_CLIENT_NAME = "deskclerk"
SPENDING_FILE = "spending.sqlite"


class UnscreenedRefused(Exception):
    pass


def erase(user_id: str) -> None:
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("DESKCLERK_TOKEN_KEY is not set; put it in .env")
        tokens = TokenStore(settings.token_file, settings.token_key.get_secret_value())
    except (ValidationError, TokenKeyError) as error:
        sys.exit(f"deskclerk: {error}")
    spending_store = SpendingStore(settings.token_file.parent / SPENDING_FILE)
    secrets = tokens.delete_user(user_id)
    sessions = SessionStore(settings.token_file).delete_user(user_id)
    threads = delete_user_threads(settings.token_file, user_id)
    costs = spending_store.delete_user(user_id)
    log_to_stdout()
    Audit(settings.audit_text, settings.audit_key).erased(
        user_id, secrets, sessions, threads, costs
    )
    print(
        f"deskclerk: erased {user_id}: {secrets} secrets, {sessions} sessions, "
        f"{threads} conversations, {costs} spending rows"
    )


def main(argv: Sequence[str] = ()) -> None:
    parser = argparse.ArgumentParser(prog="deskclerk")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("erase", help="delete one employee's data").add_argument("user_id")
    arguments = parser.parse_args(argv)
    try:
        read_secret_files()
    except (SecretFileError, ServersError) as error:
        sys.exit(f"deskclerk: {error}")
    if arguments.command == "erase":
        erase(arguments.user_id)
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
        tokens = TokenStore(settings.token_file, settings.token_key.get_secret_value())
        spending_store = SpendingStore(settings.token_file.parent / SPENDING_FILE)
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
            servers,
            connections,
            tokens,
            spending_store,
            SessionStore(settings.token_file),
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
    uvicorn.run(app, host=settings.host, port=settings.port)
    if tracing is not None:
        tracing.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
