# Runs the web front: python -m deskclerk reads the settings, the server list and the token
# store, builds the model, and serves the app with uvicorn.

import sys

import httpx
import uvicorn
from pydantic import ValidationError

from deskclerk.api import MissingSessionKey, create_app
from deskclerk.audit import log_to_stdout
from deskclerk.auth import AuthUnavailable, discover_jwks_url, jwks_signing_key
from deskclerk.config import load_settings
from deskclerk.connections import OAuthConnections
from deskclerk.jev import jev_from_environment
from deskclerk.providers import MissingApiKey, chat_model
from deskclerk.servers import ServersError, load_servers
from deskclerk.tokens import TokenKeyError, TokenStore
from deskclerk.tracing import TracingNotInstalled, tracing_from_environment
from deskclerk.writecheck import write_check

FLOW_TIMEOUT_SECONDS = 600
MCP_CLIENT_NAME = "deskclerk"
HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("DESKCLERK_TOKEN_KEY is not set; put it in .env")
        servers = load_servers(settings.servers_file)
        tokens = TokenStore(settings.token_file, settings.token_key.get_secret_value())
        connections = OAuthConnections(
            tokens, settings.redirect_url, MCP_CLIENT_NAME, FLOW_TIMEOUT_SECONDS
        )
        with httpx.Client() as client:
            jwks_url = discover_jwks_url(settings, client)
        jev = jev_from_environment()
        if jev is None:
            print(
                "deskclerk: TYPESAFE_API_KEY is not set, "
                "so tool outputs are not screened for prompt injection",
                file=sys.stderr,
            )
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
        AuthUnavailable,
        TracingNotInstalled,
        ValueError,
    ) as error:
        sys.exit(f"deskclerk: {error}")
    log_to_stdout()
    uvicorn.run(app, host=HOST, port=PORT)
    if tracing is not None:
        tracing.flush()


if __name__ == "__main__":
    main()
