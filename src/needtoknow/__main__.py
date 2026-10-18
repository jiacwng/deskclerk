# Runs the web front: python -m needtoknow reads the settings, the server list and the token
# store, builds the model, and serves the app with uvicorn.

import sys

import httpx
import uvicorn
from pydantic import ValidationError

from needtoknow.api import MissingSessionKey, create_app
from needtoknow.audit import log_to_stdout
from needtoknow.auth import AuthUnavailable, discover_jwks_url, jwks_signing_key
from needtoknow.config import load_settings
from needtoknow.connections import OAuthConnections
from needtoknow.graph import graph_from_settings
from needtoknow.providers import MissingApiKey, MissingProvider, chat_model
from needtoknow.servers import ServersError, load_servers
from needtoknow.tokens import TokenKeyError, TokenStore

FLOW_TIMEOUT_SECONDS = 600


def main() -> None:
    try:
        settings = load_settings()
        if settings.token_key is None:
            raise TokenKeyError("NEEDTOKNOW_TOKEN_KEY is not set; put it in .env")
        servers = load_servers(settings.servers_file)
        tokens = TokenStore(settings.token_file, settings.token_key.get_secret_value())
        connections = OAuthConnections(
            tokens, settings.redirect_url, settings.mcp_client_name, FLOW_TIMEOUT_SECONDS
        )
        with httpx.Client() as client:
            jwks_url = discover_jwks_url(settings, client)
        app = create_app(
            settings,
            servers,
            connections,
            tokens,
            chat_model(settings),
            jwks_signing_key(jwks_url),
            graph_from_settings(settings),
        )
    except (
        ValidationError,
        ServersError,
        TokenKeyError,
        MissingSessionKey,
        MissingApiKey,
        MissingProvider,
        AuthUnavailable,
        ValueError,
    ) as error:
        sys.exit(f"needtoknow: {error}")
    log_to_stdout()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
