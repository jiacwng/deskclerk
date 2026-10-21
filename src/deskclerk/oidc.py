# The browser half of OpenID Connect sign-in for a public client: the authorization code flow
# with PKCE (RFC 7636) against the endpoints the issuer publishes at
# /.well-known/openid-configuration. The ID token that comes back, not the access token, is
# checked by auth.py.

import base64
import hashlib
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from deskclerk.auth import AuthUnavailable
from deskclerk.config import Settings

TIMEOUT_SECONDS = 10
LOGIN_SCOPE = "openid profile email"


class SignInRefused(Exception):
    pass


class NoIdToken(Exception):
    pass


@dataclass(frozen=True)
class Endpoints:
    authorization: str
    token: str


@dataclass(frozen=True)
class SignInStart:
    url: str
    state: str
    verifier: str
    nonce: str


async def discover(settings: Settings, client: httpx.AsyncClient) -> Endpoints:
    discovery = f"{settings.issuer}/.well-known/openid-configuration"
    try:
        response = await client.get(discovery)
        response.raise_for_status()
        document = response.json()
        authorization, token = document["authorization_endpoint"], document["token_endpoint"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise AuthUnavailable(f"cannot read the endpoints from {discovery}: {error}") from None
    if not isinstance(authorization, str) or not isinstance(token, str):
        raise AuthUnavailable(f"{discovery} gives no authorization or token endpoint")
    return Endpoints(authorization, token)


def start(settings: Settings, endpoints: Endpoints) -> SignInStart:
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)
    nonce = secrets.token_urlsafe(24)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings.client_id,
            "redirect_uri": settings.sign_in_redirect_url,
            "scope": LOGIN_SCOPE,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge.decode().rstrip("="),
            "code_challenge_method": "S256",
        }
    )
    return SignInStart(f"{endpoints.authorization}?{query}", state, verifier, nonce)


async def exchange_code(
    settings: Settings, endpoints: Endpoints, client: httpx.AsyncClient, code: str, verifier: str
) -> str:
    try:
        form = {
            "grant_type": "authorization_code",
            "client_id": settings.client_id,
            "redirect_uri": settings.sign_in_redirect_url,
            "code": code,
            "code_verifier": verifier,
        }
        if settings.client_secret is not None:
            form["client_secret"] = settings.client_secret.get_secret_value()
        response = await client.post(endpoints.token, data=form)
        if 400 <= response.status_code < 500:
            raise SignInRefused("the login server refused the code")
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise AuthUnavailable(f"the login server gave no token response: {error}") from None
    id_token = body.get("id_token") if isinstance(body, dict) else None
    if not isinstance(id_token, str) or not id_token:
        raise NoIdToken("the login server gave no ID token")
    return id_token
