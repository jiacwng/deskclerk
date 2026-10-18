# The browser half of OpenID Connect sign-in for a public client: the authorization code flow
# with PKCE (RFC 7636) against the endpoints the issuer publishes at
# /.well-known/openid-configuration. The access token that comes back is checked by auth.py.

import base64
import hashlib
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from needtoknow.auth import AuthUnavailable
from needtoknow.config import Settings

TIMEOUT_SECONDS = 10


class SignInRefused(Exception):
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
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings.client_id,
            "redirect_uri": settings.sign_in_redirect_url,
            "scope": settings.login_scope,
            "state": state,
            "code_challenge": challenge.decode().rstrip("="),
            "code_challenge_method": "S256",
        }
    )
    return SignInStart(f"{endpoints.authorization}?{query}", state, verifier)


async def exchange_code(
    settings: Settings, endpoints: Endpoints, client: httpx.AsyncClient, code: str, verifier: str
) -> str:
    try:
        response = await client.post(
            endpoints.token,
            data={
                "grant_type": "authorization_code",
                "client_id": settings.client_id,
                "redirect_uri": settings.sign_in_redirect_url,
                "code": code,
                "code_verifier": verifier,
            },
        )
        if 400 <= response.status_code < 500:
            raise SignInRefused("the login server refused the code")
        response.raise_for_status()
        access_token = response.json()["access_token"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise AuthUnavailable(f"the login server gave no access token: {error}") from None
    if not isinstance(access_token, str) or not access_token:
        raise AuthUnavailable("the login server gave no access token")
    return access_token
