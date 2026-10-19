# Shared fixtures: access tokens signed with a throwaway RSA key, so tests that need one run
# without Keycloak.

import time
from collections.abc import Callable

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from deskclerk.auth import SigningKeyLookup
from deskclerk.config import Settings, load_settings

TOKEN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def token_settings() -> Settings:
    settings = load_settings()
    return settings.model_copy(
        update={
            "jwks_url": f"{settings.issuer}/protocol/openid-connect/certs",
            "issuer": "http://keycloak.test/realms/deskclerk",
            "client_id": "deskclerk-api",
            "audience": "deskclerk-api",
            "requests_per_minute": 10_000,
        }
    )


@pytest.fixture(scope="session")
def token_signing_key() -> SigningKeyLookup:
    public_key = jwt.PyJWK.from_dict(RSAAlgorithm.to_jwk(TOKEN_KEY.public_key(), as_dict=True))

    def signing_key(token: str) -> jwt.PyJWK:
        return public_key

    return signing_key


@pytest.fixture(scope="session")
def sign_token(token_settings: Settings) -> Callable[[dict[str, object]], str]:
    # The claims given replace the defaults below, and a claim given as None is left out.
    def sign(claims: dict[str, object]) -> str:
        now = int(time.time())
        base: dict[str, object] = {
            "iss": token_settings.issuer,
            "aud": token_settings.audience,
            "azp": token_settings.client_id,
            "typ": "Bearer",
            "scope": "openid profile",
            "iat": now,
            "exp": now + 300,
            "preferred_username": "district-1",
            "groups": ["district:1"],
        }
        merged = {name: value for name, value in (base | claims).items() if value is not None}
        return jwt.encode(merged, TOKEN_KEY, algorithm="RS256")

    return sign


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
