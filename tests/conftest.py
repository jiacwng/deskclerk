# Shared fixtures. Database: an admin connection creates the roles and tables, the owner loads the
# slices of public data in tests/data, the Enron emails and the BIRD databases, with their
# embeddings, and the tests read through the app role or the reader role. Tokens: access tokens
# signed with a throwaway RSA key, so tests that need one run without Keycloak.

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from psycopg.rows import TupleRow

from needtoknow import db
from needtoknow.auth import SigningKeyLookup
from needtoknow.bird import load_bird
from needtoknow.config import Settings, load_provision_settings, load_settings
from needtoknow.db import Pool, connect_app, connect_reader
from needtoknow.documents import Document
from needtoknow.enron import load_enron, mail_files, parse_email
from needtoknow.provision import connect_admin, connect_owner, create_schema, prepare_database

TOKEN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
DATA = Path(__file__).parent / "data"
MAILDIR = DATA / "enron" / "maildir"
MINIDEV = DATA / "bird"


@pytest.fixture(scope="session")
def owner() -> Iterator[psycopg.Connection[TupleRow]]:
    settings = load_settings()
    provision = load_provision_settings()
    try:
        admin = connect_admin(settings, provision)
    except psycopg.OperationalError as error:
        pytest.fail(
            f"cannot reach PostgreSQL at {settings.db_host}:{settings.db_port}, "
            f"start it with: docker compose up -d --wait db\n{error}",
            pytrace=False,
        )
    with admin:
        prepare_database(admin, settings, provision, evaluation=True)
    with connect_owner(settings, provision) as connection:
        create_schema(connection)
        load_enron(connection, MAILDIR)
        load_bird(connection, MINIDEV)
        yield connection


@pytest.fixture(scope="session")
def app(owner: psycopg.Connection[TupleRow]) -> Iterator[psycopg.Connection[TupleRow]]:
    with connect_app(load_settings()) as connection:
        yield connection


@pytest.fixture(scope="session")
def reader(owner: psycopg.Connection[TupleRow]) -> Iterator[psycopg.Connection[TupleRow]]:
    with connect_reader(load_settings(), load_provision_settings()) as connection:
        yield connection


@pytest.fixture(scope="session")
def app_pool(owner: psycopg.Connection[TupleRow]) -> Iterator[Pool]:
    with db.app_pool(load_settings(), size=1) as pool:
        yield pool


@pytest.fixture(scope="session")
def reader_pool(owner: psycopg.Connection[TupleRow]) -> Iterator[Pool]:
    with db.reader_pool(load_settings(), load_provision_settings(), size=1) as pool:
        yield pool


# Every email of the slice once, read by the custodians whose mailbox holds a copy.
@pytest.fixture(scope="session")
def emails() -> dict[str, Document]:
    found: dict[str, tuple[str, str, set[str]]] = {}
    for custodian, path in mail_files(MAILDIR):
        email = parse_email(path.read_bytes())
        found.setdefault(email.id, (email.subject, email.body, set()))[2].add(f"user:{custodian}")
    return {
        doc_id: Document(id=doc_id, title=title, body=body, readers=frozenset(readers))
        for doc_id, (title, body, readers) in found.items()
    }


@pytest.fixture(scope="session")
def token_settings() -> Settings:
    return load_settings().model_copy(
        update={
            "issuer": "http://keycloak.test/realms/needtoknow",
            "client_id": "needtoknow-api",
            "audience": "needtoknow-api",
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
