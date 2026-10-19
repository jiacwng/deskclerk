# A user in more than 200 Entra ID groups gets a token without them, and the API reads the
# groups from Microsoft Graph instead. Graph is a mock transport here, so no tenant is needed.

import time
from collections.abc import Callable

import httpx
import jwt
import pytest
from pydantic import SecretStr

from conftest import TOKEN_KEY
from deskclerk import graph as graph_module
from deskclerk.auth import AuthError, AuthUnavailable, SigningKeyLookup, authenticate
from deskclerk.config import Settings
from deskclerk.graph import (
    CACHE_SECONDS,
    GRAPH,
    LOGIN,
    GraphError,
    GraphGroups,
    graph_from_settings,
)
from test_auth import ENTRA_GROUP, ENTRA_TENANT, ENTRA_USER, entra_claims, entra_settings

OTHER_GROUP = "0a4f3a6e-3c1c-4a7e-9d2e-6f3b1c2d4e5f"
MEMBER_OF = f"{GRAPH}/users/{ENTRA_USER}/transitiveMemberOf/microsoft.graph.group"
SECOND_PAGE = f"{MEMBER_OF}?$skiptoken=page-2"


def _overflow_token(**changes: object) -> str:
    claims = entra_claims()
    del claims["groups"]
    claims["_claim_names"] = {"groups": "src1"}
    claims["_claim_sources"] = {
        "src1": {"endpoint": f"https://graph.windows.net/{ENTRA_TENANT}/users/x/getMemberObjects"}
    }
    claims |= changes
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, TOKEN_KEY, "RS256")


def _graph(answer: Callable[[httpx.Request], httpx.Response]) -> GraphGroups:
    client = httpx.Client(transport=httpx.MockTransport(answer))
    return GraphGroups(client, ENTRA_TENANT, "api-client", SecretStr("api-secret"))


def _two_pages(requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"access_token": "app-token", "expires_in": 3599})
        if str(request.url) == SECOND_PAGE:
            return httpx.Response(200, json={"value": [{"id": OTHER_GROUP}]})
        page = {"value": [{"id": ENTRA_GROUP}], "@odata.nextLink": SECOND_PAGE}
        return httpx.Response(200, json=page)

    return answer


def test_an_overflow_token_without_graph_credentials_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = entra_settings(token_settings)
    with pytest.raises(AuthError, match="DESKCLERK_GRAPH_TENANT_ID"):
        authenticate(_overflow_token(), settings, token_signing_key, settings.audience)


def test_an_overflow_token_gets_every_page_of_groups_from_graph(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = entra_settings(token_settings)
    requests: list[httpx.Request] = []
    graph = _graph(_two_pages(requests))

    for _ in range(2):
        identity = authenticate(
            _overflow_token(), settings, token_signing_key, settings.audience, graph
        )
        assert identity.principals == {
            f"user:{ENTRA_USER}",
            "group:everyone",
            f"group:{ENTRA_GROUP}",
            f"group:{OTHER_GROUP}",
        }

    [login, first, second, *_] = requests
    assert str(login.url) == f"{LOGIN}/{ENTRA_TENANT}/oauth2/v2.0/token"
    form = dict(httpx.QueryParams(login.content.decode()))
    assert form["grant_type"] == "client_credentials"
    assert form["scope"] == "https://graph.microsoft.com/.default"
    assert str(first.url) == f"{MEMBER_OF}?%24select=id"
    assert first.headers["Authorization"] == "Bearer app-token"
    assert str(second.url) == SECOND_PAGE
    assert [request.method for request in requests].count("POST") == 1


def test_groups_are_read_again_only_after_the_cache_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    requests: list[httpx.Request] = []
    graph = _graph(_two_pages(requests))

    first = graph.groups_of(ENTRA_USER)
    clock[0] += CACHE_SECONDS - 1
    cached = graph.groups_of(ENTRA_USER)
    pages_before_expiry = [request.method for request in requests].count("GET")
    clock[0] += 2
    graph.groups_of(ENTRA_USER)

    assert first == cached == [ENTRA_GROUP, OTHER_GROUP]
    assert pages_before_expiry == 2
    assert [request.method for request in requests].count("GET") == 4


def test_the_cache_drops_the_oldest_user_once_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "CACHED_USERS", 2)
    requests: list[httpx.Request] = []
    graph = _graph(_two_pages(requests))
    users = ["11111111-1111-1111-1111-111111111111", ENTRA_USER, OTHER_GROUP]

    for user in users:
        graph.groups_of(user)
    after_filling = len(requests)
    graph.groups_of(users[1])
    graph.groups_of(users[2])
    after_cached = len(requests)
    graph.groups_of(users[0])

    assert after_cached == after_filling
    assert len(requests) == after_cached + 2


def test_an_overflow_token_without_an_object_id_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    settings = entra_settings(token_settings).model_copy(update={"user_claim": "sub"})
    graph = _graph(_two_pages([]))
    for oid in (None, "../../groups"):
        with pytest.raises(AuthError, match="oid must be the user's object id"):
            authenticate(
                _overflow_token(oid=oid), settings, token_signing_key, settings.audience, graph
            )


@pytest.mark.parametrize(
    "graph_answer",
    [
        pytest.param(httpx.Response(503), id="graph-down"),
        pytest.param(httpx.Response(200, text="not json"), id="not-json"),
        pytest.param(
            httpx.Response(200, json={"value": [], "@odata.nextLink": "https://evil.test/x"}),
            id="next-page-off-graph",
        ),
    ],
)
def test_graph_failing_makes_the_login_unavailable(
    graph_answer: httpx.Response, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"access_token": "app-token", "expires_in": 3599})
        return graph_answer

    settings = entra_settings(token_settings)
    with pytest.raises(AuthUnavailable, match="cannot read the groups from Microsoft Graph"):
        authenticate(
            _overflow_token(), settings, token_signing_key, settings.audience, _graph(answer)
        )


def test_a_refused_application_login_keeps_the_secret_out_of_the_error() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_client"})

    with pytest.raises(GraphError, match="cannot read the groups") as raised:
        _graph(answer).groups_of(ENTRA_USER)
    assert "api-secret" not in str(raised.value)


def test_graph_is_used_only_with_all_three_credentials(token_settings: Settings) -> None:
    assert graph_from_settings(token_settings) is None
    configured = token_settings.model_copy(
        update={
            "graph_tenant_id": ENTRA_TENANT,
            "graph_client_id": "api-client",
            "graph_client_secret": SecretStr("api-secret"),
        }
    )
    assert isinstance(graph_from_settings(configured), GraphGroups)
