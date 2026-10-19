# Settings refuse a model named without its provider, one price setting without the other, and
# Graph credentials given in part, and the audience defaults to the client.

import pytest
from pydantic import SecretStr, ValidationError

from needtoknow.config import Settings


@pytest.mark.parametrize("model_name", ["claude-haiku-5-5", ":claude-haiku-5-5", "anthropic:"])
def test_the_model_needs_a_provider_and_a_name(model_name: str) -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_MODEL is provider:model"):
        Settings(model=model_name)


def test_one_price_without_the_other_is_refused() -> None:
    with pytest.raises(ValidationError, match="set both NEEDTOKNOW_PRICE_INPUT"):
        Settings(price_input=1.0)
    with pytest.raises(ValidationError, match="set both NEEDTOKNOW_PRICE_INPUT"):
        Settings(price_output=1.0)


def test_the_audience_is_the_client_unless_set() -> None:
    assert Settings(client_id="page").audience == "page"
    assert Settings(client_id="page", audience="api://default").audience == "api://default"


def test_graph_credentials_come_all_three_or_none() -> None:
    with pytest.raises(ValidationError, match="NEEDTOKNOW_GRAPH_CLIENT_SECRET together"):
        Settings(graph_tenant_id="tenant", graph_client_id="client")

    complete = Settings(
        graph_tenant_id="tenant", graph_client_id="client", graph_client_secret="hunter2"
    )
    assert "hunter2" not in repr(complete)


def test_the_token_key_is_a_secret() -> None:
    settings = Settings(token_key=SecretStr("a-key"))
    assert "a-key" not in repr(settings)
    assert settings.token_key is not None
    assert settings.token_key.get_secret_value() == "a-key"


def test_a_short_session_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(session_key=SecretStr("short"))
    assert Settings(session_key=SecretStr("k" * 32)).session_key is not None
