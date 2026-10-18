# The chat model for each provider is built without a network call, Anthropic alone gets low
# effort, a model is priced from the table or from the price settings and refused without
# either, a missing provider package names the extra that installs it, and the optional
# describe_data argument stays required and nullable under strict tool use.

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableBinding
from langchain_core.utils.function_calling import convert_to_openai_tool

from needtoknow import providers
from needtoknow.agent import DESCRIBE_DATA_TOOL
from needtoknow.config import Settings
from needtoknow.providers import (
    MAX_TOKENS,
    MissingApiKey,
    MissingProvider,
    Price,
    chat_model,
    price_of,
)


def _priced(model_name: str = "anthropic:claude-haiku-5-5") -> Settings:
    return Settings(model=model_name, price_input=1.25, price_output=10.0)


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.parametrize(
    ("model_name", "variable"),
    [("anthropic:claude-haiku-5-5", "ANTHROPIC_API_KEY"), ("openai:gpt-5", "OPENAI_API_KEY")],
)
def test_a_missing_key_stops_the_model_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, model_name: str, variable: str
) -> None:
    monkeypatch.delenv(variable)

    with pytest.raises(MissingApiKey, match=f"{variable} is not set; put it in .env"):
        chat_model(_priced(model_name))


def test_an_ollama_model_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")

    chat_model(_priced("ollama:llama3.2:3b"))


def test_the_default_model_is_anthropic_with_low_effort() -> None:
    model = chat_model(Settings())

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-5-5"
    assert model.effort == "low"
    assert model.max_tokens == MAX_TOKENS


def test_an_openai_model_gets_no_effort() -> None:
    # The optional providers are imported here, so collecting the suite stays as fast as before.
    from langchain_openai import ChatOpenAI

    model = chat_model(_priced("openai:gpt-5"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5"
    assert model.max_tokens == MAX_TOKENS
    assert model.model_kwargs == {}


def test_an_ollama_model_keeps_the_colon_in_its_tag_and_caps_its_answer() -> None:
    from langchain_ollama import ChatOllama

    model = chat_model(_priced("ollama:llama3.2:3b"))

    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.2:3b"
    assert model.num_predict == MAX_TOKENS


@pytest.mark.parametrize("model_name", ["anthropic:claude-haiku-4-5", "openai:gpt-5"])
def test_a_model_outside_the_table_without_prices_is_refused(model_name: str) -> None:
    settings = Settings(model=model_name)

    with pytest.raises(ValueError, match="NEEDTOKNOW_PRICE_INPUT and NEEDTOKNOW_PRICE_OUTPUT"):
        price_of(settings)
    with pytest.raises(ValueError, match="no price for model"):
        chat_model(settings)


def test_the_price_settings_price_a_model_and_override_the_table() -> None:
    assert price_of(_priced("openai:gpt-5")) == Price(1.25, 10.0)
    assert price_of(_priced()) == Price(1.25, 10.0)


def test_the_price_table_matches_the_list_prices() -> None:
    prices = providers.PRICES
    assert prices["anthropic:claude-haiku-5-5"].cost(1_000_000, 1_000_000) == pytest.approx(0.60)
    assert prices["anthropic:claude-sonnet-5-5"].cost(1_000_000, 1_000_000) == pytest.approx(12.00)
    assert prices["anthropic:claude-opus-5-5"].cost(1_000_000, 1_000_000) == pytest.approx(24.00)


def test_a_missing_provider_package_names_its_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_installed(*args: Any, **kwargs: Any) -> None:
        raise ImportError("No module named 'langchain_ollama'")

    monkeypatch.setattr(providers, "init_chat_model", not_installed)

    with pytest.raises(MissingProvider, match=r"pip install 'needtoknow\[ollama\]'"):
        chat_model(_priced("ollama:llama3.2:3b"))


def test_describe_data_takes_a_nullable_schema_under_strict_tool_use() -> None:
    openai = convert_to_openai_tool(DESCRIBE_DATA_TOOL, strict=True)["function"]
    anthropic = chat_model(Settings()).bind_tools([DESCRIBE_DATA_TOOL], strict=True)

    assert openai["strict"] is True
    assert openai["parameters"]["required"] == ["schema"]
    assert openai["parameters"]["properties"]["schema"]["type"] == ["string", "null"]
    assert isinstance(anthropic, RunnableBinding)
    (tool,) = anthropic.kwargs["tools"]
    assert tool["input_schema"] == DESCRIBE_DATA_TOOL["input_schema"]
