# The chat model is built without a network call for each provider, a call is priced by
# genai-prices under the price model, and a model or provider without a price is refused.

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from deskclerk.config import Settings
from deskclerk.providers import (
    MAX_TOKENS,
    InconsistentUsage,
    MissingApiKey,
    ProviderNotInstalled,
    Usage,
    chat_model,
    cost_of,
    usage_from,
)

HAIKU = "anthropic:claude-haiku-5-5"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_a_missing_key_stops_the_model_naming_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    with pytest.raises(MissingApiKey, match="ANTHROPIC_API_KEY is not set; put it in .env"):
        chat_model(Settings())


def test_the_default_model_is_anthropic_with_low_effort() -> None:
    model = chat_model(Settings())

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-haiku-5-5"
    assert model.effort == "low"
    assert model.max_tokens == MAX_TOKENS


@pytest.mark.parametrize("model_name", ["anthropic:claude-nope-1", "openai:gpt-nope"])
def test_a_model_without_a_price_is_refused(model_name: str) -> None:
    with pytest.raises(ValueError, match="no price for model"):
        chat_model(Settings(model=model_name))


def test_a_provider_outside_the_list_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown provider 'google'"):
        chat_model(Settings(model="google:gemini-2.5-pro"))


def test_an_openai_model_builds_with_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(MissingApiKey, match="OPENAI_API_KEY is not set; put it in .env"):
        chat_model(Settings(model="openai:gpt-5"))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = chat_model(Settings(model="openai:gpt-5"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5"
    assert model.reasoning_effort == "low"
    assert model.max_tokens == MAX_TOKENS


def test_the_prices_match_the_list_prices() -> None:
    one_hundred_thousand_each = Usage(100_000, 100_000, 0, 0)
    assert cost_of("anthropic:claude-haiku-5-5", one_hundred_thousand_each) == pytest.approx(0.06)
    assert cost_of("anthropic:claude-sonnet-5-5", one_hundred_thousand_each) == pytest.approx(1.20)
    assert cost_of("anthropic:claude-opus-5-5", one_hundred_thousand_each) == pytest.approx(2.40)


def _message(input_tokens: int, output_tokens: int, **details: int) -> AIMessage:
    usage: Any = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if details:
        usage["input_token_details"] = details
    return AIMessage(content="answer", usage_metadata=usage)


def test_a_cached_call_is_billed_at_the_cache_prices() -> None:
    message = _message(10_000, 500, cache_read=6_000, cache_creation=2_000)

    # 2,000 uncached at 0.10, 6,000 read at 0.01, 2,000 written at 0.125, 500 out at 0.50.
    expected = (2_000 * 0.10 + 6_000 * 0.01 + 2_000 * 0.125 + 500 * 0.50) / 1_000_000

    assert cost_of(HAIKU, usage_from(message)) == pytest.approx(expected)


def test_a_usage_without_cache_details_is_billed_as_plain_input() -> None:
    usage = usage_from(_message(10_000, 500))

    assert usage == Usage(10_000, 500, 0, 0)
    assert cost_of(HAIKU, usage) == pytest.approx((10_000 * 0.10 + 500 * 0.50) / 1_000_000)


def test_a_message_without_usage_metadata_costs_nothing() -> None:
    assert usage_from(AIMessage(content="answer")) == Usage(0, 0, 0, 0)


def test_cache_tokens_beyond_the_input_tokens_are_refused() -> None:
    usage = usage_from(_message(1_000, 0, cache_read=800, cache_creation=300))

    with pytest.raises(InconsistentUsage, match="exceed the 1000 input tokens"):
        cost_of(HAIKU, usage)


AZURE_VARIABLES = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION")


def _set_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-10-21")


@pytest.mark.parametrize("variable", AZURE_VARIABLES)
def test_an_azure_model_names_each_missing_variable(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_azure(monkeypatch)
    monkeypatch.delenv(variable)

    with pytest.raises(MissingApiKey, match=f"{variable} is not set; put it in .env"):
        chat_model(Settings(model="azure_openai:my-gpt", price_model="openai:gpt-5"))


def test_an_azure_deployment_builds_and_is_priced_as_the_model_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_azure(monkeypatch)
    settings = Settings(model="azure_openai:my-gpt", price_model="openai:gpt-5")

    model = chat_model(settings)

    assert isinstance(model, AzureChatOpenAI)
    assert model.deployment_name == "my-gpt"
    assert model.max_tokens == MAX_TOKENS
    assert cost_of(settings.price_model, Usage(1_000, 1_000, 0, 0)) > 0


def test_an_azure_deployment_without_a_price_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_azure(monkeypatch)

    with pytest.raises(ValueError, match="DESKCLERK_PRICE_MODEL"):
        chat_model(Settings(model="azure_openai:my-gpt"))


def test_the_price_model_defaults_to_the_model() -> None:
    assert Settings().price_model == "anthropic:claude-haiku-5-5"
    assert Settings(model="openai:gpt-5").price_model == "openai:gpt-5"


def test_a_price_model_without_a_provider_is_refused() -> None:
    with pytest.raises(ValueError, match="DESKCLERK_PRICE_MODEL is provider:model"):
        Settings(price_model="gpt-5")


def test_a_bedrock_model_needs_a_region_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    settings = Settings(
        model="bedrock_converse:eu.anthropic.claude-haiku-4-5-v1:0",
        price_model="anthropic:claude-haiku-5-5",
    )
    with pytest.raises(MissingApiKey, match="AWS_REGION is not set; put it in .env"):
        chat_model(settings)

    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    model = chat_model(settings)

    assert isinstance(model, ChatBedrockConverse)
    assert model.max_tokens == MAX_TOKENS


def test_a_provider_package_that_is_not_installed_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_installed(*args: Any, **kwargs: Any) -> None:
        raise ImportError("no module named langchain_aws")

    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setattr("deskclerk.providers.init_chat_model", not_installed)

    with pytest.raises(ProviderNotInstalled, match=r"deskclerk\[bedrock\]"):
        chat_model(
            Settings(
                model="bedrock_converse:eu.anthropic.claude-haiku-4-5-v1:0",
                price_model="anthropic:claude-haiku-5-5",
            )
        )
