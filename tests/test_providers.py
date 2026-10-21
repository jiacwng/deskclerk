# The chat model is built without a network call with low effort, a model is priced from the
# table and refused outside it, and an optional tool argument stays required and nullable under
# strict tool use.

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding

from deskclerk import providers
from deskclerk.config import Settings
from deskclerk.providers import (
    MAX_TOKENS,
    InconsistentUsage,
    MissingApiKey,
    Usage,
    chat_model,
    price_of,
    usage_from,
)

LOOKUP_TOOL: dict[str, Any] = {
    "name": "lookup",
    "description": "Look a name up, in one group or in all of them.",
    "input_schema": {
        "type": "object",
        "properties": {"group": {"type": ["string", "null"]}},
        "required": ["group"],
        "additionalProperties": False,
    },
}


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


@pytest.mark.parametrize("model_name", ["anthropic:claude-haiku-4-5", "openai:gpt-5"])
def test_a_model_outside_the_table_is_refused(model_name: str) -> None:
    settings = Settings(model=model_name)

    with pytest.raises(ValueError, match="no price for model"):
        price_of(settings)
    with pytest.raises(ValueError, match="no price for model"):
        chat_model(settings)


def test_the_price_table_matches_the_list_prices() -> None:
    prices = providers.PRICES
    one_million_each = Usage(1_000_000, 1_000_000, 0, 0)
    assert prices["anthropic:claude-haiku-5-5"].cost(one_million_each) == pytest.approx(0.60)
    assert prices["anthropic:claude-sonnet-5-5"].cost(one_million_each) == pytest.approx(12.00)
    assert prices["anthropic:claude-opus-5-5"].cost(one_million_each) == pytest.approx(24.00)


def test_a_tool_takes_a_nullable_schema_under_strict_tool_use() -> None:
    anthropic = chat_model(Settings()).bind_tools([LOOKUP_TOOL], strict=True)

    assert isinstance(anthropic, RunnableBinding)
    (tool,) = anthropic.kwargs["tools"]
    assert tool["input_schema"] == LOOKUP_TOOL["input_schema"]


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
    price = providers.PRICES["anthropic:claude-haiku-5-5"]
    message = _message(10_000, 500, cache_read=6_000, cache_creation=2_000)

    # 2,000 uncached at 0.10, 6,000 read at 0.01, 2,000 written at 0.125, 500 out at 0.50.
    expected = (2_000 * 0.10 + 6_000 * 0.01 + 2_000 * 0.125 + 500 * 0.50) / 1_000_000

    assert price.cost(usage_from(message)) == pytest.approx(expected)


def test_a_usage_without_cache_details_is_billed_as_plain_input() -> None:
    price = providers.PRICES["anthropic:claude-haiku-5-5"]

    usage = usage_from(_message(10_000, 500))

    assert usage == Usage(10_000, 500, 0, 0)
    assert price.cost(usage) == pytest.approx((10_000 * 0.10 + 500 * 0.50) / 1_000_000)


def test_a_message_without_usage_metadata_costs_nothing() -> None:
    assert usage_from(AIMessage(content="answer")) == Usage(0, 0, 0, 0)


def test_cached_tokens_cost_less_than_the_same_tokens_uncached() -> None:
    price = providers.PRICES["anthropic:claude-haiku-5-5"]

    cached = price.cost(usage_from(_message(10_000, 0, cache_read=8_000)))
    plain = price.cost(usage_from(_message(10_000, 0)))

    assert cached < plain


def test_cache_tokens_beyond_the_input_tokens_are_refused() -> None:
    price = providers.PRICES["anthropic:claude-haiku-5-5"]
    usage = usage_from(_message(1_000, 0, cache_read=800, cache_creation=300))

    with pytest.raises(InconsistentUsage, match="exceed the 1000 input tokens"):
        price.cost(usage)
