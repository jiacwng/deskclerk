# The chat model behind the agent, and what it costs. DESKCLERK_MODEL names a model as
# provider:model, with provider anthropic or openai. Every model call is priced by genai-prices
# from the token usage in its response.

import os
from dataclasses import dataclass
from typing import Any

from genai_prices import Usage as PriceUsage
from genai_prices import calc_price
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from deskclerk.config import Settings

MAX_TOKENS = 16000
EFFORT = "low"

API_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class BudgetExceeded(RuntimeError):
    pass


class MissingApiKey(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


class InconsistentUsage(ValueError):
    pass


def usage_from(message: AIMessage) -> Usage:
    metadata = message.usage_metadata
    if metadata is None:
        return Usage(0, 0, 0, 0)
    details = metadata.get("input_token_details", {})
    return Usage(
        input_tokens=metadata.get("input_tokens", 0),
        output_tokens=metadata.get("output_tokens", 0),
        cache_read_tokens=details.get("cache_read", 0),
        cache_write_tokens=details.get("cache_creation", 0),
    )


def split_model(model: str) -> tuple[str, str]:
    provider, _, name = model.partition(":")
    if provider not in API_KEYS:
        known = ", ".join(sorted(API_KEYS))
        raise ValueError(f"unknown provider {provider!r} in {model!r}: use one of {known}")
    return provider, name


def cost_of(model: str, usage: Usage) -> float:
    provider, name = split_model(model)
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0:
        raise InconsistentUsage(
            f"{usage.cache_read_tokens} cache read and {usage.cache_write_tokens} cache write "
            f"tokens exceed the {usage.input_tokens} input tokens"
        )
    try:
        calculation = calc_price(
            PriceUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            ),
            name,
            provider_id=provider,
        )
    except LookupError as error:
        raise ValueError(f"no price for model {name!r} from {provider}: {error}") from error
    return float(calculation.total_price)


def model_options(provider: str) -> dict[str, Any]:
    # Low effort keeps reasoning short; each provider names it and the output cap its own way.
    if provider == "openai":
        return {"max_tokens": MAX_TOKENS, "reasoning_effort": EFFORT}
    return {"max_tokens": MAX_TOKENS, "effort": EFFORT}


def missing_api_key(provider: str) -> str | None:
    variable = API_KEYS[provider]
    if not os.environ.get(variable):
        return f"{variable} is not set; put it in .env"
    return None


def chat_model(settings: Settings) -> BaseChatModel:
    cost_of(settings.model, Usage(0, 0, 0, 0))
    provider, name = split_model(settings.model)
    missing = missing_api_key(provider)
    if missing is not None:
        raise MissingApiKey(missing)
    model: BaseChatModel = init_chat_model(name, model_provider=provider, **model_options(provider))
    return model
