# The chat model behind the agent, and what it costs. DESKCLERK_MODEL names a model as
# provider:model, with provider anthropic, openai, azure_openai or bedrock_converse. Every model
# call is priced by genai-prices from the token usage in its response, under DESKCLERK_PRICE_MODEL.

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

REQUIRED_ENVIRONMENT = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "azure_openai": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"),
    "bedrock_converse": (),
}
EXTRAS = {"openai": "openai", "azure_openai": "openai", "bedrock_converse": "bedrock"}
PRICE_PROVIDERS = {
    "anthropic": "anthropic",
    "openai": "openai",
    "azure_openai": "azure",
    "bedrock_converse": "aws",
}
REGION_VARIABLES = ("AWS_REGION", "AWS_DEFAULT_REGION")


class BudgetExceeded(RuntimeError):
    pass


class MissingApiKey(RuntimeError):
    pass


class ProviderNotInstalled(RuntimeError):
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
    if provider not in REQUIRED_ENVIRONMENT:
        known = ", ".join(sorted(REQUIRED_ENVIRONMENT))
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
            provider_id=PRICE_PROVIDERS[provider],
        )
    except LookupError as error:
        raise ValueError(
            f"no price for model {name!r} from {provider}: {error}; "
            f"set DESKCLERK_PRICE_MODEL to a model genai-prices knows, such as openai:gpt-5"
        ) from error
    return float(calculation.total_price)


def model_options(provider: str) -> dict[str, Any]:
    # Low effort keeps reasoning short; each provider names it and the output cap its own way.
    if provider in ("openai", "azure_openai"):
        return {"max_tokens": MAX_TOKENS, "reasoning_effort": EFFORT}
    if provider == "bedrock_converse":
        return {"max_tokens": MAX_TOKENS}
    return {"max_tokens": MAX_TOKENS, "effort": EFFORT}


def missing_environment(provider: str) -> str | None:
    for variable in REQUIRED_ENVIRONMENT[provider]:
        if not os.environ.get(variable):
            return f"{variable} is not set; put it in .env"
    if provider == "bedrock_converse" and not any(os.environ.get(v) for v in REGION_VARIABLES):
        return "AWS_REGION is not set; put it in .env"
    return None


def chat_model(settings: Settings) -> BaseChatModel:
    cost_of(settings.price_model, Usage(0, 0, 0, 0))
    provider, name = split_model(settings.model)
    missing = missing_environment(provider)
    if missing is not None:
        raise MissingApiKey(missing)
    options = model_options(provider)
    if provider == "azure_openai":
        options["azure_deployment"] = name
    try:
        model: BaseChatModel = init_chat_model(name, model_provider=provider, **options)
    except ImportError as error:
        extra = EXTRAS[provider]
        raise ProviderNotInstalled(
            f"{provider} needs the {extra} extra: pip install 'deskclerk[{extra}]'"
        ) from error
    return model
