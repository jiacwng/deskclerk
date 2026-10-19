# The chat model behind the agent, and what it costs. DESKCLERK_MODEL names an Anthropic model
# as anthropic:model. Every model call is priced from the token usage in its response and
# counted against the spending cap of the employee who asked.

import os
import threading
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from deskclerk.config import Settings

MAX_TOKENS = 16000
EFFORT = "low"


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


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float
    cache_write_per_million: float

    def cost(self, usage: Usage) -> float:
        uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
        if uncached < 0:
            raise InconsistentUsage(
                f"{usage.cache_read_tokens} cache read and {usage.cache_write_tokens} cache write "
                f"tokens exceed the {usage.input_tokens} input tokens"
            )
        return (
            uncached * self.input_per_million
            + usage.cache_read_tokens * self.cache_read_per_million
            + usage.cache_write_tokens * self.cache_write_per_million
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


# Anthropic bills a 5-minute cache write at 1.25 times the input price and a cache read at 0.1
# times, or 0.05 times on Opus 5.5:
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def _anthropic_price(input_per_million: float, output_per_million: float, read: float) -> Price:
    return Price(
        input_per_million,
        output_per_million,
        input_per_million * read,
        input_per_million * CACHE_WRITE_MULTIPLIER,
    )


# List prices in USD per million input and output tokens, for prompts up to 100,000 tokens.
PRICES = {
    "anthropic:claude-haiku-5-5": _anthropic_price(0.10, 0.50, CACHE_READ_MULTIPLIER),
    "anthropic:claude-sonnet-5-5": _anthropic_price(2.00, 10.00, CACHE_READ_MULTIPLIER),
    "anthropic:claude-opus-5-5": _anthropic_price(4.00, 20.00, 0.05),
}


def price_of(settings: Settings) -> Price:
    if settings.model not in PRICES:
        known = ", ".join(sorted(PRICES))
        raise ValueError(f"no price for model {settings.model!r}: use one of {known}")
    return PRICES[settings.model]


class Spending:
    def __init__(self, price: Price, budget_usd: float) -> None:
        self.price = price
        self.budget_usd = budget_usd
        self.usage = Usage(0, 0, 0, 0)
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return self.price.cost(self.usage)

    def check(self) -> None:
        with self._lock:
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"spent {self.cost_usd:.4f} USD, the budget is {self.budget_usd:.4f} USD "
                    "(DESKCLERK_BUDGET_USD)"
                )

    def record(self, usage: Usage) -> None:
        with self._lock:
            self.usage = Usage(
                self.usage.input_tokens + usage.input_tokens,
                self.usage.output_tokens + usage.output_tokens,
                self.usage.cache_read_tokens + usage.cache_read_tokens,
                self.usage.cache_write_tokens + usage.cache_write_tokens,
            )


def missing_api_key() -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set; put it in .env"
    return None


def chat_model(settings: Settings) -> BaseChatModel:
    price_of(settings)
    missing = missing_api_key()
    if missing is not None:
        raise MissingApiKey(missing)
    name = settings.model.removeprefix("anthropic:")
    # Low effort keeps adaptive thinking short; thinking itself is left at the model's default
    # because some models refuse a request that turns it off.
    return init_chat_model(name, model_provider="anthropic", max_tokens=MAX_TOKENS, effort=EFFORT)
