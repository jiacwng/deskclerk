# The chat model behind the agent, and what it costs. NEEDTOKNOW_MODEL names any provider
# LangChain's init_chat_model knows, as provider:model. Every model call is priced from the
# token usage in its response and counted against a spending cap that holds for the whole
# process.

import threading
from dataclasses import dataclass
from typing import Any

import anthropic
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from needtoknow.config import Settings

MAX_TOKENS = 4096
EXTRAS = frozenset({"openai", "ollama"})


class BudgetExceeded(RuntimeError):
    pass


class MissingProvider(ImportError):
    pass


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


# List prices in USD per million input and output tokens, for prompts up to 100,000 tokens.
# A model missing here needs NEEDTOKNOW_PRICE_INPUT and NEEDTOKNOW_PRICE_OUTPUT, so no call
# can go unpriced.
PRICES = {
    "anthropic:claude-haiku-5-5": Price(0.10, 0.50),
    "anthropic:claude-sonnet-5-5": Price(2.00, 10.00),
    "anthropic:claude-opus-5-5": Price(4.00, 20.00),
}


def price_of(settings: Settings) -> Price:
    if settings.price_input is not None and settings.price_output is not None:
        return Price(settings.price_input, settings.price_output)
    if settings.model not in PRICES:
        known = ", ".join(sorted(PRICES))
        raise ValueError(
            f"no price for model {settings.model!r}: set NEEDTOKNOW_PRICE_INPUT and "
            f"NEEDTOKNOW_PRICE_OUTPUT in USD per million tokens, or use one of {known}"
        )
    return PRICES[settings.model]


class Spending:
    def __init__(self, price: Price, budget_usd: float) -> None:
        self.price = price
        self.budget_usd = budget_usd
        self.input_tokens = 0
        self.output_tokens = 0
        # The API answers requests on a thread pool, and every request records here.
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return self.price.cost(self.input_tokens, self.output_tokens)

    def check(self) -> None:
        with self._lock:
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"spent {self.cost_usd:.4f} USD, the budget is {self.budget_usd:.4f} USD "
                    "(NEEDTOKNOW_BUDGET_USD)"
                )

    def record(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens


def _options(provider: str) -> dict[str, Any]:
    # Each client reads its API key from the environment, such as ANTHROPIC_API_KEY. Low effort
    # keeps Anthropic's adaptive thinking short; thinking itself is left at the model's default
    # because some models refuse a request that turns it off. Ollama ignores max_tokens and
    # caps the answer with num_predict.
    if provider == "anthropic":
        return {"max_tokens": MAX_TOKENS, "effort": "low"}
    if provider == "ollama":
        return {"num_predict": MAX_TOKENS}
    return {"max_tokens": MAX_TOKENS}


def provider_errors() -> tuple[type[Exception], ...]:
    try:
        import openai
    except ImportError:
        return (anthropic.APIError,)
    return (anthropic.APIError, openai.APIError)


def chat_model(settings: Settings) -> BaseChatModel:
    price_of(settings)
    provider, _, name = settings.model.partition(":")
    try:
        model: BaseChatModel = init_chat_model(name, model_provider=provider, **_options(provider))
    except ImportError as error:
        if provider not in EXTRAS:
            raise
        raise MissingProvider(
            f"NEEDTOKNOW_MODEL {settings.model!r} needs the {provider} package: "
            f"pip install 'needtoknow[{provider}]'"
        ) from error
    return model
