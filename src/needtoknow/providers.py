# The chat model behind the agent, and what it costs. Every model call is priced from the token
# usage in its response and counted against a spending cap that holds for the whole process.

import threading
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic

from needtoknow.config import Settings

# Anthropic list prices in USD per million input and output tokens, for prompts up to 100,000
# tokens. A model missing here is refused, so no call can go unpriced.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
MAX_TOKENS = 4096


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


def price_of(model: str) -> Price:
    if model not in PRICES:
        known = ", ".join(sorted(PRICES))
        raise ValueError(f"no price for model {model!r}, known models: {known}")
    input_price, output_price = PRICES[model]
    return Price(input_per_million=input_price, output_per_million=output_price)


class Spending:
    def __init__(self, model: str, budget_usd: float) -> None:
        self.price = price_of(model)
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


def chat_model(settings: Settings) -> ChatAnthropic:
    price_of(settings.model)
    # The client reads ANTHROPIC_API_KEY from the environment. Low effort keeps adaptive
    # thinking short; thinking itself is left at the model's default because some models
    # refuse a request that turns it off.
    return ChatAnthropic(model=settings.model, max_tokens=MAX_TOKENS, effort="low")
