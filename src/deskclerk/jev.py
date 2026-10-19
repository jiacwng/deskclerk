# TypeSafe's Jev decision model through its official SDK: one yes/no question (a Noul) about a
# piece of state, answered with the probability of yes. The SDK reads TYPESAFE_API_KEY, retries
# busy or failed calls, and keeps credentials out of its errors; this module adds the running
# cost, a spending cap and one exception type for the callers.

import os
import threading
from typing import Any

from typesafe_sdk import (
    Noul,
    NoulCriteria,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeError,
)

MODEL = "jev-1.13.0"
# TypeSafe bills input tokens only.
PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000


class JevError(RuntimeError):
    pass


class JevInputTooLong(JevError):
    pass


class Jev:
    def __init__(self, client: TypeSafeClient, budget_usd: float) -> None:
        self._client = client
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self.input_tokens = 0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN

    def noul(self, state: Any, instructions: str, criteria: NoulCriteria | None = None) -> float:
        if self.cost_usd >= self.budget_usd:
            raise JevError(
                f"Jev spent {self.cost_usd:.4f} USD, its budget is {self.budget_usd} USD"
            )
        question = Noul(instructions=instructions, criteria=criteria)
        try:
            response = self._client.system_one(state, {"answer": question}, model=MODEL)
        except TypeSafeBadRequestError as error:
            if "max_tokens_exceeded" in str(error.body):
                raise JevInputTooLong(f"Jev refused the state as too long: {error}") from None
            raise JevError(f"Jev failed: {error}") from None
        except TypeSafeError as error:
            raise JevError(f"Jev failed: {error}") from None
        with self._lock:
            self.input_tokens += response.usage.input_tokens or 0
        return float(response.nouls["answer"].noul)


def jev_from_environment(budget_usd: float) -> Jev | None:
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return None
    return Jev(TypeSafeClient(), budget_usd)
