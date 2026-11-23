# Jev with a spending cap and a call that asks several questions about one state at once, for the
# bench runners. The product's Jev has neither.

import os
from collections.abc import Callable, Mapping
from typing import Any

from typesafe_sdk import Noul, TypeSafeClient

from deskclerk.jev import Jev, JevError


class BudgetedJev(Jev):
    def __init__(self, client: TypeSafeClient, budget_usd: float | None) -> None:
        super().__init__(client)
        self.budget_usd = budget_usd

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Noul],
        on_tokens: Callable[[int], None] | None,
    ) -> dict[str, float]:
        if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
            raise JevError(
                f"Jev spent {self.cost_usd:.4f} USD, its budget is {self.budget_usd} USD"
            )
        return super().ask(state, questions, on_tokens)

    def noul_many(
        self,
        state: Any,
        instructions: Mapping[str, str],
        on_tokens: Callable[[int], None] | None = None,
    ) -> dict[str, float]:
        questions = {name: Noul(instructions=text) for name, text in instructions.items()}
        return self.ask(state, questions, on_tokens)


def budgeted_jev_from_environment(budget_usd: float | None) -> BudgetedJev | None:
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        return None
    return BudgetedJev(TypeSafeClient(), budget_usd)
