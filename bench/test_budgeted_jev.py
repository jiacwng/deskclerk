# Checks the bench's Jev with a fake client in place of the paid API: the spending cap and the call
# that asks several questions about one state at once.

from types import SimpleNamespace
from typing import Any, cast

import pytest
from typesafe_sdk import TypeSafeClient

from budgeted_jev import BudgetedJev
from deskclerk.jev import PRICE_PER_INPUT_TOKEN, JevError


class AnsweringClient:
    def __init__(self, input_tokens: int) -> None:
        self.input_tokens = input_tokens

    def system_one(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=self.input_tokens),
            nouls={"answer": SimpleNamespace(noul=0.25)},
        )


class ManyAnsweringClient:
    def __init__(self) -> None:
        self.asked: list[object] = []

    def system_one(self, state: Any, questions: Any, **kwargs: Any) -> Any:
        self.asked.append((state, sorted(questions)))
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=300),
            nouls={name: SimpleNamespace(noul=0.1 * index) for index, name in enumerate(questions)},
        )


def jev_with(client: object, budget_usd: float | None = 1.0) -> BudgetedJev:
    return BudgetedJev(cast(TypeSafeClient, client), budget_usd)


def test_a_spent_budget_refuses_the_next_question() -> None:
    jev = jev_with(AnsweringClient(1_000_000), budget_usd=0.01)
    jev.noul({}, "q")
    with pytest.raises(JevError, match="budget"):
        jev.noul({}, "q")


def test_without_a_budget_there_is_no_cap() -> None:
    jev = jev_with(AnsweringClient(1_000_000), budget_usd=None)
    for _ in range(3):
        assert jev.noul({}, "q") == 0.25
    assert jev.input_tokens == 3_000_000


def test_noul_many_asks_every_question_in_one_call_and_bills_the_tokens_once() -> None:
    client = ManyAnsweringClient()
    jev = jev_with(client)
    reported: list[int] = []
    answers = jev.noul_many({"s": 1}, {"a": "qa", "b": "qb"}, on_tokens=reported.append)
    assert answers == {"a": 0.0, "b": pytest.approx(0.1)}
    assert client.asked == [({"s": 1}, ["a", "b"])]
    assert reported == [300]
    assert jev.cost_usd == pytest.approx(300 * PRICE_PER_INPUT_TOKEN)


def test_noul_many_respects_the_budget() -> None:
    jev = jev_with(ManyAnsweringClient(), budget_usd=0.0)
    with pytest.raises(JevError, match="budget"):
        jev.noul_many({}, {"a": "q"})
