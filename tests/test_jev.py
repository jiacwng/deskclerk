# Jev's wrapper around the TypeSafe SDK: how its 400 errors map to exceptions, the optional
# spending cap and the token callback. A fake client stands in for the paid API.

from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from typesafe_sdk import TypeSafeBadRequestError, TypeSafeClient

from deskclerk.jev import PRICE_PER_INPUT_TOKEN, Jev, JevError, JevInputTooLong


class RefusingClient:
    def __init__(self, body: object) -> None:
        self.body = body

    def system_one(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeSafeBadRequestError(400, self.body, httpx2.Headers())


class AnsweringClient:
    def __init__(self, input_tokens: int) -> None:
        self.input_tokens = input_tokens

    def system_one(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=self.input_tokens),
            nouls={"answer": SimpleNamespace(noul=0.25)},
        )


def jev_with(client: object, budget_usd: float | None = 1.0) -> Jev:
    return Jev(cast(TypeSafeClient, client), budget_usd)


def test_a_400_max_tokens_exceeded_from_typesafe_is_jev_input_too_long() -> None:
    body = {"detail": {"code": "max_tokens_exceeded"}}
    with pytest.raises(JevInputTooLong):
        jev_with(RefusingClient(body)).noul({}, "q")


def test_any_other_400_from_typesafe_is_a_plain_jev_error() -> None:
    with pytest.raises(JevError) as raised:
        jev_with(RefusingClient({"detail": "bad state"})).noul({}, "q")
    assert not isinstance(raised.value, JevInputTooLong)


def test_on_tokens_is_called_with_the_input_tokens_of_each_answer() -> None:
    reported: list[int] = []
    jev = jev_with(AnsweringClient(120))
    assert jev.noul({}, "q", on_tokens=reported.append) == 0.25
    jev.noul({}, "q", on_tokens=reported.append)
    assert reported == [120, 120]
    assert jev.cost_usd == pytest.approx(240 * PRICE_PER_INPUT_TOKEN)


def test_on_tokens_is_not_called_when_jev_fails() -> None:
    reported: list[int] = []
    with pytest.raises(JevError):
        jev_with(RefusingClient({"detail": "bad"})).noul({}, "q", on_tokens=reported.append)
    assert reported == []


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
