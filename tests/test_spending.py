# Spending is counted per employee per calendar month and kept in an SQLite file: it survives a
# new store on the same file, starts at zero in a new month, stops an employee at the budget, and
# includes Jev's input tokens.

from pathlib import Path

import pytest

from deskclerk.jev import PRICE_PER_INPUT_TOKEN
from deskclerk.providers import BudgetExceeded, Usage, cost_of
from deskclerk.spending import Spending, SpendingStore

MODEL = "anthropic:claude-haiku-5-5"


def spending_for(
    path: Path, user_id: str = "ada", month: str = "2026-10", budget: float = 5.0
) -> Spending:
    return Spending(MODEL, budget, SpendingStore(path), user_id, month)


def test_the_budget_counts_cache_costs(tmp_path: Path) -> None:
    first = Usage(50_000, 0, 50_000, 0)
    second = Usage(50_000, 0, 0, 50_000)
    total = cost_of(MODEL, first) + cost_of(MODEL, second)
    spending = spending_for(tmp_path / "spending.sqlite", budget=total - cost_of(MODEL, second) / 2)

    spending.record(first)
    spending.check()
    spending.record(second)

    assert spending.cost_usd == pytest.approx(total)
    with pytest.raises(BudgetExceeded):
        spending.check()


def test_the_cost_is_the_model_plus_jev(tmp_path: Path) -> None:
    spending = spending_for(tmp_path / "spending.sqlite")

    spending.record(Usage(50_000, 0, 0, 0))
    spending.record_jev(500_000)

    assert spending.cost_usd == pytest.approx(
        cost_of(MODEL, Usage(50_000, 0, 0, 0)) + 500_000 * PRICE_PER_INPUT_TOKEN
    )
    spending.check()


def test_spending_survives_a_new_store_on_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    first = spending_for(path, budget=0.04)
    first.record(Usage(90_000, 200, 3, 4))
    first.record_jev(1_000_000)

    again = spending_for(path, budget=0.04)

    assert again.cost_usd == pytest.approx(
        cost_of(MODEL, Usage(90_000, 200, 3, 4)) + 1_000_000 * PRICE_PER_INPUT_TOKEN
    )
    with pytest.raises(BudgetExceeded):
        again.check()


def test_a_spent_budget_still_stops_the_employee_after_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    spending_for(path, budget=0.005).record(Usage(90_000, 0, 0, 0))

    with pytest.raises(BudgetExceeded, match="resets at the start of next month"):
        spending_for(path, budget=0.005).check()


def test_a_new_month_starts_at_zero(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    cost = cost_of(MODEL, Usage(90_000, 0, 0, 0))
    spending_for(path, month="2026-10", budget=cost / 2).record(Usage(90_000, 0, 0, 0))

    november = spending_for(path, month="2026-11", budget=cost / 2)

    assert november.cost_usd == 0.0
    november.check()
    assert spending_for(path, month="2026-10", budget=cost / 2).cost_usd == pytest.approx(cost)


def test_each_employee_has_their_own_spending(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    spending_for(path, user_id="ada").record(Usage(90_000, 0, 0, 0))

    assert spending_for(path, user_id="grace").cost_usd == 0.0


def test_each_call_is_priced_on_its_own(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    spending = spending_for(path)

    spending.record(Usage(60_000, 0, 0, 0))
    spending.record(Usage(60_000, 0, 0, 0))

    one_call = cost_of(MODEL, Usage(60_000, 0, 0, 0))
    assert spending.cost_usd == pytest.approx(2 * one_call)
    assert spending.cost_usd < cost_of(MODEL, Usage(120_000, 0, 0, 0))
    assert spending_for(path).cost_usd == pytest.approx(2 * one_call)
