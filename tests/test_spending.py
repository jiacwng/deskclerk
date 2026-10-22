# Spending is counted per employee per calendar month and kept in an SQLite file: it survives a
# new store on the same file, starts at zero in a new month, stops an employee at the budget, and
# includes Jev's input tokens.

from pathlib import Path

import pytest

from deskclerk.jev import PRICE_PER_INPUT_TOKEN
from deskclerk.providers import BudgetExceeded, Price, Usage
from deskclerk.spending import Spending, SpendingStore

PRICE = Price(1.0, 0.0, 0.0, 10.0)


def spending_for(
    path: Path, user_id: str = "ada", month: str = "2026-10", budget: float = 5.0
) -> Spending:
    return Spending(PRICE, budget, SpendingStore(path), user_id, month)


def test_the_budget_counts_cache_costs(tmp_path: Path) -> None:
    spending = spending_for(tmp_path / "spending.sqlite")

    spending.record(Usage(1_000_000, 0, 1_000_000, 0))
    spending.check()
    spending.record(Usage(1_000_000, 0, 0, 1_000_000))

    assert spending.cost_usd == pytest.approx(10.0)
    with pytest.raises(BudgetExceeded):
        spending.check()


def test_the_cost_is_the_model_plus_jev(tmp_path: Path) -> None:
    spending = spending_for(tmp_path / "spending.sqlite")

    spending.record(Usage(1_000_000, 0, 0, 0))
    spending.record_jev(500_000)

    assert spending.cost_usd == pytest.approx(1.0 + 500_000 * PRICE_PER_INPUT_TOKEN)
    spending.check()


def test_spending_survives_a_new_store_on_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    first = spending_for(path, budget=0.04)
    first.record(Usage(1_000_000, 200, 3, 4))
    first.record_jev(1_000_000)

    again = spending_for(path, budget=0.04)

    assert again.usage == Usage(1_000_000, 200, 3, 4)
    assert again.jev_usd == pytest.approx(1_000_000 * PRICE_PER_INPUT_TOKEN)
    assert again.cost_usd == pytest.approx(
        PRICE.cost(Usage(1_000_000, 200, 3, 4)) + 1_000_000 * PRICE_PER_INPUT_TOKEN
    )
    with pytest.raises(BudgetExceeded):
        again.check()


def test_a_spent_budget_still_stops_the_employee_after_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    spending_for(path, budget=0.5).record(Usage(1_000_000, 0, 0, 0))

    with pytest.raises(BudgetExceeded, match="resets at the start of next month"):
        spending_for(path, budget=0.5).check()


def test_a_new_month_starts_at_zero(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    spending_for(path, month="2026-10", budget=0.5).record(Usage(1_000_000, 0, 0, 0))

    november = spending_for(path, month="2026-11", budget=0.5)

    assert november.cost_usd == 0.0
    november.check()
    assert spending_for(path, month="2026-10", budget=0.5).cost_usd == pytest.approx(1.0)


def test_each_employee_has_their_own_spending(tmp_path: Path) -> None:
    path = tmp_path / "spending.sqlite"
    spending_for(path, user_id="ada").record(Usage(1_000_000, 0, 0, 0))

    assert spending_for(path, user_id="grace").cost_usd == 0.0
