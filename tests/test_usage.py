# The usage command prints one tab separated line per employee for a month, highest spend first,
# with model and Jev dollars apart, the number of priced calls and a totals line; an empty month
# prints the header and a zero total.

import pytest

from deskclerk import __main__ as entry
from deskclerk.database import Database
from deskclerk.providers import Usage
from deskclerk.spending import SpendingStore

HEADER = "user_id\tmodel_usd\tjev_usd\ttotal_usd\tcalls"


def run_usage(capsys: pytest.CaptureFixture[str], month: str) -> list[str]:
    entry.main(["usage", "--month", month])
    return capsys.readouterr().out.splitlines()


def test_usage_sums_each_employee_for_the_month_highest_first(
    database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SpendingStore(database)
    store.add("ana", "2026-10", "model", Usage(1, 1, 0, 0), 0.25)
    store.add("ana", "2026-09", "model", Usage(1, 1, 0, 0), 9.0)
    store.add("bob", "2026-10", "model", Usage(1, 1, 0, 0), 1.0)
    store.add("bob", "2026-10", "jev", Usage(10, 0, 0, 0), 0.5)
    store.add("bob", "2026-10", "model", Usage(1, 1, 0, 0), 0.5)

    lines = run_usage(capsys, "2026-10")

    assert lines == [
        HEADER,
        "bob\t1.5000\t0.5000\t2.0000\t3",
        "ana\t0.2500\t0.0000\t0.2500\t1",
        "total\t1.7500\t0.5000\t2.2500\t4",
    ]


def test_usage_of_an_empty_month_prints_the_header_and_a_zero_total(
    database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    SpendingStore(database).add("ana", "2026-09", "model", Usage(1, 1, 0, 0), 1.0)

    lines = run_usage(capsys, "2026-10")

    assert lines == [HEADER, "total\t0.0000\t0.0000\t0.0000\t0"]


def test_usage_refuses_a_month_that_is_not_a_month(database: Database) -> None:
    with pytest.raises(SystemExit):
        entry.main(["usage", "--month", "October"])
