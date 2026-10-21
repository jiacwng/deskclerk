# What each employee has spent this calendar month (UTC), kept in an SQLite file so a restart does
# not reset it. Model calls are counted as tokens and priced when read; Jev input tokens are
# counted in USD. Every spend is written to the file before the next check.

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from deskclerk.jev import PRICE_PER_INPUT_TOKEN
from deskclerk.providers import BudgetExceeded, Price, Usage


def month_of(moment: datetime) -> str:
    return moment.strftime("%Y-%m")


class SpendingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS spending ("
                "user_id TEXT NOT NULL, month TEXT NOT NULL, "
                "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
                "cache_read_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL, "
                "jev_usd REAL NOT NULL, PRIMARY KEY (user_id, month))"
            )

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self._path)
        try:
            with database:
                yield database
        finally:
            database.close()

    def load(self, user_id: str, month: str) -> tuple[Usage, float]:
        with self._database() as database:
            row = database.execute(
                "SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
                "jev_usd FROM spending WHERE user_id = ? AND month = ?",
                (user_id, month),
            ).fetchone()
        if row is None:
            return Usage(0, 0, 0, 0), 0.0
        return Usage(row[0], row[1], row[2], row[3]), row[4]

    def add(self, user_id: str, month: str, usage: Usage, jev_usd: float) -> None:
        with self._database() as database:
            database.execute(
                "INSERT INTO spending (user_id, month, input_tokens, output_tokens, "
                "cache_read_tokens, cache_write_tokens, jev_usd) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (user_id, month) DO UPDATE SET "
                "input_tokens = input_tokens + excluded.input_tokens, "
                "output_tokens = output_tokens + excluded.output_tokens, "
                "cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens, "
                "cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens, "
                "jev_usd = jev_usd + excluded.jev_usd",
                (
                    user_id,
                    month,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_tokens,
                    usage.cache_write_tokens,
                    jev_usd,
                ),
            )


class Spending:
    def __init__(
        self, price: Price, budget_usd: float, store: SpendingStore, user_id: str, month: str
    ) -> None:
        self.price = price
        self.budget_usd = budget_usd
        self._store = store
        self._user_id = user_id
        self._month = month
        self.usage, self.jev_usd = store.load(user_id, month)
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return self.price.cost(self.usage) + self.jev_usd

    def check(self) -> None:
        with self._lock:
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"spent {self.cost_usd:.4f} USD this month, the budget is "
                    f"{self.budget_usd:.4f} USD (DESKCLERK_BUDGET_USD); it resets at the start "
                    "of next month (UTC)"
                )

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._store.add(self._user_id, self._month, usage, 0.0)
            self.usage = Usage(
                self.usage.input_tokens + usage.input_tokens,
                self.usage.output_tokens + usage.output_tokens,
                self.usage.cache_read_tokens + usage.cache_read_tokens,
                self.usage.cache_write_tokens + usage.cache_write_tokens,
            )

    def record_jev(self, tokens: int) -> None:
        with self._lock:
            cost = tokens * PRICE_PER_INPUT_TOKEN
            self._store.add(self._user_id, self._month, Usage(0, 0, 0, 0), cost)
            self.jev_usd += cost
