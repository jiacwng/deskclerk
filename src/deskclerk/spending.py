# What each employee has spent this calendar month (UTC), kept in an SQLite file so a restart does
# not reset it. Every model call is priced on its own when it is recorded, and Jev input tokens
# are priced at Jev's rate. One row per priced call is written before the next check.

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from deskclerk.jev import PRICE_PER_INPUT_TOKEN
from deskclerk.providers import BudgetExceeded, Usage, cost_of


def month_of(moment: datetime) -> str:
    return moment.strftime("%Y-%m")


class SpendingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS costs ("
                "user_id TEXT NOT NULL, month TEXT NOT NULL, source TEXT NOT NULL, "
                "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
                "cache_read_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL, "
                "usd REAL NOT NULL)"
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS costs_user_month ON costs (user_id, month)"
            )

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self._path)
        try:
            with database:
                yield database
        finally:
            database.close()

    def load(self, user_id: str, month: str) -> float:
        with self._database() as database:
            row = database.execute(
                "SELECT COALESCE(SUM(usd), 0.0) FROM costs WHERE user_id = ? AND month = ?",
                (user_id, month),
            ).fetchone()
        return float(row[0])

    def delete_user(self, user_id: str) -> int:
        with self._database() as database:
            return database.execute("DELETE FROM costs WHERE user_id = ?", (user_id,)).rowcount

    def add(self, user_id: str, month: str, source: str, usage: Usage, usd: float) -> None:
        with self._database() as database:
            database.execute(
                "INSERT INTO costs (user_id, month, source, input_tokens, output_tokens, "
                "cache_read_tokens, cache_write_tokens, usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    month,
                    source,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_tokens,
                    usage.cache_write_tokens,
                    usd,
                ),
            )


class Spending:
    def __init__(
        self, model: str, budget_usd: float, store: SpendingStore, user_id: str, month: str
    ) -> None:
        self.model = model
        self.budget_usd = budget_usd
        self._store = store
        self._user_id = user_id
        self._month = month
        self.spent_usd = store.load(user_id, month)
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return self.spent_usd

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
            cost = cost_of(self.model, usage)
            self._store.add(self._user_id, self._month, "model", usage, cost)
            self.spent_usd += cost

    def record_jev(self, tokens: int) -> None:
        with self._lock:
            cost = tokens * PRICE_PER_INPUT_TOKEN
            self._store.add(self._user_id, self._month, "jev", Usage(tokens, 0, 0, 0), cost)
            self.spent_usd += cost
