# What each employee has spent this calendar month (UTC), summed from a Postgres table, so a restart
# or another copy of the app sees the same total. Every model call is priced on its own when it is
# recorded, and Jev input tokens are priced at Jev's rate. One row per priced call is written
# before the next check.

from dataclasses import dataclass
from datetime import datetime

from deskclerk.database import Database
from deskclerk.jev import PRICE_PER_INPUT_TOKEN
from deskclerk.providers import BudgetExceeded, Usage, cost_of


def month_of(moment: datetime) -> str:
    return moment.strftime("%Y-%m")


@dataclass(frozen=True)
class UsageRow:
    user_id: str
    model_usd: float
    jev_usd: float
    calls: int

    @property
    def total_usd(self) -> float:
        return self.model_usd + self.jev_usd


class SpendingStore:
    def __init__(self, database: Database) -> None:
        self._pool = database.pool

    def load(self, user_id: str, month: str) -> float:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(usd), 0.0) FROM costs WHERE user_id = %s AND month = %s",
                (user_id, month),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def report(self, month: str) -> list[UsageRow]:
        with self._pool.connection() as connection:
            found = connection.execute(
                "SELECT user_id, "
                "COALESCE(SUM(CASE WHEN source = 'model' THEN usd END), 0.0), "
                "COALESCE(SUM(CASE WHEN source = 'jev' THEN usd END), 0.0), "
                "COUNT(*) FROM costs WHERE month = %s GROUP BY user_id "
                "ORDER BY SUM(usd) DESC, user_id",
                (month,),
            ).fetchall()
        return [
            UsageRow(str(user), float(model), float(jev), int(calls))
            for user, model, jev, calls in found
        ]

    def delete_user(self, user_id: str) -> int:
        with self._pool.connection() as connection:
            return connection.execute("DELETE FROM costs WHERE user_id = %s", (user_id,)).rowcount

    def add(self, user_id: str, month: str, source: str, usage: Usage, usd: float) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                "INSERT INTO costs (user_id, month, source, input_tokens, output_tokens, "
                "cache_read_tokens, cache_write_tokens, usd) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
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

    @property
    def cost_usd(self) -> float:
        return self._store.load(self._user_id, self._month)

    def check(self) -> None:
        spent = self.cost_usd
        if spent >= self.budget_usd:
            raise BudgetExceeded(
                f"spent {spent:.4f} USD this month, the budget is "
                f"{self.budget_usd:.4f} USD (DESKCLERK_BUDGET_USD); it resets at the start "
                "of next month (UTC)"
            )

    def record(self, usage: Usage) -> None:
        cost = cost_of(self.model, usage)
        self._store.add(self._user_id, self._month, "model", usage, cost)

    def record_jev(self, tokens: int) -> None:
        cost = tokens * PRICE_PER_INPUT_TOKEN
        self._store.add(self._user_id, self._month, "jev", Usage(tokens, 0, 0, 0), cost)
