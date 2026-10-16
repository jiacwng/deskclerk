# The speed and search quality runner on the Enron slice: both searches are timed on the same
# emails and checked against an exact search, the slowest get a query plan, and a full answer is
# timed and priced with a scripted model.

from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from fakes import HAIKU, final_reply, scripted
from needtoknow.agent import SEARCH_RESULTS
from needtoknow.db import Pool
from needtoknow.leak_eval import build_facts
from needtoknow.providers import Spending
from needtoknow.speed_eval import (
    slowest_plans,
    summarise,
    time_answers,
    time_searches,
    write_results,
)

Connection = psycopg.Connection[TupleRow]


def test_both_searches_are_timed_on_every_email(app: Connection, reader: Connection) -> None:
    facts = build_facts(reader, limit=3)

    secured, unfiltered = time_searches(facts, app, reader)

    assert len(secured) == len(unfiltered) == len(facts)
    assert all(timing.seconds > 0 for timing in secured + unfiltered)
    assert any(timing.found for timing in unfiltered)
    assert all(0.0 <= timing.recall <= 1.0 for timing in secured + unfiltered)
    assert all(timing.visible > 0 for timing in secured)


def test_the_slowest_secured_searches_get_their_query_plan(
    app: Connection, reader: Connection
) -> None:
    facts = build_facts(reader, limit=2)
    secured, _ = time_searches(facts, app, reader)

    plans = slowest_plans(facts, secured, app)

    assert len(plans) == 2
    assert all("Execution Time" in plan for plan in plans)
    assert all("[query embedding]" in plan for plan in plans)


def test_answers_are_timed_and_priced(
    app: Connection, app_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    facts = build_facts(reader, limit=2)
    model = scripted(final_reply("one"), final_reply("two"))
    spending = Spending(HAIKU, 1.0)

    answers = time_answers(facts, app_pool, model, spending)
    secured, unfiltered = time_searches(facts, app, reader)
    summary = summarise(secured, unfiltered, answers, "anthropic:claude-haiku-5-5")

    assert summary["answers"] == 2
    assert summary["answer_total_cost_usd"] == round(spending.cost_usd, 6)
    assert summary["results_asked"] == SEARCH_RESULTS
    assert sum(group["searches"] for group in summary["secured_by_visible"]) == 2
    write_results(tmp_path, summary, ["plan"])
    table = (tmp_path / "speed.md").read_text(encoding="utf-8")
    assert "Row-level security" in table
    assert "No filter" in table
