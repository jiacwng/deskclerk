# The speed and search quality runner on the Enron slice: both searches are timed on the same
# emails, and a full answer is timed and priced with a scripted model.

from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from fakes import final_reply, scripted
from needtoknow.agent import SEARCH_RESULTS
from needtoknow.leak_eval import build_facts
from needtoknow.providers import Spending
from needtoknow.speed_eval import summarise, time_answers, time_searches, write_results

Connection = psycopg.Connection[TupleRow]


def test_both_searches_are_timed_on_every_email(app: Connection, reader: Connection) -> None:
    facts = build_facts(reader, limit=3)

    secured, unfiltered = time_searches(facts, app, reader)

    assert len(secured) == len(unfiltered) == len(facts)
    assert all(timing.seconds > 0 for timing in secured + unfiltered)
    assert any(timing.found for timing in unfiltered)


def test_answers_are_timed_and_priced(app: Connection, reader: Connection, tmp_path: Path) -> None:
    facts = build_facts(reader, limit=2)
    model = scripted(final_reply("one"), final_reply("two"))
    spending = Spending("claude-haiku-5-5", 1.0)

    answers = time_answers(facts, app, model, spending)
    secured, unfiltered = time_searches(facts, app, reader)
    summary = summarise(secured, unfiltered, answers, "claude-haiku-5-5")

    assert summary["answers"] == 2
    assert summary["answer_total_cost_usd"] == round(spending.cost_usd, 6)
    assert summary["results_asked"] == SEARCH_RESULTS
    write_results(tmp_path, summary)
    table = (tmp_path / "speed.md").read_text(encoding="utf-8")
    assert "Row-level security" in table
    assert "No filter" in table
