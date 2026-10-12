# BIRD execution accuracy on the slice with a scripted model: the scoring function is
# order-insensitive and catches a wrong result, and the runner scores the agent against the
# single-call baseline end to end on one question from each slice database.

import psycopg
from psycopg.rows import TupleRow

from fakes import call_tool, final_reply, scripted
from needtoknow.bird_eval import (
    ANALYST,
    Question,
    execution_match,
    run_sql,
    score,
    summarise,
    write_results,
)
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]

DISTRICTS_SQL = "SELECT DISTINCT district_id FROM financial.account ORDER BY district_id"
COUNTIES_SQL = "SELECT DISTINCT county FROM california_schools.schools ORDER BY county"

FINANCIAL = Question(
    question_id=1,
    db_id="financial",
    question="Which districts hold an account?",
    evidence="",
    gold_sql=DISTRICTS_SQL,
    difficulty="simple",
)
SCHOOLS = Question(
    question_id=2,
    db_id="california_schools",
    question="Which counties have a school?",
    evidence="",
    gold_sql=COUNTIES_SQL,
    difficulty="simple",
)


def test_execution_match_is_order_insensitive_and_catches_a_wrong_result(
    app: Connection,
) -> None:
    rows, error = run_sql(app, "financial", ANALYST, DISTRICTS_SQL)
    assert error is None and rows is not None and len(rows) > 1

    assert execution_match(rows, list(reversed(rows)))
    assert not execution_match(rows, rows[:-1])


def test_the_runner_scores_the_agent_and_the_baseline(app: Connection, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def agent_then_wrong_baseline(gold: str, db_id: str, wrong: str) -> list:  # type: ignore[type-arg]
        return [
            call_tool("describe_data", {}),
            call_tool("describe_data", {"schema": db_id}),
            call_tool("query_data", {"sql": gold}),
            final_reply(f"The answer uses {gold}"),
            final_reply(wrong),
        ]

    replies = agent_then_wrong_baseline(
        DISTRICTS_SQL,
        "financial",
        "SELECT district_id FROM financial.account WHERE district_id = -1",
    )
    replies += agent_then_wrong_baseline(
        COUNTIES_SQL,
        "california_schools",
        "SELECT county FROM california_schools.schools WHERE county = 'Nowhere'",
    )
    model = scripted(*replies)
    spending = Spending("claude-haiku-5-5", 2.0)

    records = score([FINANCIAL, SCHOOLS], app, model, spending)

    summary = summarise(records, "claude-haiku-5-5")
    assert summary["agent"]["accuracy"] == 1.0
    assert summary["baseline"]["accuracy"] == 0.0
    assert summary["agent"]["correct_question_ids"] == [1, 2]
    assert summary["baseline"]["outcomes"]["wrong_result"] == 2

    write_results(tmp_path, records, summary)
    table = (tmp_path / "bird.md").read_text(encoding="utf-8")
    assert "# BIRD execution accuracy" in table
    assert (tmp_path / "bird_raw.jsonl").read_text(encoding="utf-8").count("\n") == 2
