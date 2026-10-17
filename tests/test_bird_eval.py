# BIRD execution accuracy on the slice with a scripted model: the scoring function is
# order-insensitive and catches a wrong result, and the runner scores the agent against the
# single-call baseline end to end on one question from each slice database.

import io
import json
from dataclasses import asdict
from pathlib import Path

import psycopg
import pytest
from langchain_core.messages import AIMessage
from psycopg.rows import TupleRow

from fakes import HAIKU, call_tool, final_reply, scripted
from needtoknow.bird_eval import (
    ANALYST,
    OUTPUT_CONTRACT,
    SQL_REPLY_FORMAT,
    BirdEvalError,
    Prediction,
    Question,
    Record,
    agent_prediction,
    execution_match,
    previous_records,
    read_records,
    run_sql,
    score,
    summarise,
    write_results,
)
from needtoknow.db import Pool
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]


def sql_reply(query: str) -> AIMessage:
    return final_reply(json.dumps({"sql": query}))


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


def test_the_runner_scores_the_agent_and_the_baseline(app_pool: Pool, tmp_path) -> None:  # type: ignore[no-untyped-def]
    def agent_then_wrong_baseline(gold: str, db_id: str, wrong: str) -> list:  # type: ignore[type-arg]
        return [
            call_tool("describe_data", {"schema": None}),
            call_tool("describe_data", {"schema": db_id}),
            call_tool("query_data", {"sql": gold}),
            final_reply(f"The answer uses {gold}"),
            sql_reply(wrong),
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
    spending = Spending(HAIKU, 2.0)

    raw_path = tmp_path / "bird_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        records, unscorable = score([FINANCIAL, SCHOOLS], app_pool, model, spending, raw)

    assert unscorable == []
    summary = summarise(records, "anthropic:claude-haiku-5-5", HAIKU)
    assert summary["agent"]["accuracy"] == 1.0
    assert summary["baseline"]["accuracy"] == 0.0
    assert summary["agent"]["correct_question_ids"] == [1, 2]
    assert summary["baseline"]["outcomes"]["wrong_result"] == 2

    write_results(tmp_path, summary)
    table = (tmp_path / "bird.md").read_text(encoding="utf-8")
    assert "# BIRD execution accuracy" in table
    assert "Only the agent got 2 questions right and only the baseline 0" in table
    assert "so the difference is not significant at the 5% level" in table
    assert read_records(raw_path) == records


def test_both_sides_are_asked_for_exactly_the_columns_in_the_question(app_pool: Pool) -> None:
    model = scripted(final_reply("No query."), sql_reply(DISTRICTS_SQL))

    score([FINANCIAL], app_pool, model, Spending(HAIKU, 2.0), io.StringIO())

    agent_prompt, baseline_prompt = model.prompts
    assert OUTPUT_CONTRACT in agent_prompt[-1].text
    assert OUTPUT_CONTRACT in baseline_prompt[0].text


def test_the_agent_reads_every_row_of_a_long_gold_answer(app_pool: Pool) -> None:
    many = "SELECT n FROM generate_series(1, 500) AS n"
    model = scripted(call_tool("query_data", {"sql": many}), final_reply("Done."))

    agent_prediction(FINANCIAL, app_pool, model, Spending(HAIKU, 2.0))

    rows_shown = model.prompts[-1][-1].text
    assert rows_shown.endswith("(500 rows)")


@pytest.mark.parametrize(
    ("queries", "scored"),
    [
        (["SELECT nope", DISTRICTS_SQL], DISTRICTS_SQL),
        ([DISTRICTS_SQL, "SELECT nope"], DISTRICTS_SQL),
        (["SELECT nope", "SELECT still_nope"], "SELECT still_nope"),
        ([], None),
    ],
    ids=["probe-then-answer", "answer-then-failure", "every-query-failed", "no-query"],
)
def test_the_scored_query_is_the_last_that_ran_else_the_last_sent(
    app_pool: Pool, queries: list[str], scored: str | None
) -> None:
    calls = [call_tool("query_data", {"sql": query}) for query in queries]
    model = scripted(*calls, final_reply("Done."))

    predicted, _, _ = agent_prediction(FINANCIAL, app_pool, model, Spending(HAIKU, 2.0))

    assert predicted == scored


def test_an_array_answer_is_scored_instead_of_crashing(app_pool: Pool) -> None:
    pair = "SELECT ARRAY[1, 2] AS pair"
    question = Question(
        question_id=3,
        db_id="financial",
        question="Which pair?",
        evidence="",
        gold_sql=pair,
        difficulty="simple",
    )
    model = scripted(call_tool("query_data", {"sql": pair}), final_reply("Done."), sql_reply(pair))

    records, _ = score([question], app_pool, model, Spending(HAIKU, 2.0), io.StringIO())

    assert records[0].agent.outcome == "correct"
    assert records[0].baseline.outcome == "correct"


def test_scored_questions_are_kept_only_when_resuming(tmp_path: Path) -> None:
    raw_path = tmp_path / "bird_raw.jsonl"
    assert previous_records(raw_path, resume=False) == []

    record = Record(
        question_id=1,
        db_id="financial",
        difficulty="simple",
        agent=Prediction("correct", DISTRICTS_SQL, 10, 2),
        baseline=Prediction("sql_error", None, 5, 1),
    )
    raw_path.write_text(json.dumps(asdict(record)) + "\n", encoding="utf-8")

    with pytest.raises(BirdEvalError, match="Pass --resume"):
        previous_records(raw_path, resume=False)
    assert previous_records(raw_path, resume=True) == [record]


def test_a_question_scored_twice_is_refused(tmp_path: Path) -> None:
    record = Record(
        question_id=137,
        db_id="financial",
        difficulty="simple",
        agent=Prediction("correct", DISTRICTS_SQL, 10, 2),
        baseline=Prediction("sql_error", None, 5, 1),
    )
    raw_path = tmp_path / "bird_raw.jsonl"
    line = json.dumps(asdict(record)) + "\n"
    raw_path.write_text(line + line, encoding="utf-8")

    with pytest.raises(BirdEvalError, match="more than once: 137"):
        read_records(raw_path)
    with pytest.raises(BirdEvalError, match="more than once: 137"):
        summarise([record, record], "anthropic:claude-haiku-5-5", HAIKU)


def test_the_agent_is_scored_on_the_query_its_answer_shows(app_pool: Pool) -> None:
    probe = "SELECT DISTINCT frequency FROM financial.account"
    model = scripted(
        call_tool("query_data", {"sql": DISTRICTS_SQL}),
        call_tool("query_data", {"sql": probe}),
        final_reply(f"The districts:\n\n```sql\n{DISTRICTS_SQL};\n```"),
    )

    predicted, _, _ = agent_prediction(FINANCIAL, app_pool, model, Spending(HAIKU, 2.0))

    assert predicted == DISTRICTS_SQL


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (json.dumps({"sql": DISTRICTS_SQL + ";"}), "correct"),
        (f"{DISTRICTS_SQL}\n\nWait, the question asks for something else.", "sql_error"),
        (json.dumps({"sql": ""}), "sql_error"),
    ],
    ids=["structured", "prose", "empty"],
)
def test_the_baseline_is_scored_only_on_the_sql_field_of_its_reply(
    app_pool: Pool, reply: str, outcome: str
) -> None:
    model = scripted(final_reply("No query."), final_reply(reply))

    records, _ = score([FINANCIAL], app_pool, model, Spending(HAIKU, 2.0), io.StringIO())

    assert model.options[-1]["output_config"] == {"format": SQL_REPLY_FORMAT}
    assert records[0].baseline.outcome == outcome


def test_an_agent_that_runs_no_query_scores_an_sql_error(app_pool: Pool) -> None:
    model = scripted(final_reply("I cannot answer that."), sql_reply(DISTRICTS_SQL))

    records, _ = score([FINANCIAL], app_pool, model, Spending(HAIKU, 2.0), io.StringIO())

    assert records[0].agent.outcome == "sql_error"
    summary = summarise(records, "anthropic:claude-haiku-5-5", HAIKU)
    assert summary["agent"]["outcomes"] == {"correct": 0, "wrong_result": 0, "sql_error": 1}
