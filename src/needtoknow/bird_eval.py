# BIRD Mini-Dev execution accuracy: needtoknow's agent against a single-call baseline, both
# answering as the analyst who sees every row, scored the way the BIRD repository scores it (the
# predicted result set equals the gold result set, order-insensitive). The gold SQL and both
# predictions run on the loaded data. A full run:
# NEEDTOKNOW_DB_NAME=needtoknow_full PYTHONPATH=src .venv/bin/python -m needtoknow.bird_eval \
#   --questions ~/datasets/bird/extracted/minidev/MINIDEV/mini_dev_postgresql.json --out results/

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from psycopg import sql

from needtoknow.agent import ask
from needtoknow.auth import principals
from needtoknow.config import load_settings
from needtoknow.data import TIMEOUT_MS, describe_data
from needtoknow.db import as_user, connect_app
from needtoknow.providers import BudgetExceeded, Price, Spending, chat_model, price_of

ANALYST = principals("analyst", ["analyst"])
CORRECT = "correct"
WRONG_RESULT = "wrong_result"
SQL_ERROR = "sql_error"
NO_QUERY = "no_query"
OUTCOMES = (CORRECT, WRONG_RESULT, SQL_ERROR, NO_QUERY)

BASELINE_SYSTEM = (
    "You translate a question into one read-only PostgreSQL SELECT. Reply with only the SQL, "
    "no explanation and no code fence."
)

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_log = logging.getLogger("needtoknow.bird_eval")


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


@dataclass(frozen=True)
class Prediction:
    outcome: str
    sql: str | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Record:
    question_id: int
    db_id: str
    difficulty: str
    agent: Prediction
    baseline: Prediction


def read_questions(path: Path) -> list[Question]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            question_id=int(row["question_id"]),
            db_id=row["db_id"],
            question=row["question"],
            evidence=row.get("evidence", ""),
            gold_sql=row["SQL"],
            difficulty=row.get("difficulty", ""),
        )
        for row in rows
    ]


def execution_match(predicted: Sequence[tuple[Any, ...]], gold: Sequence[tuple[Any, ...]]) -> bool:
    return {tuple(row) for row in predicted} == {tuple(row) for row in gold}


def run_sql(
    app: psycopg.Connection[Any], db_id: str, who: frozenset[str], query: str
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    try:
        with as_user(app, who), app.cursor() as cursor:
            app.execute("SET TRANSACTION READ ONLY")
            app.execute(f"SET LOCAL statement_timeout = {TIMEOUT_MS}")
            search_path = sql.SQL("SET LOCAL search_path TO {}, public")
            app.execute(search_path.format(sql.Identifier(db_id)))
            cursor.execute(query)
            return cursor.fetchall(), None
    except psycopg.Error as error:
        return None, error.diag.message_primary or "the database refused it"


def agent_prediction(
    question: Question, app: psycopg.Connection[Any], model: BaseChatModel, spending: Spending
) -> tuple[str | None, int, int]:
    text = f"Using the data in the schema named {question.db_id!r}: {question.question}"
    if question.evidence:
        text += f"\n\nEvidence: {question.evidence}"
    answer = ask(text, app, ANALYST, model, spending)
    predicted = answer.queries[-1].sql if answer.queries else None
    return predicted, answer.input_tokens, answer.output_tokens


def baseline_prediction(
    question: Question, schema: str, model: BaseChatModel, spending: Spending
) -> tuple[str | None, int, int]:
    spending.check()
    prompt = (
        f"Schema {question.db_id}:\n{schema}\n\n"
        f"Question: {question.question}\nEvidence: {question.evidence}\n\n"
        f"Write one SELECT over this schema, using schema-qualified names like {question.db_id}."
    )
    reply = model.invoke([SystemMessage(BASELINE_SYSTEM), HumanMessage(prompt)])
    if not isinstance(reply, AIMessage) or reply.usage_metadata is None:
        raise RuntimeError("the model reply carries no token usage, so it cannot be priced")
    usage = reply.usage_metadata
    spending.record(usage["input_tokens"], usage["output_tokens"])
    return _extract_sql(reply.text), usage["input_tokens"], usage["output_tokens"]


def score(
    questions: Sequence[Question],
    app: psycopg.Connection[Any],
    model: BaseChatModel,
    spending: Spending,
) -> list[Record]:
    records = []
    for question in questions:
        gold, gold_error = run_sql(app, question.db_id, ANALYST, question.gold_sql)
        if gold is None:
            raise BirdEvalError(
                f"gold SQL for question {question.question_id} failed: {gold_error}"
            )
        gold_set = {tuple(row) for row in gold}

        agent_sql, agent_in, agent_out = agent_prediction(question, app, model, spending)
        agent_outcome = _outcome(app, question.db_id, agent_sql, gold_set)
        schema = describe_data(app, question.db_id)
        baseline_sql, base_in, base_out = baseline_prediction(question, schema, model, spending)
        baseline_outcome = _outcome(app, question.db_id, baseline_sql, gold_set)

        record = Record(
            question_id=question.question_id,
            db_id=question.db_id,
            difficulty=question.difficulty,
            agent=Prediction(agent_outcome, agent_sql, agent_in, agent_out),
            baseline=Prediction(baseline_outcome, baseline_sql, base_in, base_out),
        )
        records.append(record)
        _log.info(
            "question %d (%s): agent %s, baseline %s",
            question.question_id,
            question.db_id,
            agent_outcome,
            baseline_outcome,
        )
    return records


def summarise(records: Sequence[Record], model_name: str) -> dict[str, Any]:
    price = price_of(model_name)
    return {
        "model": model_name,
        "questions": len(records),
        "agent": _side_summary(records, True, price),
        "baseline": _side_summary(records, False, price),
    }


def results_table(summary: dict[str, Any]) -> str:
    total = summary["questions"]
    lines = [
        "# BIRD execution accuracy",
        "",
        f"Model {summary['model']}. {total} BIRD Mini-Dev questions, scored as the BIRD "
        "repository scores them: the predicted result set equals the gold result set, "
        "order-insensitive. Both read as the analyst, who sees every row.",
        "",
        "| Approach | Correct | Accuracy | SQL error | Wrong result | No query | Cost (USD) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("agent", "baseline"):
        side = summary[name]
        counts = side["outcomes"]
        lines.append(
            f"| {name} | {counts[CORRECT]} | {side['accuracy']:.3f} | {counts[SQL_ERROR]} "
            f"| {counts[WRONG_RESULT]} | {counts[NO_QUERY]} | {side['cost_usd']:.4f} |"
        )
    lines += [
        "",
        f"Agent tokens: {summary['agent']['input_tokens']:,} input and "
        f"{summary['agent']['output_tokens']:,} output. Baseline tokens: "
        f"{summary['baseline']['input_tokens']:,} input and "
        f"{summary['baseline']['output_tokens']:,} output.",
        "",
    ]
    return "\n".join(lines)


def write_results(out: Path, records: Sequence[Record], summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bird_raw.jsonl").open("w", encoding="utf-8") as raw:
        for record in records:
            raw.write(json.dumps(asdict(record)) + "\n")
    (out / "bird.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "bird.md").write_text(results_table(summary), encoding="utf-8")


class BirdEvalError(RuntimeError):
    pass


def _outcome(
    app: psycopg.Connection[Any],
    db_id: str,
    predicted_sql: str | None,
    gold_set: set[tuple[Any, ...]],
) -> str:
    if not predicted_sql:
        return NO_QUERY
    rows, error = run_sql(app, db_id, ANALYST, predicted_sql)
    if error is not None or rows is None:
        return SQL_ERROR
    return CORRECT if {tuple(row) for row in rows} == gold_set else WRONG_RESULT


def _side_summary(records: Sequence[Record], agent: bool, price: Price) -> dict[str, Any]:
    predictions = [(record.agent if agent else record.baseline) for record in records]
    counts = Counter(prediction.outcome for prediction in predictions)
    input_tokens = sum(prediction.input_tokens for prediction in predictions)
    output_tokens = sum(prediction.output_tokens for prediction in predictions)
    return {
        "accuracy": counts[CORRECT] / len(records) if records else 0.0,
        "outcomes": {outcome: counts[outcome] for outcome in OUTCOMES},
        "correct_question_ids": [
            record.question_id
            for record, prediction in zip(records, predictions, strict=True)
            if prediction.outcome == CORRECT
        ],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(price.cost(input_tokens, output_tokens), 6),
    }


def _extract_sql(text: str) -> str:
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    return text.strip().rstrip(";").strip()


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Score BIRD execution accuracy, agent vs baseline."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--limit", type=_positive, default=None)
    parser.add_argument("--budget-usd", type=float, default=settings.budget_usd)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    questions = read_questions(args.questions)[: args.limit]
    spending = Spending(settings.model, args.budget_usd)
    with connect_app(settings) as app:
        try:
            records = score(questions, app, chat_model(settings), spending)
        except BudgetExceeded as error:
            raise SystemExit(f"stopped, nothing written: {error}") from error

    summary = summarise(records, settings.model)
    write_results(args.out, records, summary)
    print(results_table(summary))


if __name__ == "__main__":
    main()
