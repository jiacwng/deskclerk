# BIRD Mini-Dev execution accuracy: needtoknow's agent against a single-call baseline, both
# answering as the analyst who sees every row, scored the way the BIRD repository scores it (the
# predicted result set equals the gold result set, order-insensitive). The gold SQL and both
# predictions run on the loaded data. A full run:
# NEEDTOKNOW_DB_NAME=needtoknow_full PYTHONPATH=src .venv/bin/python -m needtoknow.bird_eval \
#   --questions ~/datasets/bird/extracted/minidev/MINIDEV/mini_dev_postgresql.json --out results/

import argparse
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from psycopg import sql

from needtoknow.agent import Answer, ask
from needtoknow.auth import principals
from needtoknow.config import load_settings
from needtoknow.data import TIMEOUT_MS, Caps, describe_data
from needtoknow.db import Pool, app_pool, as_user
from needtoknow.providers import BudgetExceeded, Price, Spending, chat_model, price_of
from needtoknow.stats import mcnemar_exact

ANALYST = principals("analyst", ["analyst"])
CORRECT = "correct"
WRONG_RESULT = "wrong_result"
SQL_ERROR = "sql_error"
OUTCOMES = (CORRECT, WRONG_RESULT, SQL_ERROR)

# The benchmark compares result sets exactly, so a column added for an employee's benefit scores
# as a wrong answer. Both the agent and the baseline are given this sentence.
OUTPUT_CONTRACT = (
    "Your query must return exactly the columns the question asks for, in that order, and no "
    "others. Do not add identifying or explanatory columns."
)
BASELINE_SYSTEM = (
    "You translate a question into one read-only PostgreSQL SELECT and put it in the sql field. "
    f"{OUTPUT_CONTRACT}"
)
SQL_REPLY_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
        "additionalProperties": False,
    },
}
# A gold answer the agent never sees is scored as a wrong answer, so the evaluation reads far
# more rows than a person ever would. Eight of the 500 answers are longer than this.
EVAL_CAPS = Caps(rows=2_000, chars=200_000)

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
    return _result_set(predicted) == _result_set(gold)


def read_records(path: Path) -> list[Record]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["agent"] = Prediction(**row["agent"])
        row["baseline"] = Prediction(**row["baseline"])
        records.append(Record(**row))
    _refuse_repeats(records)
    return records


def previous_records(raw_path: Path, resume: bool) -> list[Record]:
    if not raw_path.exists():
        return []
    if not resume:
        raise BirdEvalError(
            f"{raw_path} already holds scored questions. Pass --resume to carry on from them, "
            "or choose another --out."
        )
    return read_records(raw_path)


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
    question: Question, pool: Pool, model: BaseChatModel, spending: Spending
) -> tuple[str | None, int, int]:
    text = f"Using the data in the schema named {question.db_id!r}: {question.question}"
    if question.evidence:
        text += f"\n\nEvidence: {question.evidence}"
    text += f"\n\n{OUTPUT_CONTRACT}"
    answer = ask(text, pool, ANALYST, model, spending, caps=EVAL_CAPS)
    return _answer_query(answer), answer.input_tokens, answer.output_tokens


def baseline_prediction(
    question: Question, schema: str, model: BaseChatModel, spending: Spending
) -> tuple[str | None, int, int]:
    spending.check()
    prompt = (
        f"Schema {question.db_id}:\n{schema}\n\n"
        f"Question: {question.question}\nEvidence: {question.evidence}\n\n"
        f"Write one SELECT over this schema, using schema-qualified names like {question.db_id}."
    )
    structured = model.bind(output_config={"format": SQL_REPLY_FORMAT})
    reply = structured.invoke([SystemMessage(BASELINE_SYSTEM), HumanMessage(prompt)])
    if not isinstance(reply, AIMessage) or reply.usage_metadata is None:
        raise RuntimeError("the model reply carries no token usage, so it cannot be priced")
    usage = reply.usage_metadata
    spending.record(usage["input_tokens"], usage["output_tokens"])
    return _reply_sql(reply.text), usage["input_tokens"], usage["output_tokens"]


def score(
    questions: Sequence[Question],
    pool: Pool,
    model: BaseChatModel,
    spending: Spending,
    raw: TextIO,
) -> tuple[list[Record], list[int]]:
    records = []
    unscorable: list[int] = []
    for question in questions:
        with pool.connection() as app:
            gold, gold_error = run_sql(app, question.db_id, ANALYST, question.gold_sql)
        # A gold query that will not run scores nothing, and it must not discard the questions
        # already paid for, so it is counted and reported instead of raised.
        if gold is None:
            unscorable.append(question.question_id)
            _log.info(
                "question %d (%s): skipped, gold SQL failed: %s",
                question.question_id,
                question.db_id,
                gold_error,
            )
            continue
        gold_set = _result_set(gold)

        agent_sql, agent_in, agent_out = agent_prediction(question, pool, model, spending)
        with pool.connection() as app:
            agent_outcome = _outcome(app, question.db_id, agent_sql, gold_set)
            schema = describe_data(app, question.db_id)
        baseline_sql, base_in, base_out = baseline_prediction(question, schema, model, spending)
        with pool.connection() as app:
            baseline_outcome = _outcome(app, question.db_id, baseline_sql, gold_set)

        record = Record(
            question_id=question.question_id,
            db_id=question.db_id,
            difficulty=question.difficulty,
            agent=Prediction(agent_outcome, agent_sql, agent_in, agent_out),
            baseline=Prediction(baseline_outcome, baseline_sql, base_in, base_out),
        )
        records.append(record)
        raw.write(json.dumps(asdict(record)) + "\n")
        raw.flush()
        _log.info(
            "question %d (%s): agent %s, baseline %s",
            question.question_id,
            question.db_id,
            agent_outcome,
            baseline_outcome,
        )
    return records, unscorable


def summarise(
    records: Sequence[Record], model_name: str, price: Price, unscorable: Sequence[int] = ()
) -> dict[str, Any]:
    _refuse_repeats(records)
    return {
        "model": model_name,
        "questions": len(records),
        "unscorable": list(unscorable),
        "agent": _side_summary(records, True, price),
        "baseline": _side_summary(records, False, price),
    }


def _unscorable_note(unscorable: Sequence[int]) -> list[str]:
    if not unscorable:
        return []
    ids = ", ".join(str(question_id) for question_id in unscorable)
    return [
        f"{len(unscorable)} questions are left out because their gold SQL would not run: {ids}.",
        "",
    ]


def results_table(summary: dict[str, Any]) -> str:
    total = summary["questions"]
    lines = [
        "# BIRD execution accuracy",
        "",
        f"Model {summary['model']}. {total} BIRD Mini-Dev questions, scored as the BIRD "
        "repository scores them: the predicted result set equals the gold result set, "
        "order-insensitive. Both read as the analyst, who sees every row.",
        "",
        *_unscorable_note(summary.get("unscorable") or []),
        "| Approach | Correct | Accuracy | SQL error | Wrong result | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("agent", "baseline"):
        side = summary[name]
        counts = side["outcomes"]
        lines.append(
            f"| {name} | {counts[CORRECT]} | {side['accuracy']:.3f} | {counts[SQL_ERROR]} "
            f"| {counts[WRONG_RESULT]} | {side['cost_usd']:.4f} |"
        )
    agent_correct = set(summary["agent"]["correct_question_ids"])
    baseline_correct = set(summary["baseline"]["correct_question_ids"])
    only_agent = len(agent_correct - baseline_correct)
    only_baseline = len(baseline_correct - agent_correct)
    p_value = mcnemar_exact(only_agent, only_baseline)
    significance = "is" if p_value < 0.05 else "is not"
    cost_ratio = summary["agent"]["cost_usd"] / summary["baseline"]["cost_usd"]
    lines += [
        "",
        f"Only the agent got {only_agent} questions right and only the baseline {only_baseline}. "
        f"The exact McNemar test on these paired outcomes gives p = {p_value:.3f}, so the "
        f"difference {significance} significant at the 5% level. The agent costs "
        f"{cost_ratio:.1f} times as much as the baseline.",
        "",
        f"Agent tokens: {summary['agent']['input_tokens']:,} input and "
        f"{summary['agent']['output_tokens']:,} output. Baseline tokens: "
        f"{summary['baseline']['input_tokens']:,} input and "
        f"{summary['baseline']['output_tokens']:,} output.",
        "",
    ]
    return "\n".join(lines)


def write_results(out: Path, summary: dict[str, Any]) -> None:
    (out / "bird.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "bird.md").write_text(results_table(summary), encoding="utf-8")


class BirdEvalError(RuntimeError):
    pass


def _refuse_repeats(records: Sequence[Record]) -> None:
    counts = Counter(record.question_id for record in records)
    repeated = sorted(question_id for question_id, count in counts.items() if count > 1)
    if repeated:
        ids = ", ".join(str(question_id) for question_id in repeated)
        raise BirdEvalError(f"questions scored more than once: {ids}")


def _outcome(
    app: psycopg.Connection[Any],
    db_id: str,
    predicted_sql: str | None,
    gold_set: set[tuple[Any, ...]],
) -> str:
    if not predicted_sql:
        return SQL_ERROR
    rows, error = run_sql(app, db_id, ANALYST, predicted_sql)
    if error is not None or rows is None:
        return SQL_ERROR
    return CORRECT if _result_set(rows) == gold_set else WRONG_RESULT


def _result_set(rows: Sequence[Sequence[Any]]) -> set[tuple[Any, ...]]:
    return {tuple(_hashable(value) for value in row) for row in rows}


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


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


def _reply_sql(text: str) -> str | None:
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(reply, dict) or not isinstance(reply.get("sql"), str):
        return None
    return reply["sql"].strip().rstrip(";").strip() or None


def _answer_query(answer: Answer) -> str | None:
    shown = _squash(answer.text)
    ran = [query for query in answer.queries if query.error is None]
    for query in reversed(ran):
        if _squash(query.sql) in shown:
            return query.sql
    if ran:
        return ran[-1].sql
    if answer.queries:
        return answer.queries[-1].sql
    return None


def _squash(text: str) -> str:
    return " ".join(text.split()).rstrip(";")


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    raw_path = args.out / "bird_raw.jsonl"
    try:
        done = previous_records(raw_path, args.resume)
    except BirdEvalError as error:
        raise SystemExit(str(error)) from error
    scored = {record.question_id for record in done}
    questions = [
        question
        for question in read_questions(args.questions)[: args.limit]
        if question.question_id not in scored
    ]
    spending = Spending(price_of(settings), args.budget_usd)
    args.out.mkdir(parents=True, exist_ok=True)
    with app_pool(settings, size=1) as pool, raw_path.open("a", encoding="utf-8") as raw:
        try:
            records, unscorable = score(questions, pool, chat_model(settings), spending, raw)
        except BudgetExceeded as error:
            raise SystemExit(f"stopped: {error}. Rerun with --resume to carry on.") from error

    summary = summarise(done + records, settings.model, spending.price, unscorable)
    write_results(args.out, summary)
    print(results_table(summary))


if __name__ == "__main__":
    main()
