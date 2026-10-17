# The answer table on the Enron mailboxes, from EnronQA (Ryan et al. 2025, arXiv 2505.00263): its
# questions and reference answers were written per email of the CMU release, and each row names
# the email's file in the maildir. A sample of rows whose email is loaded is asked by the
# custodian whose mailbox holds it, under row-level security. As in the paper, a judge compares
# the answer with the reference answer; here the judge is TypeSafe's Jev. The injection screen
# runs as it ships, so the table also counts the answers it withheld passages from.
# A full run (NEEDTOKNOW_DB_NAME=needtoknow_full):
# PYTHONPATH=src .venv/bin/python -m needtoknow.evaluation.quality_eval \
#   --enronqa ~/datasets/enronqa/test/test-00000-of-00001.parquet \
#   --maildir ~/datasets/enron/extracted/maildir --limit 200 --workers 12 --out results/

import argparse
import json
import logging
import random
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import psycopg
import pyarrow.parquet as pq
from langchain_core.language_models import BaseChatModel

from needtoknow.agent import ask, employee_tools
from needtoknow.auth import principals
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.db import Pool, app_pool, connect_reader
from needtoknow.enron import MAX_FILE_BYTES, parse_email
from needtoknow.evaluation.stats import with_interval
from needtoknow.jev import Jev, JevError, jev_from_environment
from needtoknow.providers import BudgetExceeded, Spending, chat_model, price_of
from needtoknow.screen import Screen, screen_from_settings

# The paper's judge prompt (its appendix B.6) asks whether the answer matches the gold answer.
JUDGE_QUESTION = "Does the answer match the gold answer?"
RAW_FILE = "answers_enron_raw.jsonl"
SAMPLE_SEED = 0
_log = logging.getLogger("needtoknow.evaluation.quality_eval")


@dataclass(frozen=True)
class Row:
    path: str
    user: str
    question: str
    gold_answer: str


@dataclass(frozen=True)
class Question:
    doc_id: str
    user: str
    question: str
    gold_answer: str
    email: str


@dataclass(frozen=True)
class Sample:
    questions: list[Question]
    rows_read: int


@dataclass(frozen=True)
class Record:
    doc_id: str
    user: str
    question: str
    gold_answer: str
    answer: str
    retrieved: bool
    correct: bool
    refused: bool
    exhausted: bool
    withheld: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def read_rows(path: Path) -> list[Row]:
    table = pq.read_table(path, columns=["path", "user", "questions", "gold_answers"])
    rows = []
    for row in table.to_pylist():
        if not row["questions"] or not row["gold_answers"]:
            continue
        rows.append(
            Row(
                path=row["path"],
                user=row["user"],
                question=row["questions"][0],
                gold_answer=row["gold_answers"][0],
            )
        )
    return rows


# A row counts when its file is inside the maildir, was loaded, and the custodian named in the row
# reads the email; a copy of an email already taken is skipped. Rows come in a seeded shuffle, so
# --limit always takes the same ones.
def build_questions(
    reader: psycopg.Connection[Any], rows: Sequence[Row], maildir: Path, limit: int | None
) -> Sample:
    shuffled = list(rows)
    sampler = random.Random(SAMPLE_SEED)  # noqa: S311, a reproducible sample, not a secret
    sampler.shuffle(shuffled)
    root = maildir.resolve()
    questions: list[Question] = []
    taken: set[str] = set()
    read = 0
    for row in shuffled:
        if limit is not None and len(questions) == limit:
            break
        read += 1
        file = maildir / row.path
        if file.is_symlink() or not file.resolve().is_relative_to(root) or not file.is_file():
            continue
        if file.stat().st_size > MAX_FILE_BYTES:
            continue
        email = parse_email(file.read_bytes())
        held = reader.execute(
            "SELECT 1 FROM doc_access WHERE doc_id = %s AND principal = %s",
            [email.id, f"user:{row.user}"],
        ).fetchone()
        if held is None or email.id in taken:
            continue
        taken.add(email.id)
        questions.append(
            Question(
                doc_id=email.id,
                user=row.user,
                question=row.question,
                gold_answer=row.gold_answer,
                email=email.body,
            )
        )
    return Sample(questions=questions, rows_read=read)


def evaluate(
    questions: Sequence[Question],
    pool: Pool,
    workers: int,
    model: BaseChatModel,
    spending: Spending,
    jev: Jev,
    raw: TextIO,
    done: Sequence[Record] = (),
    screen: Screen | None = None,
) -> list[Record]:
    asked = {record.doc_id for record in done}
    jobs = [question for question in questions if question.doc_id not in asked]
    writing = threading.Lock()
    finished: list[tuple[int, Record]] = []

    def run(number: int, question: Question) -> None:
        record = _ask_once(question, pool, model, spending, jev, screen)
        with writing:
            finished.append((number, record))
            raw.write(json.dumps(asdict(record)) + "\n")
            raw.flush()
            if len(finished) % 50 == 0:
                _log.info(
                    "%d of %d asked; %.4f USD on the model, %.4f USD on the judge",
                    len(finished),
                    len(jobs),
                    spending.cost_usd,
                    jev.cost_usd,
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, number, job) for number, job in enumerate(jobs)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    return [record for _, record in sorted(finished, key=lambda pair: pair[0])]


def judge(question: Question, answer: str, jev: Jev) -> bool:
    state = {
        "email": question.email,
        "question": question.question,
        "gold_answer": question.gold_answer,
        "answer": answer,
    }
    return jev.noul(state, JUDGE_QUESTION) >= 0.5


def read_records(path: Path) -> list[Record]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Record(**json.loads(line)) for line in lines]


def summarise(records: Sequence[Record], sample: Sample, model_name: str) -> dict[str, Any]:
    retrieved = [record for record in records if record.retrieved]
    return {
        "model": model_name,
        "questions": len(records),
        "rows_read": sample.rows_read,
        "matched": len(sample.questions),
        "correct": sum(record.correct for record in records),
        "retrieved": len(retrieved),
        "correct_when_retrieved": sum(record.correct for record in retrieved),
        "refused": sum(record.refused for record in records),
        "exhausted": sum(record.exhausted for record in records),
        "withheld": sum(record.withheld > 0 for record in records),
        "input_tokens": sum(record.input_tokens for record in records),
        "output_tokens": sum(record.output_tokens for record in records),
        "cost_usd": round(sum(record.cost_usd for record in records), 6),
    }


def results_table(summary: dict[str, Any]) -> str:
    questions = summary["questions"]
    retrieved = summary["retrieved"]
    when_retrieved = (
        with_interval(summary["correct_when_retrieved"], retrieved) if retrieved else "n/a"
    )
    return "\n".join(
        [
            "# Answer table on Enron",
            "",
            f"Model {summary['model']}. {questions} EnronQA questions, each asked under row-level "
            "security by the custodian whose mailbox holds the email it was written from. "
            f"{summary['matched']} of the {summary['rows_read']} rows read matched a loaded email "
            "its custodian reads.",
            "",
            f"| Correct, judged by Jev (of {questions}) | Email reached the model (of {questions}) "
            f"| Correct when it did (of {retrieved}) | Refused | Out of steps | Screen withheld a "
            "passage | Cost (USD) |",
            "|---|---|---|---|---|---|---|",
            f"| {with_interval(summary['correct'], questions)} "
            f"| {with_interval(retrieved, questions)} | {when_retrieved} | {summary['refused']} "
            f"| {summary['exhausted']} | {summary['withheld']} | {summary['cost_usd']:.4f} |",
            "",
            "Counts come with their 95% Wilson score interval.",
            "Correct: TypeSafe's Jev decision model, shown the email, the question, the reference "
            f'answer and the answer, was asked "{JUDGE_QUESTION}" and gave yes a probability of at '
            "least 0.5. The paper uses Llama-3.1-70B-Instruct as this judge; how often Jev agrees "
            "with a person on this task is not measured. A refusal or an answer out of steps "
            "counts as wrong and is not judged.",
            "Email reached the model: a search result or an opened document shown to the model "
            "carries the email's id. Screen withheld a passage: answers where the injection screen "
            "kept at least one search result, opened email or query row from the model.",
            "",
            "Questions and reference answers are the first of each row of EnronQA "
            "(MichaelR207/enron_qa_0922, CC BY 4.0). We wrote only the judge question.",
            "",
        ]
    )


def write_results(out: Path, summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers_enron.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "answers_enron.md").write_text(results_table(summary), encoding="utf-8")


def _ask_once(
    question: Question,
    pool: Pool,
    model: BaseChatModel,
    spending: Spending,
    jev: Jev,
    screen: Screen | None,
) -> Record:
    toolset = employee_tools(pool, principals(question.user, []), screen)
    answer = ask(question.question, toolset, model, spending)
    answered = not answer.refused and not answer.exhausted
    label = f"[{question.doc_id}]"
    return Record(
        doc_id=question.doc_id,
        user=question.user,
        question=question.question,
        gold_answer=question.gold_answer,
        answer=answer.text,
        retrieved=any(label in text for text in answer.tool_outputs),
        correct=answered and judge(question, answer.text, jev),
        refused=answer.refused,
        exhausted=answer.exhausted,
        withheld=answer.withheld,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=answer.cost_usd,
    )


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Measure answers on EnronQA questions.")
    parser.add_argument("--enronqa", type=Path, required=True)
    parser.add_argument("--maildir", type=Path, required=True)
    parser.add_argument("--limit", type=_positive, default=None)
    parser.add_argument("--workers", type=_positive, default=1)
    parser.add_argument("--budget-usd", type=float, default=settings.budget_usd)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    raw_path = args.out / RAW_FILE
    if raw_path.exists() and not args.resume:
        raise SystemExit(
            f"{raw_path} already holds answers. Pass --resume to carry on from them, "
            "or choose another --out."
        )
    jev = jev_from_environment(args.budget_usd)
    if jev is None:
        raise SystemExit("the judge is Jev: set TYPESAFE_API_KEY")
    rows = read_rows(args.enronqa)
    provision = load_provision_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    with app_pool(settings, args.workers) as app:
        with connect_reader(settings, provision) as reader:
            sample = build_questions(reader, rows, args.maildir, args.limit)
        if not sample.questions:
            raise SystemExit("no EnronQA row matched a loaded email its custodian reads")
        sampled = {question.doc_id for question in sample.questions}
        done = []
        if raw_path.exists():
            done = [record for record in read_records(raw_path) if record.doc_id in sampled]
        spending = Spending(price_of(settings), args.budget_usd)
        model = chat_model(settings)
        with raw_path.open("a", encoding="utf-8") as raw:
            try:
                records = evaluate(
                    sample.questions,
                    app,
                    args.workers,
                    model,
                    spending,
                    jev,
                    raw,
                    done,
                    screen_from_settings(settings),
                )
            except BudgetExceeded as error:
                raise SystemExit(f"stopped: {error}. Rerun with --resume to carry on.") from error
            except JevError as error:
                raise SystemExit(f"stopped: {error}. Rerun with --resume to carry on.") from error

    summary = summarise(done + records, sample, settings.model)
    write_results(args.out, summary)
    print(results_table(summary))


if __name__ == "__main__":
    main()
