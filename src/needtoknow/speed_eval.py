# What permissions cost on the Enron mailboxes. On the leak evaluation's sample of emails, the
# document search runs once as the custodian who holds each email, under row-level security, and
# once with no filter at all; both report their time, whether the email comes back in the top
# results, and how often fewer results come back than were asked for. A smaller set of the same
# questions then goes through the whole agent to time and price a full answer. A full run
# (NEEDTOKNOW_DB_NAME=needtoknow_full):
# PYTHONPATH=src .venv/bin/python -m needtoknow.speed_eval --out results/

import argparse
import json
import logging
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from langchain_core.language_models import BaseChatModel

from needtoknow.agent import SEARCH_RESULTS, ask
from needtoknow.auth import principals
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.db import as_user, connect_app, connect_reader
from needtoknow.embed import Vector, embed_query
from needtoknow.leak_eval import Fact, build_facts
from needtoknow.providers import BudgetExceeded, Spending, chat_model
from needtoknow.retrieval import Hit, nearest_chunks

_log = logging.getLogger("needtoknow.speed_eval")


@dataclass(frozen=True)
class SearchTiming:
    seconds: float
    found: bool
    short: bool


@dataclass(frozen=True)
class AnswerTiming:
    seconds: float
    cost_usd: float


def time_searches(
    facts: Sequence[Fact], app: psycopg.Connection[Any], reader: psycopg.Connection[Any]
) -> tuple[list[SearchTiming], list[SearchTiming]]:
    secured = []
    unfiltered = []
    for number, fact in enumerate(facts):
        embedding = embed_query(fact.question)
        # Alternating which search goes first keeps the second from always finding the index
        # pages the first one just read.
        if number % 2 == 0:
            secured.append(_secured_search(app, fact, embedding))
            unfiltered.append(_unfiltered_search(reader, fact, embedding))
        else:
            unfiltered.append(_unfiltered_search(reader, fact, embedding))
            secured.append(_secured_search(app, fact, embedding))
    return secured, unfiltered


def time_answers(
    facts: Sequence[Fact],
    app: psycopg.Connection[Any],
    model: BaseChatModel,
    spending: Spending,
) -> list[AnswerTiming]:
    timings = []
    for number, fact in enumerate(facts, start=1):
        start = time.perf_counter()
        answer = ask(fact.question, app, principals(fact.allowed_user, []), model, spending)
        timings.append(AnswerTiming(time.perf_counter() - start, answer.cost_usd))
        _log.info("answer %d of %d; %.4f USD spent", number, len(facts), spending.cost_usd)
    return timings


def summarise(
    secured: Sequence[SearchTiming],
    unfiltered: Sequence[SearchTiming],
    answers: Sequence[AnswerTiming],
    model_name: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "results_asked": SEARCH_RESULTS,
        "searches": len(secured),
        "secured": _search_summary(secured),
        "unfiltered": _search_summary(unfiltered),
        "answers": len(answers),
        "answer_median_seconds": _median([timing.seconds for timing in answers]),
        "answer_p95_seconds": _p95([timing.seconds for timing in answers]),
        "answer_median_cost_usd": _median([timing.cost_usd for timing in answers], 6),
        "answer_total_cost_usd": round(sum(timing.cost_usd for timing in answers), 6),
    }


def results_table(summary: dict[str, Any]) -> str:
    searches = summary["searches"]
    asked = summary["results_asked"]
    lines = [
        "# Speed, cost and search quality on Enron",
        "",
        f"{searches} emails from the leak sample. Each email's subject is searched for as the "
        "custodian who holds it, once under row-level security and once with no filter.",
        "",
        f"| Search | Median (ms) | 95th percentile (ms) | Email in the top {asked} (of "
        f"{searches}) | Fewer than {asked} results (of {searches}) |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("Row-level security", "secured"), ("No filter", "unfiltered")):
        row = summary[key]
        lines.append(
            f"| {label} | {row['median_ms']} | {row['p95_ms']} | {row['found']} | {row['short']} |"
        )
    lines += [
        "",
        f"Full answers through the agent, model {summary['model']}, as the holder: "
        f"{summary['answers']} questions, median {summary['answer_median_seconds']} s, 95th "
        f"percentile {summary['answer_p95_seconds']} s, median cost "
        f"{summary['answer_median_cost_usd']} USD, total {summary['answer_total_cost_usd']} USD.",
        "",
    ]
    return "\n".join(lines)


def write_results(out: Path, summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "speed.md").write_text(results_table(summary), encoding="utf-8")


def _secured_search(app: psycopg.Connection[Any], fact: Fact, embedding: Vector) -> SearchTiming:
    start = time.perf_counter()
    with as_user(app, principals(fact.allowed_user, [])):
        hits = nearest_chunks(app, embedding, SEARCH_RESULTS)
    return _timing(start, hits, fact.doc_id)


def _unfiltered_search(
    reader: psycopg.Connection[Any], fact: Fact, embedding: Vector
) -> SearchTiming:
    start = time.perf_counter()
    with reader.transaction():
        hits = nearest_chunks(reader, embedding, SEARCH_RESULTS)
    return _timing(start, hits, fact.doc_id)


def _timing(start: float, hits: Sequence[Hit], doc_id: str) -> SearchTiming:
    seconds = time.perf_counter() - start
    found = any(hit.doc_id == doc_id for hit in hits)
    return SearchTiming(seconds, found, len(hits) < SEARCH_RESULTS)


def _search_summary(timings: Sequence[SearchTiming]) -> dict[str, Any]:
    milliseconds = [timing.seconds * 1000 for timing in timings]
    return {
        "median_ms": _median(milliseconds),
        "p95_ms": _p95(milliseconds),
        "found": sum(timing.found for timing in timings),
        "short": sum(timing.short for timing in timings),
    }


def _median(values: Sequence[float], digits: int = 1) -> float | None:
    return round(statistics.median(values), digits) if values else None


def _p95(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return round(statistics.quantiles(values, n=20)[-1], 1)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Time and price search and answers on Enron.")
    parser.add_argument("--searches", type=_positive, default=200)
    parser.add_argument("--answers", type=_positive, default=50)
    parser.add_argument("--budget-usd", type=float, default=settings.budget_usd)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s")
    logging.getLogger("needtoknow").setLevel(logging.INFO)
    provision = load_provision_settings()
    with connect_app(settings) as app, connect_reader(settings, provision) as reader:
        facts = build_facts(reader, args.searches)
        if not facts:
            raise SystemExit("no Enron email has a distinctive token and a non-holder")
        secured, unfiltered = time_searches(facts, app, reader)
        spending = Spending(settings.model, args.budget_usd)
        try:
            answers = time_answers(facts[: args.answers], app, chat_model(settings), spending)
        except BudgetExceeded as error:
            raise SystemExit(f"stopped, nothing written: {error}") from error

    summary = summarise(secured, unfiltered, answers, settings.model)
    write_results(args.out, summary)
    print(results_table(summary))


if __name__ == "__main__":
    main()
