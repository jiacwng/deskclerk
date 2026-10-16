# What permissions cost on the Enron mailboxes. On the leak evaluation's sample of emails, the
# document search runs once as the custodian who holds each email, under row-level security, and
# once with no filter at all, after untimed warm-up searches; both report their time, whether the
# email comes back in the top results, and how many of the exact nearest chunks the index found.
# The secured searches are also split by how many emails the custodian can read, and the slowest
# get their query plan. A smaller set of the same questions then goes through the whole agent to
# time and price a full answer. A full run (NEEDTOKNOW_DB_NAME=needtoknow_full):
# PYTHONPATH=src .venv/bin/python -m needtoknow.speed_eval --out results/

import argparse
import json
import logging
import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from langchain_core.language_models import BaseChatModel
from psycopg import sql

from needtoknow.agent import SEARCH_RESULTS, ask
from needtoknow.auth import principals
from needtoknow.config import load_provision_settings, load_settings
from needtoknow.db import Pool, app_pool, as_user, connect_app, connect_reader
from needtoknow.embed import Vector, embed_query
from needtoknow.leak_eval import Fact, build_facts
from needtoknow.providers import BudgetExceeded, Spending, chat_model
from needtoknow.retrieval import Hit, nearest_chunks, prepare_nearest_chunks

WARMUP = 20
SLOWEST_PLANS = 5
GROUPS = 3
_VECTOR = re.compile(r"'\[[^\]]*\]'")
_log = logging.getLogger("needtoknow.speed_eval")


@dataclass(frozen=True)
class SearchTiming:
    seconds: float
    found: bool
    short: bool
    recall: float
    visible: int


@dataclass(frozen=True)
class AnswerTiming:
    seconds: float
    cost_usd: float


def time_searches(
    facts: Sequence[Fact], app: psycopg.Connection[Any], reader: psycopg.Connection[Any]
) -> tuple[list[SearchTiming], list[SearchTiming]]:
    visible = _visible_emails(reader)
    for fact in facts[:WARMUP]:
        embedding = embed_query(fact.question)
        _secured_search(app, fact, embedding)
        _unfiltered_search(reader, fact, embedding)

    secured_runs = []
    unfiltered_runs = []
    embeddings = []
    for number, fact in enumerate(facts):
        embedding = embed_query(fact.question)
        embeddings.append(embedding)
        # Alternating which search goes first keeps the second from always finding the index
        # pages the first one just read.
        if number % 2 == 0:
            secured_runs.append(_timed(app, fact, embedding, secured=True))
            unfiltered_runs.append(_timed(reader, fact, embedding, secured=False))
        else:
            unfiltered_runs.append(_timed(reader, fact, embedding, secured=False))
            secured_runs.append(_timed(app, fact, embedding, secured=True))

    # An exact search reads every chunk and pushes the index out of memory, so all of them run
    # after the timed searches.
    secured = []
    unfiltered = []
    for fact, embedding, secured_run, unfiltered_run in zip(
        facts, embeddings, secured_runs, unfiltered_runs, strict=True
    ):
        seen = visible.get(fact.allowed_user, 0)
        secured.append(_scored(app, fact, embedding, secured_run, seen, secured=True))
        unfiltered.append(_scored(reader, fact, embedding, unfiltered_run, seen, secured=False))
    return secured, unfiltered


def slowest_plans(
    facts: Sequence[Fact], secured: Sequence[SearchTiming], app: psycopg.Connection[Any]
) -> list[str]:
    ranked = sorted(zip(secured, facts, strict=True), key=lambda pair: -pair[0].seconds)
    plans = []
    for timing, fact in ranked[:SLOWEST_PLANS]:
        embedding = embed_query(fact.question)
        with as_user(app, principals(fact.allowed_user, [])):
            query, parameters = prepare_nearest_chunks(app, embedding, SEARCH_RESULTS)
            explain = sql.SQL("EXPLAIN (ANALYZE, BUFFERS) ") + query
            lines = [
                _VECTOR.sub("'[query embedding]'", row[0])
                for row in app.execute(explain, parameters).fetchall()
            ]
        header = (
            f"{fact.allowed_user}, who can read {timing.visible} emails, "
            f"{timing.seconds * 1000:.1f} ms when timed"
        )
        plans.append("\n".join([header, *lines]))
    return plans


def time_answers(
    facts: Sequence[Fact],
    pool: Pool,
    model: BaseChatModel,
    spending: Spending,
) -> list[AnswerTiming]:
    timings = []
    for number, fact in enumerate(facts, start=1):
        start = time.perf_counter()
        answer = ask(fact.question, pool, principals(fact.allowed_user, []), model, spending)
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
        "warmup": min(WARMUP, len(secured)),
        "searches": len(secured),
        "secured": _search_summary(secured),
        "unfiltered": _search_summary(unfiltered),
        "secured_by_visible": _by_visible(secured),
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
        "custodian who holds it, once under row-level security and once with no filter, after "
        f"{summary['warmup']} untimed warm-up searches of each. One search at a time, warm cache.",
        "",
        f"| Search | Median (ms) | 95th percentile (ms) | Email in the top {asked} (of "
        f"{searches}) | Fewer than {asked} results (of {searches}) | Recall of the exact top "
        f"{asked} |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (("Row-level security", "secured"), ("No filter", "unfiltered")):
        row = summary[key]
        lines.append(
            f"| {label} | {row['median_ms']} | {row['p95_ms']} | {row['found']} | {row['short']} "
            f"| {row['recall']} |"
        )
    lines += [
        "",
        "Recall of the exact top results: the share of the chunks an exact search returns, under "
        "the same permissions, that the index search also returned.",
        "",
        "Row-level security searches by how many emails the custodian can read:",
        "",
        "| Emails readable | Searches | Median (ms) | 95th percentile (ms) | Recall |",
        "|---|---|---|---|---|",
    ]
    for group in summary["secured_by_visible"]:
        lines.append(
            f"| {group['visible_min']:,} to {group['visible_max']:,} | {group['searches']} "
            f"| {group['median_ms']} | {group['p95_ms']} | {group['recall']} |"
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


def write_results(out: Path, summary: dict[str, Any], plans: Sequence[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "speed.md").write_text(results_table(summary), encoding="utf-8")
    (out / "speed_plans.txt").write_text("\n\n".join(plans) + "\n", encoding="utf-8")


def _timed(
    connection: psycopg.Connection[Any], fact: Fact, embedding: Vector, secured: bool
) -> tuple[float, list[Hit]]:
    start = time.perf_counter()
    hits = (
        _secured_search(connection, fact, embedding)
        if secured
        else _unfiltered_search(connection, fact, embedding)
    )
    return time.perf_counter() - start, hits


def _scored(
    connection: psycopg.Connection[Any],
    fact: Fact,
    embedding: Vector,
    run: tuple[float, list[Hit]],
    visible: int,
    secured: bool,
) -> SearchTiming:
    seconds, hits = run
    exact = _exact_search(connection, fact, embedding, secured)
    return SearchTiming(
        seconds=seconds,
        found=any(hit.doc_id == fact.doc_id for hit in hits),
        short=len(hits) < SEARCH_RESULTS,
        recall=_recall(hits, exact),
        visible=visible,
    )


def _secured_search(app: psycopg.Connection[Any], fact: Fact, embedding: Vector) -> list[Hit]:
    with as_user(app, principals(fact.allowed_user, [])):
        return nearest_chunks(app, embedding, SEARCH_RESULTS)


def _unfiltered_search(reader: psycopg.Connection[Any], fact: Fact, embedding: Vector) -> list[Hit]:
    with reader.transaction():
        return nearest_chunks(reader, embedding, SEARCH_RESULTS)


# With index scans off, PostgreSQL reads every permitted chunk and sorts by distance: the exact
# answer the HNSW index approximates.
def _exact_search(
    connection: psycopg.Connection[Any], fact: Fact, embedding: Vector, secured: bool
) -> list[Hit]:
    if secured:
        with as_user(connection, principals(fact.allowed_user, [])):
            connection.execute("SET LOCAL enable_indexscan = off")
            return nearest_chunks(connection, embedding, SEARCH_RESULTS)
    with connection.transaction():
        connection.execute("SET LOCAL enable_indexscan = off")
        return nearest_chunks(connection, embedding, SEARCH_RESULTS)


def _recall(hits: Sequence[Hit], exact: Sequence[Hit]) -> float:
    if not exact:
        return 1.0
    found = {(hit.doc_id, hit.text) for hit in hits}
    return sum((hit.doc_id, hit.text) in found for hit in exact) / len(exact)


def _visible_emails(reader: psycopg.Connection[Any]) -> dict[str, int]:
    rows = reader.execute(
        "SELECT principal, count(*) FROM doc_access WHERE doc_id LIKE 'enron-%' AND principal "
        "LIKE 'user:%' GROUP BY principal"
    ).fetchall()
    return {principal.removeprefix("user:"): count for principal, count in rows}


def _search_summary(timings: Sequence[SearchTiming]) -> dict[str, Any]:
    milliseconds = [timing.seconds * 1000 for timing in timings]
    return {
        "median_ms": _median(milliseconds),
        "p95_ms": _p95(milliseconds),
        "found": sum(timing.found for timing in timings),
        "short": sum(timing.short for timing in timings),
        "recall": _mean_recall(timings),
    }


def _by_visible(timings: Sequence[SearchTiming]) -> list[dict[str, Any]]:
    ranked = sorted(timings, key=lambda timing: timing.visible)
    size = -(-len(ranked) // GROUPS)
    groups = []
    for start in range(0, len(ranked), size):
        group = ranked[start : start + size]
        milliseconds = [timing.seconds * 1000 for timing in group]
        groups.append(
            {
                "visible_min": group[0].visible,
                "visible_max": group[-1].visible,
                "searches": len(group),
                "median_ms": _median(milliseconds),
                "p95_ms": _p95(milliseconds),
                "recall": _mean_recall(group),
            }
        )
    return groups


def _mean_recall(timings: Sequence[SearchTiming]) -> float | None:
    return round(statistics.mean(timing.recall for timing in timings), 3) if timings else None


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
    with (
        connect_app(settings) as app,
        connect_reader(settings, provision) as reader,
        app_pool(settings, size=1) as pool,
    ):
        facts = build_facts(reader, args.searches)
        if not facts:
            raise SystemExit("no Enron email has a distinctive token and a non-holder")
        secured, unfiltered = time_searches(facts, app, reader)
        plans = slowest_plans(facts, secured, app)
        write_results(args.out, summarise(secured, unfiltered, [], settings.model), plans)
        spending = Spending(settings.model, args.budget_usd)
        try:
            answers = time_answers(facts[: args.answers], pool, chat_model(settings), spending)
        except BudgetExceeded as error:
            raise SystemExit(f"stopped, only the searches written: {error}") from error

    summary = summarise(secured, unfiltered, answers, settings.model)
    write_results(args.out, summary, plans)
    print(results_table(summary))


if __name__ == "__main__":
    main()
