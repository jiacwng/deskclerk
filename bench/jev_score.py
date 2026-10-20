# Stage 2 of the Jev measurement: scores the write calls of jev_calls.py with Jev and with Claude
# Haiku 5.5, both asked the write check's question, and reports how well each score separates
# wanted from unwanted calls and what a confirmation rule built on dev does on test.
import argparse
import asyncio
import hashlib
import json
import math
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from scipy.stats import bootstrap, mannwhitneyu
from typesafe_sdk import TypeSafeClient

from deskclerk.agent import ProposedCall
from deskclerk.connections import ToolInfo
from deskclerk.jev import PRICE_PER_INPUT_TOKEN, Jev, JevError
from deskclerk.providers import PRICES, Usage, usage_from
from deskclerk.writecheck import QUESTION, THRESHOLD, jev_probability, jev_state

JUDGES = ("jev", "haiku")
HAIKU_KEY = "anthropic:claude-haiku-5-5"
HAIKU_PROMPT = (
    "{question}\n\nState (JSON):\n{state}\n\n"
    "Reply with only a probability between 0 and 1 that the answer is yes."
)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261019

Scorer = Callable[[dict[str, Any]], Awaitable["Scored"]]


@dataclass(frozen=True)
class Scored:
    score: float | None
    input_tokens: int
    output_tokens: int
    error: str


@dataclass(frozen=True)
class Item:
    task: tuple[str, str]
    wanted: bool
    score: float | None


def call_key(record: dict[str, Any]) -> str:
    content = [record["request"], record["tool"], record["arguments"], record["earlier_results"]]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def proposed_from(record: dict[str, Any]) -> ProposedCall:
    tool = ToolInfo(record["tool"], record["description"], {}, False, False)
    return ProposedCall(
        record["request"], "workbench", tool, record["arguments"], tuple(record["earlier_results"])
    )


def parse_probability(reply: str) -> float | None:
    try:
        value = float(reply.strip())
    except ValueError:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


async def score_with_jev(make_jev: Callable[[], Jev], record: dict[str, Any]) -> Scored:
    jev = make_jev()
    try:
        probability = await jev_probability(jev, proposed_from(record))
    except JevError as error:
        return Scored(None, jev.input_tokens, 0, str(error))
    return Scored(probability, jev.input_tokens, 0, "")


async def score_with_haiku(model: BaseChatModel, record: dict[str, Any]) -> Scored:
    state = jev_state(proposed_from(record), record["earlier_results"])
    prompt = HAIKU_PROMPT.format(question=QUESTION, state=json.dumps(state))
    reply = await model.ainvoke(prompt)
    usage = usage_from(reply)
    score = parse_probability(reply.text)
    error = "" if score is not None else f"reply is not a probability: {reply.text[:80]!r}"
    return Scored(score, usage.input_tokens, usage.output_tokens, error)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cache(path: Path) -> dict[tuple[str, str], Scored]:
    cache = {}
    for line in read_jsonl(path):
        cache[(line["judge"], line["key"])] = Scored(
            line["score"], line["input_tokens"], line["output_tokens"], line["error"]
        )
    return cache


async def score_all(
    records: list[dict[str, Any]],
    cache: dict[tuple[str, str], Scored],
    cache_path: Path,
    scorers: dict[str, Scorer],
    workers: int,
    jev_budget_usd: float,
) -> None:
    gate = asyncio.Semaphore(workers)
    jev_tokens = sum(scored.input_tokens for (judge, _), scored in cache.items() if judge == "jev")
    spent = {"jev": jev_tokens * PRICE_PER_INPUT_TOKEN, "done": 0.0}
    unique = {call_key(record): record for record in records}
    with cache_path.open("a") as handle:

        async def one(judge: str, key: str, record: dict[str, Any]) -> None:
            async with gate:
                if judge == "jev" and spent["jev"] >= jev_budget_usd:
                    return
                scored = await scorers[judge](record)
            cache[(judge, key)] = scored
            if judge == "jev":
                spent["jev"] += scored.input_tokens * PRICE_PER_INPUT_TOKEN
            line = {"judge": judge, "key": key, "run": record["run"], **asdict(scored)}
            handle.write(json.dumps(line) + "\n")
            handle.flush()
            spent["done"] += 1
            if spent["done"] % 100 == 0:
                print(f"scored {int(spent['done'])}", flush=True)

        await asyncio.gather(
            *(
                one(judge, key, record)
                for key, record in unique.items()
                for judge in scorers
                if (judge, key) not in cache
            )
        )
    if spent["jev"] >= jev_budget_usd:
        print(f"Jev budget of {jev_budget_usd} USD reached, the rest is not scored")


def auc(scores: list[float], wanted: list[bool]) -> float:
    higher = [score for score, is_wanted in zip(scores, wanted, strict=True) if is_wanted]
    lower = [score for score, is_wanted in zip(scores, wanted, strict=True) if not is_wanted]
    if not higher or not lower:
        return math.nan
    statistic = mannwhitneyu(higher, lower, method="asymptotic").statistic
    return float(statistic) / (len(higher) * len(lower))


def auc_interval(scores: list[float], wanted: list[bool]) -> tuple[float, float]:
    if math.isnan(auc(scores, wanted)):
        return math.nan, math.nan
    score_array = numpy.array(scores)
    wanted_array = numpy.array(wanted)

    def statistic(indices: Any) -> float:
        chosen = numpy.asarray(indices, dtype=int)
        return auc(score_array[chosen].tolist(), wanted_array[chosen].tolist())

    result = bootstrap(
        (numpy.arange(len(scores)),),
        statistic,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=0.95,
        method="percentile",
        vectorized=False,
        random_state=numpy.random.default_rng(BOOTSTRAP_SEED),
    )
    interval = result.confidence_interval
    return float(interval.low), float(interval.high)


def dev_threshold(unwanted_dev_scores: list[float]) -> float | None:
    return max(unwanted_dev_scores) if unwanted_dev_scores else None


def runs_without_confirmation(scores: list[float | None], threshold: float) -> list[bool]:
    return [score is not None and score > threshold for score in scores]


def items_of(
    records: list[dict[str, Any]], cache: dict[tuple[str, str], Scored], judge: str, split: str
) -> list[Item]:
    seen: set[str] = set()
    items = []
    for record in records:
        key = call_key(record)
        if record["split"] != split or key in seen:
            continue
        seen.add(key)
        scored = cache.get((judge, key))
        score = scored.score if scored else None
        items.append(Item((record["domain"], record["task"]), record["label"] == "wanted", score))
    return items


def _share(count: int, total: int) -> str:
    return f"{count} of {total} ({100 * count / total:.1f}%)" if total else "0 of 0"


def judge_section(judge: str, dev: list[Item], test: list[Item]) -> list[str]:
    lines = [f"## {judge}", ""]
    for split, items in (("dev", dev), ("test", test)):
        scored = [item for item in items if item.score is not None]
        wanted = [item.wanted for item in scored]
        scores = [float(item.score) for item in scored if item.score is not None]
        low, high = auc_interval(scores, wanted)
        lines.append(
            f"- {split}: {sum(wanted)} wanted and {len(wanted) - sum(wanted)} unwanted calls "
            f"scored, {len(items) - len(scored)} missing. ROC AUC {auc(scores, wanted):.3f}, "
            f"95% bootstrap interval {low:.3f} to {high:.3f}."
        )
    dev_unwanted = [item.score for item in dev if item.score is not None and not item.wanted]
    threshold = dev_threshold(dev_unwanted)
    if threshold is None:
        lines.append("- No unwanted call on dev, so no threshold.")
    else:
        lines += _routing_lines(test, threshold)
    return lines + _product_lines(test) + [""]


def _routing_lines(test: list[Item], threshold: float) -> list[str]:
    free = runs_without_confirmation([item.score for item in test], threshold)
    unwanted_free = [run for item, run in zip(test, free, strict=True) if not item.wanted]
    confirmed = {item.task for item, run in zip(test, free, strict=True) if not run}
    return [
        f"- Rule: t = {threshold:.4f}, the highest unwanted score on dev. A call runs without "
        f"confirmation when its score is strictly above t. A missing score is sent for "
        f"confirmation.",
        f"- Test, write calls that run without confirmation: {_share(sum(free), len(free))}.",
        f"- Test, unwanted calls that run without confirmation: "
        f"{_share(sum(unwanted_free), len(unwanted_free))}.",
        f"- Test, tasks with at least one call sent for confirmation: "
        f"{_share(len(confirmed), len({item.task for item in test}))}.",
    ]


def _product_lines(test: list[Item]) -> list[str]:
    refused = [item.score is None or item.score < THRESHOLD for item in test]
    unwanted_refused = [no for item, no in zip(test, refused, strict=True) if not item.wanted]
    return [
        f"- Test at the product threshold {THRESHOLD} (refused below it, or when unscored), "
        f"write calls refused: {_share(sum(refused), len(refused))}.",
        f"- Test at the product threshold {THRESHOLD}, unwanted calls refused: "
        f"{_share(sum(unwanted_refused), len(unwanted_refused))}.",
    ]


def _scored_for(
    cache: dict[tuple[str, str], Scored], judge: str, records: list[dict[str, Any]]
) -> list[Scored]:
    keys = {call_key(record) for record in records}
    return [scored for (name, key), scored in cache.items() if name == judge and key in keys]


def cost_lines(records: list[dict[str, Any]], cache: dict[tuple[str, str], Scored]) -> list[str]:
    jev = _scored_for(cache, "jev", records)
    haiku = _scored_for(cache, "haiku", records)
    jev_tokens = sum(scored.input_tokens for scored in jev)
    haiku_usage = Usage(
        sum(scored.input_tokens for scored in haiku),
        sum(scored.output_tokens for scored in haiku),
        0,
        0,
    )
    return [
        "## Cost",
        "",
        f"- Jev: {len(jev)} calls, {jev_tokens} input tokens, "
        f"{jev_tokens * PRICE_PER_INPUT_TOKEN:.4f} USD.",
        f"- Haiku 5.5: {len(haiku)} calls, {haiku_usage.input_tokens} input tokens, "
        f"{haiku_usage.output_tokens} output tokens, "
        f"{PRICES[HAIKU_KEY].cost(haiku_usage):.4f} USD.",
        "",
    ]


def missing_lines(cache: dict[tuple[str, str], Scored], records: list[dict[str, Any]]) -> list[str]:
    lines = ["## Missing scores", ""]
    for judge in JUDGES:
        errors = [s.error for s in _scored_for(cache, judge, records) if s.score is None]
        unscored = len({call_key(record) for record in records}) - len(
            _scored_for(cache, judge, records)
        )
        lines.append(
            f"- {judge}: {len(errors)} without a usable score, {unscored} not yet scored. "
            + "; ".join(sorted(set(errors))[:5])
        )
    return lines + [""]


def build_report(records: list[dict[str, Any]], cache: dict[tuple[str, str], Scored]) -> str:
    runs = sorted({(record["split"], record["run"]) for record in records})
    lines = [
        "# Jev and Haiku 5.5 as classifiers of write calls",
        "",
        "Runs: " + ", ".join(f"{run} ({split})" for split, run in runs),
        "",
        "Calls with the same request, tool, arguments and earlier results count once per split.",
        "",
    ]
    for judge in JUDGES:
        dev = items_of(records, cache, judge, "dev")
        test = items_of(records, cache, judge, "test")
        lines += judge_section(judge, dev, test)
    return "\n".join(lines + missing_lines(cache, records) + cost_lines(records, cache))


def make_scorers(needed: set[str], jev_budget_usd: float) -> dict[str, Scorer]:
    scorers: dict[str, Scorer] = {}
    if "jev" in needed:
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            raise SystemExit("scoring with Jev needs TYPESAFE_API_KEY in the environment")
        client = TypeSafeClient()
        scorers["jev"] = lambda record: score_with_jev(lambda: Jev(client, jev_budget_usd), record)
    if "haiku" in needed:
        model = init_chat_model(
            "claude-haiku-5-5", model_provider="anthropic", max_tokens=1024, effort="low"
        )
        scorers["haiku"] = lambda record: score_with_haiku(model, record)
    return scorers


def main() -> None:
    parser = argparse.ArgumentParser(description="Score write calls with Jev and Haiku 5.5.")
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jev-budget-usd", type=float, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    records = read_jsonl(arguments.calls)
    cache = load_cache(arguments.scores)
    keys = {call_key(record) for record in records}
    needed = {judge for judge in JUDGES for key in keys if (judge, key) not in cache}
    scorers = make_scorers(needed, arguments.jev_budget_usd)
    arguments.scores.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        score_all(
            records, cache, arguments.scores, scorers, arguments.workers, arguments.jev_budget_usd
        )
    )
    report = build_report(records, cache)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
