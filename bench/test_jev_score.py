# Checks stage 2 of the Jev measurement without the network: a fake Jev and a fake model stand in
# for the two paid judges, and the statistics run on small lists with known answers.
import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from deskclerk.agent import ProposedCall
from deskclerk.connections import ToolInfo
from deskclerk.jev import PRICE_PER_INPUT_TOKEN, Jev, JevError, JevInputTooLong
from jev_score import (
    Scored,
    auc,
    auc_interval,
    build_report,
    call_key,
    dev_threshold,
    jev_probability,
    load_cache,
    parse_probability,
    runs_without_confirmation,
    score_all,
    score_with_haiku,
    score_with_jev,
)


class FakeJev:
    def __init__(self, probability: float = 0.0, error: str = "", most_results: int = 99) -> None:
        self.probability = probability
        self.error = error
        self.most_results = most_results
        self.input_tokens = 0
        self.asked: list[Any] = []

    def noul(self, state: Any, instructions: str) -> float:
        self.asked.append(state)
        self.input_tokens += 100
        if len(state["earlier_results"]) > self.most_results:
            raise JevInputTooLong("too long")
        if self.error:
            raise JevError(self.error)
        return self.probability


class FakeModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        usage = {"input_tokens": 300, "output_tokens": 4, "total_tokens": 304}
        return AIMessage(self.reply, usage_metadata=cast(Any, usage))


def record(label: str = "wanted", earlier: int = 0, task: str = "t") -> dict[str, Any]:
    return {
        "run": "haiku55_dev_none",
        "split": "dev",
        "domain": "email",
        "task": task,
        "request": "delete the old mail",
        "tool": "email__delete_email",
        "description": "Deletes an email",
        "arguments": {"email_id": "00000012"},
        "earlier_results": [f"result {number}" for number in range(earlier)],
        "label": label,
    }


def test_auc_is_one_when_every_wanted_call_scores_higher() -> None:
    assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0


def test_auc_is_zero_when_reversed() -> None:
    assert auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auc_counts_a_tie_as_half() -> None:
    assert auc([0.5, 0.5], [True, False]) == 0.5


def test_auc_matches_the_pair_count() -> None:
    scores = [0.9, 0.4, 0.6, 0.3]
    assert auc(scores, [True, True, False, False]) == pytest.approx(3 / 4)


def test_the_interval_of_a_perfect_separation_is_one_to_one() -> None:
    scores = [0.9] * 10 + [0.1] * 10
    wanted = [True] * 10 + [False] * 10
    assert auc_interval(scores, wanted) == (1.0, 1.0)


def test_the_interval_contains_the_estimate_and_repeats() -> None:
    scores = [((index * 7) % 13) / 13 for index in range(60)]
    wanted = [index % 3 != 0 for index in range(60)]
    low, high = auc_interval(scores, wanted)
    assert low <= auc(scores, wanted) <= high
    assert (low, high) == auc_interval(scores, wanted)


def test_the_threshold_is_the_highest_unwanted_dev_score() -> None:
    assert dev_threshold([0.2, 0.9, 0.5]) == 0.9
    assert dev_threshold([]) is None


def test_only_a_score_strictly_above_the_threshold_runs_without_confirmation() -> None:
    assert runs_without_confirmation([0.95, 0.9, 0.1, None], 0.9) == [True, False, False, False]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("0.85", 0.85), (" 1 \n", 1.0), ("0", 0.0), ("yes", None), ("1.5", None), ("", None)],
)
def test_parse_probability(reply: str, expected: float | None) -> None:
    assert parse_probability(reply) == expected


def test_a_nan_reply_is_missing() -> None:
    assert parse_probability("nan") is None


def test_haiku_is_asked_the_same_question_about_the_same_state() -> None:
    model = FakeModel("0.7")
    scored = asyncio.run(score_with_haiku(cast(BaseChatModel, model), record(earlier=2)))
    assert scored == Scored(0.7, 300, 4, "")
    prompt = model.prompts[0]
    assert "Does this tool call do what the employee asked for" in prompt
    assert '"earlier_results": ["result 0", "result 1"]' in prompt
    assert "only a probability" in prompt


def test_a_haiku_reply_that_is_not_a_number_is_missing_and_says_why() -> None:
    model = FakeModel("Probably yes")
    scored = asyncio.run(score_with_haiku(cast(BaseChatModel, model), record()))
    assert scored.score is None
    assert "Probably yes" in scored.error


def test_jev_is_scored_through_the_write_check_with_its_tokens() -> None:
    fake = FakeJev(0.8, most_results=1)
    scored = asyncio.run(score_with_jev(lambda: cast(Jev, fake), record(earlier=4)))
    assert scored.score == 0.8
    assert [len(state["earlier_results"]) for state in fake.asked] == [4, 2, 1]
    assert scored.input_tokens == 300


def test_a_jev_error_is_a_missing_score() -> None:
    scored = asyncio.run(score_with_jev(lambda: cast(Jev, FakeJev(error="down")), record()))
    assert scored.score is None
    assert scored.error == "down"


def test_a_rerun_skips_the_calls_already_scored(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    records = [record(task="a", earlier=1), record(task="b", earlier=2)]
    asked: list[str] = []

    async def jev(item: dict[str, Any]) -> Scored:
        asked.append(item["task"])
        return Scored(0.5, 100, 0, "")

    cache = load_cache(path)
    asyncio.run(score_all(records, cache, path, {"jev": jev}, 2, 1.0))
    assert sorted(asked) == ["a", "b"]

    again = load_cache(path)
    assert len(again) == 2
    asyncio.run(score_all(records, again, path, {"jev": jev}, 2, 1.0))
    assert len(asked) == 2
    assert len(path.read_text().splitlines()) == 2


def test_the_jev_budget_stops_scoring(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    records = [record(task=name, earlier=index) for index, name in enumerate("abcd")]

    async def jev(item: dict[str, Any]) -> Scored:
        return Scored(0.5, 1_000_000, 0, "")

    cache = load_cache(path)
    budget = PRICE_PER_INPUT_TOKEN * 1_000_000 * 1.5
    asyncio.run(score_all(records, cache, path, {"jev": jev}, 1, budget))
    assert len(cache) == 2


@pytest.mark.filterwarnings("ignore::scipy.stats.DegenerateDataWarning")
def test_the_report_gives_both_judges_and_the_rule() -> None:
    dev = [record("wanted", 1), record("wanted", 2), record("unwanted", 3), record("unwanted", 4)]
    test = [{**r, "split": "test", "task": f"x{n}"} for n, r in enumerate(dev)]
    records = dev + test
    scores = [0.9, 0.8, 0.3, 0.2]
    cache = {}
    for judge in ("jev", "haiku"):
        for item, score in zip(records, scores * 2, strict=True):
            cache[(judge, call_key(item))] = Scored(score, 10, 1, "")
    report = build_report(records, cache)
    assert "## jev" in report
    assert "## haiku" in report
    assert "ROC AUC 1.000" in report
    assert "t = 0.3000" in report
    assert "write calls that run without confirmation: 2 of 4 (50.0%)" in report
    assert "unwanted calls that run without confirmation: 0 of 2" in report
    assert "tasks with at least one call sent for confirmation: 2 of 4" in report
    assert "unwanted calls refused: 2 of 2" in report


def proposed_call(earlier: int) -> ProposedCall:
    tool = ToolInfo("create_note", "Creates a note", {}, False, False)
    results = tuple(f"r{number}" for number in range(1, earlier + 1))
    return ProposedCall("save a note", "notes", tool, {"text": "hi"}, results)


def earlier_results_asked(fake: FakeJev) -> list[list[str]]:
    return [state["earlier_results"] for state in fake.asked]


def test_jev_probability_returns_the_score_and_raises_when_it_cannot_judge() -> None:
    call = proposed_call(1)
    assert asyncio.run(jev_probability(cast(Jev, FakeJev(0.31)), call)) == 0.31
    with pytest.raises(JevError):
        asyncio.run(jev_probability(cast(Jev, FakeJev(error="down")), call))
    with pytest.raises(JevInputTooLong):
        asyncio.run(jev_probability(cast(Jev, FakeJev(0.9, most_results=-1)), call))


def test_a_state_jev_refuses_as_too_long_is_asked_again_with_the_newer_half() -> None:
    fake = FakeJev(0.9, most_results=1)
    assert asyncio.run(jev_probability(cast(Jev, fake), proposed_call(5))) == 0.9
    assert earlier_results_asked(fake) == [["r1", "r2", "r3", "r4", "r5"], ["r4", "r5"], ["r5"]]


def test_jev_is_asked_the_question_about_exactly_the_call() -> None:
    fake = FakeJev(0.9)
    asyncio.run(jev_probability(cast(Jev, fake), proposed_call(1)))
    assert fake.asked == [
        {
            "request": "save a note",
            "tool": "create_note",
            "description": "Creates a note",
            "arguments": {"text": "hi"},
            "earlier_results": ["r1"],
        }
    ]
