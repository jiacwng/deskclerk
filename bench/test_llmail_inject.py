# Checks the LLMail-Inject measurement without the network: tiny dataset files under tmp_path,
# the sampling rule, the group labels and the report sections built from fixed scores.
import hashlib
import json
from pathlib import Path

from llmail_inject import (
    ATTACK,
    CHALLENGE_CLEAN,
    DOJO_CLEAN,
    build_outputs,
    category_of,
    grouped_items,
    label_lines,
    rate_lines,
    read_clean_emails,
    read_submissions,
    read_user_request,
    sample_submissions,
)
from screen_outputs import Cache, Output, Scored, output_key


def test_sample_takes_the_lowest_sha256_ranks_and_ignores_input_order() -> None:
    texts = [f"attack {number}" for number in range(20)]
    sample = sample_submissions(texts, 5)
    expected = sorted(texts, key=lambda text: hashlib.sha256(text.encode()).hexdigest())[:5]
    assert sample == expected
    assert sample_submissions(list(reversed(texts)), 5) == expected
    assert sample_submissions(texts, 50) == sorted(
        texts, key=lambda text: hashlib.sha256(text.encode()).hexdigest()
    )


def test_readers_parse_the_three_files(tmp_path: Path) -> None:
    labelled = {
        "Subject of the email: a. Body: b": {"attack_attempt": "True", "reason": "api_triggered"},
        "Subject of the email: c. Body: d": {
            "attack_attempt": "Unclear",
            "judge_answer": "x",
            "judge_category": "social engineering",
            "reason": "judge",
        },
    }
    (tmp_path / "labelled.json").write_text(json.dumps(labelled))
    (tmp_path / "clean.json").write_text(json.dumps(["z", "y", "z"]))
    (tmp_path / "scenarios.json").write_text(
        json.dumps({"scenario_1": {"user_query": "Summarize all following emails."}})
    )
    submissions = read_submissions(tmp_path / "labelled.json")
    assert [category_of(label) for label in submissions.values()] == [
        "api_triggered",
        "judge: Unclear",
    ]
    assert read_clean_emails(tmp_path / "clean.json") == ["y", "z"]
    assert read_user_request(tmp_path / "scenarios.json") == "Summarize all following emails."


def test_outputs_carry_their_group_and_a_text_in_two_groups_counts_once() -> None:
    dojo = [
        Output("pay the bill", "rent", False, True),
        Output("pay the bill", "rent", False, False),
        Output("pay the bill", "rent", False, False),
    ]
    outputs, groups = build_outputs("summarise", ["attack"], ["hello"], dojo)
    assert groups[output_key(Output("summarise", "attack", True, False))] == ATTACK
    assert groups[output_key(Output("summarise", "hello", False, True))] == CHALLENGE_CLEAN
    assert groups[output_key(dojo[0])] == DOJO_CLEAN
    items = grouped_items(outputs, groups)
    assert sorted(item.group for item in items) == sorted([ATTACK, CHALLENGE_CLEAN, DOJO_CLEAN])


def test_rates_and_label_split_come_from_the_cached_scores() -> None:
    outputs, groups = build_outputs("summarise", ["a1", "a2"], ["c1"], [])
    items = grouped_items(outputs, groups)
    keys = {item.group: [] for item in items}
    for item in items:
        keys[item.group].append(item.key)
    first, second = keys[ATTACK]
    cache: Cache = {}
    for screen in ("jev", "promptguard", "deberta"):
        cache[(screen, first)] = Scored(0.9, 0, "")
        cache[(screen, second)] = Scored(0.1, 0, "")
        cache[(screen, keys[CHALLENGE_CLEAN][0])] = Scored(0.1, 0, "")
    text = "\n".join(rate_lines(items, cache))
    assert "| challenge attacks | 50.0% (1 of 2)" in text
    assert "| challenge clean emails | 0.0% (0 of 1)" in text
    assert "| AgentDojo clean outputs | no outputs" in text
    categories = {first: "api_triggered", second: "judge: Unclear"}
    split = "\n".join(label_lines(items, cache, categories))
    assert "| api_triggered (1 sampled) | 100.0% (1 of 1)" in split
    assert "| judge: Unclear (1 sampled) | 0.0% (0 of 1)" in split
