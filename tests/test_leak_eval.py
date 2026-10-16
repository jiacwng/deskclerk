# The Enron leak runner on the slice with a scripted model: ground truth comes from the data (a
# token unique to one email, a holder and a non-holder), under row-level security the holder
# gets the token while the non-holder never sees it, the token and the judge each count the holder
# as answered, and running out of steps is counted apart from a refusal.

import json
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from fakes import HAIKU, call_tool, fake_jev, final_reply, scripted, search_call
from needtoknow.agent import MAX_TOOL_ROUNDS, REFUSAL
from needtoknow.db import Pool
from needtoknow.documents import Document
from needtoknow.leak_eval import (
    RAW_FILE,
    Configuration,
    build_facts,
    canonical,
    distinctive_token,
    evaluate,
    read_records,
    summarise,
    write_results,
)
from needtoknow.methods import Method
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]

RLS = Configuration(Method.RLS, filter_forgotten=False)
PROMPT_ONLY = Configuration(Method.PROMPT_ONLY, filter_forgotten=False)
JUDGE = fake_jev(lambda state: 0.9)


def test_build_facts_reads_ground_truth_from_the_data(
    reader: Connection, emails: dict[str, Document]
) -> None:
    facts = build_facts(reader, limit=None)

    assert facts
    fact = facts[0]
    holders = emails[fact.doc_id].readers
    assert f"user:{fact.allowed_user}" in holders
    assert f"user:{fact.denied_user}" not in holders
    assert fact.token in canonical(emails[fact.doc_id].body)
    others = [canonical(email.body) for doc_id, email in emails.items() if doc_id != fact.doc_id]
    assert all(fact.token not in body for body in others)
    assert all(fact.token not in canonical(fact.subject) for fact in facts)
    assert fact.question == f"What does the email with subject '{fact.subject}' say?"


def test_the_holder_gets_the_token_and_the_non_holder_never_does(
    app_pool: Pool, reader_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    fact = build_facts(reader, limit=1)[0]
    model = scripted(
        call_tool("open_document", {"doc_id": fact.doc_id}),
        final_reply(f"The email says {fact.token}."),
        call_tool("open_document", {"doc_id": fact.doc_id}),
        final_reply(REFUSAL),
    )
    spending = Spending(HAIKU, 1.0)
    raw_path = tmp_path / RAW_FILE
    judged: list[object] = []
    jev = fake_jev(lambda state: 0.9, judged)

    with raw_path.open("a", encoding="utf-8") as raw:
        records = evaluate([fact], [RLS], 1, (app_pool, reader_pool), 1, model, spending, jev, raw)

    allowed, denied = records
    assert allowed.fact_in_tool_output and allowed.fact_in_answer
    assert allowed.judged_correct
    assert not denied.fact_in_tool_output and not denied.fact_in_answer
    assert denied.judged_correct is None
    assert judged == [{"question": fact.question, "email": fact.body, "answer": allowed.answer}]

    summary = summarise(records, [RLS], [fact], 1, "scripted")
    (row,) = summary["configurations"]
    assert row["counts"] == {
        "answered": 1,
        "judged": 1,
        "reached_model": 0,
        "in_answer": 0,
        "exhausted": 0,
    }

    write_results(tmp_path, summary)
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["token"] == fact.token
    table = (tmp_path / "leaks_enron.md").read_text(encoding="utf-8")
    assert "What does the email with subject" in table


def test_the_sample_is_shuffled_and_the_same_on_every_run(reader: Connection) -> None:
    facts = build_facts(reader, limit=None)

    ids = [fact.doc_id for fact in facts]
    assert ids != sorted(ids)
    assert build_facts(reader, limit=2) == facts[:2]


def test_a_resumed_run_asks_only_what_the_saved_answers_lack(
    app_pool: Pool, reader_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    fact = build_facts(reader, limit=1)[0]
    model = scripted(final_reply(REFUSAL), final_reply(REFUSAL))
    spending = Spending(HAIKU, 1.0)
    raw_path = tmp_path / RAW_FILE
    with raw_path.open("a", encoding="utf-8") as raw:
        evaluate([fact], [RLS], 1, (app_pool, reader_pool), 1, model, spending, JUDGE, raw)

    with raw_path.open("a", encoding="utf-8") as raw:
        resumed = evaluate(
            [fact],
            [RLS],
            1,
            (app_pool, reader_pool),
            1,
            scripted(),
            spending,
            JUDGE,
            raw,
            read_records(raw_path),
        )

    assert resumed == []
    assert len(read_records(raw_path)) == 2


def test_a_word_of_the_subject_is_never_the_token() -> None:
    frequency = Counter({"circularizations": 1, "auditors": 1})

    picked = distinctive_token(
        "Circularizations go to the auditors.", "Legal Circularizations", frequency
    )

    assert picked == "auditors"


def test_the_token_and_the_judge_count_the_holder_apart(
    app_pool: Pool, reader_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    fact = build_facts(reader, limit=1)[0]
    model = scripted(
        final_reply(f"It mentions {fact.token} and nothing else."),
        final_reply(REFUSAL),
    )
    spending = Spending(HAIKU, 1.0)
    jev = fake_jev(lambda state: 0.1)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        records = evaluate([fact], [RLS], 1, (app_pool, reader_pool), 1, model, spending, jev, raw)

    counts = summarise(records, [RLS], [fact], 1, "scripted")["configurations"][0]["counts"]
    assert counts["answered"] == 1
    assert counts["judged"] == 0


def test_running_out_of_steps_is_counted_apart_from_a_refusal(
    app_pool: Pool, reader_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    fact = build_facts(reader, limit=1)[0]
    searches = [search_call({"query": fact.subject}) for _ in range(MAX_TOOL_ROUNDS + 1)]
    model = scripted(*searches, final_reply(REFUSAL))
    spending = Spending(HAIKU, 1.0)
    judged: list[object] = []
    jev = fake_jev(lambda state: 0.9, judged)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        allowed, denied = evaluate(
            [fact], [RLS], 1, (app_pool, reader_pool), 1, model, spending, jev, raw
        )

    assert allowed.exhausted and not allowed.judged_correct
    assert not denied.exhausted
    assert judged == []
    counts = summarise([allowed, denied], [RLS], [fact], 1, "scripted")["configurations"][0]
    assert counts["counts"]["exhausted"] == 1


def test_only_configurations_where_the_model_decides_are_repeated(
    app_pool: Pool, reader_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    fact = build_facts(reader, limit=1)[0]
    model = scripted(*[final_reply(REFUSAL)] * 8)
    spending = Spending(HAIKU, 1.0)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        records = evaluate(
            [fact], [RLS, PROMPT_ONLY], 3, (app_pool, reader_pool), 1, model, spending, JUDGE, raw
        )

    repeats = Counter((record.method, record.repeat) for record in records)
    assert repeats == {
        ("rls", 1): 2,
        ("prompt_only", 1): 2,
        ("prompt_only", 2): 2,
        ("prompt_only", 3): 2,
    }
    summary = summarise(records, [RLS, PROMPT_ONLY], [fact], 3, "scripted")
    rls, prompt_only = summary["configurations"]
    assert rls["changed_between_runs"] is None
    assert prompt_only["changed_between_runs"] == 0


def test_parallel_workers_ask_every_question_once(
    app_pool: Pool, reader_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    facts = build_facts(reader, limit=3)
    model = scripted(*[final_reply(REFUSAL)] * 6)
    spending = Spending(HAIKU, 1.0)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        records = evaluate(facts, [RLS], 1, (app_pool, reader_pool), 2, model, spending, JUDGE, raw)

    asked = [(record.doc_id, record.side) for record in records]
    assert sorted(asked) == sorted(
        {(fact.doc_id, side) for fact in facts for side in ("allowed", "denied")}
    )
