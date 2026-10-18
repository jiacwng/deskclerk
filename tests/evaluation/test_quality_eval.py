# The EnronQA answer runner on the slice with a scripted model and a scripted judge. No EnronQA
# row names a file of the slice, so the rows here point at slice files with questions and
# answers that are labelled test inputs: a row matches only a loaded email its custodian reads,
# the judge sees the email beside the reference answer, a refusal is wrong without a judge call,
# and a resumed run asks nothing twice.

from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import TupleRow

from conftest import MAILDIR
from fakes import HAIKU, call_tool, fake_jev, final_reply, scripted
from needtoknow.agent import REFUSAL
from needtoknow.db import Pool
from needtoknow.documents import Document
from needtoknow.enron import mail_files, parse_email
from needtoknow.evaluation.quality_eval import (
    RAW_FILE,
    Row,
    Sample,
    build_questions,
    evaluate,
    read_records,
    read_rows,
    results_table,
    summarise,
    write_results,
)
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]

QUESTION = "test question"
GOLD = "test reference answer"
JUDGE = fake_jev(lambda state: 0.9)


def _slice_rows(count: int) -> list[Row]:
    rows = []
    seen: set[str] = set()
    for custodian, path in mail_files(MAILDIR):
        doc_id = parse_email(path.read_bytes()).id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        relative = path.relative_to(MAILDIR).as_posix()
        rows.append(Row(path=relative, user=custodian, question=QUESTION, gold_answer=GOLD))
        if len(rows) == count:
            break
    return rows


def test_read_rows_takes_the_first_question_and_skips_rows_without_one(tmp_path: Path) -> None:
    table = pa.table(
        {
            "path": ["kean-s/aadepartment/1", "kean-s/aadepartment/2"],
            "user": ["kean-s", "kean-s"],
            "questions": [[QUESTION, "second test question"], []],
            "gold_answers": [[GOLD, "second test answer"], []],
        }
    )
    path = tmp_path / "rows.parquet"
    pq.write_table(table, path)

    assert read_rows(path) == [
        Row(path="kean-s/aadepartment/1", user="kean-s", question=QUESTION, gold_answer=GOLD)
    ]


def test_a_row_matches_only_a_loaded_email_its_custodian_reads(
    reader: Connection, emails: dict[str, Document]
) -> None:
    (held,) = _slice_rows(1)
    doc_id = parse_email((MAILDIR / held.path).read_bytes()).id
    outsider = next(
        user
        for user in ("dasovich-j", "kaminski-v", "kean-s")
        if f"user:{user}" not in emails[doc_id].readers
    )
    rows = [
        held,
        Row(path=held.path, user=held.user, question=QUESTION, gold_answer=GOLD),
        Row(path=held.path, user=outsider, question=QUESTION, gold_answer=GOLD),
        Row(path="kean-s/no-such-folder/1", user="kean-s", question=QUESTION, gold_answer=GOLD),
        Row(path="../../../../pyproject.toml", user="kean-s", question=QUESTION, gold_answer=GOLD),
    ]

    sample = build_questions(reader, rows, MAILDIR, limit=None)

    (question,) = sample.questions
    assert question.doc_id == doc_id
    assert question.user == held.user
    assert question.email == emails[doc_id].body
    assert sample.rows_read == len(rows)


def test_the_sample_is_shuffled_and_the_same_on_every_run(reader: Connection) -> None:
    rows = _slice_rows(10)

    sample = build_questions(reader, rows, MAILDIR, limit=None)

    paths = {parse_email((MAILDIR / row.path).read_bytes()).id: row.path for row in rows}
    order = [paths[question.doc_id] for question in sample.questions]
    assert order != [row.path for row in rows]
    again = build_questions(reader, rows, MAILDIR, limit=3)
    assert again.questions == sample.questions[:3]
    assert again.rows_read == 3


def test_the_judge_sees_the_email_and_the_reference_answer(
    app_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    sample = build_questions(reader, _slice_rows(1), MAILDIR, limit=None)
    (question,) = sample.questions
    model = scripted(
        call_tool("open_document", {"doc_id": question.doc_id}),
        final_reply("A test answer."),
    )
    judged: list[object] = []
    jev = fake_jev(lambda state: 0.9, judged)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        (record,) = evaluate(sample.questions, app_pool, 1, model, Spending(HAIKU, 1.0), jev, raw)

    assert record.retrieved and record.correct
    assert judged == [
        {
            "email": question.email,
            "question": QUESTION,
            "gold_answer": GOLD,
            "answer": record.answer,
        }
    ]
    summary = summarise([record], sample, "scripted")
    assert summary["correct"] == 1
    assert summary["retrieved"] == 1
    assert summary["correct_when_retrieved"] == 1
    write_results(tmp_path, summary)
    table = (tmp_path / "answers_enron.md").read_text(encoding="utf-8")
    assert table == results_table(summary)
    assert "1 of the 1 rows read matched" in table


def test_a_judged_mismatch_counts_as_wrong(
    app_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    sample = build_questions(reader, _slice_rows(1), MAILDIR, limit=None)
    model = scripted(final_reply("A test answer."))
    jev = fake_jev(lambda state: 0.1)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        (record,) = evaluate(sample.questions, app_pool, 1, model, Spending(HAIKU, 1.0), jev, raw)

    assert not record.correct
    assert not record.retrieved


def test_a_refusal_is_wrong_and_never_judged(
    app_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    sample = build_questions(reader, _slice_rows(1), MAILDIR, limit=None)
    judged: list[object] = []
    jev = fake_jev(lambda state: 0.9, judged)

    with (tmp_path / RAW_FILE).open("a", encoding="utf-8") as raw:
        (record,) = evaluate(
            sample.questions,
            app_pool,
            1,
            scripted(final_reply(REFUSAL)),
            Spending(HAIKU, 1.0),
            jev,
            raw,
        )

    assert record.refused and not record.correct
    assert judged == []


def test_a_resumed_run_asks_only_what_the_saved_answers_lack(
    app_pool: Pool, reader: Connection, tmp_path: Path
) -> None:
    sample = build_questions(reader, _slice_rows(3), MAILDIR, limit=None)
    first = Sample(questions=sample.questions[:2], rows_read=2)
    raw_path = tmp_path / RAW_FILE
    model = scripted(final_reply(REFUSAL), final_reply(REFUSAL))
    with raw_path.open("a", encoding="utf-8") as raw:
        evaluate(first.questions, app_pool, 2, model, Spending(HAIKU, 1.0), JUDGE, raw)

    with raw_path.open("a", encoding="utf-8") as raw:
        resumed = evaluate(
            sample.questions,
            app_pool,
            1,
            scripted(final_reply(REFUSAL)),
            Spending(HAIKU, 1.0),
            JUDGE,
            raw,
            read_records(raw_path),
        )

    assert [record.doc_id for record in resumed] == [sample.questions[2].doc_id]
    assert len(read_records(raw_path)) == 3
