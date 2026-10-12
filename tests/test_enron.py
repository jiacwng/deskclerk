# The Enron loader on the slice in tests/data: one document per email however many copies the
# mailboxes hold, readable by exactly the custodians who hold one, every email embedded, and a
# load that stops part way carries on to the same rows. Row-level security on the emails too.

from pathlib import Path

import numpy as np
import psycopg
import pytest
from pgvector import Vector
from psycopg import sql
from psycopg.rows import TupleRow

from conftest import MAILDIR
from needtoknow.auth import principals
from needtoknow.db import as_user
from needtoknow.documents import CHUNK_CHARS, Document, split_into_chunks
from needtoknow.embed import DIMENSIONS
from needtoknow.enron import (
    MAX_FILE_BYTES,
    TITLE_CHARS,
    load_enron,
    mail_files,
    parse_email,
)

Connection = psycopg.Connection[TupleRow]

CUSTODIANS = ("dasovich-j", "kaminski-v", "kean-s")
EMAIL = (
    b"Message-ID: <1.1075.JavaMail.evans@thyme>\r\n"
    b"Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)\r\n"
    b"From: phillip.allen@enron.com\r\n"
    b"To: tim.belden@enron.com\r\n"
    b"Subject: \r\n"
    b"\tForecast\r\n"
    b"X-Folder: \\Phillip_Allen_Jan2002_1\\Allen, Phillip K.\\'Sent Mail\r\n"
    b"\r\n"
    b"Here is our forecast.  \r\n"
    b"\r\n"
    b"Phillip\r\n"
)


def _count(
    connection: Connection, query: str | sql.Composed, parameters: list[str] | None = None
) -> int:
    row = connection.execute(query, parameters).fetchone()
    assert row is not None
    return int(row[0])


def _bodies(document: Document) -> list[str]:
    prefix = len(document.title) + 2 if document.title else 0
    return [chunk.text[prefix:] for chunk in split_into_chunks(document)]


def test_every_chunk_fits_and_keeps_every_word_of_its_email() -> None:
    for _, path in mail_files(MAILDIR):
        email = parse_email(path.read_bytes())
        document = Document(id=email.id, title=email.subject, body=email.body, readers=frozenset())

        bodies = _bodies(document)

        assert all(len(chunk.text) <= CHUNK_CHARS for chunk in split_into_chunks(document))
        assert " ".join(bodies).split() == email.body.split()


def test_a_run_with_no_space_is_cut_to_fit() -> None:
    run = "x" * (CHUNK_CHARS * 3)
    document = Document(id="d", title="Title", body=run, readers=frozenset())

    bodies = _bodies(document)

    assert len(bodies) == 4
    assert "".join(bodies) == run


def test_an_email_is_its_subject_and_its_body_without_headers() -> None:
    email = parse_email(EMAIL)

    assert email.subject == "Forecast"
    assert email.body == "Here is our forecast.\n\nPhillip"
    assert email.id.startswith("enron-")


def test_copies_in_other_folders_and_mailboxes_are_one_email() -> None:
    other_copy = EMAIL.replace(b"<1.1075.", b"<2.2041.").replace(b"'Sent Mail", b"All documents")

    assert parse_email(other_copy).id == parse_email(EMAIL).id
    assert parse_email(EMAIL.replace(b"\r\n", b"\n")).id == parse_email(EMAIL).id
    assert parse_email(EMAIL.replace(b"forecast.", b"forecast!")).id != parse_email(EMAIL).id


def test_the_slice_holds_fifty_emails_five_of_them_in_two_mailboxes(
    emails: dict[str, Document],
) -> None:
    files = list(mail_files(MAILDIR))
    shared = [doc for doc in emails.values() if len(doc.readers) == 2]

    assert len(files) == 55
    assert len(emails) == 50
    assert len(shared) == 5
    for document in shared:
        assert document.readers == {"user:dasovich-j", "user:kean-s"}


def test_limit_reads_the_first_files_in_path_order() -> None:
    assert list(mail_files(MAILDIR, limit=3)) == list(mail_files(MAILDIR))[:3]


def test_the_id_is_the_whole_digest_and_a_long_subject_is_cut() -> None:
    email = parse_email(EMAIL.replace(b"\tForecast", b"\t" + b"Forecast " * 40))

    assert len(email.id) == len("enron-") + 64
    assert len(email.subject) == TITLE_CHARS


def test_a_file_over_the_size_limit_is_skipped_and_counted(
    owner: Connection, tmp_path: Path
) -> None:
    folder = tmp_path / "maildir" / "allen-p" / "inbox"
    folder.mkdir(parents=True)
    (folder / "1").write_bytes(EMAIL + b"x" * MAX_FILE_BYTES)

    loaded = load_enron(owner, tmp_path / "maildir")

    assert (loaded.files, loaded.skipped, loaded.emails, loaded.chunks) == (1, 1, 0, 0)


def test_the_database_holds_each_email_once_with_its_readers(
    owner: Connection, emails: dict[str, Document]
) -> None:
    rows = owner.execute(
        "SELECT doc_id, principal FROM doc_access WHERE doc_id LIKE 'enron-%'"
    ).fetchall()
    readers: dict[str, set[str]] = {}
    for doc_id, principal in rows:
        readers.setdefault(doc_id, set()).add(principal)

    assert readers == {doc_id: set(doc.readers) for doc_id, doc in emails.items()}


def test_every_email_has_its_embedded_chunks(
    owner: Connection, emails: dict[str, Document]
) -> None:
    rows = owner.execute(
        "SELECT doc_id, text, embedding FROM chunks WHERE doc_id LIKE 'enron-%' "
        "ORDER BY doc_id, position"
    ).fetchall()
    texts: dict[str, list[str]] = {}
    for doc_id, text, embedding in rows:
        texts.setdefault(doc_id, []).append(text)
        assert isinstance(embedding, Vector)
        assert embedding.dimensions() == DIMENSIONS
        assert np.isfinite(embedding.to_numpy()).all()

    assert texts == {
        doc_id: [chunk.text for chunk in split_into_chunks(doc)] for doc_id, doc in emails.items()
    }


def test_a_load_cut_short_carries_on_to_the_same_rows(owner: Connection) -> None:
    query = (
        "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM doc_access), "
        "(SELECT count(*) FROM chunks)"
    )
    before = owner.execute(query).fetchone()
    with owner.transaction():
        owner.execute("DELETE FROM chunks WHERE doc_id LIKE 'enron-%'")
        owner.execute("DELETE FROM doc_access WHERE doc_id LIKE 'enron-%'")
        owner.execute("DELETE FROM documents WHERE id LIKE 'enron-%'")

    cut = load_enron(owner, MAILDIR, limit=20)
    rest = load_enron(owner, MAILDIR)
    again = load_enron(owner, MAILDIR)

    assert cut.files == 20
    assert rest.files == 55
    assert cut.emails + rest.emails == 50
    assert again.emails == 0
    assert again.chunks == 0
    assert owner.execute(query).fetchone() == before


@pytest.mark.parametrize("custodian", CUSTODIANS)
def test_a_custodian_sees_exactly_the_emails_in_their_mailbox(
    app: Connection, emails: dict[str, Document], custodian: str
) -> None:
    expected = {doc_id for doc_id, doc in emails.items() if f"user:{custodian}" in doc.readers}
    with as_user(app, principals(custodian, [])):
        documents = {row[0] for row in app.execute("SELECT id FROM documents")}
        chunks = {row[0] for row in app.execute("SELECT doc_id FROM chunks")}

    assert {doc_id for doc_id in documents if doc_id.startswith("enron-")} == expected
    assert {doc_id for doc_id in chunks if doc_id.startswith("enron-")} == expected
    assert 15 <= len(expected) <= 20


def test_an_email_in_two_mailboxes_is_visible_to_both_and_no_one_else(
    app: Connection, emails: dict[str, Document]
) -> None:
    shared = next(doc_id for doc_id, doc in emails.items() if len(doc.readers) == 2)
    seen = {}
    for custodian in CUSTODIANS:
        with as_user(app, principals(custodian, [])):
            seen[custodian] = _count(app, "SELECT count(*) FROM documents WHERE id = %s", [shared])

    assert seen == {"dasovich-j": 1, "kaminski-v": 0, "kean-s": 1}


@pytest.mark.parametrize("table", ["documents", "doc_access", "chunks"])
def test_no_principals_see_no_email(app: Connection, owner: Connection, table: str) -> None:
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
    assert _count(owner, query) > 0
    assert _count(app, query) == 0
    with as_user(app, ["user:nobody"]):
        assert _count(app, query) == 0
