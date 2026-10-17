# A company's documents.jsonl against the real database and embedding model: a load adds,
# re-embeds, re-grants and deletes as the file says, and a reader taken off a document loses it
# in search and in open_document on the next load.

import json
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.auth import EVERYONE
from needtoknow.company import CompanyError, load_documents, read_documents
from needtoknow.retrieval import open_document, search

Connection = psycopg.Connection[TupleRow]
ALICE = "user:company-test-alice"
QUESTION = "When does the Lisbon office move to the new building?"
MOVE = {
    "id": "office-move",
    "title": "Lisbon office move",
    "body": "The Lisbon office moves to the new building on Rua Augusta in March.",
    "users": ["company-test-alice"],
    "groups": [],
    "public": False,
}


@pytest.fixture
def owner_clean(owner: Connection) -> Iterator[Connection]:
    yield owner
    with owner.transaction():
        owner.execute("DELETE FROM chunks WHERE starts_with(doc_id, 'company-')")
        owner.execute("DELETE FROM doc_access WHERE starts_with(doc_id, 'company-')")
        owner.execute("DELETE FROM documents WHERE starts_with(id, 'company-')")


def _write(path: Path, *entries: dict[str, object]) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def _found(app: Connection, principal: str) -> tuple[list[str], tuple[str, str] | None]:
    hits = search(app, [principal], QUESTION, 5)
    return [hit.doc_id for hit in hits], open_document(app, [principal], "company-office-move")


def test_a_reader_taken_off_a_document_no_longer_finds_it(
    owner_clean: Connection, app: Connection, tmp_path: Path
) -> None:
    load_documents(owner_clean, _write(tmp_path / "documents.jsonl", MOVE), delete_missing=False)

    assert _found(app, ALICE) == (["company-office-move"], (MOVE["title"], MOVE["body"]))

    revoked = MOVE | {"users": ["company-test-bob"]}
    loaded = load_documents(
        owner_clean, _write(tmp_path / "documents.jsonl", revoked), delete_missing=False
    )

    assert (loaded.written, loaded.regranted) == (0, 1)
    assert _found(app, ALICE) == ([], None)
    assert _found(app, "user:company-test-bob")[0] == ["company-office-move"]


def test_a_changed_body_is_split_and_embedded_again(
    owner_clean: Connection, tmp_path: Path
) -> None:
    path = tmp_path / "documents.jsonl"
    load_documents(owner_clean, _write(path, MOVE), delete_missing=False)
    edited = MOVE | {"body": "The move is postponed to June."}

    loaded = load_documents(owner_clean, _write(path, edited), delete_missing=False)
    again = load_documents(owner_clean, path, delete_missing=False)

    texts = owner_clean.execute(
        "SELECT text FROM chunks WHERE doc_id = 'company-office-move'"
    ).fetchall()
    assert (loaded.written, loaded.chunks) == (1, 1)
    assert texts == [("Lisbon office move\n\nThe move is postponed to June.",)]
    assert (again.written, again.regranted, again.chunks) == (0, 0, 0)


def test_documents_missing_from_the_file_are_deleted_only_when_asked(
    owner_clean: Connection, tmp_path: Path
) -> None:
    path = tmp_path / "documents.jsonl"
    other = MOVE | {"id": "parking", "title": "Parking", "body": "Parking is on level two."}
    load_documents(owner_clean, _write(path, MOVE, other), delete_missing=False)

    kept = load_documents(owner_clean, _write(path, MOVE), delete_missing=False)
    deleted = load_documents(owner_clean, path, delete_missing=True)

    ids = owner_clean.execute(
        "SELECT id FROM documents WHERE starts_with(id, 'company-') ORDER BY id"
    ).fetchall()
    leftovers = owner_clean.execute(
        "SELECT (SELECT count(*) FROM chunks WHERE doc_id = 'company-parking'), "
        "(SELECT count(*) FROM doc_access WHERE doc_id = 'company-parking')"
    ).fetchone()
    enron = owner_clean.execute(
        "SELECT count(*) FROM documents WHERE starts_with(id, 'enron-')"
    ).fetchone()
    assert (kept.deleted, deleted.deleted) == (0, 1)
    assert ids == [("company-office-move",)]
    assert leftovers == (0, 0)
    assert enron is not None and enron[0] > 0


def test_users_groups_and_public_become_principals(tmp_path: Path) -> None:
    entry = MOVE | {"users": ["alice"], "groups": ["finance"], "public": True}

    (document,) = read_documents(_write(tmp_path / "documents.jsonl", entry))

    assert document.id == "company-office-move"
    assert document.readers == frozenset({"user:alice", "group:finance", EVERYONE})


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        (MOVE | {"users": ["alice,bob"]}, "line 1: 1 validation error"),
        (MOVE | {"users": [""]}, "line 1: 1 validation error"),
        (MOVE | {"public": "yes"}, "line 1: 1 validation error"),
        (MOVE | {"owner": "alice"}, "line 1: 1 validation error"),
        ({"id": "office-move", "title": "Lisbon office move"}, "line 1: 1 validation error"),
        (MOVE | {"id": ""}, "line 1: 1 validation error"),
    ],
)
def test_a_malformed_line_is_refused_with_its_number(
    tmp_path: Path, entry: dict[str, object], error: str
) -> None:
    with pytest.raises(CompanyError, match=error):
        read_documents(_write(tmp_path / "documents.jsonl", entry))


def test_an_id_given_twice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CompanyError, match="line 2: id 'office-move' appears twice"):
        read_documents(_write(tmp_path / "documents.jsonl", MOVE, MOVE))
