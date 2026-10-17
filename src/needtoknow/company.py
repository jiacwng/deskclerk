# Loads a company's own documents, read from a documents.jsonl file of one JSON object per line
# in the shape of the Onyx and Microsoft Graph connectors (id, title, body, and the users and
# groups who may read it, or public for everyone) or from a folder of files. Each source is the
# truth for its id prefix: a load adds new documents, re-embeds changed ones, replaces each
# document's readers with the source's, and with delete_missing removes the documents under that
# prefix the source no longer lists.

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from needtoknow.auth import EVERYONE, EVERYONE_GROUP
from needtoknow.documents import Chunk, Document, embed_chunks, insert_documents
from needtoknow.embed import Vector

ID_PREFIX = "company-"
BATCH = 200

Principal = Annotated[str, Field(min_length=1, pattern=r"^[^,]+$")]


class CompanyError(ValueError):
    pass


class Readers(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    users: list[Principal] = []
    groups: list[Principal] = []
    public: bool = False

    @field_validator("groups")
    @classmethod
    def _no_everyone_group(cls, groups: list[str]) -> list[str]:
        if EVERYONE_GROUP in groups:
            raise ValueError(f"group {EVERYONE_GROUP!r} is reserved; set public to true instead")
        return groups

    def principals(self) -> frozenset[str]:
        readers = {f"user:{user}" for user in self.users}
        readers.update(f"group:{group}" for group in self.groups)
        if self.public:
            readers.add(EVERYONE)
        return frozenset(readers)


class DocumentLine(Readers):
    id: str = Field(min_length=1)
    title: str
    body: str


@dataclass(frozen=True)
class Loaded:
    documents: int
    written: int
    regranted: int
    deleted: int
    chunks: int


def read_documents(path: Path) -> list[Document]:
    documents: dict[str, Document] = {}
    with path.open(encoding="utf-8-sig") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entry = DocumentLine.model_validate_json(line)
            except ValidationError as error:
                raise CompanyError(f"{path.name} line {number}: {error}") from None
            if entry.id in documents:
                raise CompanyError(f"{path.name} line {number}: id {entry.id!r} appears twice")
            documents[entry.id] = Document(
                id=ID_PREFIX + entry.id,
                title=entry.title,
                body=entry.body,
                readers=entry.principals(),
            )
    return list(documents.values())


def load_documents(
    owner: psycopg.Connection[Any], documents: Sequence[Document], prefix: str, delete_missing: bool
) -> Loaded:
    stored = _stored(owner, prefix)
    rewrite = []
    regrant = []
    for document in documents:
        before = stored.get(document.id)
        if before is None or (before.title, before.body) != (document.title, document.body):
            rewrite.append(document)
        elif before.readers != document.readers:
            regrant.append(document)
    chunks = 0
    for start in range(0, len(rewrite), BATCH):
        batch = rewrite[start : start + BATCH]
        embedded: dict[str, list[tuple[Chunk, Vector]]] = {}
        for chunk, embedding in embed_chunks(batch):
            embedded.setdefault(chunk.doc_id, []).append((chunk, embedding))
        for document in batch:
            _rewrite(owner, document, embedded.get(document.id, []))
            chunks += len(embedded.get(document.id, []))
    for document in regrant:
        _regrant(owner, document)
    missing = []
    if delete_missing:
        listed = {document.id for document in documents}
        missing = sorted(set(stored) - listed)
        _delete(owner, missing)
    return Loaded(
        documents=len(documents),
        written=len(rewrite),
        regranted=len(regrant),
        deleted=len(missing),
        chunks=chunks,
    )


def _stored(owner: psycopg.Connection[Any], prefix: str) -> dict[str, Document]:
    readers: dict[str, set[str]] = {}
    for doc_id, principal in owner.execute(
        "SELECT doc_id, principal FROM doc_access WHERE starts_with(doc_id, %s)", [prefix]
    ):
        readers.setdefault(doc_id, set()).add(principal)
    stored = {}
    for doc_id, title, body in owner.execute(
        "SELECT id, title, body FROM documents WHERE starts_with(id, %s)", [prefix]
    ):
        stored[doc_id] = Document(
            id=doc_id, title=title, body=body, readers=frozenset(readers.get(doc_id, set()))
        )
    return stored


# Each document is replaced in its own transaction, so a search sees either its old text and
# readers or its new ones, never a mix of the two.
def _rewrite(
    owner: psycopg.Connection[Any], document: Document, embedded: Sequence[tuple[Chunk, Vector]]
) -> None:
    with owner.transaction(), owner.cursor() as cursor:
        _delete_rows(cursor, [document.id])
        insert_documents(cursor, [document], embedded)


def _regrant(owner: psycopg.Connection[Any], document: Document) -> None:
    with owner.transaction(), owner.cursor() as cursor:
        cursor.execute("DELETE FROM doc_access WHERE doc_id = %s", [document.id])
        cursor.executemany(
            "INSERT INTO doc_access (doc_id, principal) VALUES (%s, %s)",
            [(document.id, reader) for reader in document.readers],
        )


def _delete(owner: psycopg.Connection[Any], doc_ids: list[str]) -> None:
    with owner.transaction(), owner.cursor() as cursor:
        _delete_rows(cursor, doc_ids)


def _delete_rows(cursor: psycopg.Cursor[Any], doc_ids: list[str]) -> None:
    cursor.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", [doc_ids])
    cursor.execute("DELETE FROM doc_access WHERE doc_id = ANY(%s)", [doc_ids])
    cursor.execute("DELETE FROM documents WHERE id = ANY(%s)", [doc_ids])
